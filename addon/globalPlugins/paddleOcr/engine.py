# Paddle OCR add-on for NVDA
# Copyright (C) 2026 fdrv161
# SPDX-License-Identifier: GPL-2.0-or-later

"""OCR engine on PaddleOCR models (DB detector + CTC recognizer) via onnxruntime.

Imports nothing from NVDA, so it can be tested with plain Python.
Input is a BGR image (numpy, uint8, h×w×3). Output is the screen lines; each line is
a list of words with coordinates in pixels of the same image (NVDA LinesWordsResult format).
"""

import itertools
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Detector post-processing settings from the inference.yml of the PP-OCRv6 models,
# except the box threshold: 0.45 here instead of 0.4 there.
DET_THRESH = 0.2
DET_BOX_THRESH = 0.45
DET_UNCLIP_RATIO = 1.4
DET_MIN_SIZE = 3
# Images with a longer side above this are downscaled for the detector.
# Downscaling a screen hurts (small fonts vanish), so the limit is high: only for 8K and the like.
DET_MAX_SIDE = 4000
# Small images (a button, a label) get a margin of this many pixels for the detector: it often misses text
# that touches the image edge. Measured on lines cut from screenshots: empty results 10–25% → ~1%.
# Larger images are left as they are: there a margin only shuffles the result.
DET_PAD = 16
DET_PAD_BELOW = 100
# The detector's working memory grows with the image area (~180 MB for a 1080p screen, ~680 MB for 4K),
# so tall images go through it in horizontal strips of about DET_STRIP rows. Each strip also takes
# DET_OVERLAP rows of its neighbours on both sides, so a line up to twice that tall is whole in at least
# one strip (see _join_strips). Measured on 1080p: detector peak 120 → 32 MB, time +6%;
# an overlap of 64 costs +18% with the same accuracy.
DET_STRIP = 384
DET_OVERLAP = 32
REC_HEIGHT = 48
# Measured: batches of 4–8 are fastest; larger ones are slower (more width padding).
REC_BATCH = 4
# Memory of one run grows with batch × width (and × dictionary size, 18k characters in the Latin mode,
# without a slim head: tools/slim_head.py). So a batch is at most REC_BATCH_WIDTH pixels wide in total
# (lines of the 48 px input; a full-width 1080p line is ~4500): long lines go two or one at a time.
# Measured on a full-width console: peak 430 → 290 MB, the Cyrillic mode 18% faster (less padding),
# other screens the same. Lines wider than REC_MAX_RATIO heights (not text: separators) are squeezed
# to that width.
REC_BATCH_WIDTH = 6000
REC_MAX_RATIO = 320
# Lines the recognizer is unsure about (usually icons and borders) are dropped.
REC_MIN_SCORE = 0.5
# Pieces at the same height separated by more than this many line heights are separate lines (columns).
LINE_GAP = 10

_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DET_SCALE = 1.0 / (255.0 * _DET_STD)
_DET_SHIFT = _DET_MEAN / _DET_STD


def _load_charset(ymlPath):
	"""Reads the character dictionary from inference.yml without a yaml library.

	The dictionary is a list of lines like "  - 'a'" after "character_dict:".
	"""
	chars = []
	inDict = False
	for line in Path(ymlPath).read_text(encoding="utf-8").splitlines():
		if line.strip() == "character_dict:":
			inDict = True
			continue
		if inDict:
			# ASCII spaces only: strip() would eat space characters from the dictionary itself (e.g. U+3000).
			s = line.lstrip(" ")
			if not s.startswith("- "):
				break
			chars.append(_yaml_scalar(s[2:]))
	if not chars:
		raise ValueError(f"No character_dict in {ymlPath}")
	return chars


def _yaml_scalar(s):
	# Characters in the yml may be in single quotes ('' inside means one quote),
	# in double quotes (with escape sequences like "\\") or unquoted.
	if len(s) >= 2 and s[0] == s[-1] == "'":
		return s[1:-1].replace("''", "'")
	if len(s) >= 2 and s[0] == s[-1] == '"':
		return s[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")
	return s


def _session(path, threads, oneShot=False):
	opts = ort.SessionOptions()
	opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
	if threads:
		opts.intra_op_num_threads = threads
	if oneShot:
		# For a model that runs once per recognition (the detector). Without the memory arena its large
		# buffers go back to the system right away (lower peak too). Without spinning its idle threads
		# do not take CPU from the recognizer that runs next: measured 2–4% faster, ~20% less CPU.
		opts.enable_cpu_mem_arena = False
		opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
	opts.log_severity_level = 3
	return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _run_options():
	# Return working memory to the system after each run,
	# otherwise the peak of a large screen stays until exit.
	opts = ort.RunOptions()
	opts.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
	return opts


class OcrEngine:
	"""Detector + recognizer. allowed(c) tells which characters the recognizer may output (None means all)."""

	def __init__(self, detPath, recPath, recYml, threads=0, allowed=None):
		self._det = _session(detPath, threads, oneShot=True)
		self._detInput = self._det.get_inputs()[0].name
		self._rec = Reader(recPath, recYml, threads, allowed)

	def recognize(self, bgr):
		"""Returns a list of lines; a line is a list of words {x, y, width, height, text}."""
		h, w = bgr.shape[:2]
		if min(h, w) < DET_PAD_BELOW:
			p = DET_PAD
			boxes = self._detect(_add_margin(bgr, p))
			# Back to the coordinates of the original image; the recognizer reads lines from it.
			boxes = [
				(max(0, x0 - p), max(0, y0 - p), min(w, x1 - p), min(h, y1 - p)) for x0, y0, x1, y1 in boxes
			]
			boxes = [b for b in boxes if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
		else:
			boxes = self._detect(bgr)
		if not boxes:
			return []
		items = [words for words, score in self._rec.read(bgr, boxes) if words and score >= REC_MIN_SCORE]
		_pick_twin_script(items)
		return _group_lines(items)

	# --- detector ---

	def _detect(self, bgr):
		h, w = bgr.shape[:2]
		scale = min(1.0, DET_MAX_SIDE / max(h, w))
		# The model needs sides that are multiples of 32.
		rh = max(32, round(h * scale / 32) * 32)
		rw = max(32, round(w * scale / 32) * 32)
		img = cv2.resize(bgr, (rw, rh)) if (rw, rh) != (w, h) else bgr
		sy = h / rh
		parts = []
		for top, bottom, s0, s1 in _strips(rh):
			# (pixel/255 - mean)/std per channel straight into the 1×3×h×w layout: no intermediate float
			# copies of the whole image, which is noticeable on a 1080p screen.
			x = np.empty((1, 3, s1 - s0, rw), dtype=np.float32)
			for c in range(3):
				np.multiply(img[s0:s1, :, c], _DET_SCALE[c], out=x[0, c], dtype=np.float32)
				x[0, c] -= _DET_SHIFT[c]
			pred = self._det.run(None, {self._detInput: x})[0][0, 0]
			del x
			parts.append((top * sy, bottom * sy, s0 * sy, s1 * sy, _db_boxes(pred, w / rw, sy, w, h, s0)))
		return _join_strips(parts, DET_OVERLAP / 2 * sy)


class Reader:
	"""Line recognizer (CTC) of one model."""

	def __init__(self, recPath, recYml, threads=0, allowed=None):
		self._rec = _session(recPath, threads)
		inputs = self._rec.get_inputs()
		self._recInput = inputs[0].name
		# A model with a slim head (tools/slim_head.py) returns only the columns given in its other inputs.
		slim = len(inputs) > 1
		# Index 0 is the CTC blank; the models also know a space at the end.
		self._chars = ["", *_load_charset(recYml), " "]
		classes = self._rec.get_outputs()[0].shape[-1]
		if slim:
			meta = self._rec.get_modelmeta().custom_metadata_map
			classes = int(meta["classes"])
		if isinstance(classes, int) and classes != len(self._chars):
			raise ValueError(f"Dictionary ({len(self._chars)}) does not match the model ({classes})")
		# Banned characters (allowed(c) is false) are never chosen when decoding.
		# Keep only the allowed columns: the Latin mode keeps ~760 of 18.7k, so decoding
		# is many times cheaper, and the chosen character is the same as with banned ones zeroed.
		# A slim head returns only these columns itself, so the full 18.7k-wide output never exists.
		self._keep = None
		if allowed:
			self._keep = np.array([i for i, c in enumerate(self._chars) if i == 0 or allowed(c)])
			self._chars = [self._chars[i] for i in self._keep]
		# For a slim head: the kept columns split among its inputs keep0, keep1, ... (one per slice
		# of the dictionary), numbered from the slice start. The output then has them in order.
		self._keepFeeds = {}
		if slim:
			# The model inputs are int64; numpy's default int is not on every platform and version.
			keep = (self._keep if self._keep is not None else np.arange(classes)).astype(np.int64)
			step = int(meta["slice"])
			self._keepFeeds = {
				f"keep{i}": keep[(keep >= start) & (keep < start + step)] - start
				for i, start in enumerate(range(0, classes, step))
			}
			if set(self._keepFeeds) != {inp.name for inp in inputs[1:]}:
				raise ValueError(f"Unknown slim head inputs in {recPath}")
		# Indices of Latin and Cyrillic letters in the dictionary, for fixing mixed-script words.
		self._latin = np.array([i for i, c in enumerate(self._chars) if _script(c) == "latin"])
		self._cyr = np.array([i for i, c in enumerate(self._chars) if _script(c) == "cyr"])
		# Return memory to the system only after the last batch of lines: after each batch,
		# onnxruntime would allocate it again for the next one, and a screen has ~25 batches.
		self._runOpts = ort.RunOptions()
		self._lastRunOpts = _run_options()

	def read(self, bgr, boxes):
		"""For each box: (words, confidence); there may be no words."""
		crops = [bgr[y0:y1, x0:x1] for (x0, y0, x1, y1) in boxes]
		# Sort by aspect ratio so a batch holds lines of similar length (less empty padding).
		ratio = [min(c.shape[1] / c.shape[0], REC_MAX_RATIO) for c in crops]
		order = sorted(range(len(crops)), key=lambda i: ratio[i])
		batches = []
		for i in order:
			# The lines go from narrow to wide, so the new line sets the batch width.
			if (
				not batches
				or len(batches[-1]) == REC_BATCH
				or (len(batches[-1]) + 1) * math.ceil(REC_HEIGHT * ratio[i]) > REC_BATCH_WIDTH
			):
				batches.append([i])
			else:
				batches[-1].append(i)
		results = [None] * len(crops)
		# Widest batches first. The last run returns memory to the system but keeps its own output,
		# which is then the narrowest one, so the larger outputs of wide batches do not stay
		# until the next recognition (with an original 18k-class head they reach ~100 MB).
		for n, idx in reversed(list(enumerate(batches))):
			batchW = max(math.ceil(REC_HEIGHT * max(ratio[i] for i in idx)), REC_HEIGHT)
			batch = np.zeros((len(idx), 3, REC_HEIGHT, batchW), dtype=np.float32)
			widths = []
			for j, i in enumerate(idx):
				ch, cw = crops[i].shape[:2]
				rw = min(batchW, max(1, math.ceil(REC_HEIGHT * cw / ch)))
				img = cv2.resize(crops[i], (rw, REC_HEIGHT)).astype(np.float32)
				batch[j, :, :, :rw] = ((img / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
				widths.append(rw)
			last = n == 0
			feeds = {self._recInput: batch, **self._keepFeeds}
			probs = self._rec.run(None, feeds, self._lastRunOpts if last else self._runOpts)[0]
			if self._keep is not None and not self._keepFeeds:
				probs = probs[:, :, self._keep]
			for j, i in enumerate(idx):
				results[i] = self._ctc_words(probs[j], batchW, widths[j], boxes[i])
		return results

	def _ctc_words(self, prob, batchW, resizedW, box):
		"""CTC decoding + word coordinates from the positions of characters in the line."""
		ids = prob.argmax(axis=1)
		conf = prob.max(axis=1)
		steps = len(ids)
		# One CTC step covers batchW/steps pixels of the 48 px high image;
		# convert to pixels of the original line.
		x0, y0, x1, y1 = box
		toBox = (batchW / steps) * ((x1 - x0) / resizedW)
		chars = []  # (character, step)
		prev = 0
		for t in range(steps):
			c = int(ids[t])
			if c != 0 and c != prev:
				chars.append((self._chars[c], t))
			prev = c
		if not chars:
			return [], 0.0
		score = float(np.mean([conf[t] for _char, t in chars]))
		words = []
		cur = []
		for ch, t in chars + [(" ", steps)]:
			if ch == " ":
				if cur:
					cur = self._fix_mixed_script(cur, prob)
					wx0 = x0 + int(cur[0][1] * toBox)
					wx1 = min(x1, x0 + math.ceil((cur[-1][1] + 1) * toBox))
					word = {
						"x": wx0,
						"y": y0,
						"width": max(1, wx1 - wx0),
						"height": y1 - y0,
						"text": _fix_case(_fix_il("".join(c for c, _t in cur))),
					}
					alt = self._twin_spelling(cur, prob)
					if alt:
						word["alt"] = _fix_case(_fix_il(alt))
					words.append(word)
					cur = []
			else:
				cur.append((ch, t))
		return words, score

	def _twin_spelling(self, word, prob):
		"""A word made only of twin letters ("Сору", "Bce") → the same word in the other script, else None.

		Which script to keep is decided by _pick_twin_script from the neighbouring words.
		"""
		if not len(self._latin) or not len(self._cyr):
			return None
		letters = [c for c, _t in word if _script(c)]
		scripts = {_script(c) for c in letters}
		if len(scripts) != 1 or not all(c in _TWINS for c in letters):
			return None
		idx = self._latin if scripts == {"cyr"} else self._cyr
		return "".join(self._chars[int(idx[prob[t, idx].argmax()])] if _script(c) else c for c, t in word)

	def _fix_mixed_script(self, word, prob):
		"""Removes twin letters of the wrong script: "Itет" → "Item", "Windоws" → "Windows".

		The Cyrillic model knows both Latin and Cyrillic, and with lookalike letters (е/e, а/a, т/m)
		it sometimes picks the wrong script. If a word has letters of both scripts, pick the main one
		by the sum of probabilities at the letter positions and replace each foreign letter
		with the best letter of the main script.
		"""
		if not len(self._latin) or not len(self._cyr):
			return word
		# Look at each run of consecutive letters separately: in "Автотюнчик-1727.wav" Cyrillic and Latin
		# legitimately sit in different runs; only a run with mixed scripts needs fixing.
		fixed = list(word)
		i = 0
		while i < len(fixed):
			if _script(fixed[i][0]) is None:
				i += 1
				continue
			j = i
			while j < len(fixed) and _script(fixed[j][0]) is not None:
				j += 1
			run = fixed[i:j]
			if {"latin", "cyr"} <= {_script(c) for c, _t in run}:
				steps = [t for _char, t in run]
				latinMass = prob[steps][:, self._latin].sum()
				cyrMass = prob[steps][:, self._cyr].sum()
				main, mainIdx = ("latin", self._latin) if latinMass >= cyrMass else ("cyr", self._cyr)
				for k in range(i, j):
					c, t = fixed[k]
					if _script(c) != main:
						fixed[k] = (self._chars[int(mainIdx[prob[t, mainIdx].argmax()])], t)
			i = j
		return fixed


def _fix_il(text):
	"""Capital I and lowercase l look the same in sans-serif fonts; the model mixes them up: "AlI", "AII".

	Only clear cases are fixed:
	- I after a lowercase letter and not before another letter is l ("AlI" → "All", "FiII" → "Fill";
		"AddIn" and "userID" are left alone: a letter follows the I);
	- a capital followed only by I/l, 3+ letters in total, means lowercase l ("AII" → "All");
	- in an all-caps word (2+ capitals) a lowercase l is I ("TlGHT" → "TIGHT", "MlDl" → "MIDI").
	"""
	# Roman numerals ("XII", "VIII") are left alone; a stray l in them ("XlI") is fixed by the next rule.
	roman = set(text.replace("l", "I")) <= set("IVXLCDM")
	if (
		len(text) >= 3
		and text[0].isupper()
		and text[0] not in "Il"
		and set(text[1:]) <= {"I", "l"}
		and not roman
	):
		return text[0] + "l" * (len(text) - 1)
	letters = [c for c in text if c.isalpha()]
	if (
		"l" in letters
		and sum(c.isupper() for c in letters) >= 2
		and all(c.isupper() or c == "l" for c in letters)
	):
		return text.replace("l", "I")
	# An all-caps word with one stray lowercase letter ("tIGHT") is left alone: _fix_case will fix it.
	if "I" not in text or sum(c.isupper() and c != "I" for c in letters) >= 2:
		return text
	chars = list(text)
	# Left to right: in "FiII" the first I becomes l, and then the second one also follows a lowercase letter.
	for i in range(1, len(chars)):
		nxt = chars[i + 1] if i + 1 < len(chars) else ""
		if (
			chars[i] == "I"
			and chars[i - 1].islower()
			and _script(chars[i - 1]) == "latin"
			# Before another capital the I is real: "userID", "getID".
			and (not nxt.isalpha() or nxt == "I")
		):
			chars[i] = "l"
	return "".join(chars)


def _fix_case(text):
	"""'DIREcTorY' → 'DIRECTORY': in small all-caps labels the model mixes up the case.

	Fixed only if more than half of the letters are capitals AND a capital follows a lowercase letter again,
	so normal "iPhone", "GitHub", "IDs", "macOS" are left alone.
	"""
	letters = [c for c in text if c.isalpha()]
	if len(letters) < 4:
		return text
	upper = sum(c.isupper() for c in letters)
	if upper * 2 <= len(letters) or upper == len(letters):
		return text
	if not any(a.islower() and b.isupper() for a, b in itertools.pairwise(letters)):
		return text
	return text.upper()


# Letters that look the same in Cyrillic and Latin (the model also confuses т and m).
_TWINS = set("АВЕКМНОРСТХУаеорсхуктABEKMHOPCTXYaeopcxykm")


def _word_script(text):
	"""The script of a word if it is clearly visible: at least one letter is not a twin. Else None."""
	for c in text:
		s = _script(c)
		if s and c not in _TWINS:
			return s
	return None


def _pick_twin_script(items):
	"""Words made only of twin letters take the script of their line neighbours, or, with no neighbours,
	the script of most words on the screen: "Сору Paste" → "Copy Paste", "Bce программы" → "Все программы".
	"""

	def majority(words):
		counts = {"latin": 0, "cyr": 0}
		for w in words:
			s = _word_script(w["text"])
			if s:
				counts[s] += 1
		if counts["latin"] == counts["cyr"]:
			return None
		return max(counts, key=counts.get)

	screen = majority(w for ws in items for w in ws)
	for ws in items:
		target = majority(ws) or screen
		for w in ws:
			alt = w.pop("alt", None)
			# alt is written in the other script than text; take it if the neighbours use that script.
			if alt and target and _word_script(w["text"]) is None and _script_of_letters(alt) == target:
				w["text"] = alt


def _script_of_letters(text):
	"""The script of the first letter (in a word of twins only, all letters share one script)."""
	return next((_script(c) for c in text if _script(c)), None)


def latinOnly(c):
	"""Latin, digits, punctuation and arrows.

	CJK and the rest are banned: v6 confidently reads icons as "平".
	"""
	o = ord(c)
	return o < 0x250 or 0x2000 <= o < 0x2200


# Russian, Ukrainian and Belarusian letters; Bulgarian uses the Russian ones.
_EAST_SLAVIC = set("АБВГҐДЕЁЄЖЗИІЇЙКЛМНОПРСТУЎФХЦЧШЩЪЫЬЭЮЯабвгґдеёєжзиіїйклмнопрстуўфхцчшщъыьэюя")


def _is_cyrillic(o):
	# Main block with its supplement, and the two blocks of historic letters (Extended-A and -B).
	return 0x400 <= o <= 0x52F or 0x2DE0 <= o <= 0x2DFF or 0xA640 <= o <= 0xA69F


def anyCyrillic(c):
	"""Cyrillic of all modern languages, Latin and symbols.

	Historic letters (Old Church Slavonic: "Ѣ", "Ѫ", combining letters) are banned: no screen uses them,
	and the model could read icons as them.
	"""
	o = ord(c)
	return not (_is_cyrillic(o) and (o >= 0x2DE0 or 0x460 <= o <= 0x489))


def eastSlavicOnly(c):
	"""Russian, Ukrainian, Belarusian and Bulgarian letters, Latin and symbols.

	Letters of other Cyrillic alphabets are banned: without this the model sometimes puts
	a lookalike into Russian text, for example Kazakh "һ" instead of "н" or Serbian "Ј".
	"""
	return not _is_cyrillic(ord(c)) or c in _EAST_SLAVIC


def bgraToBgr(buf, width, height):
	"""NVDA screenshot (RGBQUAD bytes: blue, green, red, unused) → BGR image for the engine."""
	return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()


def _script(c):
	"""The script of a character: 'latin', 'cyr' or None (digits, punctuation and others)."""
	if not c.isalpha():
		return None
	o = ord(c)
	if 0x400 <= o <= 0x52F:
		return "cyr"
	if o < 0x250:
		return "latin"
	return None


def _add_margin(bgr, pad):
	"""The image with a margin in its background colour (the median of the edge pixels)."""
	edge = np.concatenate([bgr[0], bgr[-1], bgr[:, 0], bgr[:, -1]])
	color = [int(v) for v in np.median(edge, axis=0)]
	return cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=color)


def _strips(rows):
	"""Strips of the detector input: (own top, own bottom, top, bottom) in rows, multiples of 32.

	The own parts cover the image without gaps; a strip runs DET_OVERLAP rows past them on both sides.
	"""
	n = max(1, round(rows / DET_STRIP))
	cuts = [round(k * rows / n / 32) * 32 for k in range(n)] + [rows]
	return [(a, b, max(0, a - DET_OVERLAP), min(rows, b + DET_OVERLAP)) for a, b in itertools.pairwise(cuts)]


def _join_strips(parts, margin):
	"""Boxes of the detector strips → one list with each line once.

	parts: (own top, own bottom, top, bottom, boxes) per strip, in image pixels. A line near a cut is found
	by both strips, and they may disagree: one of them cuts the line at its edge, splits it into other
	pieces or puts its centre a pixel apart (the detector looks at the whole strip). So the centre alone
	could give a line twice or not at all. Boxes of neighbouring strips that cover each other by half
	on both axes are one line; of such a group the boxes of the strip that cuts it least are kept.
	A box without a match is kept only from the strip that owns its centre.
	"""
	found = []  # (box, strip, own centre)
	for k, (ownTop, ownBottom, _top, _bottom, boxes) in enumerate(parts):
		for box in boxes:
			cy = (box[1] + box[3]) / 2
			# A centre a little past the own part is a candidate: the neighbour strip may confirm it.
			if ownTop - margin <= cy < ownBottom + margin:
				found.append((box, k, ownTop <= cy < ownBottom))
	# Groups of the same line (union-find over the boxes).
	group = list(range(len(found)))

	def root(i):
		while group[i] != i:
			group[i] = group[group[i]]
			i = group[i]
		return i

	for k in range(len(parts) - 1):
		cut = parts[k][1]
		# Only boxes that reach into the overlap around the cut can be the same line.
		above = [i for i, f in enumerate(found) if f[1] == k and f[0][3] > cut - 2 * margin]
		below = [i for i, f in enumerate(found) if f[1] == k + 1 and f[0][1] < cut + 2 * margin]
		for i in above:
			ax0, ay0, ax1, ay1 = found[i][0]
			for j in below:
				bx0, by0, bx1, by1 = found[j][0]
				dx = min(ax1, bx1) - max(ax0, bx0)
				dy = min(ay1, by1) - max(ay0, by0)
				if dx >= 0.5 * min(ax1 - ax0, bx1 - bx0) and dy >= 0.5 * min(ay1 - ay0, by1 - by0):
					group[root(j)] = root(i)
	groups = {}
	for i in range(len(found)):
		groups.setdefault(root(i), []).append(found[i])
	boxes = []
	for members in groups.values():
		if len(members) == 1:
			box, _k, own = members[0]
			if own:
				boxes.append(box)
			continue
		y0 = min(box[1] for box, _k, _own in members)
		y1 = max(box[3] for box, _k, _own in members)
		cy = (y0 + y1) / 2
		# How many rows of the line each strip misses; on a tie, the strip that owns the centre wins.
		missed = {}
		for _box, k, _own in members:
			ownTop, ownBottom, top, bottom, _boxes = parts[k]
			missed[k] = (max(0, top - y0) + max(0, y1 - bottom), not ownTop <= cy < ownBottom)
		best = min(missed, key=missed.get)
		_ownTop, _ownBottom, top, bottom, _boxes = parts[best]
		for box, k, _own in members:
			if k != best:
				continue
			bx0, by0, bx1, by1 = box
			# A line taller than both strips can see (huge text) is cut in each of them: a box that
			# reaches the edge of its strip gets the rows the neighbour strip saw past that edge.
			if top > y0 and by0 <= top + margin:
				by0 = y0
			if bottom < y1 and by1 >= bottom - margin:
				by1 = y1
			boxes.append((bx0, by0, bx1, by1))
	return boxes


def _db_boxes(pred, sx, sy, w, h, top=0):
	"""Detector probability map → line rectangles (x0, y0, x1, y1) in pixels of the original image.

	top is the row of the map's first row in the detector input (for a strip).
	"""
	bitmap = (pred > DET_THRESH).astype(np.uint8)
	contours, _hierarchy = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	boxes = []
	for cnt in contours:
		bx, by, bw, bh = cv2.boundingRect(cnt)
		if min(bw, bh) < DET_MIN_SIZE:
			continue
		# Confidence is the mean probability inside the contour (as in PaddleOCR, fast mode).
		mask = np.zeros((bh, bw), dtype=np.uint8)
		cv2.drawContours(mask, [cnt - (bx, by)], -1, 1, -1)
		if cv2.mean(pred[by : by + bh, bx : bx + bw], mask)[0] < DET_BOX_THRESH:
			continue
		# The detector finds the "shrunk" middle of a line; expand it back (unclip).
		# For a rectangle, distance = area * ratio / perimeter.
		(cx, cy), (rw, rh), angle = cv2.minAreaRect(cnt)
		d = rw * rh * DET_UNCLIP_RATIO / (2 * (rw + rh))
		pts = cv2.boxPoints(((cx, cy), (rw + 2 * d, rh + 2 * d), angle))
		if min(rw, rh) + 2 * d < DET_MIN_SIZE + 2:
			continue
		# Text on screen is horizontal, so take an upright rectangle around the rotated one.
		x0 = max(0, math.floor(pts[:, 0].min() * sx))
		y0 = max(0, math.floor((pts[:, 1].min() + top) * sy))
		x1 = min(w, math.ceil(pts[:, 0].max() * sx))
		y1 = min(h, math.ceil((pts[:, 1].max() + top) * sy))
		if x1 - x0 >= 2 and y1 - y0 >= 2:
			boxes.append((x0, y0, x1, y1))
	return boxes


def _group_lines(items):
	"""Groups text pieces at the same height into one line, top to bottom and left to right."""
	items.sort(key=lambda ws: ws[0]["y"] + ws[0]["height"] / 2)
	lines = []  # [top, bottom, [pieces]]
	for ws in items:
		top = ws[0]["y"]
		bottom = top + ws[0]["height"]
		if lines:
			lt, lb, parts = lines[-1]
			# Same line if the pieces overlap vertically by more than half of the smaller one.
			overlap = min(lb, bottom) - max(lt, top)
			if overlap > 0.5 * min(lb - lt, bottom - top):
				parts.append(ws)
				lines[-1][0] = min(lt, top)
				lines[-1][1] = max(lb, bottom)
				continue
		lines.append([top, bottom, [ws]])
	result = []
	for _top, _bottom, parts in lines:
		parts.sort(key=lambda ws: ws[0]["x"])
		cur = list(parts[0])
		for ws in parts[1:]:
			# A large gap means another column (a list on the left, a hint on the right):
			# read it as a separate line so we do not get "Human Voices Make a Category Selection".
			gap = ws[0]["x"] - (cur[-1]["x"] + cur[-1]["width"])
			if gap > LINE_GAP * min(cur[-1]["height"], ws[0]["height"]):
				result.append(cur)
				cur = []
			cur.extend(ws)
		result.append(cur)
	return result
