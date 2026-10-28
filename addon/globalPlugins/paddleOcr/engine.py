# Paddle OCR add-on for NVDA
# Copyright (C) 2026 fdrv161
# SPDX-License-Identifier: GPL-2.0-or-later

"""OCR engine on PaddleOCR models (DB detector + CTC recognizer) via onnxruntime.

Imports nothing from NVDA, so it can be tested with plain Python.
Input is a BGR image (numpy, uint8, h×w×3). Output is the screen lines; each line is
a list of words with coordinates in pixels of the same image (NVDA LinesWordsResult format).
"""

import functools
import itertools
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

# The official onnxruntime build sends trace events through Windows ETW by default. Turned off before any
# session is created; a single start-up event may still be sent when the library loads.
ort.disable_telemetry_events()

# Detector post-processing settings from the inference.yml of the PP-OCRv6 models,
# except the box threshold: 0.45 here instead of 0.4 there.
DET_THRESH = 0.2
DET_BOX_THRESH = 0.45
DET_UNCLIP_RATIO = 1.4
DET_MIN_SIZE = 3
# Images wider than this are downscaled for the detector.
# Downscaling a screen hurts (small fonts vanish), so the limit is high: only for 8K and the like.
# Only the width counts: the detector reads tall images in strips (DET_STRIP), so their memory does not grow,
# and a long scrolled screenshot (1920 x 7560) downscaled by its height lost 7% of its words.
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
# With reuse=True the engine keeps a copy of an image up to this many pixels until the next one,
# to compare them (3 bytes per pixel: 6 MB for 1080p, 25 MB for 4K); larger pictures are not kept.
REUSE_MAX_PIXELS = 3840 * 2160

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


def _session(model, threads, oneShot=False):
	"""A session for a model file, or for the model itself given as bytes."""
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
	if not isinstance(model, bytes):
		model = str(model)
	return ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])


def _run_options():
	# Return working memory to the system after each run,
	# otherwise the peak of a large screen stays until exit.
	opts = ort.RunOptions()
	opts.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
	return opts


class _Seen:
	"""The previous image of an OcrEngine with reuse=True and what was found on it."""

	def __init__(self, bgr, parts, boxes, reads, lines):
		self.bgr = bgr
		self.parts = parts  # detector strips (see _join_strips); None for small images
		self.boxes = boxes
		self.reads = reads  # Reader.read results for boxes, untouched by the text fixes that follow it
		self.lines = lines


class OcrEngine:
	"""Detector + recognizer. allowed(c) tells which characters the recognizer may output (None means all).

	reuse: parts of an image that are the same as in the previous image of the same size are not
	recognized again (NVDA's auto refresh asks for the same place every 1.5 s).
	"""

	def __init__(self, detPath, recPath, recYml, threads=0, allowed=None, reuse=False):
		self._det = _session(detPath, threads, oneShot=True)
		self._detInput = self._det.get_inputs()[0].name
		self._resize = _session(_resize_model(), threads, oneShot=True)
		self._rec = Reader(recPath, recYml, threads, allowed)
		self._reuse = reuse
		self._seen = None

	def recognize(self, bgr):
		"""Returns a list of lines; a line is a list of words {x, y, width, height, text, charEdges}."""
		h, w = bgr.shape[:2]
		# Taken away first: after a failed run the next one starts from scratch.
		prev, self._seen = self._seen, None
		diff = rowsChanged = None
		if prev is not None and prev.bgr.shape == bgr.shape:
			# True where the image changed, the three channels of a pixel side by side.
			diff = np.not_equal(prev.bgr, bgr).reshape(h, w * 3)
			rowsChanged = diff.any(axis=1)
			if not rowsChanged.any():
				self._seen = prev
				return _copy_lines(prev.lines)
		else:
			prev = None
		parts = None
		if min(h, w) < DET_PAD_BELOW:
			p = DET_PAD
			boxes, _parts = self._detect(_add_margin(bgr, p))
			# Back to the coordinates of the original image; the recognizer reads lines from it.
			boxes = [
				(max(0, x0 - p), max(0, y0 - p), min(w, x1 - p), min(h, y1 - p)) for x0, y0, x1, y1 in boxes
			]
			boxes = [b for b in boxes if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
		else:
			boxes, parts = self._detect(bgr, None if prev is None else (rowsChanged, prev.parts))
		kept = []
		if prev is not None:
			old, new = _changed_groups(prev.boxes, boxes, diff)
			kept = [(prev.boxes[i], prev.reads[i]) for i in old]
			boxes = [boxes[i] for i in new]
		reads = [read for _box, read in kept] + self._rec.read(bgr, boxes)
		boxes = [box for box, _read in kept] + boxes
		# Copies: the text fixes change the words, and the reads may be used again.
		items = [_copy_words(words) for words, score in reads if words and score >= REC_MIN_SCORE]
		_pick_twin_script(items)
		lines = _group_lines(items)
		if self._reuse and h * w <= REUSE_MAX_PIXELS:
			# A copy: the caller may change its array, and a changed image must not look unchanged.
			self._seen = _Seen(bgr.copy(), parts, boxes, reads, _copy_lines(lines))
		return lines

	# --- detector ---

	def _detect(self, bgr, reuse=None):
		"""Returns (boxes, strips).

		reuse: (for each row of the image whether it changed, the strips of the previous image
		of this size); strips without changed rows are taken from it instead of running the detector.
		"""
		h, w = bgr.shape[:2]
		scale = min(1.0, DET_MAX_SIDE / w)
		# The model needs sides that are multiples of 32.
		rh = max(32, round(h * scale / 32) * 32)
		rw = max(32, round(w * scale / 32) * 32)
		img = bgr
		if (rw, rh) != (w, h):
			sizes = np.array([1, rh, rw, 3], dtype=np.int64)
			img = self._resize.run(None, {"image": np.ascontiguousarray(bgr)[None], "sizes": sizes})[0][0]
		sy = h / rh
		strips = _strips(rh)
		# One buffer for all strips: a new one for each would make the system zero its pages again.
		buf = np.empty(3 * max(s1 - s0 for *_own, s0, s1 in strips) * rw, dtype=np.float32)
		parts = []
		for k, (top, bottom, s0, s1) in enumerate(strips):
			if reuse is not None:
				rowsChanged, oldParts = reuse
				# The rows of the image this strip is resized from; the resize also reads a row past each end.
				if not rowsChanged[max(0, math.floor(s0 * sy) - 1) : math.ceil(s1 * sy) + 1].any():
					parts.append(oldParts[k])
					continue
			x = buf[: 3 * (s1 - s0) * rw].reshape(1, 3, s1 - s0, rw)
			# (pixel/255 - mean)/std per channel straight into the 1×3×h×w layout: no intermediate float
			# copies of the whole image, which is noticeable on a 1080p screen.
			for c in range(3):
				np.multiply(img[s0:s1, :, c], _DET_SCALE[c], out=x[0, c], dtype=np.float32)
				x[0, c] -= _DET_SHIFT[c]
			pred = self._det.run(None, {self._detInput: x})[0][0, 0]
			parts.append((top * sy, bottom * sy, s0 * sy, s1 * sy, _db_boxes(pred, w / rw, sy, w, h, s0)))
		return _join_strips(parts, DET_OVERLAP / 2 * sy), parts


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
				_line_input(crops[i], rw, batch[j, :, :, :rw])
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
		for ch, t in [*chars, (" ", steps)]:
			if ch == " ":
				if cur:
					cur = self._fix_mixed_script(cur, prob)
					wx0 = x0 + int(cur[0][1] * toBox)
					wx1 = min(x1, x0 + math.ceil((cur[-1][1] + 1) * toBox))
					# Where each character starts, plus the word's end: a character ends halfway between
					# the CTC steps where it and the next one were read. Approximate (a step is 2-3 screen
					# pixels), but enough to put the mouse on a character. Kept monotonic and inside the word.
					edges = [wx0]
					for (_c, ta), (_d, tb) in itertools.pairwise(cur):
						edges.append(min(wx1, max(edges[-1], x0 + int((ta + tb + 1) / 2 * toBox))))
					edges.append(max(edges[-1], wx0 + max(1, wx1 - wx0)))
					word = {
						"x": wx0,
						"y": y0,
						"width": max(1, wx1 - wx0),
						"height": y1 - y0,
						"text": _fix_case(_fix_il("".join(c for c, _t in cur))),
						# One more than the characters of the text; the result checks this, since a text fix
						# could change the length (upper() of "ß" is "SS").
						"charEdges": edges,
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


def rgbToBgr(rgb, alpha, width, height):
	"""Picture from the clipboard (wx.Image: RGB bytes, alpha bytes or None) → BGR image for the engine.
	Transparent parts become white: dark text on a transparent background would otherwise
	be black on black."""
	img = np.frombuffer(rgb, dtype=np.uint8).reshape(height, width, 3)
	if alpha is not None:
		# In place, in 16-bit integers: float math on a 12-megapixel photo peaked at ~370 MB, this at ~160 MB.
		a = np.frombuffer(alpha, dtype=np.uint8).reshape(height, width, 1).astype(np.uint16)
		out = img.astype(np.uint16)
		out *= a
		a = 255 * (255 - a) + 127
		out += a
		out //= 255
		img = out.astype(np.uint8)
	return np.ascontiguousarray(img[:, :, ::-1])


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
	h, w = bgr.shape[:2]
	edge = np.concatenate([bgr[0], bgr[-1], bgr[:, 0], bgr[:, -1]])
	out = np.empty((h + 2 * pad, w + 2 * pad, 3), dtype=np.uint8)
	out[:] = np.median(edge, axis=0).astype(np.uint8)
	out[pad : pad + h, pad : pad + w] = bgr
	return out


def _taps(src, dst, channels=0, scale=1.0):
	"""A bilinear resize of src samples to dst as OpenCV's INTER_LINEAR does it (pixel centres aligned,
	the edge sample repeated): out[k] = (a[i0[k]] * w0[k] + a[i1[k]] * w1[k]) * scale.

	With channels, the indices are for rows of interleaved pixels (bgrbgr...), one row of them per channel
	(channels x dst), so picking them gives the channels apart.
	"""
	# Few numpy calls: this runs for every line, and on short lines the calls cost more than the work.
	pos = np.arange(dst) * (src / dst)
	pos += 0.5 * src / dst - 0.5
	np.clip(pos, 0, src - 1, out=pos)
	i0 = pos.astype(np.intp)
	pos -= i0
	w1 = np.multiply(pos, scale, dtype=np.float32)
	w0 = np.subtract(np.float32(scale), w1)
	i1 = i0 + 1
	np.minimum(i1, src - 1, out=i1)
	if channels:
		k = np.arange(channels)[:, None]
		i0 = i0 * channels + k
		i1 = i1 * channels + k
	return i0, i1, w0, w1


# Taps for the height of a line, cached: lines of a screen have few heights (the widths are all different,
# and caching them grew the memory by megabytes). The arrays are read-only.
@functools.lru_cache(maxsize=256)
def _line_rows(h):
	"""Taps from h rows to REC_HEIGHT with the scaling to -1..1 (see _line_input), for h > REC_HEIGHT."""
	taps = _taps(h, REC_HEIGHT, 0, 1 / 127.5)
	for a in taps:
		a.flags.writeable = False
	return taps


@functools.lru_cache(maxsize=64)
def _line_rows_matrix(h):
	"""Taps from h rows to REC_HEIGHT as a matrix (REC_HEIGHT rows of two weights), for h <= REC_HEIGHT."""
	i0, i1, w0, w1 = _taps(h, REC_HEIGHT)
	m = np.zeros((REC_HEIGHT, h), dtype=np.float32)
	k = np.arange(REC_HEIGHT)
	np.add.at(m, (k, i0), w0)
	np.add.at(m, (k, i1), w1)
	m.flags.writeable = False
	return m


def _mix(a, taps, axis, out=None):
	"""One axis of a bilinear resize of a 2-D array (see _taps), as float32 (into out if given)."""
	i0, i1, w0, w1 = taps
	if axis == 0:
		lo, hi, w0, w1 = a[i0], a[i1], w0[:, None], w1[:, None]
	else:
		lo, hi = a[:, i0], a[:, i1]
	# Converted first: numpy multiplies uint8 by float32 several times slower than float32 by float32.
	lo = lo.astype(np.float32, copy=False)
	hi = hi.astype(np.float32, copy=False)
	if out is None:
		out = lo
	np.multiply(lo, w0, out=out)
	hi *= w1
	out += hi
	return out


def _resize_model():
	"""A one-operator ONNX model: Resize of a uint8 image (1 x h x w x 3, sizes [1, h', w', 3]),
	bilinear as OpenCV's INTER_LINEAR does it (pixel centres aligned, edge pixels repeated, no antialiasing).

	For the detector input. Numpy took 3-4 times longer than OpenCV to stretch a window to sides
	that are multiples of 32; onnxruntime does it about as fast. Written here as protobuf bytes,
	since the onnx package is not in the add-on (tests check it with that package).
	"""

	def varint(n):
		out = bytearray()
		while n > 0x7F:
			out.append(n & 0x7F | 0x80)
			n >>= 7
		out.append(n)
		return bytes(out)

	def field(number, value):
		# Integers as varints (wire type 0), strings and messages length-delimited (wire type 2).
		if isinstance(value, int):
			return varint(number << 3) + varint(value)
		if isinstance(value, str):
			value = value.encode()
		return varint(number << 3 | 2) + varint(len(value)) + value

	def tensor(name, elemType, dims):
		# ValueInfoProto: name, type {tensor_type {elem_type, shape {dim {dim_value or dim_param}}}}.
		shape = b"".join(field(1, field(1 if isinstance(d, int) else 2, d)) for d in dims)
		return field(1, name) + field(2, field(1, field(1, elemType) + field(2, shape)))

	def attribute(name, text):
		return field(1, name) + field(4, text) + field(20, 3)  # type 3: STRING

	uint8, int64 = 2, 7
	node = (
		b"".join(field(1, name) for name in ("image", "", "", "sizes"))  # inputs X, roi, scales, sizes
		+ field(2, "resized")
		+ field(4, "Resize")
		+ field(5, attribute("mode", "linear"))
		+ field(5, attribute("coordinate_transformation_mode", "half_pixel"))
	)
	graph = (
		field(1, node)
		+ field(2, "resize")
		+ field(11, tensor("image", uint8, (1, "h", "w", 3)))
		+ field(11, tensor("sizes", int64, (4,)))
		+ field(12, tensor("resized", uint8, (1, "rh", "rw", 3)))
	)
	# ModelProto: ir_version 8, opset_import {version 18}, graph.
	return field(1, 8) + field(8, field(2, 18)) + field(7, graph)


def _line_input(crop, width, out):
	"""A line (h x w x 3 uint8) as the recognizer takes it, written into out (3 x REC_HEIGHT x width):
	resized to REC_HEIGHT x width, (pixel/255 - 0.5)/0.5, channels first.

	The scaling is in the weights of the first step. Picking columns is the slow step, so it runs
	on whichever is fewer: the rows of the line or the REC_HEIGHT rows of the result. The second step
	goes channel by channel straight into out: copies over ~1 MB would make the system zero
	new memory each time, which took longer than the resize itself. Measured on the lines
	of the test screens: as fast as cv2.resize with the old normalisation, long lines faster.
	"""
	h, w = crop.shape[:2]
	rows = crop.reshape(h, w * 3)
	if h <= REC_HEIGHT:
		# Columns of the few rows, then the rows by a matrix product (cheaper than picking rows here).
		a = _mix(rows.astype(np.float32), _taps(w, width, 3, 1 / 127.5), 1)  # h x 3 x width
		a -= 1
		m = _line_rows_matrix(h)
		for c in range(3):
			np.matmul(m, a[:, c], out=out[c])
	else:
		a = _mix(rows, _line_rows(h), 0).reshape(REC_HEIGHT, w, 3)
		a -= 1
		xs = _taps(w, width)
		for c in range(3):
			_mix(a[:, :, c], xs, 1, out[c])


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


def _changed_groups(old, new, diff):
	"""After a partial change of the image: (indices of old boxes to keep with their text,
	indices of new boxes to read).

	A change anywhere in a detector strip moves the other boxes of the strip by a pixel (the detector looks
	at the whole strip), and lines whose pixels did not change would be read a little differently: NVDA's
	auto refresh would speak them as new text. So old and new boxes that overlap (shared columns and at
	least half the rows of the lower box) form groups: a group without a changed pixel in any of its boxes
	keeps its old boxes, otherwise its new boxes are read.
	diff: rows × (columns × 3), non-zero where the image changed.
	"""
	boxes = np.array([*old, *new], dtype=np.int32).reshape(-1, 4)
	x0, y0, x1, y1 = boxes.T
	dx = np.minimum.outer(x1, x1) - np.maximum.outer(x0, x0)
	dy = np.minimum.outer(y1, y1) - np.maximum.outer(y0, y0)
	linked = (dx > 0) & (2 * dy >= np.minimum.outer(y1 - y0, y1 - y0))
	group = list(range(len(boxes)))

	def root(i):
		while group[i] != i:
			group[i] = group[group[i]]
			i = group[i]
		return i

	for i, j in zip(*np.nonzero(np.triu(linked, 1)), strict=True):
		group[root(int(j))] = root(int(i))
	changed = set()
	for i, (bx0, by0, bx1, by1) in enumerate(boxes.tolist()):
		if diff[by0:by1, 3 * bx0 : 3 * bx1].any():
			changed.add(root(i))
	n = len(old)
	return (
		[i for i in range(n) if root(i) not in changed],
		[i - n for i in range(n, len(boxes)) if root(i) in changed],
	)


def _copy_words(words):
	return [{**w, "charEdges": list(w["charEdges"])} for w in words]


def _copy_lines(lines):
	return [_copy_words(line) for line in lines]


def _db_boxes(pred, sx, sy, w, h, top=0):
	"""Detector probability map → line rectangles (x0, y0, x1, y1) in pixels of the original image.

	top is the row of the map's first row in the detector input (for a strip).

	As PaddleOCR does it with OpenCV (findContours, minAreaRect), in numpy and for all blobs at once:
	blobs of pixels above DET_THRESH (8-connected), their mean probability, the smallest rotated rectangle
	around each. Unlike findContours, the border of a hole inside a blob is not a box of its own
	(it gave a second box inside the blob's box; 1 of 7300 boxes on the test screens).
	"""
	bitmap, rows, c0, c1, comp = _blobs(pred, DET_THRESH)
	if not len(rows):
		return []
	# Sum of probabilities over each run: the map values above the threshold come in run order.
	lens = c1 - c0
	runSum = np.add.reduceat(pred[bitmap], np.cumsum(lens) - lens).astype(np.float64)
	total = np.bincount(comp, weights=runSum)
	area = np.bincount(comp, weights=lens)
	_add_holes(pred, rows, c0, c1, comp, total, area)
	# Confidence is the mean probability inside the blob (as in PaddleOCR, fast mode).
	mean = total / area
	# Runs grouped by blob, still top to bottom and left to right inside a blob.
	order = np.argsort(comp, kind="stable")
	rows, c0, c1, comp = rows[order], c0[order], c1[order], comp[order]
	first = np.flatnonzero(_starts(comp))
	bw = np.maximum.reduceat(c1, first) - np.minimum.reduceat(c0, first)
	bh = np.maximum.reduceat(rows, first) - rows[first] + 1
	keep = (np.minimum(bw, bh) >= DET_MIN_SIZE) & (mean >= DET_BOX_THRESH)
	if not keep.any():
		return []
	m = keep[comp]
	rows, c0, c1, comp = rows[m], c0[m], c1[m], comp[m]
	cx, cy, rw, rh, ux, uy = _min_area_rect(rows, c0, c1, comp)
	# The detector finds the "shrunk" middle of a line; expand it back (unclip).
	# For a rectangle, distance = area * ratio / perimeter.
	with np.errstate(divide="ignore", invalid="ignore"):
		d = rw * rh * DET_UNCLIP_RATIO / (2 * (rw + rh))
	good = (np.minimum(rw, rh) > 0) & (np.minimum(rw, rh) + 2 * d >= DET_MIN_SIZE + 2)
	# Text on screen is horizontal, so take an upright rectangle around the rotated one:
	# its half sides along x and y.
	hu = rw / 2 + d
	hn = rh / 2 + d
	ex = np.abs(ux) * hu + np.abs(uy) * hn
	ey = np.abs(uy) * hu + np.abs(ux) * hn
	x0 = np.maximum(0, np.floor((cx - ex) * sx))
	y0 = np.maximum(0, np.floor((cy - ey + top) * sy))
	x1 = np.minimum(w, np.ceil((cx + ex) * sx))
	y1 = np.minimum(h, np.ceil((cy + ey + top) * sy))
	good &= (x1 - x0 >= 2) & (y1 - y0 >= 2)
	# Last blob first, the order findContours gives: the order of equal lines further on depends on it.
	boxes = np.stack([x0, y0, x1, y1], axis=1)[good][::-1].astype(np.int64)
	return [tuple(b) for b in boxes.tolist()]


def _starts(*keys):
	"""For sorted keys: True where a group starts (any key differs from the one before)."""
	out = np.ones(len(keys[0]), dtype=bool)
	out[1:] = keys[0][1:] != keys[0][:-1]
	for key in keys[1:]:
		out[1:] |= key[1:] != key[:-1]
	return out


def _blobs(pred, thresh):
	"""8-connected blobs of pixels above thresh as horizontal runs.

	Returns (pred > thresh, then per run: row, first column, column after the last, blob number);
	runs top to bottom and left to right, blobs numbered in the same order by their first pixel.
	"""
	h, w = pred.shape
	# Rows of the map one after another with a False after each, so a run never goes on to the next row.
	stride = w + 1
	flat = np.zeros(h * stride + 1, dtype=bool)
	bitmap = flat[1:].reshape(h, stride)[:, :w]
	np.greater(pred, thresh, out=bitmap)
	edges = np.flatnonzero(flat[1:] != flat[:-1])
	starts = edges[0::2]
	ends = edges[1::2]
	n = len(starts)
	# The runs of the row above that touch a run, corners too: the ones that end at or after its start
	# and start at or before its end (in flat positions one row up, so only that row can match).
	lo = np.searchsorted(ends, starts - stride, "left")
	hi = np.searchsorted(starts, ends - stride, "right")
	count = np.maximum(hi - lo, 0)
	total = int(count.sum())
	parent = np.arange(n)
	if total:
		below = np.repeat(np.arange(n), count)
		above = np.arange(total) + np.repeat(lo - (np.cumsum(count) - count), count)
		parent = _union(parent, above, below)
	roots = np.flatnonzero(parent == np.arange(n))
	number = np.empty(n, dtype=np.intp)
	number[roots] = np.arange(len(roots))
	rows = starts // stride
	return bitmap, rows, starts - rows * stride, ends - rows * stride, number[parent]


def _add_holes(pred, rows, c0, c1, comp, total, area):
	"""Adds the map values (sum to total, count to area, per blob) of the holes of each blob and of what
	lies in them: findContours fills the outer border of a blob, so PaddleOCR's mean counts them.
	Without it a noisy blob full of holes (a picture) got a high mean and became a box.

	The runs are as _blobs returns them.
	"""
	# In the top row of a hole two runs of its blob come one after another; most maps have no such row.
	if not ((rows[1:] == rows[:-1]) & (comp[1:] == comp[:-1])).any():
		return
	h, w = pred.shape
	n = len(rows)
	stride = w + 1
	starts = rows * stride + c0
	ends = rows * stride + c1
	# Background runs in the flat layout of _blobs: run k is the one before blob run k, the last one
	# is after the last blob run. Runs that reach a row end (the False column) or start a row,
	# and all runs of the first and the last row, touch the image border: they are outside.
	bs = np.concatenate(([0], ends))
	be = np.append(starts, h * stride)
	row = bs // stride
	border = (be - row * stride > w) | (bs == row * stride) | (row == 0) | (row == h - 1)
	# 4-connected background: the runs of the row above that share a column with a run.
	lo = np.searchsorted(be, bs - stride, "right")
	hi = np.searchsorted(bs, be - stride, "left")
	count = np.maximum(hi - lo, 0)
	group = np.arange(n + 1)
	pairs = int(count.sum())
	if pairs:
		below = np.repeat(np.arange(n + 1), count)
		above = np.arange(pairs) + np.repeat(lo - (np.cumsum(count) - count), count)
		group = _union(group, above, below)
	outside = np.zeros(n + 1, dtype=bool)
	outside[group[border]] = True
	hole = ~outside[group]
	if not hole.any():
		return
	# The blob around a hole is the one left of the hole's first run: the top row of a hole has nothing
	# else in it. A group is named after its first run, so that run is group - 1 in the blob runs.
	owner = comp[group - 1]
	t = np.flatnonzero(hole)
	lens = be[t] - bs[t]
	offsets = np.cumsum(lens) - lens
	first = row[t] * w + bs[t] - row[t] * stride
	values = pred.ravel()[np.repeat(first - offsets, lens) + np.arange(lens.sum())]
	np.add.at(total, owner[t], np.add.reduceat(values, offsets))
	np.add.at(area, owner[t], lens)
	# A blob in a hole (an island) counts in the blob around the hole. Blobs are numbered by their first
	# run, an island after the blob around it, so from the last one, islands of islands come first.
	firstRun = np.flatnonzero(np.concatenate(([True], comp[1:] > np.maximum.accumulate(comp)[:-1])))
	for b in np.flatnonzero(hole[firstRun])[::-1].tolist():
		around = owner[firstRun[b]]
		total[around] += total[b]
		area[around] += area[b]


def _union(parent, a, b):
	"""Union-find over the pairs (a[k], b[k]) for all pairs at once.

	Every set ends up pointing straight at its smallest member. Each round joins the roots of the pairs
	still apart (the larger root under the smaller one), then flattens the chains; a few rounds suffice.
	"""
	while True:
		ra = parent[a]
		rb = parent[b]
		apart = ra != rb
		if not apart.any():
			return parent
		a, b, ra, rb = a[apart], b[apart], ra[apart], rb[apart]
		np.minimum.at(parent, np.maximum(ra, rb), np.minimum(ra, rb))
		while True:
			up = parent[parent]
			if np.array_equal(up, parent):
				break
			parent = up


def _min_area_rect(rows, c0, c1, comp):
	"""The smallest rotated rectangle around each blob (as cv2.minAreaRect, on pixel centres).

	Runs grouped by blob (comp ascending). Returns arrays per blob: centre x, y, the sides along
	the direction (ux, uy) and across it. A blob of one line or point gets a side of 0.
	"""
	# The hull of a blob is made of the first and last pixel of each of its rows.
	g = np.flatnonzero(_starts(comp, rows))
	y = rows[g]
	xl = c0[g]
	xr = c1[np.append(g[1:], len(rows)) - 1] - 1
	gc = comp[g]
	left = _hull_side(y, xl, gc, 1)
	right = _hull_side(y, xr, gc, -1)
	# The hull of each blob: its left side top to bottom, then its right side bottom to top.
	hy = np.concatenate((y[left], y[right]))
	hx = np.concatenate((xl[left], xr[right]))
	hc = np.concatenate((gc[left], gc[right]))
	below = 2 * (rows.max() + 1)
	o = np.lexsort((np.concatenate((y[left], below - y[right])), hc))
	hy, hx, hc = hy[o], hx[o], hc[o]
	# A top or bottom row of one pixel is in both sides: drop the repeat (also the last point = the first).
	dup = ~_starts(hc, hx, hy)
	start = _starts(hc)
	firstOf = np.maximum.accumulate(np.where(start, np.arange(len(hc)), 0))
	end = np.append(start[1:], True)
	dup |= end & ~start & (hx == hx[firstOf]) & (hy == hy[firstOf])
	hy, hx, hc = hy[~dup], hx[~dup], hc[~dup]
	# Blobs with a hull of fewer than 3 points (a line or a point) keep zero sides.
	out = np.zeros((6, comp[-1] + 1))
	s = np.flatnonzero(_starts(hc))
	blob = hc[s]
	k = np.diff(np.append(s, len(hc)))
	m = np.repeat(k >= 3, k)
	hy, hx, hc = hy[m].astype(np.float64), hx[m].astype(np.float64), hc[m]
	blob, k = blob[k >= 3], k[k >= 3]
	if not len(blob):
		return tuple(out)
	hfirst = np.cumsum(k) - k
	# Rotating calipers by brute force: each edge of a hull against every point of that hull.
	# The smallest rectangle has a side along one of the edges.
	at = np.repeat(np.arange(len(blob)), k)  # hull point -> position of its blob in blob
	nxt = np.arange(len(hc)) + 1
	nxt[hfirst + k - 1] = hfirst
	ex = hx[nxt] - hx
	ey = hy[nxt] - hy
	length = np.hypot(ex, ey)
	ux = ex / length
	uy = ey / length
	per = k[at]
	edge = np.repeat(np.arange(len(hc)), per)
	block = np.cumsum(per) - per
	point = hfirst[at[edge]] + np.arange(len(edge)) - block[edge]
	along = ux[edge] * hx[point] + uy[edge] * hy[point]
	across = ux[edge] * hy[point] - uy[edge] * hx[point]
	amax, amin = np.maximum.reduceat(along, block), np.minimum.reduceat(along, block)
	cmax, cmin = np.maximum.reduceat(across, block), np.minimum.reduceat(across, block)
	area = (amax - amin) * (cmax - cmin)
	# Of rectangles with the same area (small round blobs), the most slanted one: OpenCV mostly picks it
	# (its float rounding makes the slanted area a little smaller), and so the boxes stay as they were.
	smallest = np.minimum.reduceat(area, hfirst)[at]
	slant = np.where(area <= smallest * (1 + 1e-9) + 1e-9, -np.maximum(np.abs(ux), np.abs(uy)), -2)
	best = np.lexsort((slant, at))[hfirst + k - 1]
	ux, uy = ux[best], uy[best]
	along = (amax[best] + amin[best]) / 2
	across = (cmax[best] + cmin[best]) / 2
	# The centre back from (along, across) to (x, y): across is along (-uy, ux).
	out[:, blob] = (
		ux * along - uy * across,
		uy * along + ux * across,
		amax[best] - amin[best],
		cmax[best] - cmin[best],
		ux,
		uy,
	)
	return tuple(out)


def _hull_side(y, x, chain, side):
	"""Indices of the points that make the left (side=1) or right (side=-1) side of a convex hull,
	for many chains at once. A chain has one point per row, top to bottom.

	Points that are not strictly outside the line between their neighbours are dropped, all such points
	in a round (each lies inside the hull either way), until none are left.
	"""
	idx = np.arange(len(y))
	while len(idx) >= 3:
		a, b, c = idx[:-2], idx[1:-1], idx[2:]
		cross = (x[b] - x[a]) * (y[c] - y[a]) - (x[c] - x[a]) * (y[b] - y[a])
		drop = (chain[a] == chain[c]) & (cross * side >= 0)
		if not drop.any():
			break
		keep = np.ones(len(idx), dtype=bool)
		keep[1:-1] = ~drop
		idx = idx[keep]
	return idx


def _stacked(a, b):
	"""Two pieces one under the other: they share columns but barely share rows.
	A line grows with every piece added, so a taller piece beside it (a switch label next to a title and
	its description) stretches it down to the next row; without this check that row joined the line
	and, sorted by x, was read before it."""
	ax0, bx0 = a[0]["x"], b[0]["x"]
	ax1, bx1 = a[-1]["x"] + a[-1]["width"], b[-1]["x"] + b[-1]["width"]
	at, bt = a[0]["y"], b[0]["y"]
	ah, bh = a[0]["height"], b[0]["height"]
	rows = min(at + ah, bt + bh) - max(at, bt)
	return min(ax1, bx1) > max(ax0, bx0) and rows <= 0.5 * min(ah, bh)


def _group_lines(items):
	"""Groups text pieces at the same height into one line, top to bottom and left to right."""
	items.sort(key=lambda ws: ws[0]["y"] + ws[0]["height"] / 2)
	lines = []  # [top, bottom, [pieces]]
	for ws in items:
		top = ws[0]["y"]
		bottom = top + ws[0]["height"]
		if lines:
			lt, lb, parts = lines[-1]
			# Same line if the pieces overlap vertically by more than half of the smaller one,
			# and the piece is not under another piece of the line (see _stacked).
			overlap = min(lb, bottom) - max(lt, top)
			if overlap > 0.5 * min(lb - lt, bottom - top) and not any(_stacked(p, ws) for p in parts):
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
