# Paddle OCR add-on for NVDA
# Copyright (C) 2026 fdrv161
# SPDX-License-Identifier: GPL-2.0-or-later

"""OCR engine on PaddleOCR models (DB detector + CTC recognizer) via onnxruntime.

Imports nothing from NVDA, so it can be tested with plain Python.
Input is a BGR image (numpy, uint8, h×w×3). Output is the screen lines; each line is
a list of words with coordinates in pixels of the same image (NVDA LinesWordsResult format).
"""
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import itertools

# Detector post-processing settings from the inference.yml of the PP-OCRv6 models.
DET_THRESH = 0.2
DET_BOX_THRESH = 0.45
DET_UNCLIP_RATIO = 1.4
DET_MIN_SIZE = 3
# Images with a longer side above this are downscaled for the detector.
# Downscaling a screen hurts (small fonts vanish), so the limit is high: only for 8K and the like.
DET_MAX_SIDE = 4000
REC_HEIGHT = 48
# Measured: batches of 4–8 are fastest; larger ones are slower (more width padding).
REC_BATCH = 4
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


def _session(path, threads):
	opts = ort.SessionOptions()
	opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
	if threads:
		opts.intra_op_num_threads = threads
	opts.log_severity_level = 3
	return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _run_options():
	# Return working memory to the system after each run, otherwise the peak of a large screen stays until exit.
	opts = ort.RunOptions()
	opts.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
	return opts


class OcrEngine:
	"""Detector + recognizer. allowed(c) tells which characters the recognizer may output (None means all)."""

	def __init__(self, detPath, recPath, recYml, threads=0, allowed=None):
		self._det = _session(detPath, threads)
		self._detInput = self._det.get_inputs()[0].name
		self._rec = Reader(recPath, recYml, threads, allowed)
		self._runOpts = _run_options()

	def recognize(self, bgr):
		"""Returns a list of lines; a line is a list of words {x, y, width, height, text}."""
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
		# (pixel/255 - mean)/std per channel straight into the 1×3×h×w layout: no intermediate float
		# copies of the whole image, which is noticeable on a 1080p screen.
		x = np.empty((1, 3, rh, rw), dtype=np.float32)
		for c in range(3):
			np.multiply(img[:, :, c], _DET_SCALE[c], out=x[0, c], dtype=np.float32)
			x[0, c] -= _DET_SHIFT[c]
		pred = self._det.run(None, {self._detInput: x}, self._runOpts)[0][0, 0]
		return _db_boxes(pred, w / rw, h / rh, w, h)


class Reader:
	"""Line recognizer (CTC) of one model."""

	def __init__(self, recPath, recYml, threads=0, allowed=None):
		self._rec = _session(recPath, threads)
		self._recInput = self._rec.get_inputs()[0].name
		# Index 0 is the CTC blank; the models also know a space at the end.
		self._chars = ["", *_load_charset(recYml), " "]
		classes = self._rec.get_outputs()[0].shape[-1]
		if isinstance(classes, int) and classes != len(self._chars):
			raise ValueError(f"Dictionary ({len(self._chars)}) does not match the model ({classes})")
		# Banned characters (allowed(c) is false) are never chosen when decoding.
		# Keep only the allowed columns: the multilingual model has ~600 of 18k, so decoding
		# is many times cheaper, and the chosen character is the same as with banned ones zeroed.
		self._keep = None
		if allowed:
			self._keep = np.array([i for i, c in enumerate(self._chars) if i == 0 or allowed(c)])
			self._chars = [self._chars[i] for i in self._keep]
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
		order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / crops[i].shape[0])
		results = [None] * len(crops)
		for start in range(0, len(order), REC_BATCH):
			idx = order[start:start + REC_BATCH]
			maxRatio = max(crops[i].shape[1] / crops[i].shape[0] for i in idx)
			batchW = max(math.ceil(REC_HEIGHT * maxRatio), REC_HEIGHT)
			batch = np.zeros((len(idx), 3, REC_HEIGHT, batchW), dtype=np.float32)
			widths = []
			for j, i in enumerate(idx):
				ch, cw = crops[i].shape[:2]
				rw = min(batchW, max(1, math.ceil(REC_HEIGHT * cw / ch)))
				img = cv2.resize(crops[i], (rw, REC_HEIGHT)).astype(np.float32)
				batch[j, :, :, :rw] = ((img / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
				widths.append(rw)
			last = start + REC_BATCH >= len(order)
			probs = self._rec.run(None, {self._recInput: batch}, self._lastRunOpts if last else self._runOpts)[0]
			if self._keep is not None:
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
		score = float(np.mean([conf[t] for _, t in chars]))
		words = []
		cur = []
		for ch, t in chars + [(" ", steps)]:
			if ch == " ":
				if cur:
					cur = self._fix_mixed_script(cur, prob)
					wx0 = x0 + int(cur[0][1] * toBox)
					wx1 = min(x1, x0 + math.ceil((cur[-1][1] + 1) * toBox))
					word = {
						"x": wx0, "y": y0, "width": max(1, wx1 - wx0), "height": y1 - y0,
						"text": _fix_case(_fix_il("".join(c for c, _ in cur))),
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
		letters = [c for c, _ in word if _script(c)]
		scripts = {_script(c) for c in letters}
		if len(scripts) != 1 or not all(c in _TWINS for c in letters):
			return None
		idx = self._latin if scripts == {"cyr"} else self._cyr
		return "".join(self._chars[int(idx[prob[t, idx].argmax()])] if _script(c) else c for c, t in word)

	def _fix_mixed_script(self, word, prob):
		"""Removes twin letters of the wrong script: "Itет" → "Item", "Windоws" → "Windows".

		The Russian model knows both Latin and Cyrillic, and with lookalike letters (е/e, а/a, т/m)
		it sometimes picks the wrong script. If a word has letters of both scripts, pick the main one by the sum
		of probabilities at the letter positions and replace each foreign letter with the best letter of the main script.
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
			if {"latin", "cyr"} <= {_script(c) for c, _ in run}:
				steps = [t for _, t in run]
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
	- I after a lowercase letter and not before a lowercase one is l ("AlI" → "All", "FiII" → "Fill";
	    "AddIn" is left alone: a lowercase letter follows the I);
	- a capital followed only by I/l, 3+ letters in total, means lowercase l ("AII" → "All");
	- in an all-caps word (2+ capitals) a lowercase l is I ("TlGHT" → "TIGHT", "MlDl" → "MIDI").
	"""
	if len(text) >= 3 and text[0].isupper() and text[0] not in "Il" and set(text[1:]) <= {"I", "l"}:
		return text[0] + "l" * (len(text) - 1)
	letters = [c for c in text if c.isalpha()]
	if "l" in letters and sum(c.isupper() for c in letters) >= 2 and all(c.isupper() or c == "l" for c in letters):
		return text.replace("l", "I")
	# An all-caps word with one stray lowercase letter ("tIGHT") is left alone: _fix_case will fix it.
	if "I" not in text or sum(c.isupper() and c != "I" for c in letters) >= 2:
		return text
	chars = list(text)
	# Left to right: in "FiII" the first I becomes l, and then the second one also follows a lowercase letter.
	for i in range(1, len(chars)):
		nxt = chars[i + 1] if i + 1 < len(chars) else ""
		if chars[i] == "I" and chars[i - 1].islower() and _script(chars[i - 1]) == "latin" and not nxt.islower():
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
_TWINS = set("АВЕКМНОРСТХУаеорсхукт" "ABEKMHOPCTXYaeopcxykm")


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


def latin_only(c):
	"""Latin, digits, punctuation and arrows. CJK and the rest are banned: v6 confidently reads icons as "平"."""
	o = ord(c)
	return o < 0x250 or 0x2000 <= o < 0x2200



def bgra_to_bgr(buf, width, height):
	"""NVDA screenshot (RGBQUAD bytes: blue, green, red, unused) → BGR image for the engine."""
	return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()


def _script(c):
	"""The script of a character: 'latin', 'cyr' or None (digits, punctuation and others)."""
	if not c.isalpha():
		return None
	o = ord(c)
	if 0x400 <= o <= 0x4FF:
		return "cyr"
	if o < 0x250:
		return "latin"
	return None


def _db_boxes(pred, sx, sy, w, h):
	"""Detector probability map → line rectangles (x0, y0, x1, y1) in pixels of the original image."""
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
		if cv2.mean(pred[by:by + bh, bx:bx + bw], mask)[0] < DET_BOX_THRESH:
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
		y0 = max(0, math.floor(pts[:, 1].min() * sy))
		x1 = min(w, math.ceil(pts[:, 0].max() * sx))
		y1 = min(h, math.ceil(pts[:, 1].max() * sy))
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
