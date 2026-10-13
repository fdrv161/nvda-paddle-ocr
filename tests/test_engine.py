"""Engine checks without NVDA: capture conversion, word coordinates, character masks and text fixes.

Run: .venv/Scripts/python -m pytest tests/
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "addon" / "globalPlugins" / "paddleOcr"
sys.path.insert(0, str(PLUGIN))
import engine

MODELS = PLUGIN / "models"


@pytest.fixture(scope="module")
def ocr():
	if not (MODELS / "rec_cyrillic_v5").is_dir():
		pytest.skip("No models in the add-on: run tools/fetch_libs.py")
	rec = MODELS / "rec_cyrillic_v5"
	# As in the add-on with NVDA in Russian.
	return engine.OcrEngine(
		MODELS / "det_v6_tiny" / "inference.onnx",
		rec / "inference.onnx",
		rec / "inference.yml",
		allowed=engine.eastSlavicOnly,
	)


def test_bgraToBgr():
	bgr = np.random.default_rng(1).integers(0, 256, (7, 5, 3), dtype=np.uint8)
	bgra = np.dstack([bgr, np.full((7, 5), 255, np.uint8)])
	out = engine.bgraToBgr(bgra.tobytes(), 5, 7)
	assert out.shape == (7, 5, 3)
	assert out.flags["C_CONTIGUOUS"]
	assert np.array_equal(out, bgr)


@pytest.mark.parametrize(
	"src, expected",
	[
		("DIREcTorY", "DIRECTORY"),
		("НАСТРоЙКИ", "НАСТРОЙКИ"),
		("iPhone", "iPhone"),
		("GitHub", "GitHub"),
		("IDs", "IDs"),
		("macOS", "macOS"),
		("NVDAObjects", "NVDAObjects"),
		("Settings", "Settings"),
	],
)
def test_fix_case(src, expected):
	assert engine._fix_case(src) == expected


@pytest.mark.parametrize("size", [(1920, 1080), (3840, 2160)])
def test_word_positions(ocr, size):
	w, h = size
	img = np.full((h, w, 3), 255, np.uint8)
	k = h / 1080
	# Words at known places: (text, left edge, baseline).
	placed = [("Hello", 100, 200), ("World", 700, 200), ("Settings", 1200, 600)]
	for text, x, y in placed:
		cv2.putText(
			img,
			text,
			(int(x * k), int(y * k)),
			cv2.FONT_HERSHEY_SIMPLEX,
			1.2 * k,
			(0, 0, 0),
			max(1, int(2 * k)),
		)
	words = [wd for line in ocr.recognize(img) for wd in line]
	for text, x, y in placed:
		found = [wd for wd in words if wd["text"] == text]
		assert found, f"Not found '{text}', got: {[wd['text'] for wd in words]}"
		wd = found[0]
		cx, cy = wd["x"] + wd["width"] / 2, wd["y"] + wd["height"] / 2
		# The middle of a word box must fall on the word itself, otherwise a click misses.
		(tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * k, max(1, int(2 * k)))
		assert x * k <= cx <= x * k + tw
		assert y * k - th <= cy <= y * k + th * 0.3


@pytest.mark.parametrize("text", ["Settings", "Cancel", "Open file"])
def test_small_object(ocr, text):
	# A button cut tightly around its text: the word must be found and its box must stay inside the image.
	font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
	(tw, th), base = cv2.getTextSize(text, font, scale, thick)
	img = np.full((th + base + 4, tw + 4, 3), 240, np.uint8)
	cv2.putText(img, text, (2, th + 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
	words = [wd for line in ocr.recognize(img) for wd in line]
	assert " ".join(wd["text"] for wd in words) == text
	h, w = img.shape[:2]
	for wd in words:
		assert 0 <= wd["x"] and wd["x"] + wd["width"] <= w
		assert 0 <= wd["y"] and wd["y"] + wd["height"] <= h


def test_wide_lines(ocr):
	# Very wide boxes are read one at a time and squeezed past REC_MAX_RATIO: words stay inside their boxes.
	img = np.full((20, 9000, 3), 255, np.uint8)
	cv2.putText(img, "Settings", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
	boxes = [(0, 0, 9000, 20), (0, 0, 2000, 20), (0, 0, 80, 20), (0, 0, 60, 20)]
	results = ocr._rec.read(img, boxes)
	assert len(results) == len(boxes)
	for (x0, _y0, x1, _y1), (words, _score) in zip(boxes, results, strict=True):
		for wd in words:
			assert x0 <= wd["x"] and wd["x"] + wd["width"] <= x1


def test_latinOnly():
	# Language "en": Latin with diacritics, punctuation and arrows are allowed; CJK, Greek and Cyrillic are not.
	assert all(engine.latinOnly(c) for c in "Aé9+…→")
	assert not any(engine.latinOnly(c) for c in "平γП∏")


def test_cyrillic_masks():
	# Cyrillic mode: East Slavic letters only, except with NVDA in another Cyrillic language or Serbian,
	# then all modern Cyrillic.
	assert all(engine.eastSlavicOnly(c) for c in "ЁёЇїЎўҐґЪъAz9+…→№")
	assert not any(engine.eastSlavicOnly(c) for c in "һЈјҚқӘәЂѣ")
	assert all(engine.anyCyrillic(c) for c in "ЁёһЈјҚқӘәЂђЉљ№")
	# Historic letters stay banned in both.
	assert not any(engine.anyCyrillic(c) for c in "ѢѣѪѫꙀⷠ")


@pytest.mark.parametrize(
	"raw, fixed",
	[
		("AlI", "All"),
		("AII", "All"),
		("FiII", "Fill"),
		("AddIn", "AddIn"),
		("ID", "ID"),
		("AI", "AI"),
		("III", "III"),
		("Illegal", "Illegal"),
		("iPhone", "iPhone"),
		("HeIlo", "HeIlo"),
		("TlGHT", "TIGHT"),
		("MlDl", "MIDI"),
		("Al", "Al"),
		("tIGHT", "tIGHT"),
		("XII", "XII"),
		("VIII", "VIII"),
		("XlI", "XII"),
		("userID", "userID"),
		("getID", "getID"),
	],
)
def test_fix_il(raw, fixed):
	assert engine._fix_il(raw) == fixed


def test_pick_twin_script():
	# "Сору" among English words → Latin; "Bce" among Russian ones → Cyrillic; unclear with no neighbours → by screen.
	w = lambda text, alt=None: {"text": text, **({"alt": alt} if alt else {})}
	items = [
		[w("Сору", "Copy"), w("Paste")],
		[w("Bce", "Все"), w("программы")],
		[w("Туре", "Type")],
		[w("Edit")],
	]
	engine._pick_twin_script(items)
	assert [x["text"] for ws in items for x in ws] == ["Copy", "Paste", "Все", "программы", "Type", "Edit"]
	assert not any("alt" in x for ws in items for x in ws)
