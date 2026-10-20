"""Engine checks without NVDA: capture conversion, word coordinates, character masks and text fixes.

Run: .venv/Scripts/python -m pytest tests/
"""

import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "addon" / "globalPlugins" / "paddleOcr"
sys.path.insert(0, str(PLUGIN))
import engine  # noqa: E402 (needs the sys.path line above)

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
		assert wd["x"] >= 0 and wd["x"] + wd["width"] <= w
		assert wd["y"] >= 0 and wd["y"] + wd["height"] <= h


@pytest.mark.parametrize("rows", [32, 416, 1088, 2176])
def test_strips(rows):
	# The own parts cover the detector input without gaps; every strip has a side that is a multiple of 32.
	strips = engine._strips(rows)
	assert strips[0][0] == 0 and strips[-1][1] == rows
	assert all(s[1] == t[0] for s, t in itertools.pairwise(strips))
	for a, b, s0, s1 in strips:
		assert s0 <= a < b <= s1 and s0 % 32 == 0 and s1 % 32 == 0
		assert b - a <= 1.5 * engine.DET_STRIP


def test_join_strips():
	# Two strips with a cut at row 100: own rows 0–100 and 100–200, the detector saw 0–132 and 68–200.
	def join(above, below):
		parts = [(0, 100, 0, 132, above), (100, 200, 68, 200, below)]
		return sorted(engine._join_strips(parts, 16))

	far = [(10, 20, 300, 40)], [(10, 160, 300, 180)]
	assert join(*far) == [(10, 20, 300, 40), (10, 160, 300, 180)]
	# The same line with its centre a pixel above the cut in one strip and below it in the other:
	# once, whichever side the centres land on.
	assert len(join([(10, 89, 300, 110)], [(11, 90, 301, 111)])) == 1
	assert len(join([(10, 90, 300, 111)], [(11, 89, 301, 110)])) == 1
	# A tall line cut by the bottom edge of the upper strip: the lower strip sees it whole.
	assert join([(10, 80, 300, 132)], [(10, 80, 300, 140)]) == [(10, 80, 300, 140)]
	# The upper strip merged two words that the lower one sees apart (it cuts them less): two boxes.
	assert join([(10, 96, 300, 132)], [(10, 96, 140, 136), (160, 96, 300, 136)]) == [
		(10, 96, 140, 136),
		(160, 96, 300, 136),
	]
	# Huge text cut by both strips: the box of the strip that misses less gets the rows the other one saw.
	assert join([(10, 40, 300, 134)], [(10, 66, 300, 170)]) == [(10, 40, 300, 170)]
	# A piece of a line whose centre is past the cut, not found by the owning strip: dropped.
	assert join([(10, 100, 300, 130)], []) == []
	# Neighbouring lines that touch at the cut stay two lines.
	assert join([(10, 80, 300, 101)], [(10, 99, 300, 120)]) == [(10, 80, 300, 101), (10, 99, 300, 120)]
	# One strip: the boxes as they are.
	assert engine._join_strips([(0, 50, 0, 50, [(1, 2, 3, 4)])], 16) == [(1, 2, 3, 4)]


def test_text_on_strip_cuts(ocr):
	# Words of several sizes right on the cuts between detector strips are read once each.
	h, w = 1080, 1920
	rows = round(h / 32) * 32
	cuts = [b * h / rows for _a, b, _s0, _s1 in engine._strips(rows)[:-1]]
	assert cuts
	for shift in (-1, 0, 1):
		img = np.full((h, w, 3), 255, np.uint8)
		placed = []
		for ci, cut in enumerate(cuts):
			x = 20
			for si, scale in enumerate((0.6, 1.2, 2.4)):
				word = ("Open", "Save", "Tools", "Help", "View", "Edit")[(ci * 3 + si) % 6]
				thick = max(1, int(scale * 2))
				(tw, th), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
				cv2.putText(
					img,
					word,
					(x, int(cut + shift + th / 2)),
					cv2.FONT_HERSHEY_SIMPLEX,
					scale,
					(0, 0, 0),
					thick,
				)
				placed.append(word)
				x += tw + 150
		words = [wd["text"] for line in ocr.recognize(img) for wd in line]
		for word in placed:
			assert words.count(word) == 1, f"shift {shift}: '{word}' in {words}"


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


def test_batch_width(ocr):
	# A batch of lines stays within REC_BATCH_WIDTH pixels (memory of a run grows with it);
	# only a single line may be wider. Every line is read once.
	reader = ocr._rec
	session = reader._rec
	shapes = []

	class Recording:
		def run(self, names, feeds, opts):
			shapes.append(feeds[reader._recInput].shape)
			return session.run(names, feeds, opts)

	img = np.full((20, 2600, 3), 255, np.uint8)
	boxes = [(0, 0, w, 20) for w in (60, 300, 900, 1200, 1300, 1500, 2600, 120, 2500)]
	reader._rec = Recording()
	try:
		assert len(reader.read(img, boxes)) == len(boxes)
	finally:
		reader._rec = session
	assert sum(n for n, _c, _h, _w in shapes) == len(boxes)
	for n, _c, _h, w in shapes:
		assert n == 1 or n * w <= engine.REC_BATCH_WIDTH


def test_slim_head():
	# The add-on's Latin recognizer has a slim head (tools/slim_head.py): it must read exactly
	# like the original model from models/, with the same confidence.
	slim = MODELS / "rec_v6_small"
	orig = ROOT / "models" / "rec_v6_small"
	if not slim.is_dir() or not orig.is_dir():
		pytest.skip("Needs rec_v6_small in the add-on and in models/")
	readers = [
		engine.Reader(d / "inference.onnx", d / "inference.yml", allowed=engine.latinOnly)
		for d in (orig, slim)
	]
	assert readers[1]._keepFeeds and not readers[0]._keepFeeds
	img = np.full((24, 1400, 3), 255, np.uint8)
	cv2.putText(img, "Open file Settings Cancel 2026", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
	# Boxes of different widths: a batch of narrow ones and a wide one read alone.
	boxes = [(0, 0, 1400, 24), (0, 0, 120, 24), (0, 0, 300, 24), (0, 0, 500, 24), (0, 0, 60, 24)]
	(a, b) = (r.read(img, boxes) for r in readers)
	for (wordsA, scoreA), (wordsB, scoreB) in zip(a, b, strict=True):
		assert wordsA == wordsB
		assert scoreA == pytest.approx(scoreB, abs=1e-4)
	assert "Settings" in [wd["text"] for wd in a[0][0]]


def test_slim_head_refuses_new_opset(tmp_path):
	# slim_head.py builds opset 11-12 nodes: a newer model must stop with a clear message, not a broken file.
	onnx = pytest.importorskip("onnx")
	sys.path.insert(0, str(ROOT / "tools"))
	from slim_head import slim_head

	graph = onnx.helper.make_graph(
		[onnx.helper.make_node("Softmax", ["x"], ["y"])],
		"g",
		[onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 2])],
		[onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2])],
	)
	src = tmp_path / "new.onnx"
	onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)]), str(src))
	with pytest.raises(SystemExit, match="opset 17"):
		slim_head(src, tmp_path / "out.onnx")
	assert not (tmp_path / "out.onnx").exists()


def test_latinOnly():
	# Language "en": Latin with diacritics, punctuation and arrows are allowed;
	# CJK, Greek and Cyrillic are not.
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
	# "Сору" among English words → Latin; "Bce" among Russian ones → Cyrillic;
	# unclear with no neighbours → by screen.
	def w(text, alt=None):
		return {"text": text, **({"alt": alt} if alt else {})}

	items = [
		[w("Сору", "Copy"), w("Paste")],
		[w("Bce", "Все"), w("программы")],
		[w("Туре", "Type")],
		[w("Edit")],
	]
	engine._pick_twin_script(items)
	assert [x["text"] for ws in items for x in ws] == ["Copy", "Paste", "Все", "программы", "Type", "Edit"]
	assert not any("alt" in x for ws in items for x in ws)


def _piece(text, x, y, width, height):
	return [{"x": x, "y": y, "width": width, "height": height, "text": text}]


def test_stacked_rows_stay_apart():
	# Windows Settings: a title, a switch label beside it that is taller than the title,
	# and a description right under the title. The label stretched the title's line down to the
	# description, which then joined it and was read first ("Description Title").
	items = [
		_piece("Title", 602, 490, 525, 23),
		_piece("Off", 1649, 497, 40, 26),
		_piece("Description", 601, 514, 938, 17),
	]
	lines = [" ".join(w["text"] for w in line) for line in engine._group_lines(items)]
	assert lines == ["Title", "Off", "Description"]


def test_row_of_different_heights_stays_one_line():
	items = [
		_piece("Big", 10, 100, 60, 30),
		_piece("small", 80, 108, 40, 14),
		_piece("next", 130, 104, 40, 20),
	]
	lines = [" ".join(w["text"] for w in line) for line in engine._group_lines(items)]
	assert lines == ["Big small next"]


def test_clipboard_picture_to_bgr():
	# Two pixels: opaque red, and fully transparent black, which must become white.
	rgb = bytes([255, 0, 0, 0, 0, 0])
	alpha = bytes([255, 0])
	bgr = engine.rgbToBgr(rgb, alpha, 2, 1)
	assert bgr.tolist() == [[[0, 0, 255], [255, 255, 255]]]
	assert engine.rgbToBgr(rgb, None, 2, 1).tolist() == [[[0, 0, 255], [0, 0, 0]]]
