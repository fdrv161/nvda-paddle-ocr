"""The engine's image operations without OpenCV (numpy and an onnxruntime Resize) against OpenCV itself.

The add-on no longer ships OpenCV; the tests keep its results as the reference.
Run: .venv/Scripts/python -m pytest tests/test_image_ops.py
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "addon" / "globalPlugins" / "paddleOcr"))
import engine  # noqa: E402 (needs the sys.path line above)


def _cvBoxes(pred, sx, sy, w, h, top=0):
	"""engine._db_boxes as it was with OpenCV (PaddleOCR's way)."""
	bitmap = (pred > engine.DET_THRESH).astype(np.uint8)
	contours, _hierarchy = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	boxes = []
	for cnt in contours:
		bx, by, bw, bh = cv2.boundingRect(cnt)
		if min(bw, bh) < engine.DET_MIN_SIZE:
			continue
		mask = np.zeros((bh, bw), dtype=np.uint8)
		cv2.drawContours(mask, [cnt - (bx, by)], -1, 1, -1)
		if cv2.mean(pred[by : by + bh, bx : bx + bw], mask)[0] < engine.DET_BOX_THRESH:
			continue
		(cx, cy), (rw, rh), angle = cv2.minAreaRect(cnt)
		d = rw * rh * engine.DET_UNCLIP_RATIO / (2 * (rw + rh))
		pts = cv2.boxPoints(((cx, cy), (rw + 2 * d, rh + 2 * d), angle))
		if min(rw, rh) + 2 * d < engine.DET_MIN_SIZE + 2:
			continue
		x0 = max(0, math.floor(pts[:, 0].min() * sx))
		y0 = max(0, math.floor((pts[:, 1].min() + top) * sy))
		x1 = min(w, math.ceil(pts[:, 0].max() * sx))
		y1 = min(h, math.ceil((pts[:, 1].max() + top) * sy))
		if x1 - x0 >= 2 and y1 - y0 >= 2:
			boxes.append((x0, y0, x1, y1))
	return boxes


def _hasHoles(pred):
	bitmap = (pred > engine.DET_THRESH).astype(np.uint8)
	_contours, hierarchy = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	return hierarchy is not None and (hierarchy[0][:, 3] != -1).any()


def _maps():
	"""Detector-like maps: lines of text as soft blocks, some blurred, some tilted, some at the edges."""
	rng = np.random.default_rng(7)
	for t in range(200):
		h, w = int(rng.integers(8, 200)), int(rng.integers(8, 400))
		pred = np.zeros((h, w), dtype=np.float32)
		for _ in range(int(rng.integers(1, 30))):
			y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
			pred[y : y + int(rng.integers(1, 20)), x : x + int(rng.integers(1, 120))] = rng.uniform(0.3, 1.0)
		if t % 3 == 0:
			pred = cv2.GaussianBlur(pred, (0, 0), rng.uniform(0.5, 3))
		if t % 5 == 0:
			turn = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-30, 30), 1)
			pred = cv2.warpAffine(pred, turn, (w, h))
		args = (
			rng.uniform(0.5, 2),
			rng.uniform(0.5, 2),
			int(w * 1.5),
			int(h * 1.5),
			int(rng.integers(0, 50)),
		)
		yield np.clip(pred, 0, 1).astype(np.float32), args


def test_db_boxes_like_opencv():
	compared = 0
	for pred, args in _maps():
		# The border of a hole is a box of its own only with OpenCV (see _db_boxes).
		if _hasHoles(pred):
			continue
		assert engine._db_boxes(pred, *args) == _cvBoxes(pred, *args)
		compared += 1
	assert compared > 100


@pytest.mark.parametrize("shape", [(1, 1), (1, 50), (50, 1), (3, 3), (64, 200)])
@pytest.mark.parametrize("value", [0.0, 0.3, 0.9])
def test_db_boxes_flat_maps(shape, value):
	pred = np.full(shape, value, dtype=np.float32)
	assert engine._db_boxes(pred, 1.0, 1.0, shape[1], shape[0]) == _cvBoxes(
		pred, 1.0, 1.0, shape[1], shape[0]
	)


def test_db_boxes_hole_counts_in_mean():
	"""A frame around an empty middle is no line: the middle counts in its mean, as with OpenCV,
	and so does a blob inside it."""
	pred = np.zeros((60, 90), dtype=np.float32)
	pred[5:55, 5:85] = 0.95
	pred[7:53, 7:83] = 0.0
	pred[25:35, 30:60] = 0.9
	boxes = engine._db_boxes(pred, 1.0, 1.0, 90, 60)
	assert boxes == _cvBoxes(pred, 1.0, 1.0, 90, 60)
	assert len(boxes) == 1  # only the blob in the middle
	# The frame alone has a high mean: without counting the hole it would be a box.
	frame = pred.copy()
	frame[25:35, 30:60] = 0.0
	assert engine._db_boxes(frame, 1.0, 1.0, 90, 60) == []


@pytest.mark.parametrize(
	"h, w, width",
	[
		(21, 650, 1486),
		(14, 120, 412),
		(30, 1900, 3040),
		(80, 600, 360),
		(12, 40, 160),
		(45, 300, 320),
		(48, 100, 100),
		(2, 9, 216),
		(20, 4000, 960),
	],
)
def test_line_input_like_opencv(h, w, width):
	rng = np.random.default_rng(h * w)
	screen = rng.integers(0, 256, (300, 4100, 3), dtype=np.uint8)
	crop = screen[10 : 10 + h, 20 : 20 + w]  # a view, as in the engine
	batch = np.zeros((2, 3, engine.REC_HEIGHT, width + 5), dtype=np.float32)
	engine._line_input(crop, width, batch[0, :, :, :width])
	ref = cv2.resize(crop, (width, engine.REC_HEIGHT)).astype(np.float32).transpose(2, 0, 1) / 127.5 - 1
	diff = np.abs(batch[0, :, :, :width] - ref)
	# OpenCV rounds the resized pixels to whole levels, the engine does not: at most one level apart.
	assert diff.max() <= 1 / 127.5 + 1e-5
	assert not batch[0, :, :, width:].any()


@pytest.mark.parametrize(
	"w, h, rw, rh",
	[(1920, 1080, 1920, 1088), (1013, 737, 1024, 736), (60, 40, 64, 32), (7680, 400, 4000, 192)],
)
def test_resize_model_like_opencv(w, h, rw, rh):
	rng = np.random.default_rng(w)
	img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
	session = engine._session(engine._resize_model(), 0, oneShot=True)
	out = session.run(None, {"image": img[None], "sizes": np.array([1, rh, rw, 3], dtype=np.int64)})[0][0]
	ref = cv2.resize(img, (rw, rh))
	assert out.shape == ref.shape
	assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1


def test_resize_model_is_valid_onnx():
	onnx = pytest.importorskip("onnx")
	onnx.checker.check_model(onnx.load_from_string(engine._resize_model()), full_check=True)


def test_add_margin_like_opencv():
	rng = np.random.default_rng(3)
	img = rng.integers(0, 256, (20, 50, 3), dtype=np.uint8)
	edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
	color = [int(v) for v in np.median(edge, axis=0)]
	ref = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=color)
	assert np.array_equal(engine._add_margin(img, 16), ref)
