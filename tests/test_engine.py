"""Проверки движка без NVDA: перевод снимка NVDA в картинку и координаты слов.

Запуск: .venv/Scripts/python -m pytest tests/
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "addon" / "globalPlugins" / "paddleOcr"
sys.path.insert(0, str(PLUGIN))
import engine  # noqa: E402

MODELS = PLUGIN / "models"


@pytest.fixture(scope="module")
def ocr():
	if not (MODELS / "rec_eslav_v5").is_dir():
		pytest.skip("Нет моделей в дополнении: запустите tools/fetch_libs.py")
	rec = MODELS / "rec_eslav_v5"
	return engine.OcrEngine(MODELS / "det_v6_tiny" / "inference.onnx", rec / "inference.onnx", rec / "inference.yml")


def test_bgra_to_bgr():
	bgr = np.random.default_rng(1).integers(0, 256, (7, 5, 3), dtype=np.uint8)
	bgra = np.dstack([bgr, np.full((7, 5), 255, np.uint8)])
	out = engine.bgra_to_bgr(bgra.tobytes(), 5, 7)
	assert out.shape == (7, 5, 3)
	assert out.flags["C_CONTIGUOUS"]
	assert np.array_equal(out, bgr)


@pytest.mark.parametrize("src, expected", [
	("DIREcTorY", "DIRECTORY"),
	("НАСТРоЙКИ", "НАСТРОЙКИ"),
	("iPhone", "iPhone"),
	("GitHub", "GitHub"),
	("IDs", "IDs"),
	("macOS", "macOS"),
	("NVDAObjects", "NVDAObjects"),
	("Settings", "Settings"),
])
def test_fix_case(src, expected):
	assert engine._fix_case(src) == expected


@pytest.mark.parametrize("size", [(1920, 1080), (3840, 2160)])
def test_word_positions(ocr, size):
	w, h = size
	img = np.full((h, w, 3), 255, np.uint8)
	k = h / 1080
	# Слова в известных местах: (текст, левый край, базовая линия).
	placed = [("Hello", 100, 200), ("World", 700, 200), ("Settings", 1200, 600)]
	for text, x, y in placed:
		cv2.putText(img, text, (int(x * k), int(y * k)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * k, (0, 0, 0), max(1, int(2 * k)))
	words = [wd for line in ocr.recognize(img) for wd in line]
	for text, x, y in placed:
		found = [wd for wd in words if wd["text"] == text]
		assert found, f"Не найдено «{text}», есть: {[wd['text'] for wd in words]}"
		wd = found[0]
		cx, cy = wd["x"] + wd["width"] / 2, wd["y"] + wd["height"] / 2
		# Середина рамки слова должна попадать в само слово, иначе клик промахнётся.
		(tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * k, max(1, int(2 * k)))
		assert x * k <= cx <= x * k + tw
		assert y * k - th <= cy <= y * k + th * 0.3


def test_latin_only():
	# Язык «en»: латиница с диакритикой, знаки и стрелки можно; иероглифы, греческий и кириллицу нельзя.
	assert all(engine.latin_only(c) for c in "Aé9+…→")
	assert not any(engine.latin_only(c) for c in "平γП∏")
