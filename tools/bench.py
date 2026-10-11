"""Measures engine speed and memory on screenshots.

Run: .venv/Scripts/python tools/bench.py [images...]
Without arguments takes tests/screens/*.png; if there are none, draws a synthetic 1920×1080 screen.
Output is line by line, without tables.
"""
import ctypes
import ctypes.wintypes
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "addon" / "globalPlugins" / "paddleOcr"))
import engine  # noqa: E402

MODELS = ROOT / "models"
COMBOS = [
	("tiny", "det_v6_tiny", "rec_v6_tiny"),
	("small", "det_v6_small", "rec_v6_small"),
	("small + eslav", "det_v6_small", "rec_eslav_v5"),
	("tiny + eslav", "det_v6_tiny", "rec_eslav_v5"),
]


class _Mem(ctypes.Structure):
	_fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
		"PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
		"QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
		"PagefileUsage", "PeakPagefileUsage")]


def rss_mb():
	# Process working set (how much RAM it takes) via WinAPI, without psutil.
	m = _Mem()
	m.cb = ctypes.sizeof(m)
	k32 = ctypes.WinDLL("kernel32")
	psapi = ctypes.WinDLL("psapi")
	k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
	psapi.GetProcessMemoryInfo.argtypes = [ctypes.wintypes.HANDLE, ctypes.POINTER(_Mem), ctypes.wintypes.DWORD]
	psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(m), m.cb)
	return m.WorkingSetSize / 2**20


def synthetic():
	img = np.full((1080, 1920, 3), 255, np.uint8)
	words = "File Edit View Settings Help Open Save Close Window Tools Search Options".split()
	rng = np.random.default_rng(0)
	for row in range(40):
		x = 10
		while x < 1800:
			text = " ".join(rng.choice(words, 3))
			cv2.putText(img, text, (x, 25 + row * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
			x += 330
	return img


def main(args):
	paths = [Path(a) for a in args] or sorted((ROOT / "tests" / "screens").glob("*.png"))
	images = [(p.name, cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)) for p in paths]
	if not images:
		images = [("синтетический 1920x1080", synthetic())]
	print(f"Память до загрузки: {rss_mb():.0f} МБ")
	for label, det, rec in COMBOS:
		t = time.perf_counter()
		eng = engine.OcrEngine(MODELS / det / "inference.onnx", MODELS / rec / "inference.onnx", MODELS / rec / "inference.yml")
		load = time.perf_counter() - t
		print(f"\nМодели {label}: загрузка {load:.2f} с, память {rss_mb():.0f} МБ")
		for name, img in images:
			times = []
			for _ in range(3):
				t = time.perf_counter()
				lines = eng.recognize(img)
				times.append(time.perf_counter() - t)
			text = " | ".join(" ".join(w["text"] for w in ln) for ln in lines[:3])
			print(f"  {name}: {statistics.median(times):.2f} с, строк {len(lines)}, память {rss_mb():.0f} МБ. Начало: {text[:150]}")
		del eng


if __name__ == "__main__":
	main(sys.argv[1:])
