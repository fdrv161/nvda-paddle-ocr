"""Measures engine speed and memory with the add-on's real modes.

Run: .venv/Scripts/python tools/bench.py [images...]
  --mode en|ru       only one mode (default: both, Latin and Cyrillic)
  --models DET REC   another pair of models from models/ instead of the modes, e.g.
                     --models det_v6_small rec_v6_small (download it first with tools/fetch_models.py)
Images: the given files, otherwise tests/screens/*.png, otherwise a synthetic 1920×1080 screen.
A synthetic console screen with full-width lines is always added: wide lines are the worst case for memory.
Each mode runs in its own process, so its memory peak is not mixed with another mode.
Output is line by line, without tables.
"""

import argparse
import ctypes
import ctypes.wintypes
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "addon" / "globalPlugins" / "paddleOcr"))
import engine

MODELS = ROOT / "models"
# The same models and character masks as in recognizer._getEngine (NVDA in Russian for the Cyrillic mode).
MODES = {
	"en": ("Латиница", "det_v6_tiny", "rec_v6_small", engine.latinOnly),
	"ru": ("Кириллица", "det_v6_tiny", "rec_cyrillic_v5", engine.eastSlavicOnly),
}


class _Mem(ctypes.Structure):
	_fields_ = [("cb", ctypes.wintypes.DWORD), ("PageFaultCount", ctypes.wintypes.DWORD)] + [
		(n, ctypes.c_size_t)
		for n in (
			"PeakWorkingSetSize",
			"WorkingSetSize",
			"QuotaPeakPagedPoolUsage",
			"QuotaPagedPoolUsage",
			"QuotaPeakNonPagedPoolUsage",
			"QuotaNonPagedPoolUsage",
			"PagefileUsage",
			"PeakPagefileUsage",
		)
	]


def memory():
	"""(now, peak) of the process working set in MB, via WinAPI without psutil."""
	m = _Mem()
	m.cb = ctypes.sizeof(m)
	k32 = ctypes.WinDLL("kernel32")
	psapi = ctypes.WinDLL("psapi")
	k32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
	psapi.GetProcessMemoryInfo.argtypes = [
		ctypes.wintypes.HANDLE,
		ctypes.POINTER(_Mem),
		ctypes.wintypes.DWORD,
	]
	psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(m), m.cb)
	return m.WorkingSetSize / 2**20, m.PeakWorkingSetSize / 2**20


def synthetic():
	img = np.full((1080, 1920, 3), 255, np.uint8)
	words = ["File", "Edit", "View", "Settings", "Help", "Open", "Save", "Close", "Window", "Tools", "Search", "Options"]
	rng = np.random.default_rng(0)
	for row in range(40):
		x = 10
		while x < 1800:
			text = " ".join(rng.choice(words, 3))
			cv2.putText(
				img, text, (x, 25 + row * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
			)
			x += 330
	return img


def console():
	"""A maximized console: light text on black, every line across the whole width."""
	img = np.zeros((1080, 1920, 3), np.uint8)
	words = ["Get-Process", "nvda", "WorkingSet64", "PS", "D:\\Nvda", "Select-Object", "Name", "-First", "10"]
	rng = np.random.default_rng(1)
	for row in range(45):
		text = " ".join(rng.choice(words, 30))
		cv2.putText(
			img, text, (5, 20 + row * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA
		)
	return img


def load_images(paths):
	paths = paths or sorted((ROOT / "tests" / "screens").glob("*.png"))
	images = [(p.name, cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)) for p in paths]
	if not images:
		images = [("синтетический экран 1920x1080", synthetic())]
	images.append(("синтетическая консоль 1920x1080, строки во всю ширину", console()))
	return images


def run_one(label, det, rec, allowed, paths):
	"""One mode in the current process: load time, speed, memory now and at the peak."""
	images = load_images(paths)
	now, _peak = memory()
	print(f"\n{label} ({det} + {rec}). Память до загрузки: {now:.0f} МБ")
	t = time.perf_counter()
	eng = engine.OcrEngine(
		MODELS / det / "inference.onnx",
		MODELS / rec / "inference.onnx",
		MODELS / rec / "inference.yml",
		allowed=allowed,
	)
	now, peak = memory()
	print(f"Загрузка {time.perf_counter() - t:.2f} с, память после загрузки {now:.0f} МБ")
	for name, img in images:
		times = []
		for _ in range(3):
			t = time.perf_counter()
			lines = eng.recognize(img)
			times.append(time.perf_counter() - t)
		before = peak
		now, peak = memory()
		grown = f"пик вырос до {peak:.0f} МБ" if peak > before + 1 else f"пик прежний, {peak:.0f} МБ"
		text = " | ".join(" ".join(w["text"] for w in ln) for ln in lines[:2])
		seconds = statistics.median(times)
		print(f"  {name}: {seconds:.2f} с, строк {len(lines)}, сейчас {now:.0f} МБ, {grown}.")
		print(f"    Начало: {text[:120]}")


def main():
	# Recognized text may hold characters the console code page lacks (arrows): print "?" instead of failing.
	sys.stdout.reconfigure(errors="replace")
	parser = argparse.ArgumentParser(description="Скорость и память движка в режимах дополнения.")
	parser.add_argument("images", nargs="*", type=Path)
	parser.add_argument("--mode", choices=list(MODES))
	parser.add_argument("--models", nargs=2, metavar=("DET", "REC"))
	# Internal: run one mode in this process (the parent starts a process per mode).
	parser.add_argument("--one", help=argparse.SUPPRESS)
	args = parser.parse_args()
	if args.models:
		det, rec = args.models
		run_one("Свои модели", det, rec, None, args.images)
		return 0
	if args.one:
		label, det, rec, allowed = MODES[args.one]
		run_one(label, det, rec, allowed, args.images)
		return 0
	for mode in [args.mode] if args.mode else list(MODES):
		cmd = [sys.executable, __file__, "--one", mode, *map(str, args.images)]
		if subprocess.run(cmd, check=False).returncode:
			return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
