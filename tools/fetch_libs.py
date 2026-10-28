"""Prepares the add-on contents: libraries for the NVDA Python and the needed models.

Run: .venv/Scripts/python tools/fetch_libs.py
1. Downloads cp313 win_amd64 wheels (NVDA 2026.3 runs Python 3.13, while .venv may be newer)
   and unpacks them into addon/globalPlugins/paddleOcr/lib/.
2. Copies the chosen models from models/ into addon/globalPlugins/paddleOcr/models/
   (download them first: tools/fetch_models.py). The Latin recognizer gets a slim head
   on the way (slim_head.py, needs the onnx package).
Running it again rebuilds the add-on lib/ and models/ from scratch.
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "addon" / "globalPlugins" / "paddleOcr"
WHEELS = ROOT / "tmp" / "wheels"

PACKAGES = ("onnxruntime==1.30.0", "numpy==2.5.3")
MODELS = ("det_v6_tiny", "rec_cyrillic_v5", "rec_v6_small")
MODEL_FILES = ("inference.onnx", "inference.yml")
# Models with a large dictionary (18.7k characters) get a slim head, see slim_head.py.
# The Cyrillic model has 852 characters: there the head is small anyway.
SLIM_HEAD = ("rec_v6_small",)


def fetch_wheels():
	WHEELS.mkdir(parents=True, exist_ok=True)
	# --no-deps: onnxruntime's Python dependencies (protobuf, sympy etc.) are not needed to run a model.
	subprocess.run(
		[
			sys.executable,
			"-m",
			"pip",
			"download",
			"--quiet",
			"--no-deps",
			"--only-binary=:all:",
			"--python-version",
			"3.13",
			"--platform",
			"win_amd64",
			"--dest",
			str(WHEELS),
			*PACKAGES,
		],
		check=True,
	)


def unpack_wheels():
	lib = PLUGIN / "lib"
	if lib.exists():
		shutil.rmtree(lib)
	lib.mkdir()
	for pkg in PACKAGES:
		name, version = pkg.split("==")
		pattern = f"{name.replace('-', '_')}-{version}-*cp313*win_amd64.whl"
		found = list(WHEELS.glob(pattern))
		if not found:
			raise SystemExit(f"Не найдено колесо {pattern} в {WHEELS}")
		with zipfile.ZipFile(found[0]) as z:
			# .dist-info folders are not needed at runtime; keep only the licenses from them
			# (numpy has them only there).
			for item in z.namelist():
				if ".dist-info/" not in item:
					z.extract(item, lib)
				elif "/licenses/" in item and not item.endswith("/"):
					dest = lib / "licenses" / name / item.split("/licenses/", 1)[1]
					dest.parent.mkdir(parents=True, exist_ok=True)
					dest.write_bytes(z.read(item))
		print(f"Распаковано: {found[0].name}")
	# Tests, C headers and tools are not needed in the add-on.
	# f2py and the PyInstaller hook are loaded by numpy only on demand, never by the engine.
	for junk in (
		"numpy/_core/include",
		"numpy/f2py",
		"numpy/_pyinstaller",
		"onnxruntime/tools",
		"onnxruntime/transformers",
		"onnxruntime/quantization",
		"onnxruntime/datasets",
	):
		shutil.rmtree(lib / junk, ignore_errors=True)
	for d in [d for d in lib.rglob("tests") if d.is_dir()]:
		shutil.rmtree(d, ignore_errors=True)
	# Type hints for editors, numpy's pytest setup and its C test modules (used only by numpy's own tests).
	for pattern in ("*.pyi", "py.typed", "numpy/conftest.py", "numpy/_core/_*_tests.*"):
		for f in lib.glob(pattern if "/" in pattern else f"**/{pattern}"):
			f.unlink()


def copy_models():
	# Needs the onnx package, which only this step uses.
	from slim_head import slim_head

	dest = PLUGIN / "models"
	if dest.exists():
		shutil.rmtree(dest)
	for name in MODELS:
		(dest / name).mkdir(parents=True)
		for fname in MODEL_FILES:
			src = ROOT / "models" / name / fname
			if not src.exists():
				raise SystemExit(f"Нет {src}. Сначала запустите tools/fetch_models.py")
			if name in SLIM_HEAD and fname == "inference.onnx":
				slim_head(src, dest / name / fname)
			else:
				shutil.copy2(src, dest / name / fname)
		print(f"Скопирована модель: {name}" + (" (облегчённая голова)" if name in SLIM_HEAD else ""))


def size_mb(path):
	return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**20


def main():
	fetch_wheels()
	unpack_wheels()
	copy_models()
	print(f"Готово. lib: {size_mb(PLUGIN / 'lib'):.0f} МБ, models: {size_mb(PLUGIN / 'models'):.0f} МБ")
	return 0


if __name__ == "__main__":
	sys.exit(main())
