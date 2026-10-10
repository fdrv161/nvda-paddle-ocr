"""Готовит содержимое дополнения: библиотеки под Python NVDA и нужные модели.

Запуск: .venv/Scripts/python tools/fetch_libs.py
1. Скачивает колёса cp313 win_amd64 (NVDA 2026.3 работает на Python 3.13, а .venv может быть новее)
   и распаковывает их в addon/globalPlugins/paddleOcr/lib/.
2. Копирует выбранные на этапе 0 модели из models/ в addon/globalPlugins/paddleOcr/models/
   (сначала скачать их: tools/fetch_models.py).
Повторный запуск пересобирает lib/ и models/ дополнения с нуля.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "addon" / "globalPlugins" / "paddleOcr"
WHEELS = ROOT / "tmp" / "wheels"

PACKAGES = ("onnxruntime==1.30.0", "numpy==2.5.3", "opencv-python-headless==5.0.0.93")
MODELS = ("det_v6_tiny", "rec_eslav_v5", "rec_v6_tiny")
MODEL_FILES = ("inference.onnx", "inference.yml")


def fetch_wheels():
	WHEELS.mkdir(parents=True, exist_ok=True)
	# --no-deps: питоновские зависимости onnxruntime (protobuf, sympy и т.п.) для запуска модели не нужны.
	subprocess.run([
		sys.executable, "-m", "pip", "download", "--quiet", "--no-deps",
		"--only-binary=:all:", "--python-version", "3.13", "--platform", "win_amd64",
		"--dest", str(WHEELS), *PACKAGES,
	], check=True)


def unpack_wheels():
	lib = PLUGIN / "lib"
	if lib.exists():
		shutil.rmtree(lib)
	lib.mkdir()
	for pkg in PACKAGES:
		name, version = pkg.split("==")
		pattern = f"{name.replace('-', '_')}-{version}-*cp313*win_amd64.whl"
		found = list(WHEELS.glob(pattern)) or list(WHEELS.glob(f"{name.replace('-', '_')}-{version}-*abi3*win_amd64.whl"))
		if not found:
			raise SystemExit(f"Не найдено колесо {pattern} в {WHEELS}")
		with zipfile.ZipFile(found[0]) as z:
			# Папки .dist-info не нужны для работы, только место занимают.
			for item in z.namelist():
				if ".dist-info/" not in item:
					z.extract(item, lib)
		print(f"Распаковано: {found[0].name}")
	# Тесты, заголовки C и инструменты в дополнении не нужны.
	for junk in ("numpy/_core/tests", "numpy/tests", "numpy/_core/include", "onnxruntime/tools",
			"onnxruntime/transformers", "onnxruntime/quantization", "onnxruntime/datasets", "cv2/data"):
		shutil.rmtree(lib / junk, ignore_errors=True)
	# ffmpeg нужен OpenCV только для видео и грузится лишь при работе с видео; это ~1/4 размера cv2.
	for f in (lib / "cv2").glob("opencv_videoio_ffmpeg*.dll"):
		f.unlink()


def copy_models():
	dest = PLUGIN / "models"
	if dest.exists():
		shutil.rmtree(dest)
	for name in MODELS:
		(dest / name).mkdir(parents=True)
		for fname in MODEL_FILES:
			src = ROOT / "models" / name / fname
			if not src.exists():
				raise SystemExit(f"Нет {src}. Сначала запустите tools/fetch_models.py")
			shutil.copy2(src, dest / name / fname)
		print(f"Скопирована модель: {name}")


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
