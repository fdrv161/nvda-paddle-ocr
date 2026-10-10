"""Скачивает официальные ONNX-модели PaddleOCR с Hugging Face в папку models/.

Запуск: .venv/Scripts/python tools/fetch_models.py
Повторный запуск не качает заново то, что уже есть.
В конце пишет models/manifest.json с размерами и sha256 — потом по нему
проверяем, что в дополнение попали ровно те файлы.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Имя папки -> репозиторий PaddlePaddle на Hugging Face. Все Apache 2.0.
REPOS = {
	"det_v6_tiny": "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
	"det_v6_small": "PaddlePaddle/PP-OCRv6_small_det_onnx",
	"det_v6_medium": "PaddlePaddle/PP-OCRv6_medium_det_onnx",
	"rec_v6_tiny": "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
	"rec_v6_small": "PaddlePaddle/PP-OCRv6_small_rec_onnx",
	"rec_v6_medium": "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
	"rec_eslav_v5": "PaddlePaddle/eslav_PP-OCRv5_mobile_rec_onnx",
}
# В inference.yml лежат настройки и словарь символов распознавателя.
FILES = ("inference.onnx", "inference.yml")


def sha256(path):
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def download(url, dest):
	# Сначала во временный файл: если связь оборвётся, битый файл не останется под настоящим именем.
	tmp = dest.with_suffix(dest.suffix + ".part")
	with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
		while chunk := resp.read(1 << 20):
			out.write(chunk)
	tmp.replace(dest)


def main():
	manifest = {}
	total = len(REPOS) * len(FILES)
	done = 0
	for name, repo in REPOS.items():
		folder = MODELS_DIR / name
		folder.mkdir(parents=True, exist_ok=True)
		entry = {"repo": repo, "files": {}}
		for fname in FILES:
			dest = folder / fname
			if not dest.exists():
				download(f"https://huggingface.co/{repo}/resolve/main/{fname}", dest)
			entry["files"][fname] = {"size": dest.stat().st_size, "sha256": sha256(dest)}
			done += 1
			print(f"Готово {done} из {total}: {name}/{fname}")
		manifest[name] = entry
	(MODELS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
	print("Все модели на месте.")
	return 0


if __name__ == "__main__":
	sys.exit(main())
