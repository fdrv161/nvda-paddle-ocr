"""Скачивает официальные ONNX-модели PaddleOCR с Hugging Face в папку models/.

Запуск: .venv/Scripts/python tools/fetch_models.py [имена моделей]
  без имён — все модели (для замеров); только для дополнения:
  .venv/Scripts/python tools/fetch_models.py det_v6_tiny rec_v6_small rec_eslav_v5
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


def main(names):
	unknown = [n for n in names if n not in REPOS]
	if unknown:
		print(f"Ошибка: неизвестные модели {', '.join(unknown)}. Есть: {', '.join(REPOS)}")
		return 2
	names = names or list(REPOS)
	# Дополняем уже записанный список, чтобы скачивание части моделей не стирало сведения об остальных.
	manifestPath = MODELS_DIR / "manifest.json"
	manifest = json.loads(manifestPath.read_text(encoding="utf-8")) if manifestPath.exists() else {}
	total = len(names) * len(FILES)
	done = 0
	for name in names:
		repo = REPOS[name]
		folder = MODELS_DIR / name
		folder.mkdir(parents=True, exist_ok=True)
		entry = {"repo": repo, "files": {}}
		for fname in FILES:
			dest = folder / fname
			if not dest.exists():
				download(f"https://huggingface.co/{repo}/resolve/main/{fname}", dest)
			entry["files"][fname] = {"size": dest.stat().st_size, "sha256": sha256(dest)}
			# Модель на Hugging Face могли заменить: тогда стоп, а не тихая сборка с другой моделью.
			known = manifest.get(name, {}).get("files", {}).get(fname, {}).get("sha256")
			if known and known != entry["files"][fname]["sha256"]:
				print(f"Ошибка: {name}/{fname} не совпадает с models/manifest.json (хэш другой). "
					"Проверьте модель; если замена нужна, удалите её запись из manifest.json.")
				return 1
			done += 1
			print(f"Готово {done} из {total}: {name}/{fname}")
		manifest[name] = entry
	manifestPath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
	print("Все модели на месте.")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))
