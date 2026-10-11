"""Downloads the official PaddleOCR ONNX models from Hugging Face into models/.

Run: .venv/Scripts/python tools/fetch_models.py [model names]
  without names: all models (for measurements); only those for the add-on:
  .venv/Scripts/python tools/fetch_models.py det_v6_tiny rec_v6_small rec_eslav_v5
Running it again does not re-download what is already there.
At the end writes models/manifest.json with sizes and sha256, used later to
check that exactly those files went into the add-on.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# Folder name -> PaddlePaddle repository on Hugging Face. All Apache 2.0.
REPOS = {
	"det_v6_tiny": "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
	"det_v6_small": "PaddlePaddle/PP-OCRv6_small_det_onnx",
	"det_v6_medium": "PaddlePaddle/PP-OCRv6_medium_det_onnx",
	"rec_v6_tiny": "PaddlePaddle/PP-OCRv6_tiny_rec_onnx",
	"rec_v6_small": "PaddlePaddle/PP-OCRv6_small_rec_onnx",
	"rec_v6_medium": "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
	"rec_eslav_v5": "PaddlePaddle/eslav_PP-OCRv5_mobile_rec_onnx",
}
# inference.yml holds the recognizer settings and character dictionary.
FILES = ("inference.onnx", "inference.yml")


def sha256(path):
	h = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(1 << 20), b""):
			h.update(chunk)
	return h.hexdigest()


def download(url, dest):
	# Into a temporary file first: if the connection drops, no broken file stays under the real name.
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
	# Merge into the existing list so downloading some models does not erase the records of the others.
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
			# The model on Hugging Face may have been replaced: then stop rather than silently build with another model.
			known = manifest.get(name, {}).get("files", {}).get(fname, {}).get("sha256")
			if known and known != entry["files"][fname]["sha256"]:
				print(
					f"Ошибка: {name}/{fname} не совпадает с models/manifest.json (хэш другой). "
					"Проверьте модель; если замена нужна, удалите её запись из manifest.json."
				)
				return 1
			done += 1
			print(f"Готово {done} из {total}: {name}/{fname}")
		manifest[name] = entry
	manifestPath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
	print("Все модели на месте.")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))
