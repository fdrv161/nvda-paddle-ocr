"""Собирает папку addon/ в файл .nvda-addon в корне проекта.

Запуск: .venv/Scripts/python tools/build_addon.py
Перед первой сборкой: tools/fetch_models.py, затем tools/fetch_libs.py.
"""
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADDON = ROOT / "addon"


def main():
	manifest = (ADDON / "manifest.ini").read_text(encoding="utf-8")
	name = re.search(r"^name\s*=\s*(\S+)", manifest, re.M).group(1)
	version = re.search(r"^version\s*=\s*(\S+)", manifest, re.M).group(1)
	plugin = ADDON / "globalPlugins" / name
	for need in ("lib", "models"):
		if not (plugin / need).is_dir():
			print(f"Ошибка: нет папки {plugin / need}. Запустите tools/fetch_libs.py")
			return 1
	out = ROOT / f"{name}-{version}.nvda-addon"
	count = 0
	with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
		for f in sorted(ADDON.rglob("*")):
			if f.is_file() and "__pycache__" not in f.parts:
				z.write(f, f.relative_to(ADDON).as_posix())
				count += 1
	print(f"Готово: {out.name}, файлов {count}, {out.stat().st_size / 2**20:.0f} МБ")
	return 0


if __name__ == "__main__":
	sys.exit(main())
