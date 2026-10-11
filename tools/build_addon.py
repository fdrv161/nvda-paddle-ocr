"""Packs the addon/ folder into a .nvda-addon file in the project root.

Run: .venv/Scripts/python tools/build_addon.py
Before the first build: tools/fetch_models.py, then tools/fetch_libs.py.
"""
import ast
import re
import struct
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADDON = ROOT / "addon"


def parse_po(path):
	"""Simple .po parser: msgid/msgstr, quoted continuation lines. Empty translations are skipped."""
	entries, key, cur = {}, None, {}
	for line in path.read_text(encoding="utf-8").splitlines() + [""]:
		line = line.strip()
		m = re.match(r'(msgid|msgstr)\s+(".*")$', line)
		if m:
			if m.group(1) == "msgid" and "msgstr" in cur:
				entries[cur["msgid"]] = cur["msgstr"]
				cur = {}
			key = m.group(1)
			cur[key] = ast.literal_eval(m.group(2))
		elif line.startswith('"') and key:
			cur[key] += ast.literal_eval(line)
		elif not line and "msgstr" in cur:
			entries[cur["msgid"]] = cur["msgstr"]
			cur, key = {}, None
	return {k: v for k, v in entries.items() if v}


def compile_po(path):
	"""A .mo file (GNU gettext format): header, two tables (length, offset), then the strings."""
	entries = sorted((k.encode("utf-8"), v.encode("utf-8")) for k, v in parse_po(path).items())
	n = len(entries)
	keysStart = 7 * 4 + 16 * n
	ids = strs = b""
	offsets = []
	for k, v in entries:
		offsets.append((len(k), len(ids), len(v), len(strs)))
		ids += k + b"\0"
		strs += v + b"\0"
	valuesStart = keysStart + len(ids)
	table = b"".join(struct.pack("<II", kl, keysStart + ko) for kl, ko, _, _ in offsets)
	table += b"".join(struct.pack("<II", vl, valuesStart + vo) for _, _, vl, vo in offsets)
	header = struct.pack("<7I", 0x950412DE, 0, n, 7 * 4, 7 * 4 + 8 * n, 0, 0)
	return header + table + ids + strs


def main():
	manifest = (ADDON / "manifest.ini").read_text(encoding="utf-8")
	# Values may be quoted (the NVDA guide asks for quotes), so take what is inside them.
	name = re.search(r'^name\s*=\s*"?([^"\s]+)', manifest, re.MULTILINE).group(1)
	version = re.search(r'^version\s*=\s*"?([^"\s]+)', manifest, re.MULTILINE).group(1)
	plugin = ADDON / "globalPlugins" / name
	for need in ("lib", "models"):
		if not (plugin / need).is_dir():
			print(f"Ошибка: нет папки {plugin / need}. Запустите tools/fetch_libs.py")
			return 1
	# Remind about untranslated strings so English does not show up in the Russian interface.
	used = set()
	for f in plugin.glob("*.py"):
		used |= {ast.literal_eval(s) for s in re.findall(r'_\(("(?:[^"\\]|\\.)*")\)', f.read_text(encoding="utf-8"))}
	for po in ADDON.glob("locale/*/LC_MESSAGES/nvda.po"):
		for s in sorted(used - set(parse_po(po))):
			print(f"Нет перевода ({po.parent.parent.name}): {s}")
	out = ROOT / f"{name}-{version}.nvda-addon"
	count = 0
	with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
		for f in sorted(ADDON.rglob("*")):
			if f.is_file() and "__pycache__" not in f.parts:
				arc = f.relative_to(ADDON).as_posix()
				z.write(f, arc)
				count += 1
				if f.suffix == ".po":
					# NVDA reads translations from .mo; build it here, without separate gettext tools.
					z.writestr(arc[:-3] + ".mo", compile_po(f))
					count += 1
		# The add-on's own license (GPL), next to the licenses of the bundled models and libraries.
		z.write(ROOT / "LICENSE", "COPYING.txt")
		count += 1
	print(f"Готово: {out.name}, файлов {count}, {out.stat().st_size / 2**20:.0f} МБ")
	return 0


if __name__ == "__main__":
	sys.exit(main())
