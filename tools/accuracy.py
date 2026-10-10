"""Линейка точности: картинки с заранее известным текстом → процент ошибок движка.

Рисует «экраны» шрифтами Windows (светлые и тёмные темы, разные размеры, русский и английский),
распознаёт и сравнивает с правильным текстом построчно.

Запуск:
  .venv/Scripts/python tools/accuracy.py                   # модели «Русский и английский»
  .venv/Scripts/python tools/accuracy.py --rec rec_v6_small --latin  # язык «en»
  .venv/Scripts/python tools/accuracy.py --set REC_MIN_SCORE=0.3 --show 20
--set меняет настройку движка (имя из engine.py), --show печатает столько худших строк.
Вывод построчный, без таблиц.
"""
import argparse
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "addon" / "globalPlugins" / "paddleOcr"))
import engine  # noqa: E402

MODELS = ROOT / "addon" / "globalPlugins" / "paddleOcr" / "models"
FONTS = ["segoeui.ttf", "tahoma.ttf", "arial.ttf", "verdana.ttf"]
# (название, фон, текст): обычные сочетания из программ.
THEMES = [
	("светлая", (255, 255, 255), (0, 0, 0)),
	("светлая серый текст", (240, 240, 240), (110, 110, 110)),
	("тёмная", (32, 32, 32), (230, 230, 230)),
	("тёмная бледный текст", (45, 45, 48), (150, 150, 150)),
	("цветная кнопка", (0, 103, 192), (255, 255, 255)),
]
# Высота шрифта в пикселях: 12 — это 9 пунктов при 100% (обычный текст Windows).
SIZES = [11, 12, 13, 15, 18]

EN = """File|Edit|View|Settings|Open Recent Files|Save As...|Check for updates|Preferences
Output Device: Speakers (Realtek Audio)|Sample Rate 48000 Hz|Buffer Size 256 samples|Master Volume -6.0 dB
C:\\Program Files\\Common Files\\VST3|Version 2.4.1 (build 1187)|Add to Favorites|Remove Selected
Export Audio Mixdown|Track 12 - Lead Synth|Cutoff Frequency|Resonance|Attack Decay Sustain Release
Low Pass 24 dB/oct|Ctrl+Shift+S|Show hidden files|Last modified: 10/09/2026 14:32|Download complete
Sign in with your account|Terms of Service|Privacy Policy|OK|Cancel|Apply|Next >|< Back
Install for all users|Enable MIDI input|Velocity Curve|Polyphony: 16 voices|Quality: High
Reset to default|Learn more about licensing|Unable to connect to server|Retry in 5 seconds
Pitch Bend Range +/- 2|Reverb Mix 35%|Delay Time 1/8 dotted|Bypass|Solo|Mute|Arm Recording
Snap to Grid|Zoom In|Zoom Out|Import MIDI File|Project Name: Untitled_01|Audio Engine Running
CPU 12% Disk 3%|Inbox (24)|Search messages|Mark all as read|Archive|Account Settings|All
Notifications|Dark Mode|Language: English|Keyboard Shortcuts|Help Center|Report a Problem
Visit www.example.com for details|support@example.com|Illegal file name|Fill level 100%
Allow all cookies|Install Illustrator|Bells and Vibes|Pads + Strings|Electronic Mayhem""".replace("\n", "|").split("|")

RU = """Файл|Правка|Вид|Настройки|Открыть недавние файлы|Сохранить как...|Проверить обновления
Параметры звука|Устройство вывода: Динамики|Частота дискретизации 48000 Гц|Громкость 75%
Добавить в избранное|Удалить выбранное|Экспорт проекта|Подключение к серверу не удалось
Повторить через 5 секунд|Войти в учётную запись|Политика конфиденциальности|Условия использования
Отмена|Применить|Далее|Назад|Установить для всех пользователей|Показать скрытые файлы
Дата изменения: 09.10.2026 14:32|Загрузка завершена|Сообщения|Отметить все как прочитанные
Входящие (24)|Поиск по сообщениям|Уведомления|Тёмная тема|Язык интерфейса: русский
Сочетания клавиш|Справка и поддержка|Сообщить о проблеме|Свободно 12,4 ГБ из 256 ГБ
Копирование файлов: осталось 3 минуты|Ещё|Разрешить доступ к микрофону|Все программы
Папка «Документы»|Щёлкните, чтобы продолжить|Объём памяти|Съёмка, вьюга, жёлтый хвощ
Вы уверены, что хотите выйти?|Сохранить изменения перед закрытием?|Новая вкладка""".replace("\n", "|").split("|")

MIXED = """Открыть в Visual Studio Code|Плагин Omnisphere 2.8|Версия NVDA 2026.3|Скачать Firefox
Настройки Wi-Fi|Файл README.md|Нажмите Ctrl+C, чтобы скопировать|Подключено: Bluetooth Headphones
Учётная запись Microsoft|Папка Program Files|Открыть с помощью Notepad++""".replace("\n", "|").split("|")


def make_screens():
	"""Список (название, картинка BGR, [(x0, y0, x1, y1, текст, язык)])."""
	rng = random.Random(0)
	screens = []
	for theme, bg, fg in THEMES:
		for size in SIZES:
			img = Image.new("RGB", (1400, 800), bg)
			draw = ImageDraw.Draw(img)
			truth = []
			for col, x in enumerate((20, 720)):
				y = 15
				i = 0
				while y < 800 - size * 2:
					r = rng.random()
					lang, pool = ("en", EN) if r < 0.5 else ("ru", RU) if r < 0.85 else ("mix", MIXED)
					text = rng.choice(pool)
					font = ImageFont.truetype(f"C:/Windows/Fonts/{FONTS[(col + i) % len(FONTS)]}", size)
					draw.text((x, y), text, font=font, fill=fg)
					x0, y0, x1, y1 = draw.textbbox((x, y), text, font=font)
					truth.append((x0, y0, x1, y1, text, lang))
					y += int(size * 1.9)
					i += 1
			bgr = np.array(img)[:, :, ::-1].copy()
			screens.append((f"{theme}, {size} пикс", bgr, truth))
	return screens


def levenshtein(a, b):
	prev = list(range(len(b) + 1))
	for i, ca in enumerate(a, 1):
		cur = [i]
		for j, cb in enumerate(b, 1):
			cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
		prev = cur
	return prev[-1]


def match(lines, truth):
	"""Слова распознанного текста раскладывает по правильным строкам (по центру слова)."""
	got = [[] for _ in truth]
	extra = 0
	for line in lines:
		for w in line:
			cx = w["x"] + w["width"] / 2
			cy = w["y"] + w["height"] / 2
			for k, (x0, y0, x1, y1, _, _) in enumerate(truth):
				if y0 - 4 <= cy <= y1 + 4 and x0 - 8 <= cx <= x1 + 8:
					got[k].append(w)
					break
			else:
				extra += 1
	return [" ".join(w["text"] for w in sorted(ws, key=lambda w: w["x"])) for ws in got], extra


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--det", default="det_v6_tiny")
	ap.add_argument("--rec", default="rec_eslav_v5")
	ap.add_argument("--latin", action="store_true", help="только латиница (как язык «en» в дополнении)")
	ap.add_argument("--set", action="append", default=[], metavar="ИМЯ=ЗНАЧЕНИЕ")
	ap.add_argument("--show", type=int, default=0, help="сколько худших строк напечатать")
	args = ap.parse_args()
	for s in args.set:
		name, value = s.split("=", 1)
		old = getattr(engine, name)
		setattr(engine, name, type(old)(value) if not isinstance(old, bool) else value.lower() in ("1", "true", "yes"))
	# Модели, которых нет в дополнении (small, medium), берём из общей папки models/.
	det, rec = (MODELS / m if (MODELS / m).exists() else ROOT / "models" / m for m in (args.det, args.rec))
	eng = engine.OcrEngine(det / "inference.onnx", rec / "inference.onnx", rec / "inference.yml",
		allowed=engine.latin_only if args.latin else None)
	screens = make_screens()
	errs, lens, times, rows = {}, {}, [], []
	totalExtra = 0
	for name, img, truth in screens:
		t = time.perf_counter()
		lines = eng.recognize(img)
		times.append(time.perf_counter() - t)
		hyps, extra = match(lines, truth)
		totalExtra += extra
		theme, size = name.rsplit(", ", 1)
		for (_, _, _, _, text, lang), hyp in zip(truth, hyps):
			# Многоточие одним знаком читается так же, это не ошибка.
			e = levenshtein(hyp.replace("…", "..."), text)
			for key in ("всё", f"язык {lang}", f"тема {theme}", f"размер {size}"):
				errs[key] = errs.get(key, 0) + e
				lens[key] = lens.get(key, 0) + len(text)
			rows.append((e / len(text), text, hyp, name))
	print(f"Модели {args.det} + {args.rec}{', только латиница' if args.latin else ''}, настройки: {' '.join(args.set) or 'обычные'}")
	print(f"Картинок {len(screens)}, строк {len(rows)}, среднее время {statistics.mean(times):.2f} с")
	for key in errs:
		print(f"{key}: ошибок {100 * errs[key] / lens[key]:.2f}%")
	perfect = sum(1 for r in rows if r[0] == 0)
	print(f"Строк без единой ошибки: {perfect} из {len(rows)}; лишних слов вне строк: {totalExtra}")
	for r, text, hyp, name in sorted(rows, reverse=True)[:args.show]:
		print(f"[{name}] надо «{text}», получено «{hyp}»")


if __name__ == "__main__":
	main()
