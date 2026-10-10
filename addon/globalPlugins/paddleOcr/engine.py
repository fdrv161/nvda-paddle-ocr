"""Движок OCR на моделях PaddleOCR (детектор DB + распознаватель CTC) через onnxruntime.

Ничего не импортирует из NVDA, поэтому его можно проверять обычным Python.
Вход — картинка BGR (numpy, uint8, h×w×3). Выход — строки экрана, каждая строка —
список слов с координатами в пикселях этой же картинки (формат LinesWordsResult NVDA).
"""
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Настройки постобработки детектора из inference.yml моделей PP-OCRv6.
DET_THRESH = 0.2
DET_BOX_THRESH = 0.45
DET_UNCLIP_RATIO = 1.4
DET_MIN_SIZE = 3
# Больше этого по длинной стороне картинку для детектора уменьшаем.
# Экран уменьшать вредно (мелкий шрифт пропадает), так что предел высокий — только против 8K и т.п.
DET_MAX_SIDE = 4000
REC_HEIGHT = 48
# По замерам пачка 4–8 быстрее всего; больше — медленнее (лишнее добивание по ширине).
REC_BATCH = 4
# Строки, в которых распознаватель не уверен (обычно иконки и рамки), выбрасываем.
REC_MIN_SCORE = 0.5
# Куски одной высоты с промежутком больше стольких высот строки считаем разными строками (колонками).
LINE_GAP = 10

_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_charset(ymlPath):
	"""Достаёт словарь символов из inference.yml без библиотеки yaml.

	Словарь там записан списком строк вида "  - 'а'" после "character_dict:".
	"""
	chars = []
	inDict = False
	for line in Path(ymlPath).read_text(encoding="utf-8").splitlines():
		if line.strip() == "character_dict:":
			inDict = True
			continue
		if inDict:
			# Только ASCII-пробелы: strip() съел бы символы-пробелы из самого словаря (например, U+3000).
			s = line.lstrip(" ")
			if not s.startswith("- "):
				break
			chars.append(_yaml_scalar(s[2:]))
	if not chars:
		raise ValueError(f"В {ymlPath} нет character_dict")
	return chars


def _yaml_scalar(s):
	# Символы в yml бывают в одинарных кавычках ('' внутри = одна кавычка),
	# в двойных (с escape-последовательностями вроде "\\") или без кавычек.
	if len(s) >= 2 and s[0] == s[-1] == "'":
		return s[1:-1].replace("''", "'")
	if len(s) >= 2 and s[0] == s[-1] == '"':
		return s[1:-1].encode("latin-1", "backslashreplace").decode("unicode_escape")
	return s


def _session(path, threads):
	opts = ort.SessionOptions()
	opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
	if threads:
		opts.intra_op_num_threads = threads
	opts.log_severity_level = 3
	return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _run_options():
	# После каждого запуска отдаём системе рабочую память, иначе пик большого экрана висит до конца.
	opts = ort.RunOptions()
	opts.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0")
	return opts


class OcrEngine:
	"""Детектор + распознаватель. allowed(c) — какие символы распознавателю можно выдавать (None — все)."""

	def __init__(self, detPath, recPath, recYml, threads=0, allowed=None):
		self._det = _session(detPath, threads)
		self._detInput = self._det.get_inputs()[0].name
		self._rec = Reader(recPath, recYml, threads, allowed)
		self._runOpts = _run_options()

	def recognize(self, bgr):
		"""Возвращает список строк; строка — список слов {x, y, width, height, text}."""
		boxes = self._detect(bgr)
		if not boxes:
			return []
		items = [words for words, score in self._rec.read(bgr, boxes) if words and score >= REC_MIN_SCORE]
		return _group_lines(items)

	# --- детектор ---

	def _detect(self, bgr):
		h, w = bgr.shape[:2]
		scale = min(1.0, DET_MAX_SIDE / max(h, w))
		# Модель требует стороны, кратные 32.
		rh = max(32, int(round(h * scale / 32)) * 32)
		rw = max(32, int(round(w * scale / 32)) * 32)
		img = cv2.resize(bgr, (rw, rh)) if (rw, rh) != (w, h) else bgr
		x = (img.astype(np.float32) / 255.0 - _DET_MEAN) / _DET_STD
		x = x.transpose(2, 0, 1)[None]
		pred = self._det.run(None, {self._detInput: x}, self._runOpts)[0][0, 0]
		return _db_boxes(pred, w / rw, h / rh, w, h)


class Reader:
	"""Распознаватель строк (CTC) одной модели."""

	def __init__(self, recPath, recYml, threads=0, allowed=None):
		self._rec = _session(recPath, threads)
		self._recInput = self._rec.get_inputs()[0].name
		# Индекс 0 — «пусто» (blank) в CTC, в конце модели знают ещё пробел.
		self._chars = ["", *_load_charset(recYml), " "]
		classes = self._rec.get_outputs()[0].shape[-1]
		if isinstance(classes, int) and classes != len(self._chars):
			raise ValueError(f"Словарь ({len(self._chars)}) не совпадает с моделью ({classes})")
		# Запрещённые символы (allowed(c) ложно) при расшифровке не выбираются никогда.
		self._banned = None
		if allowed:
			self._banned = np.array([i > 0 and not allowed(c) for i, c in enumerate(self._chars)])
		# Номера латинских и кириллических букв в словаре — для исправления смешанных слов.
		self._latin = np.array([i for i, c in enumerate(self._chars) if _script(c) == "latin"])
		self._cyr = np.array([i for i, c in enumerate(self._chars) if _script(c) == "cyr"])
		self._runOpts = _run_options()

	def read(self, bgr, boxes):
		"""Для каждой рамки — (слова, уверенность); слов может не быть."""
		crops = [bgr[y0:y1, x0:x1] for (x0, y0, x1, y1) in boxes]
		# Сортируем по пропорциям, чтобы в одной пачке были строки похожей длины (меньше пустого добивания).
		order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / crops[i].shape[0])
		results = [None] * len(crops)
		for start in range(0, len(order), REC_BATCH):
			idx = order[start:start + REC_BATCH]
			maxRatio = max(crops[i].shape[1] / crops[i].shape[0] for i in idx)
			batchW = max(int(math.ceil(REC_HEIGHT * maxRatio)), REC_HEIGHT)
			batch = np.zeros((len(idx), 3, REC_HEIGHT, batchW), dtype=np.float32)
			widths = []
			for j, i in enumerate(idx):
				ch, cw = crops[i].shape[:2]
				rw = min(batchW, max(1, int(math.ceil(REC_HEIGHT * cw / ch))))
				img = cv2.resize(crops[i], (rw, REC_HEIGHT)).astype(np.float32)
				batch[j, :, :, :rw] = ((img / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)
				widths.append(rw)
			probs = self._rec.run(None, {self._recInput: batch}, self._runOpts)[0]
			for j, i in enumerate(idx):
				results[i] = self._ctc_words(probs[j], batchW, widths[j], boxes[i])
		return results

	def _ctc_words(self, prob, batchW, resizedW, box):
		"""CTC-декодирование + координаты слов по позициям букв в строке."""
		if self._banned is not None:
			prob = prob.copy()
			prob[:, self._banned] = 0
		ids = prob.argmax(axis=1)
		conf = prob.max(axis=1)
		steps = len(ids)
		# Один шаг CTC покрывает batchW/steps пикселей картинки высотой 48;
		# переводим в пиксели исходной строки.
		x0, y0, x1, y1 = box
		toBox = (batchW / steps) * ((x1 - x0) / resizedW)
		chars = []  # (символ, шаг)
		prev = 0
		for t in range(steps):
			c = int(ids[t])
			if c != 0 and c != prev:
				chars.append((self._chars[c], t))
			prev = c
		if not chars:
			return [], 0.0
		score = float(np.mean([conf[t] for _, t in chars]))
		words = []
		cur = []
		for ch, t in chars + [(" ", steps)]:
			if ch == " ":
				if cur:
					cur = self._fix_mixed_script(cur, prob)
					wx0 = x0 + int(cur[0][1] * toBox)
					wx1 = min(x1, x0 + int(math.ceil((cur[-1][1] + 1) * toBox)))
					words.append({
						"x": wx0, "y": y0, "width": max(1, wx1 - wx0), "height": y1 - y0,
						"text": _fix_case("".join(c for c, _ in cur)),
					})
					cur = []
			else:
				cur.append((ch, t))
		return words, score

	def _fix_mixed_script(self, word, prob):
		"""Убирает буквы-двойники чужой азбуки: «Itет» → «Item», «Windоws» → «Windows».

		Модель для русского знает и латиницу, и кириллицу, и в похожих буквах (е/e, а/a, т/m)
		иногда ошибается азбукой. Если в слове есть буквы обеих азбук, выбираем главную по сумме
		вероятностей на позициях букв и каждую чужую букву заменяем лучшей буквой главной азбуки.
		"""
		if not len(self._latin) or not len(self._cyr):
			return word
		# Смотрим отдельно каждый кусок из букв подряд: в «Автотюнчик-1727.wav» кириллица и латиница
		# законно стоят в разных кусках, а чинить надо только кусок, где азбуки перемешаны.
		fixed = list(word)
		i = 0
		while i < len(fixed):
			if _script(fixed[i][0]) is None:
				i += 1
				continue
			j = i
			while j < len(fixed) and _script(fixed[j][0]) is not None:
				j += 1
			run = fixed[i:j]
			if {"latin", "cyr"} <= {_script(c) for c, _ in run}:
				steps = [t for _, t in run]
				latinMass = prob[steps][:, self._latin].sum()
				cyrMass = prob[steps][:, self._cyr].sum()
				main, mainIdx = ("latin", self._latin) if latinMass >= cyrMass else ("cyr", self._cyr)
				for k in range(i, j):
					c, t = fixed[k]
					if _script(c) != main:
						fixed[k] = (self._chars[int(mainIdx[prob[t, mainIdx].argmax()])], t)
			i = j
		return fixed


def _fix_case(text):
	"""«DIREcTorY» → «DIRECTORY»: в мелких заглавных надписях модель путает регистр.

	Чиним, только если заглавных больше половины И после строчной снова идёт заглавная —
	так не трогаем нормальные «iPhone», «GitHub», «IDs», «macOS».
	"""
	letters = [c for c in text if c.isalpha()]
	if len(letters) < 4:
		return text
	upper = sum(c.isupper() for c in letters)
	if upper * 2 <= len(letters) or upper == len(letters):
		return text
	if not any(a.islower() and b.isupper() for a, b in zip(letters, letters[1:])):
		return text
	return text.upper()


def latin_only(c):
	"""Латиница, цифры, знаки и стрелки. Иероглифы и прочее запрещаем: иконки v6 уверенно читает как «平»."""
	o = ord(c)
	return o < 0x250 or 0x2000 <= o < 0x2200



def bgra_to_bgr(buf, width, height):
	"""Снимок экрана NVDA (байты RGBQUAD: синий, зелёный, красный, пусто) → картинка BGR для движка."""
	return np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)[:, :, :3].copy()


def _script(c):
	"""Азбука буквы: 'latin', 'cyr' или None (цифры, знаки и прочее)."""
	if not c.isalpha():
		return None
	o = ord(c)
	if 0x400 <= o <= 0x4FF:
		return "cyr"
	if o < 0x250:
		return "latin"
	return None


def _db_boxes(pred, sx, sy, w, h):
	"""Карта вероятностей детектора → прямоугольники строк (x0, y0, x1, y1) в пикселях исходной картинки."""
	bitmap = (pred > DET_THRESH).astype(np.uint8)
	contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	boxes = []
	for cnt in contours:
		bx, by, bw, bh = cv2.boundingRect(cnt)
		if min(bw, bh) < DET_MIN_SIZE:
			continue
		# Уверенность — средняя вероятность внутри контура (как в PaddleOCR, режим fast).
		mask = np.zeros((bh, bw), dtype=np.uint8)
		cv2.drawContours(mask, [cnt - (bx, by)], -1, 1, -1)
		if cv2.mean(pred[by:by + bh, bx:bx + bw], mask)[0] < DET_BOX_THRESH:
			continue
		# Детектор находит «сжатую» середину строки; расширяем её обратно (unclip).
		# Для прямоугольника расстояние = площадь * ratio / периметр.
		(cx, cy), (rw, rh), angle = cv2.minAreaRect(cnt)
		d = rw * rh * DET_UNCLIP_RATIO / (2 * (rw + rh))
		pts = cv2.boxPoints(((cx, cy), (rw + 2 * d, rh + 2 * d), angle))
		if min(rw, rh) + 2 * d < DET_MIN_SIZE + 2:
			continue
		# На экране текст горизонтальный, поэтому берём обычный прямоугольник вокруг повёрнутого.
		x0 = max(0, int(math.floor(pts[:, 0].min() * sx)))
		y0 = max(0, int(math.floor(pts[:, 1].min() * sy)))
		x1 = min(w, int(math.ceil(pts[:, 0].max() * sx)))
		y1 = min(h, int(math.ceil(pts[:, 1].max() * sy)))
		if x1 - x0 >= 2 and y1 - y0 >= 2:
			boxes.append((x0, y0, x1, y1))
	return boxes


def _group_lines(items):
	"""Куски текста на одной высоте собирает в одну строку, сверху вниз и слева направо."""
	items.sort(key=lambda ws: ws[0]["y"] + ws[0]["height"] / 2)
	lines = []  # [верх, низ, [куски]]
	for ws in items:
		top = ws[0]["y"]
		bottom = top + ws[0]["height"]
		if lines:
			lt, lb, parts = lines[-1]
			# Одна строка, если по высоте куски перекрываются больше чем наполовину меньшего.
			overlap = min(lb, bottom) - max(lt, top)
			if overlap > 0.5 * min(lb - lt, bottom - top):
				parts.append(ws)
				lines[-1][0] = min(lt, top)
				lines[-1][1] = max(lb, bottom)
				continue
		lines.append([top, bottom, [ws]])
	result = []
	for _, _, parts in lines:
		parts.sort(key=lambda ws: ws[0]["x"])
		cur = list(parts[0])
		for ws in parts[1:]:
			# Большой промежуток — это уже другая колонка (список слева, подсказка справа):
			# читаем её отдельной строкой, чтобы не получалось «Human Voices Make a Category Selection».
			gap = ws[0]["x"] - (cur[-1]["x"] + cur[-1]["width"])
			if gap > LINE_GAP * min(cur[-1]["height"], ws[0]["height"]):
				result.append(cur)
				cur = []
			cur.extend(ws)
		result.append(cur)
	return result
