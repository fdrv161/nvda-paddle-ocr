"""Распознаватель для NVDA: принимает снимок от contentRecog и отдаёт LinesWordsResult."""
import ctypes
import os
import sys
import threading

import ui
from contentRecog import ContentRecognizer, LinesWordsResult
from logHandler import log

import addonHandler

addonHandler.initTranslation()

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_DIR, "models")
# Язык → (детектор, распознаватель). Детектор общий.
LANGUAGES = {
	"ru": ("det_v6_tiny", "rec_eslav_v5"),
	"multi": ("det_v6_tiny", "rec_v6_tiny"),
}

# Движок держим один на всё время работы NVDA: загрузка долгая, а после неё запросы быстрые.
_engine = None
_engineLang = None
# Один запрос за раз, даже если два потока успели стартовать.
_runLock = threading.Lock()


def _getEngine(lang):
	global _engine, _engineLang
	if _engine is None or _engineLang != lang:
		lib = os.path.join(_DIR, "lib")
		# В конец, а не в начало: свои модули NVDA не должны подменяться нашими.
		if lib not in sys.path:
			sys.path.append(lib)
		from . import engine
		det, rec = LANGUAGES[lang]
		_engine = None  # Сначала отпускаем старый движок, чтобы в памяти не было двух сразу.
		_engine = engine.OcrEngine(
			os.path.join(_MODELS, det, "inference.onnx"),
			os.path.join(_MODELS, rec, "inference.onnx"),
			os.path.join(_MODELS, rec, "inference.yml"),
		)
		_engineLang = lang
	return _engine


class PaddleOcr(ContentRecognizer):
	allowAutoRefresh = False
	# Пока нет настроек (этап 3), результат читаем сразу.
	autoSayAllOnResult = True

	def __init__(self, language="ru"):
		self.language = language
		# Метка текущего запроса: если запрос отменили, его результат молча выбрасываем.
		self._current = None

	def getResizeFactor(self, width, height):
		# Масштаб выбирает сам движок, а координаты он возвращает в пикселях снимка.
		return 1

	def recognize(self, pixels, imageInfo, onResult):
		w, h = imageInfo.recogWidth, imageInfo.recogHeight
		# Копируем снимок сразу: буфер NVDA после возврата из этого метода может исчезнуть.
		buf = ctypes.string_at(ctypes.addressof(pixels), w * h * 4)
		if _engine is None or _engineLang != self.language:
			# Translators: Spoken once, while OCR models are loaded for the first recognition.
			ui.message(_("Loading OCR models"))
		token = object()
		self._current = token
		threading.Thread(
			target=self._run, args=(buf, imageInfo, onResult, token),
			name="paddleOcr", daemon=True,
		).start()

	def _run(self, buf, imageInfo, onResult, token):
		with _runLock:
			if self._current is not token:
				return
			try:
				eng = _getEngine(self.language)
				from . import engine
				bgr = engine.bgra_to_bgr(buf, imageInfo.recogWidth, imageInfo.recogHeight)
				result = LinesWordsResult(eng.recognize(bgr), imageInfo)
			except Exception as e:
				log.error("Paddle OCR: recognition failed", exc_info=True)
				result = e
		if self._current is token:
			self._current = None
			# recogUi сам переносит результат в главный поток NVDA.
			onResult(result)

	def cancel(self):
		self._current = None
