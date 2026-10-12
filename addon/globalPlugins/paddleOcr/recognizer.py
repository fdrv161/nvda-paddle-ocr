# Paddle OCR add-on for NVDA
# Copyright (C) 2026 fdrv161
# SPDX-License-Identifier: GPL-2.0-or-later

"""Recognizer for NVDA: takes a capture from contentRecog and returns a LinesWordsResult."""

import ctypes
import os
import sys
import threading

import addonHandler
import config
import languageHandler
import ui
from contentRecog import ContentRecognizer, LinesWordsResult
from logHandler import log

addonHandler.initTranslation()

_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_DIR, "models")
# Language → (detector, recognizer). The detector is shared. The internal keys stay "en" and "ru"
# so that saved settings keep working; for the user these are the Latin and Cyrillic modes.
# "en" is the best model for Latin script; CJK is banned for it, otherwise icons are read as "平".
# "ru" is the Cyrillic model: 33 Cyrillic languages and English (English slightly worse than "en").
LANGUAGES = {
	"en": ("det_v6_tiny", "rec_v6_small"),
	"ru": ("det_v6_tiny", "rec_cyrillic_v5"),
}
# NVDA languages written in Cyrillic: for them the Cyrillic mode is the default.
_CYRILLIC_NVDA = {
	"ru",
	"uk",
	"be",
	"bg",
	"sr",
	"mk",
	"kk",
	"ky",
	"tg",
	"mn",
	"tt",
	"ba",
	"cv",
	"sah",
	"os",
	"ce",
}
# With these NVDA languages only their letters are allowed (see engine.eastSlavicOnly).
_EAST_SLAVIC_NVDA = {"ru", "uk", "be", "bg"}


def _nvdaLanguage():
	return languageHandler.getLanguage().split("_")[0]


def languageName(lang):
	# Translators: OCR language names, spoken when switching the Paddle OCR language.
	# The models read alphabets, not languages, so the modes are named by script.
	return {"en": _("Latin"), "ru": _("Cyrillic")}[lang]


def getLanguage(auto=False):
	"""The chosen language; if none (or auto), follow the NVDA language, like Windows OCR."""
	lang = config.conf["paddleOcr"]["language"]
	if lang in LANGUAGES and not auto:
		return lang
	return "ru" if _nvdaLanguage() in _CYRILLIC_NVDA else "en"


# One engine for the whole NVDA session: loading is slow, but requests after it are fast.
_engine = None
_engineLang = None
# One request at a time, even if two threads managed to start.
_runLock = threading.Lock()


def _getEngine(lang):
	global _engine, _engineLang
	if _engine is None or _engineLang != lang:
		lib = os.path.join(_DIR, "lib")
		# Append rather than prepend: NVDA's own modules must not be shadowed by ours.
		if lib not in sys.path:
			sys.path.append(lib)
		# numpy's OpenBLAS starts a thread per CPU core and reserves ~30 MB for each (~0.5 GB and 15 threads
		# on 16 cores). The engine does no matrix math in numpy, so load it single-threaded.
		# The variable is read only while loading, so it is restored right away
		# (processes started by NVDA later do not inherit it).
		old = os.environ.get("OPENBLAS_NUM_THREADS")
		os.environ["OPENBLAS_NUM_THREADS"] = "1"
		try:
			from . import engine
		finally:
			if old is None:
				del os.environ["OPENBLAS_NUM_THREADS"]
			else:
				os.environ["OPENBLAS_NUM_THREADS"] = old

		det, rec = LANGUAGES[lang]
		if lang == "en":
			allowed = engine.latinOnly
		elif _nvdaLanguage() in _CYRILLIC_NVDA - _EAST_SLAVIC_NVDA:
			# NVDA in Kazakh, Serbian, etc.: the letters of that language are needed.
			allowed = engine.anyCyrillic
		else:
			allowed = engine.eastSlavicOnly
		_engine = None  # Release the old engine first so two are never in memory at once.
		_engine = engine.OcrEngine(
			os.path.join(_MODELS, det, "inference.onnx"),
			os.path.join(_MODELS, rec, "inference.onnx"),
			os.path.join(_MODELS, rec, "inference.yml"),
			allowed=allowed,
		)
		_engineLang = lang
	return _engine


class PaddleOcr(ContentRecognizer):
	# recogUi reads these properties on every recognition, so settings take effect immediately.
	@property
	def allowAutoRefresh(self):
		return config.conf["paddleOcr"]["autoRefresh"]

	@property
	def autoSayAllOnResult(self):
		return config.conf["paddleOcr"]["autoSayAllOnResult"]

	def __init__(self):
		# Tag of the current request: if the request was cancelled, its result is silently dropped.
		self._current = None

	def getResizeFactor(self, width, height):
		# The engine picks the scale itself and returns coordinates in capture pixels.
		return 1

	def recognize(self, pixels, imageInfo, onResult):
		w, h = imageInfo.recogWidth, imageInfo.recogHeight
		# Copy the capture right away: NVDA's buffer may be gone after this method returns.
		buf = ctypes.string_at(ctypes.addressof(pixels), w * h * 4)
		lang = getLanguage()
		if _engine is None or _engineLang != lang:
			# Translators: Spoken once, while OCR models are loaded for the first recognition.
			ui.message(_("Loading OCR models"))
		token = object()
		self._current = token
		threading.Thread(
			target=self._run,
			args=(buf, imageInfo, onResult, token, lang),
			name="paddleOcr",
			daemon=True,
		).start()

	def _run(self, buf, imageInfo, onResult, token, lang):
		with _runLock:
			if self._current is not token:
				return
			try:
				eng = _getEngine(lang)
				from . import engine

				bgr = engine.bgraToBgr(buf, imageInfo.recogWidth, imageInfo.recogHeight)
				result = LinesWordsResult(eng.recognize(bgr), imageInfo)
			except Exception as e:
				log.exception("Paddle OCR: recognition failed")
				result = e
		if self._current is token:
			self._current = None
			# recogUi itself moves the result to the NVDA main thread.
			onResult(result)

	def cancel(self):
		self._current = None
