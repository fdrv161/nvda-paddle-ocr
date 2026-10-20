# Paddle OCR add-on for NVDA
# Copyright (C) 2026 fdrv161
# SPDX-License-Identifier: GPL-2.0-or-later

"""Recognizer for NVDA: takes a capture from contentRecog and returns a LinesWordsResult."""

import ctypes
import os
import sys
import threading
import time

import addonHandler
import config
import languageHandler
import queueHandler
import ui
from contentRecog import ContentRecognizer, LinesWordsResult, LwrTextInfo
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
# NVDA interface languages written in Cyrillic: for them the Cyrillic mode is the default.
# Only languages NVDA has a translation for: for any other Windows language NVDA falls back to English.
# Serbian is left out on purpose: NVDA's Serbian translation is in Latin script,
# and the Cyrillic model cannot read š, č, ć, ž, đ.
_CYRILLIC_NVDA = {
	"ru",
	"uk",
	"be",
	"bg",
	"mk",
	"ky",
	"mn",
}
# With these NVDA languages the Cyrillic mode allows letters of all Cyrillic alphabets without the setting:
# Macedonian, Kyrgyz and Mongolian need their own letters. NVDA's Serbian is in Latin script, but those who
# switch to the Cyrillic mode by hand need Serbian Cyrillic letters. Otherwise only Russian, Ukrainian,
# Belarusian and Bulgarian letters are allowed (see engine.eastSlavicOnly).
_ALL_CYRILLIC_NVDA = {"mk", "ky", "mn", "sr"}


def _nvdaLanguage():
	return languageHandler.getLanguage().split("_")[0]


def languageName(lang):
	# Translators: Recognition mode names, spoken when switching the Paddle OCR recognition mode.
	# The models read alphabets, not languages, so the modes are named by script.
	return {"en": _("Latin"), "ru": _("Cyrillic")}[lang]


def getLanguage(auto=False):
	"""The chosen language; if none (or auto), follow the NVDA language, like Windows OCR."""
	lang = config.conf["paddleOcr"]["language"]
	if lang in LANGUAGES and not auto:
		return lang
	return "ru" if _nvdaLanguage() in _CYRILLIC_NVDA else "en"


def _charset(lang):
	"""Which characters the recognizer may output: the name of a filter in engine.py.

	Called on NVDA's main thread, so the recognition thread does not read the config.
	"""
	if lang == "en":
		return "latinOnly"
	if config.conf["paddleOcr"]["allCyrillic"] or _nvdaLanguage() in _ALL_CYRILLIC_NVDA:
		return "anyCyrillic"
	return "eastSlavicOnly"


# One engine for the whole NVDA session: loading is slow, but requests after it are fast.
# It is rebuilt when its key (language, charset) changes.
_engine = None
_engineKey = None
# One request at a time, even if two threads managed to start.
_runLock = threading.Lock()
# A recognition running longer than this many seconds is taken as stuck: the command may start
# a new one, and a new one does not wait longer than this for the old one (a running model cannot
# be stopped; its result is dropped).
STUCK_AFTER = 30
# Where numpy, cv2 and onnxruntime came from is logged once per NVDA session.
_libsLogged = False


class MissingRuntimeError(Exception):
	"""The Visual C++ runtime that onnxruntime needs is not installed; spoken carries the advice."""

	def __init__(self, message, spoken):
		super().__init__(message)
		self.spoken = spoken


def _checkRuntime():
	# onnxruntime.dll needs MSVCP140_1.dll. NVDA ships MSVCP140.dll and vcruntime140*.dll but not this one,
	# so it comes from System32 (Visual C++ Redistributable). Without it Python only says
	# "DLL load failed ... The specified module could not be found" without the library name.
	try:
		ctypes.WinDLL("msvcp140_1.dll")
	except OSError:
		raise MissingRuntimeError(
			"Paddle OCR: MSVCP140_1.dll not found, install Microsoft Visual C++ Redistributable x64 "
			"and restart NVDA",
			# Translators: Spoken after "Recognition failed" when a system library
			# the add-on needs is missing.
			_(
				"Microsoft Visual C++ Redistributable x64 is missing. "
				"Install it from the Microsoft website and restart NVDA",
			),
		) from None


def _logLibraries(lib):
	# numpy, cv2 and onnxruntime are shared top-level modules in NVDA: if another add-on loaded its own copy
	# first, that copy is used. Logged so such a conflict can be found.
	for name in ("numpy", "cv2", "onnxruntime"):
		module = sys.modules.get(name)
		path = getattr(module, "__file__", None) or "?"
		log.debug(f"Paddle OCR: {name} {getattr(module, '__version__', '?')} from {path}")
		if not os.path.normcase(os.path.abspath(path)).startswith(os.path.normcase(lib) + os.sep):
			log.warning(f"Paddle OCR: {name} is loaded from {path}, not from the add-on; it may not work")


def _importEngine():
	"""Imports engine.py with numpy, cv2 and onnxruntime from the add-on lib/ folder."""
	lib = os.path.join(_DIR, "lib")
	# Append rather than prepend: NVDA's own modules must not be shadowed by ours.
	if lib not in sys.path:
		sys.path.append(lib)
	global _libsLogged
	# If onnxruntime is already loaded (by another add-on), the runtime is there.
	if "onnxruntime" not in sys.modules:
		_checkRuntime()
	# numpy's OpenBLAS starts a thread per CPU core and reserves ~30 MB for each (~0.5 GB and 15 threads
	# on 16 cores). The engine does no matrix math in numpy, so load it single-threaded.
	# cv2 prepends a non-existent folder to PATH for good; cv2.pyd needs only system DLLs.
	# Both are restored right away, so processes started by NVDA later do not inherit them.
	saved = {name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "PATH")}
	os.environ["OPENBLAS_NUM_THREADS"] = "1"
	try:
		from . import engine
	finally:
		for name, value in saved.items():
			if value is None:
				os.environ.pop(name, None)
			else:
				os.environ[name] = value
	if not _libsLogged:
		_libsLogged = True
		_logLibraries(lib)
	return engine


def _getEngine(key):
	global _engine, _engineKey
	if _engine is None or _engineKey != key:
		engine = _importEngine()
		lang, charset = key
		det, rec = LANGUAGES[lang]
		_engine = None  # Release the old engine first so two are never in memory at once.
		_engine = engine.OcrEngine(
			os.path.join(_MODELS, det, "inference.onnx"),
			os.path.join(_MODELS, rec, "inference.onnx"),
			os.path.join(_MODELS, rec, "inference.yml"),
			allowed=getattr(engine, charset),
		)
		_engineKey = key
	return _engine


# After a click the mouse goes back to where it was, but not at once: some programs handle the click
# a moment later and would see the mouse already gone.
CLICK_SETTLE_MS = 300
# Where the mouse was before the clicks, while its return is pending, and when the last click was:
# quick repeated clicks must return it to the user's place, not to the previous click point.
_mouseHome = None
_lastClick = 0.0


def _returnMouse():
	global _mouseHome
	import core
	import winUser

	wait = _lastClick + CLICK_SETTLE_MS / 1000 - time.monotonic()
	if wait > 0:
		core.callLater(int(wait * 1000) + 1, _returnMouse)
		return
	winUser.setCursorPos(*_mouseHome)
	_mouseHome = None


class PaddleTextInfo(LwrTextInfo):
	"""Enter in the result window clicks the middle of the word at the cursor."""

	def activate(self):
		# The same checks as NVDA's own activate: NotImplementedError is what recogUi expects.
		if not self.obj.isInForeground or not self.result.words:
			raise NotImplementedError
		import core
		import mouseHandler
		import winUser

		global _mouseHome, _lastClick
		# NVDA clicks pointAtStart, the top left corner of the word. On small buttons, links and dense lists
		# that corner is often just outside the element or on the line above; the middle is inside.
		point = self._getPointFromOffset(self.bookmark.startOffset)
		if _mouseHome is None:
			_mouseHome = tuple(winUser.getCursorPos())
			core.callLater(CLICK_SETTLE_MS, _returnMouse)
		_lastClick = time.monotonic()
		winUser.setCursorPos(point.x, point.y)
		mouseHandler.doPrimaryClick()


class PaddleResult(LinesWordsResult):
	def makeTextInfo(self, obj, position):
		return PaddleTextInfo(obj, position, self)


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
		# Checking and clearing the tag must not interleave with a new request setting it.
		self._tokenLock = threading.Lock()
		self._started = 0.0
		# Whether the current request came from the command (not from an auto refresh of a result).
		self._fromCommand = False
		# Set only while the command runs recogUi: a request asked then is the command's.
		# Both are used on NVDA's main thread only (auto refresh also asks from there).
		self._inCommand = False
		self._notice = None

	def pending(self):
		"""(seconds since the current request was asked, whether the command asked it), or None
		if there is no current request (delivered or cancelled)."""
		with self._tokenLock:
			if self._current is None:
				return None
			return time.monotonic() - self._started, self._fromCommand

	def recognizeFromCommand(self):
		"""The recognition command: the same as NVDA+R with Windows OCR, but a running recognition
		is kept instead of being cancelled and started over."""
		import api
		from contentRecog import recogUi

		if isinstance(api.getFocusObject(), recogUi.RecogResultNVDAObject):
			# NVDA itself says "Already in a content recognition result".
			recogUi.recognizeNavigatorObject(self)
			return
		notice = None
		pending = self.pending()
		if pending is not None:
			seconds, fromCommand = pending
			if seconds < STUCK_AFTER:
				# An auto refresh still running after the result was closed is not a reason to wait:
				# the new request replaces it.
				if fromCommand:
					# Translators: Spoken when the recognition command is pressed
					# while a recognition is running.
					ui.message(_("Recognition in progress"))
					return
			else:
				# Translators: Spoken when the command is pressed and the previous recognition runs too long:
				# it is dropped and a new one starts. {seconds} is a number of seconds.
				notice = _(
					"The previous recognition did not finish in {seconds} seconds, starting again. "
					"If it hangs again, restart NVDA",
				).format(seconds=STUCK_AFTER)
		# From here on everything is as with Windows OCR: checks of the object and Screen Curtain,
		# capture, result window, clicking words. The notice is spoken only if recogUi really
		# asks for a recognition (see recognize).
		self._inCommand = True
		self._notice = notice
		try:
			recogUi.recognizeNavigatorObject(self)
		finally:
			self._inCommand = False
			self._notice = None

	def getResizeFactor(self, width, height):
		# The engine picks the scale itself and returns coordinates in capture pixels.
		return 1

	def recognize(self, pixels, imageInfo, onResult):
		w, h = imageInfo.recogWidth, imageInfo.recogHeight
		# Copy the capture right away: NVDA's buffer may be gone after this method returns.
		buf = ctypes.string_at(ctypes.addressof(pixels), w * h * 4)
		if self._notice:
			ui.message(self._notice)
			self._notice = None
		lang = getLanguage()
		key = (lang, _charset(lang))
		if _engine is None or _engineKey != key:
			# Translators: Spoken once, while OCR models are loaded for the first recognition.
			ui.message(_("Loading OCR models"))
		token = object()
		with self._tokenLock:
			self._current = token
			self._started = time.monotonic()
			self._fromCommand = self._inCommand
		threading.Thread(
			target=self._run,
			args=(buf, imageInfo, onResult, token, key),
			name="paddleOcr",
			daemon=True,
		).start()

	def _run(self, buf, imageInfo, onResult, token, key):
		# A previous recognition stuck in the model keeps the lock: do not wait for it forever.
		if not _runLock.acquire(timeout=STUCK_AFTER):
			# recogUi itself logs the error and says "Recognition failed"; the advice to restart NVDA
			# was already spoken by the command (see recognizeFromCommand) and is in the help.
			if self._take(token):
				error = f"Paddle OCR: the previous recognition hangs over {STUCK_AFTER} s, restart NVDA"
				onResult(TimeoutError(error))
			return
		try:
			if self._current is not token:
				return
			try:
				start = time.perf_counter()
				eng = _getEngine(key)
				loaded = time.perf_counter()
				from . import engine

				bgr = engine.bgraToBgr(buf, imageInfo.recogWidth, imageInfo.recogHeight)
				lines = eng.recognize(bgr)
				# Only with the NVDA log level set to debug: to compare speed on other computers.
				# Numbers only, never the recognized text.
				log.debug(
					f"Paddle OCR: {key[0]} ({key[1]}), {imageInfo.recogWidth}x{imageInfo.recogHeight}, "
					f"{len(lines)} lines, loading {loaded - start:.2f} s, "
					f"recognition {time.perf_counter() - loaded:.2f} s",
				)
				result = PaddleResult(lines, imageInfo)
			except MissingRuntimeError as e:
				# No traceback: recogUi logs the message itself.
				result = e
			except Exception as e:
				log.exception("Paddle OCR: recognition failed")
				result = e
		finally:
			_runLock.release()
		if self._take(token):
			# recogUi itself moves the result to the NVDA main thread.
			onResult(result)
			if isinstance(result, MissingRuntimeError):
				# Queued the same way as recogUi's "Recognition failed", so the advice is spoken after it.
				queueHandler.queueFunction(queueHandler.eventQueue, ui.message, result.spoken)

	def _take(self, token):
		"""True if token is still the current request; it is then no longer pending."""
		with self._tokenLock:
			if self._current is not token:
				return False
			self._current = None
			return True

	def cancel(self):
		with self._tokenLock:
			self._current = None
