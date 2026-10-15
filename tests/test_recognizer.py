"""recognizer.py checks with stand-ins for NVDA modules: repeated presses, cancel, a stuck recognition,
auto refresh, the missing Visual C++ runtime and the library log.

Run: .venv/Scripts/python -m pytest tests/test_recognizer.py
"""

import ctypes
import importlib
import sys
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "addon" / "globalPlugins" / "paddleOcr"
# A package name of its own: the real paddleOcr/__init__.py needs wx and NVDA's GUI.
PKG = "paddleOcrUnderTest"
WAIT = 5

# Everything the stand-ins do, in order: ("say", text), ("result", value), ("recognize", None).
events = []


def _say(text):
	events.append(("say", text))


def _module(name, **attrs):
	module = types.ModuleType(name)
	module.__dict__.update(attrs)
	return module


class _Log:
	def __init__(self):
		self.warnings = []

	def debug(self, msg, *args, **kwargs):
		pass

	def error(self, msg, *args, **kwargs):
		pass

	def exception(self, msg, *args, **kwargs):
		pass

	def warning(self, msg, *args, **kwargs):
		self.warnings.append(msg)


class _ResultObject:
	"""Stand-in for recogUi.RecogResultNVDAObject."""


class _RecogUi:
	"""The parts of NVDA 2026.3 contentRecog.recogUi.recognizeNavigatorObject that matter here."""

	def __init__(self):
		self.active = None
		self.visible = True
		self.done = threading.Event()

	def recognizeNavigatorObject(self, recognizer):
		if isinstance(api.focus, _ResultObject):
			_say("Already in a content recognition result")
			return
		if not self.visible:
			_say("Content is not visible")
			return
		if self.active:
			self.active.cancel()
		_say("Recognizing")
		self.active = recognizer
		recognizer.recognize(_pixels(), _imageInfo(), self.onFirstResult)

	def onFirstResult(self, result):
		self.active = None
		events.append(("result", result))
		self.done.set()


def _pixels():
	return (ctypes.c_ubyte * (4 * 3 * 4))()


def _imageInfo():
	return types.SimpleNamespace(recogWidth=4, recogHeight=3)


class _FakeEngine:
	"""Stand-in for engine.py: recognition waits until the test opens the gate."""

	gate = threading.Event()
	latinOnly = eastSlavicOnly = anyCyrillic = None

	@staticmethod
	def bgraToBgr(buf, w, h):
		return buf

	class OcrEngine:
		def __init__(self, *args, **kwargs):
			pass

		def recognize(self, bgr):
			assert _FakeEngine.gate.wait(WAIT)
			return ["line"]


api = _module("api", focus=None)
api.getFocusObject = lambda: api.focus
log = _Log()
recogUi = _RecogUi()


def _initTranslation():
	# Like NVDA: _ appears in the calling module.
	sys._getframe(1).f_globals["_"] = lambda text: text


_STUBS = {
	"addonHandler": _module("addonHandler", initTranslation=_initTranslation),
	"config": _module("config", conf={"paddleOcr": {"language": "en", "autoRefresh": False}}),
	"languageHandler": _module("languageHandler", getLanguage=lambda: "en"),
	"queueHandler": _module(
		"queueHandler",
		eventQueue=object(),
		queueFunction=lambda queue, func, *args: func(*args),
	),
	"ui": _module("ui", message=_say),
	"logHandler": _module("logHandler", log=log),
	"api": api,
	"contentRecog": _module(
		"contentRecog",
		ContentRecognizer=object,
		LinesWordsResult=lambda lines, info: ("lines", tuple(lines)),
		recogUi=_module("contentRecog.recogUi"),
	),
}


@pytest.fixture
def rec(monkeypatch):
	"""The recognizer module with NVDA stand-ins; a fresh state for every test."""
	for name, module in _STUBS.items():
		monkeypatch.setitem(sys.modules, name, module)
	ui = _STUBS["contentRecog"].recogUi
	ui.RecogResultNVDAObject = _ResultObject
	ui.recognizeNavigatorObject = recogUi.recognizeNavigatorObject
	monkeypatch.setitem(sys.modules, "contentRecog.recogUi", ui)
	pkg = _module(PKG, __path__=[str(PLUGIN)])
	monkeypatch.setitem(sys.modules, PKG, pkg)
	monkeypatch.setitem(sys.modules, f"{PKG}.engine", _FakeEngine)
	monkeypatch.delitem(sys.modules, f"{PKG}.recognizer", raising=False)
	# _importEngine appends lib/ to sys.path: keep it out of the test process.
	monkeypatch.setattr(sys, "path", list(sys.path))
	module = importlib.import_module(f"{PKG}.recognizer")
	# Tests do not depend on the runtime of this computer; test_missing_runtime_is_explained puts it back.
	module.realCheckRuntime = module._checkRuntime
	monkeypatch.setattr(module, "_checkRuntime", lambda: None)
	events.clear()
	log.warnings.clear()
	api.focus = None
	recogUi.__init__()
	_FakeEngine.gate.clear()
	yield module
	# Let worker threads of the test finish.
	_FakeEngine.gate.set()


def _said():
	return [value for kind, value in events if kind == "say"]


def _results():
	return [value for kind, value in events if kind == "result"]


def _waitPending(ocr, value):
	end = time.monotonic() + WAIT
	while (ocr.pending() is None) != (value is None):
		assert time.monotonic() < end
		time.sleep(0.01)


def test_second_press_keeps_running(rec):
	ocr = rec.PaddleOcr()
	ocr.recognizeFromCommand()
	ocr.recognizeFromCommand()
	assert _said() == ["Recognizing", "Loading OCR models", "Recognition in progress"]
	_FakeEngine.gate.set()
	assert recogUi.done.wait(WAIT)
	assert _results() == [("lines", ("line",))]
	assert ocr.pending() is None


def test_press_after_result_starts_again(rec):
	ocr = rec.PaddleOcr()
	_FakeEngine.gate.set()
	ocr.recognizeFromCommand()
	assert recogUi.done.wait(WAIT)
	recogUi.done.clear()
	events.clear()
	ocr.recognizeFromCommand()
	assert recogUi.done.wait(WAIT)
	assert _said() == ["Recognizing"]
	assert len(_results()) == 1


def test_cancel_drops_result(rec):
	ocr = rec.PaddleOcr()
	ocr.recognizeFromCommand()
	ocr.cancel()
	assert ocr.pending() is None
	_FakeEngine.gate.set()
	time.sleep(0.2)
	assert _results() == []


def test_stuck_recognition_starts_over(rec, monkeypatch):
	monkeypatch.setattr(rec, "STUCK_AFTER", 0.2)
	ocr = rec.PaddleOcr()
	ocr.recognizeFromCommand()
	time.sleep(0.3)
	events.clear()
	ocr.recognizeFromCommand()
	said = _said()
	assert said[0] == "Recognizing"
	assert said[1].startswith("The previous recognition did not finish in 0.2 seconds")
	# The new request does not wait longer than STUCK_AFTER for the stuck one: NVDA gets an error.
	assert recogUi.done.wait(WAIT)
	assert isinstance(_results()[0], TimeoutError)
	# The stuck one finishes later; its result is dropped.
	_FakeEngine.gate.set()
	time.sleep(0.2)
	assert len(_results()) == 1


def test_stuck_notice_only_when_started(rec, monkeypatch):
	monkeypatch.setattr(rec, "STUCK_AFTER", 0.2)
	ocr = rec.PaddleOcr()
	ocr.recognizeFromCommand()
	time.sleep(0.3)
	events.clear()
	recogUi.visible = False
	ocr.recognizeFromCommand()
	assert _said() == ["Content is not visible"]
	# The notice is not left over for a later request.
	recogUi.visible = True
	ocr.recognizeFromCommand()
	said = _said()
	assert said[:2] == ["Content is not visible", "Recognizing"], said
	assert said[2].startswith("The previous recognition"), said


def test_auto_refresh_is_not_busy(rec):
	ocr = rec.PaddleOcr()
	_FakeEngine.gate.set()
	ocr.recognizeFromCommand()
	assert recogUi.done.wait(WAIT)
	# The result window asks for a refresh by itself; then the user closes it with Escape and presses
	# the command at once.
	_FakeEngine.gate.clear()
	refreshed = []
	ocr.recognize(_pixels(), _imageInfo(), refreshed.append)
	assert ocr.pending()[1] is False
	recogUi.done.clear()
	events.clear()
	ocr.recognizeFromCommand()
	assert _said() == ["Recognizing"]
	assert ocr.pending()[1] is True
	_FakeEngine.gate.set()
	assert recogUi.done.wait(WAIT)
	assert len(_results()) == 1
	assert refreshed == []


def test_in_result_window_nvda_answers(rec):
	ocr = rec.PaddleOcr()
	_FakeEngine.gate.set()
	ocr.recognizeFromCommand()
	assert recogUi.done.wait(WAIT)
	_FakeEngine.gate.clear()
	ocr.recognize(_pixels(), _imageInfo(), lambda result: None)
	api.focus = _ResultObject()
	events.clear()
	ocr.recognizeFromCommand()
	assert _said() == ["Already in a content recognition result"]


def test_missing_runtime_is_explained(rec, monkeypatch):
	def noDll(name):
		raise OSError(name)

	monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
	monkeypatch.setattr(ctypes, "WinDLL", noDll)
	monkeypatch.setattr(rec, "_checkRuntime", rec.realCheckRuntime)
	ocr = rec.PaddleOcr()
	ocr.recognizeFromCommand()
	assert recogUi.done.wait(WAIT)
	time.sleep(0.05)
	result = _results()[0]
	assert isinstance(result, rec.MissingRuntimeError)
	assert "MSVCP140_1.dll" in str(result)
	# The advice comes after the result, that is after NVDA's "Recognition failed".
	assert events[-1] == ("say", result.spoken)
	assert "Visual C++" in result.spoken


def test_library_outside_addon_is_logged(rec, monkeypatch):
	lib = str(PLUGIN / "lib")
	monkeypatch.setitem(sys.modules, "numpy", _module("numpy", __file__=f"{lib}/numpy/__init__.py"))
	monkeypatch.setitem(sys.modules, "cv2", _module("cv2", __file__=f"{lib}/cv2/__init__.py"))
	monkeypatch.setitem(
		sys.modules, "onnxruntime", _module("onnxruntime", __file__="C:/other/onnxruntime/__init__.py")
	)
	rec._logLibraries(lib)
	assert len(log.warnings) == 1
	assert "onnxruntime" in log.warnings[0]
