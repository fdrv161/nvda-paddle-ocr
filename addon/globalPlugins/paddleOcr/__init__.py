# Paddle OCR add-on for NVDA
# Copyright (C) 2026 fdrv161
# SPDX-License-Identifier: GPL-2.0-or-later

"""Paddle OCR: recognizes screen text with PaddleOCR models, like NVDA+R with Windows OCR."""

import contextlib

import addonHandler
import api
import config
import globalPluginHandler
import globalVars
import gui
import textInfos
import ui
import wx
from contentRecog import recogUi
from gui import guiHelper
from gui.settingsDialogs import NVDASettingsDialog, SettingsPanel
from scriptHandler import script

# recognizer.py is light: it loads the heavy libraries and models only on the first recognition.
from .recognizer import LANGUAGES, PaddleResult, getLanguage, languageName

addonHandler.initTranslation()

# An empty language means "same as NVDA" (see recognizer.getLanguage). The rest mirrors Windows OCR.
config.conf.spec["paddleOcr"] = {
	"language": "string(default='')",
	# Cyrillic mode: letters of all Cyrillic alphabets, not only Russian, Ukrainian, Belarusian, Bulgarian.
	"allCyrillic": "boolean(default=false)",
	"autoRefresh": "boolean(default=false)",
	"autoSayAllOnResult": "boolean(default=true)",
}

# Keys of the add-on's result window, besides NVDA's own (Enter clicks, Escape closes). They are bound
# only while that window has the focus, so these keys keep working as usual everywhere else.
_RESULT_GESTURES = {
	"kb:shift+enter": "doubleClickResult",
	"kb:applications": "rightClickResult",
	"kb:shift+f10": "rightClickResult",
}


def _ourResult(obj):
	return isinstance(obj, recogUi.RecogResultNVDAObject) and isinstance(obj.result, PaddleResult)


class PaddleOcrPanel(SettingsPanel):
	# Translators: Title of the Paddle OCR category in NVDA settings.
	title = _("Paddle OCR")

	def makeSettings(self, settingsSizer):
		sHelper = guiHelper.BoxSizerHelper(self, sizer=settingsSizer)
		conf = config.conf["paddleOcr"]
		self._langs = ["", *LANGUAGES]
		# Translators: Recognition mode choice in Paddle OCR settings: by the NVDA interface language.
		names = [_("Follow NVDA language ({language})").format(language=languageName(getLanguage(auto=True)))]
		names += [languageName(lang) for lang in LANGUAGES]
		self.languageList = sHelper.addLabeledControl(
			# Translators: Label of the recognition mode list in Paddle OCR settings.
			_("Recognition &mode:"),
			wx.Choice,
			choices=names,
		)
		lang = conf["language"]
		self.languageList.SetSelection(self._langs.index(lang) if lang in self._langs else 0)
		self.allCyrillicCheckBox = sHelper.addItem(
			# Translators: Checkbox in Paddle OCR settings. Without it the Cyrillic mode reads only
			# Russian, Ukrainian, Belarusian and Bulgarian letters.
			wx.CheckBox(self, label=_("&Letters of all Cyrillic alphabets in the Cyrillic mode")),
		)
		self.allCyrillicCheckBox.SetValue(conf["allCyrillic"])
		self.autoRefreshCheckBox = sHelper.addItem(
			# Translators: Checkbox in Paddle OCR settings.
			wx.CheckBox(self, label=_("Periodically &refresh recognized content")),
		)
		self.autoRefreshCheckBox.SetValue(conf["autoRefresh"])
		self.autoSayAllCheckBox = sHelper.addItem(
			# Translators: Checkbox in Paddle OCR settings.
			wx.CheckBox(self, label=_("Automatically &say all when recognition finishes")),
		)
		self.autoSayAllCheckBox.SetValue(conf["autoSayAllOnResult"])

	def onSave(self):
		conf = config.conf["paddleOcr"]
		conf["language"] = self._langs[self.languageList.GetSelection()]
		conf["allCyrillic"] = self.allCyrillicCheckBox.GetValue()
		conf["autoRefresh"] = self.autoRefreshCheckBox.GetValue()
		conf["autoSayAllOnResult"] = self.autoSayAllCheckBox.GetValue()


def disableInSecureMode(pluginClass):
	"""On sign-in, lock and UAC screens load an empty plugin: no commands, no settings."""
	if globalVars.appArgs.secure:
		return globalPluginHandler.GlobalPlugin
	return pluginClass


@disableInSecureMode
class GlobalPlugin(globalPluginHandler.GlobalPlugin):
	# Translators: Category of Paddle OCR commands in the Input Gestures dialog.
	scriptCategory = _("Paddle OCR")

	def __init__(self):
		super().__init__()
		# Create the recognizer on the first command so NVDA startup is not slowed down.
		self._recognizer = None
		NVDASettingsDialog.categoryClasses.append(PaddleOcrPanel)

	def terminate(self):
		self._bindResultGestures(False)
		if PaddleOcrPanel in NVDASettingsDialog.categoryClasses:
			NVDASettingsDialog.categoryClasses.remove(PaddleOcrPanel)
		super().terminate()

	@script(
		# Translators: Description of the command in the Input Gestures dialog.
		description=_("Recognizes the content of the current navigator object with Paddle OCR"),
		speakOnDemand=True,
	)
	def script_recognizeNavigatorObject(self, gesture):
		self._recognize()

	@script(
		# Translators: Description of the command in the Input Gestures dialog.
		description=_("Recognizes the content of the foreground window with Paddle OCR"),
		speakOnDemand=True,
	)
	def script_recognizeForegroundWindow(self, gesture):
		self._recognize(api.getForegroundObject())

	@script(
		# Translators: Description of the command in the Input Gestures dialog.
		description=_("Recognizes the whole screen with Paddle OCR"),
		speakOnDemand=True,
	)
	def script_recognizeScreen(self, gesture):
		self._recognize(api.getDesktopObject())

	def event_gainFocus(self, obj, nextHandler):
		self._bindResultGestures(_ourResult(obj))
		nextHandler()

	def _bindResultGestures(self, bind):
		for gesture, scriptName in _RESULT_GESTURES.items():
			if bind:
				self.bindGesture(gesture, scriptName)
			else:
				with contextlib.suppress(LookupError):
					self.removeGestureBinding(gesture)

	# No descriptions: these work only in the result window, like NVDA's own Enter there.
	def script_doubleClickResult(self, gesture):
		self._clickResult("double", gesture)

	def script_rightClickResult(self, gesture):
		self._clickResult("right", gesture)

	def _clickResult(self, kind, gesture):
		obj = api.getFocusObject()
		if not _ourResult(obj):
			# The focus has just left the result window: the key does what it usually does.
			gesture.send()
			return
		# Not in the foreground or nothing recognized: nothing to click, as with NVDA's Enter.
		with contextlib.suppress(NotImplementedError):
			obj.makeTextInfo(textInfos.POSITION_CARET).click(kind)

	def _recognize(self, obj=None):
		from .recognizer import PaddleOcr

		if self._recognizer is None:
			self._recognizer = PaddleOcr()
		self._recognizer.recognizeFromCommand(obj)

	@script(
		# Translators: Description of the command in the Input Gestures dialog.
		description=_("Cycles the Paddle OCR recognition mode"),
	)
	def script_cycleLanguage(self, gesture):
		langs = list(LANGUAGES)
		lang = langs[(langs.index(getLanguage()) + 1) % len(langs)]
		config.conf["paddleOcr"]["language"] = lang
		ui.message(languageName(lang))

	@script(
		# Translators: Description of the command in the Input Gestures dialog.
		description=_("Shows the Paddle OCR settings"),
	)
	def script_showSettings(self, gesture):
		wx.CallAfter(gui.mainFrame.popupSettingsDialog, NVDASettingsDialog, PaddleOcrPanel)
