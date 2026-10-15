# Paddle OCR add-on for NVDA
# Copyright (C) 2026 fdrv161
# SPDX-License-Identifier: GPL-2.0-or-later

"""Paddle OCR: recognizes screen text with PaddleOCR models, like NVDA+R with Windows OCR."""

import addonHandler
import config
import globalPluginHandler
import globalVars
import gui
import ui
import wx
from gui import guiHelper
from gui.settingsDialogs import NVDASettingsDialog, SettingsPanel
from scriptHandler import script

# recognizer.py is light: it loads the heavy libraries and models only on the first recognition.
from .recognizer import LANGUAGES, getLanguage, languageName

addonHandler.initTranslation()

# An empty language means "same as NVDA" (see recognizer.getLanguage). The rest mirrors Windows OCR.
config.conf.spec["paddleOcr"] = {
	"language": "string(default='')",
	"autoRefresh": "boolean(default=false)",
	"autoSayAllOnResult": "boolean(default=true)",
}


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
		if PaddleOcrPanel in NVDASettingsDialog.categoryClasses:
			NVDASettingsDialog.categoryClasses.remove(PaddleOcrPanel)
		super().terminate()

	@script(
		# Translators: Description of the command in the Input Gestures dialog.
		description=_("Recognizes the content of the current navigator object with Paddle OCR"),
		speakOnDemand=True,
	)
	def script_recognizeNavigatorObject(self, gesture):
		from .recognizer import PaddleOcr

		if self._recognizer is None:
			self._recognizer = PaddleOcr()
		self._recognizer.recognizeFromCommand()

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
