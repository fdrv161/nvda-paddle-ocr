"""Paddle OCR: распознавание текста на экране моделями PaddleOCR, как NVDA+R у Windows OCR."""
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

# recognizer.py лёгкий: тяжёлые библиотеки и модели он грузит только при первом распознавании.
from .recognizer import LANGUAGES, getLanguage, languageName

addonHandler.initTranslation()

# Пустой язык — «как язык NVDA» (см. recognizer.getLanguage). Остальное — как у Windows OCR.
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
		# Translators: Language choice in Paddle OCR settings: use the NVDA interface language.
		names = [_("Same as NVDA ({language})").format(language=languageName(getLanguage(auto=True)))]
		names += [languageName(lang) for lang in LANGUAGES]
		self.languageList = sHelper.addLabeledControl(
			# Translators: Label of the recognition language list in Paddle OCR settings.
			_("Recognition &language:"), wx.Choice, choices=names,
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


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
	# Translators: Category of Paddle OCR commands in the Input Gestures dialog.
	scriptCategory = _("Paddle OCR")

	def __init__(self):
		super().__init__()
		# Распознаватель создаём при первой команде, чтобы не тормозить старт NVDA.
		self._recognizer = None
		# На защищённом экране настройки менять нельзя.
		if not globalVars.appArgs.secure:
			NVDASettingsDialog.categoryClasses.append(PaddleOcrPanel)

	def terminate(self):
		if PaddleOcrPanel in NVDASettingsDialog.categoryClasses:
			NVDASettingsDialog.categoryClasses.remove(PaddleOcrPanel)
		super().terminate()

	@script(
		# Translators: Description of the command in the Input Gestures dialog.
		description=_("Recognizes the content of the current navigator object with Paddle OCR"),
	)
	def script_recognizeNavigatorObject(self, gesture):
		# На защищённом экране (вход в систему, UAC) ничего не делаем.
		if globalVars.appArgs.secure:
			return
		from contentRecog import recogUi
		from .recognizer import PaddleOcr
		if self._recognizer is None:
			self._recognizer = PaddleOcr()
		# Дальше всё как у Windows OCR: захват (в том числе при шторке), окно результата, клик по словам.
		recogUi.recognizeNavigatorObject(self._recognizer)

	@script(
		# Translators: Description of the command in the Input Gestures dialog.
		description=_("Cycles through the available languages for Paddle OCR"),
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
