"""Paddle OCR: распознавание текста на экране моделями PaddleOCR, как NVDA+R у Windows OCR."""
import addonHandler
import config
import globalPluginHandler
import globalVars
import ui
from scriptHandler import script

addonHandler.initTranslation()

# Пустой язык — «как язык NVDA» (см. recognizer.getLanguage).
config.conf.spec["paddleOcr"] = {"language": "string(default='')"}


class GlobalPlugin(globalPluginHandler.GlobalPlugin):
	# Translators: Category of Paddle OCR commands in the Input Gestures dialog.
	scriptCategory = _("Paddle OCR")

	def __init__(self):
		super().__init__()
		# Распознаватель (и тяжёлые библиотеки за ним) создаём при первой команде, чтобы не тормозить старт NVDA.
		self._recognizer = None

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
		# recognizer.py лёгкий: тяжёлые библиотеки он грузит только при распознавании.
		from .recognizer import LANGUAGES, getLanguage, languageName
		langs = list(LANGUAGES)
		lang = langs[(langs.index(getLanguage()) + 1) % len(langs)]
		config.conf["paddleOcr"]["language"] = lang
		ui.message(languageName(lang))
