# Paddle OCR

## About

Paddle OCR is an alternative to the built-in Windows OCR (NVDA+R) for [NVDA](https://www.nvaccess.org/) 2026.3 and later. The add-on uses compact recognition models from Baidu ([PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)) that run locally through [onnxruntime](https://github.com/microsoft/onnxruntime) right inside NVDA — no internet and no third-party applications. The engine reads small fonts and text on dark backgrounds more accurately. The result window and the behaviour with Screen Curtain on are the same as you are used to in NVDA.

## Commands

There are no default gestures, so nothing conflicts with other add-ons. Assign them in the NVDA menu: Preferences → Input gestures → category "Paddle OCR".

* **Recognize the navigator object.** Opens the result window: arrow keys move through the text, Enter clicks a word, Escape closes the window.
* **Cycle the recognition language:** Latin or Cyrillic.
* **Open Paddle OCR settings.**

The first recognition in a session says "Loading OCR models". On processors like the Ryzen 7 7840HS a full screen is recognized in about 0.4 seconds, and small windows in a fraction of a second. On slower hardware it takes longer.

## Languages and alphabets

The add-on has two modes, by alphabet:

* **Latin:** English, German, French, Spanish, Polish and others. The best accuracy, but Cyrillic is not recognized.
* **Cyrillic:** Russian, Ukrainian, Belarusian, Bulgarian, Serbian, Kazakh, Tatar, Bashkir and others. English text is recognized too, but slightly less accurately than in the Latin mode.

By default the mode follows the NVDA language. When NVDA is in Russian, Ukrainian, Belarusian or Bulgarian, letters of the other Cyrillic alphabets are turned off, so random artifacts like "һ" instead of "н" do not get into the text.

## Settings

NVDA menu: Preferences → Settings → category "Paddle OCR".

* **Recognition language:** same as NVDA, Latin or Cyrillic.
* **Periodically refresh recognized content:** the result window re-reads itself while it is open. This keeps the processor busy the whole time, so it is off by default.
* **Automatically say all when recognition finishes:** the text is read in full right away. If you prefer to read it yourself, turn this off. On by default.

## Limitations

* Icons and interface borders are sometimes recognized as random characters like "x", "o", "+".
* Short single words made of letters that look the same in both alphabets are sometimes read with the wrong letters.
* The system library `MSVCP140_1.dll` is required. If recognition reports an error and the NVDA log mentions this library, install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) from Microsoft.

## Version history

### 1.0.5

* A new model for Cyrillic: fewer errors in mixed lines and support for more languages (Serbian, Macedonian, Kazakh and others).
* Reliable recognition of small objects (buttons, labels): text at the very edges is no longer lost.
* Lower memory use, slightly faster processing.

### 1.0.4

* The recognize command now also speaks in the "on demand" speech mode.
* The code follows NVDA standards.

### 1.0.3

* The modes are named by script: "Latin" and "Cyrillic".
* Cyrillic is chosen automatically also when NVDA is in Ukrainian, Belarusian or Bulgarian.
* The help lists the supported languages.

### 1.0.2

* The add-on now includes the GPL license, the repository link and the changelog.
* The Russian and English help became shorter and simpler.

### 1.0.1

* Faster recognition.

### 1.0.0

* First release: recognition, two languages, settings, translated interface and help.

## Links and licenses

Source code, bug tracker and discussions are in the [GitHub repository](https://github.com/fdrv161/nvda-paddle-ocr).

The add-on code is licensed under the [GPL v2 or later](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html), the models and [OpenCV](https://opencv.org/) under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), [onnxruntime](https://github.com/microsoft/onnxruntime) under [MIT](https://opensource.org/licenses/MIT), [NumPy](https://numpy.org/) under [BSD](https://opensource.org/licenses/BSD-3-Clause).
