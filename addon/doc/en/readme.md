# Paddle OCR

## About the add-on

Paddle OCR is an alternative to the built-in Windows OCR (NVDA+R) for NVDA 2026.3 and newer. The add-on uses compact PaddleOCR recognition models from Baidu that run locally. This engine usually reads more accurately, especially small fonts, text on dark backgrounds, and non-standard interfaces such as VST plugins. The result window and screen curtain behavior are the same as with Windows OCR.

## Commands and controls

There are no default hotkeys to avoid conflicts with other add-ons. You can assign them in the NVDA menu: Preferences, Input Gestures, "Paddle OCR" category.

* **Recognizes the content of the current navigator object with Paddle OCR.** Opens the result window: arrows move through the text, Enter clicks on the word, Escape closes the window.
* **Cycles the Paddle OCR recognition mode:** Latin or Cyrillic.
* **Shows the Paddle OCR settings.**

On the first recognition after starting NVDA, "Loading OCR models" will be announced. After that, the models remain in memory.

If you press the recognition command again while recognition is still running, "Recognition in progress" will be announced and the running recognition continues. If recognition takes longer than 30 seconds, the next press starts it again. If it hangs again, restart NVDA.

On my laptop (Ryzen 7 7840HS, 32 GB RAM, Windows 11 25H2), the entire screen is recognized in about 0.4 seconds, and small windows in fractions of a second. On weak hardware it may be slower.

## Languages and alphabets

The add-on works in two modes, by alphabet:

* **Latin:** English, German, French, Spanish, Polish, and others. Maximum accuracy, but Cyrillic is not recognized.
* **Cyrillic:** Russian, Ukrainian, Belarusian, Bulgarian, Serbian, Kazakh, Tatar, Bashkir, and others. English text is also recognized, but accuracy is slightly lower than in "Latin" mode.

By default the mode follows the NVDA language: Latin for English, Cyrillic for Russian.

If NVDA is in Russian, Ukrainian, Belarusian, or Bulgarian, letters from other Cyrillic alphabets are disabled so that random artifacts like "һ" instead of "н" do not get into the text.

## Settings

NVDA menu: Preferences, Settings, "Paddle OCR" category.

* **Recognition mode:** follow NVDA language, Latin or Cyrillic.
* **Periodically refresh recognized content:** the result window re-reads itself while open. It loads the processor, so it is disabled by default.
* **Automatically say all when recognition finishes:** text is spoken immediately after recognition. Enabled by default.

## Privacy

The add-on does not connect to the internet, does not send anything anywhere, and does not save either screenshots or recognized text to disk. Only its settings are stored in the NVDA config. Only error messages go into the NVDA log, and at the "debug" log level, also the image size and the time it took for OCR to occur.

## Limitations and nuances

* Icons and interface frames are sometimes recognized as random characters like "x", "o", "+".
* Short single words made of letters that are the same in both alphabets are sometimes read with the wrong letters.
* The system library `MSVCP140_1.dll` is required. If an error sounds during recognition and this library is mentioned in the NVDA log, install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) package from the Microsoft website.

## Version history

What's new in each version is on the [releases page](https://github.com/fdrv161/nvda-paddle-ocr/releases).

## Links and licenses

Source code: [GitHub](https://github.com/fdrv161/nvda-paddle-ocr).

Projects used: [NVDA](https://www.nvaccess.org/), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [onnxruntime](https://github.com/microsoft/onnxruntime), [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/).

The add-on code is distributed under [GPL v2 or later](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html), PaddleOCR models and OpenCV under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), onnxruntime under [MIT](https://opensource.org/licenses/MIT), NumPy under [BSD](https://opensource.org/licenses/BSD-3-Clause).

Library versions included in the add-on: onnxruntime 1.30.0, NumPy 2.5.3, opencv-python-headless 5.0.0.93, taken unchanged from PyPI. Where each file was taken from is described in the repository README.