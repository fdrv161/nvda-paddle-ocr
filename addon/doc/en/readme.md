# Paddle OCR

* Author: fdrv161
* NVDA compatibility: 2026.3 and later

## About the add-on

Paddle OCR is an alternative to the built-in Windows OCR (NVDA+R). The add-on uses compact PaddleOCR recognition models from Baidu that run locally. This engine usually reads more accurately, especially small fonts, text on dark backgrounds, and non-standard interfaces such as VST plugins. The result window and screen curtain behavior are the same as with Windows OCR.

## Requirements

* NVDA 2026.3 or later.
* The [Microsoft Visual C++ Redistributable x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) package. It is usually already installed; if it is missing, the add-on says so after "Recognition failed".
* Memory: about 100 MB after the first recognition, and briefly up to 300 MB while recognizing.

## Commands and controls

There are no default hotkeys to avoid conflicts with other add-ons. You can assign them in the NVDA menu: Preferences, Input Gestures, "Paddle OCR" category.

* **Recognizes the content of the current navigator object with Paddle OCR.** Opens the result window: arrows move through the text, Enter clicks the middle of the word at the cursor, Shift+Enter double-clicks it, the Applications key or Shift+F10 right-clicks it, Escape closes the window.
* **Recognizes the content of the foreground window with Paddle OCR** and **Recognizes the whole screen with Paddle OCR.** The same, but for the whole window or screen, without moving the navigator object first. The navigator object and the review cursor stay where they were.
* **Cycles the Paddle OCR recognition mode:** Latin or Cyrillic.
* **Shows the Paddle OCR settings.**

On the first recognition after starting NVDA, "Loading OCR models" will be announced. After that, the models remain in memory.

Pressing the command again during a recognition does not start it over: "Recognition in progress" is announced. If a recognition hangs for more than 30 seconds, the next press starts a new one; if it hangs again, restart NVDA.

On my laptop (Ryzen 7 7840HS, 32 GB RAM, Windows 11 25H2), the entire screen is recognized in about 0.4 seconds, and small windows in fractions of a second. On weak hardware it may be slower.

## Languages and alphabets

The add-on works in two modes, by alphabet:

* **Latin:** English, German, French, Spanish, Polish, and others. Maximum accuracy, but Cyrillic is not recognized.
* **Cyrillic:** Russian, Ukrainian, Belarusian, Bulgarian, Serbian, Kazakh, Tatar, Bashkir, and others. English text is also recognized, but accuracy is slightly lower than in "Latin" mode.

By default the mode follows the NVDA language: Latin for English, Cyrillic for Russian.

In the Cyrillic mode, only Russian, Ukrainian, Belarusian and Bulgarian letters are recognized by default, so that random artifacts like "һ" instead of "н" do not get into the text. For Kazakh, Serbian, Tatar, Bashkir and other languages, turn on "Letters of all Cyrillic alphabets in the Cyrillic mode" in the settings. With NVDA in Macedonian, Kyrgyz, Mongolian or Serbian, these letters are always on.

## Settings

NVDA menu: Preferences, Settings, "Paddle OCR" category.

* **Recognition mode:** follow NVDA language, Latin or Cyrillic.
* **Letters of all Cyrillic alphabets in the Cyrillic mode:** for languages other than Russian, Ukrainian, Belarusian and Bulgarian. Disabled by default.
* **Periodically refresh recognized content:** the result window re-reads itself while open. It loads the processor, so it is disabled by default.
* **Automatically say all when recognition finishes:** text is spoken immediately after recognition. Enabled by default.

## Privacy

The add-on does not connect to the internet. Screenshots and recognized text are not sent anywhere and are not saved to disk. Only its settings are stored in the NVDA configuration. Only errors go into the NVDA log, and at the "debug" level also technical details: image size, recognition time, library versions.

## Limitations and nuances

* Icons and interface frames are sometimes recognized as random characters like "x", "o", "+".
* Short single words made of letters that are the same in both alphabets are sometimes read with the wrong letters.

## Version history

What's new in each version is on the [releases page](https://github.com/fdrv161/nvda-paddle-ocr/releases).

## Links and licenses

Source code: [GitHub](https://github.com/fdrv161/nvda-paddle-ocr).

Built on [NVDA](https://www.nvaccess.org/), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [ONNX Runtime](https://github.com/microsoft/onnxruntime), [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/).

The add-on code is distributed under [GPL v2 or later](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html), PaddleOCR models and OpenCV under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), onnxruntime under [MIT](https://opensource.org/licenses/MIT), NumPy under [BSD](https://opensource.org/licenses/BSD-3-Clause).

Details about third-party components are in the repository README.
