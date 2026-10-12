# Paddle OCR for NVDA

## About

Paddle OCR is an alternative to the built-in Windows OCR (NVDA+R) for [NVDA](https://www.nvaccess.org/) 2026.3 and later. The add-on uses compact recognition models from Baidu ([PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)) that run locally through [onnxruntime](https://github.com/microsoft/onnxruntime) right inside NVDA — no internet and no third-party applications. The engine reads small fonts and text on dark backgrounds more accurately. The result window and the behaviour with Screen Curtain on are the same as you are used to in NVDA.

## Installation

1. Open the [latest release](../../releases/latest) (always the current version) and download the `paddleOcr-x.y.z.nvda-addon` file.
2. Open it, confirm the installation and restart NVDA.
3. Assign gestures: NVDA menu, Preferences, Input gestures, category "Paddle OCR". There are no default gestures.

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

See the [releases](../../releases) page; the add-on help lists all versions too.

## Building

Libraries and models are not stored in the repository; scripts download them.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install numpy opencv-python-headless onnxruntime pytest pillow markdown
```

```bash
.venv/Scripts/python tools/fetch_models.py det_v6_tiny rec_v6_small rec_cyrillic_v5
```

```bash
.venv/Scripts/python tools/fetch_libs.py
```

```bash
.venv/Scripts/python tools/build_addon.py
```

To release, set `version` in `addon/manifest.ini` and push a tag with the same version, for example `v1.1.0`. GitHub Actions runs the tests, builds the add-on and publishes the release.

## License

The add-on code is licensed under the [GNU GPL version 2 or later](LICENSE). Bundled components keep their own licenses: PaddleOCR models and OpenCV (Apache 2.0), onnxruntime (MIT), NumPy (BSD). Their license texts are included in the add-on.
