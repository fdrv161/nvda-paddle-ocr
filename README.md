# Paddle OCR for NVDA

Recognizes text on the screen with [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models. It works like the built-in Windows OCR (NVDA+R): the same result window, the same navigation and clicking on words, but it is usually more accurate, especially with small text, dark themes and custom interfaces of audio plugins and other apps.

Everything works offline: the models and libraries are inside the add-on.

## Requirements

- NVDA 2026.3 or later (64-bit).

## Installation

1. Open the [latest release](../../releases/latest) (always the current version) and download the `paddleOcr-x.y.z.nvda-addon` file.
2. Open it, confirm the installation and restart NVDA.
3. Assign gestures: NVDA menu, Preferences, Input gestures, category "Paddle OCR". There are no default gestures.

## Commands

- **Recognize the current navigator object.** Opens the result window. Arrow keys move through the text, Enter clicks a word, Escape closes the window.
- **Cycle the recognition language**: Latin or Cyrillic.
- **Open Paddle OCR settings.**

The first recognition says "Loading OCR models". After that a full screen takes about 0.4 seconds on the author's laptop (AMD Ryzen 7 7840HS); on slower computers it may take several times longer.

## Languages

The models read alphabets rather than single languages:

- **Latin**: English, German, French, Spanish, Polish and other Latin-script languages. The best accuracy. Does not read Cyrillic.
- **Cyrillic**: Russian, Ukrainian, Belarusian, Bulgarian, Serbian, Macedonian, Kazakh, Kyrgyz, Tajik, Mongolian, Tatar, Bashkir and other Cyrillic-script languages, plus English (slightly worse than in Latin mode).

By default the mode follows the NVDA language. When NVDA is in Russian, Ukrainian, Belarusian or Bulgarian, the Cyrillic mode reads only the letters of these languages; letters of the other Cyrillic languages are read when NVDA itself runs in that language.

## Settings

NVDA menu, Preferences, Settings, category "Paddle OCR":

- Recognition language.
- Periodically refresh recognized content (off by default).
- Automatically say all when recognition finishes (on by default).

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
