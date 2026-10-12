# Paddle OCR for NVDA

## About

Paddle OCR is an alternative to the built-in Windows OCR (NVDA+R) for NVDA 2026.3 and later. The add-on uses compact PaddleOCR recognition models from Baidu that run locally through onnxruntime right inside NVDA — no internet and no third-party applications. The engine is usually more accurate, especially with small fonts, text on dark backgrounds and custom interfaces such as audio plugins. The result window and Screen Curtain support work the same as with Windows OCR.

## Installation

1. Open the [latest release](../../releases/latest) (always the current version) and download the `paddleOcr-x.y.z.nvda-addon` file.
2. Open it, confirm the installation and restart NVDA.
3. Assign gestures: NVDA menu, Preferences, Input gestures, category "Paddle OCR". There are no default gestures.

## Commands

There are no default gestures, so nothing conflicts with other add-ons. Assign them in the NVDA menu: Preferences, Input gestures, category "Paddle OCR".

* **Recognize the navigator object.** Opens the result window: arrow keys move through the text, Enter clicks a word, Escape closes the window.
* **Cycle the recognition language:** Latin or Cyrillic.
* **Open Paddle OCR settings.**

The first recognition in a session says "Loading OCR models". On the author's laptop (Ryzen 7 7840HS) a full screen is recognized in about 0.4 seconds, and small windows in a fraction of a second. On slower computers it may take several times longer.

## Languages and alphabets

The add-on has two modes, by alphabet:

* **Latin:** English, German, French, Spanish, Polish and others. The best accuracy, but Cyrillic is not recognized.
* **Cyrillic:** Russian, Ukrainian, Belarusian, Bulgarian, Serbian, Kazakh, Tatar, Bashkir and others. English text is recognized too, but slightly less accurately than in the Latin mode.

By default the mode follows the NVDA language. For an application in English, Latin works best; for one in Russian, Cyrillic.

When NVDA is in Russian, Ukrainian, Belarusian or Bulgarian, letters of the other Cyrillic alphabets are turned off, so random artifacts like "һ" instead of "н" do not get into the text. Letters of, for example, Kazakh or Serbian are read when NVDA itself runs in that language.

## Settings

NVDA menu: Preferences, Settings, category "Paddle OCR".

* **Recognition language:** same as NVDA, Latin or Cyrillic.
* **Periodically refresh recognized content:** the result window re-reads itself while it is open. This keeps the processor busy the whole time, so it is off by default.
* **Automatically say all when recognition finishes:** the text is read in full right away. If you prefer to read it yourself, turn this off. On by default.

## Privacy

Recognition runs entirely on your computer. The add-on does not connect to the internet, sends nothing anywhere and saves neither screenshots nor recognized text to disk; only its settings are stored, in the NVDA configuration. The NVDA log gets only error messages and, at the debug log level, the image size and timings, never the recognized text.

## Limitations

* Icons and interface borders are sometimes recognized as random characters like "x", "o", "+".
* Short single words made of letters that look the same in both alphabets are sometimes read with the wrong letters.
* The system library `MSVCP140_1.dll` is required. If recognition reports an error and the NVDA log mentions this library, install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) from Microsoft.

## Version history

What is new in each version is on the [releases page](https://github.com/fdrv161/nvda-paddle-ocr/releases).

## Building

Libraries and models are not stored in the repository; scripts download them.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install numpy==2.5.3 opencv-python-headless==5.0.0.93 onnxruntime==1.30.0 pytest pillow markdown
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

## Third-party components

The add-on's `lib/` and `models/` folders are not stored in git. `tools/fetch_libs.py` and `tools/fetch_models.py` download them from the official sources below, with pinned versions, so running these two scripts rebuilds the same folders. Files are copied unchanged; only parts not needed at runtime are left out: tests, C headers, onnxruntime tools, OpenCV data files and the OpenCV ffmpeg video library. Licenses of the libraries go into the package with them.

* **onnxruntime 1.30.0** (MIT), PyPI wheel `onnxruntime-1.30.0-cp313-cp313-win_amd64.whl`. Binaries: `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `onnxruntime_pybind11_state.pyd`. Licenses: `lib/onnxruntime/LICENSE`, `ThirdPartyNotices.txt`.
* **NumPy 2.5.3** (BSD-3-Clause), PyPI wheel `numpy-2.5.3-cp313-cp313-win_amd64.whl`. Binaries: the `.pyd` modules of NumPy and, in `numpy.libs`, OpenBLAS (`libscipy_openblas64_*.dll`) and the Microsoft C++ runtime (`msvcp140-*.dll`) that NumPy ships. Licenses: `lib/licenses/numpy/`.
* **opencv-python-headless 5.0.0.93** (Apache 2.0), PyPI wheel `opencv_python_headless-5.0.0.93-cp37-abi3-win_amd64.whl`. Binary: `cv2.pyd`. Licenses: `lib/cv2/LICENSE.txt`, `LICENSE-3RD-PARTY.txt`.
* **PaddleOCR models** (Apache 2.0) `det_v6_tiny`, `rec_v6_small`, `rec_cyrillic_v5`: ONNX files from the [PaddlePaddle organization on Hugging Face](https://huggingface.co/PaddlePaddle). Repository names and SHA-256 hashes are in `models/manifest.json`; the license is in `addon/licenses/PaddleOCR-models.txt`.

## Links and licenses

Source code, bug tracker and discussions are in the [GitHub repository](https://github.com/fdrv161/nvda-paddle-ocr).

Projects used: [NVDA](https://www.nvaccess.org/), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [onnxruntime](https://github.com/microsoft/onnxruntime), [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/).

The add-on code is licensed under the [GPL v2 or later](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html), PaddleOCR models and OpenCV under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), onnxruntime under [MIT](https://opensource.org/licenses/MIT), NumPy under [BSD](https://opensource.org/licenses/BSD-3-Clause).
