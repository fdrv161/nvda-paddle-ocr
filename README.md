# Paddle OCR for NVDA

* Author: fdrv161
* NVDA compatibility: 2026.3 and later

Screen OCR for the NVDA screen reader with local PaddleOCR models.

## About the add-on

Paddle OCR is an alternative to the built-in Windows OCR (NVDA+R). The add-on uses compact PaddleOCR recognition models from Baidu that run locally. This engine usually reads more accurately, especially small fonts, text on dark backgrounds, and non-standard interfaces such as VST plugins. The result window and screen curtain behavior are the same as with Windows OCR.

## Requirements

* NVDA 2026.3 or later.
* The [Microsoft Visual C++ Redistributable x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) package. It is usually already installed; if it is missing, the add-on says so after "Recognition failed".
* Memory: about 100 MB after the first recognition, and briefly up to 300 MB while recognizing.

## Installation

1. Open the [latest release](../../releases/latest) (always the current version) and download the `paddleOcr-x.y.z.nvda-addon` file.
2. Open it, confirm the installation and restart NVDA.
3. Assign gestures to the commands, see [Commands and controls](#commands-and-controls).

## Commands and controls

There are no default hotkeys to avoid conflicts with other add-ons. You can assign them in the NVDA menu: Preferences, Input Gestures, "Paddle OCR" category.

* **Recognizes the content of the current navigator object with Paddle OCR.** Opens the result window: arrows move through the text, Enter clicks the word at the cursor, Escape closes the window.
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

If NVDA is in Russian, Ukrainian, Belarusian, or Bulgarian, letters from other Cyrillic alphabets are disabled so that random artifacts like "һ" instead of "н" do not get into the text.

## Settings

NVDA menu: Preferences, Settings, "Paddle OCR" category.

* **Recognition mode:** follow NVDA language, Latin or Cyrillic.
* **Periodically refresh recognized content:** the result window re-reads itself while open. It loads the processor, so it is disabled by default.
* **Automatically say all when recognition finishes:** text is spoken immediately after recognition. Enabled by default.

## Privacy

The add-on does not connect to the internet, does not send anything anywhere, and does not save either screenshots or recognized text to disk. Only its settings are stored in the NVDA configuration. Only errors go into the NVDA log, and at the "debug" level also technical details: image size, recognition time, library versions.

## Limitations and nuances

* Icons and interface frames are sometimes recognized as random characters like "x", "o", "+".
* Short single words made of letters that are the same in both alphabets are sometimes read with the wrong letters.

## Version history

What's new in each version is on the [releases page](https://github.com/fdrv161/nvda-paddle-ocr/releases).

## How this add-on was made

The code, tools and tests of this add-on were written by Claude (Anthropic) in Claude Code. Every commit in the history is marked "Co-Authored-By: Claude". I am not a programmer and cannot review the Python code myself.

What I did:

* the idea, inspired by the discussion in [nvaccess/nvda#20780](https://github.com/nvaccess/nvda/issues/20780);
* what the add-on can do and how to use it;
* testing every version with significant changes in NVDA; most fixes come from that;
* the Russian help (written with AI, then read and edited by me) and its translation into English with AI;
* checking the Russian interface in NVDA.

The models and the split into Latin and Cyrillic modes (instead of a list of languages, which I wanted at first) were chosen from accuracy and speed measurements that Claude made with the tools in this repository (`tools/accuracy.py`, `tools/bench.py`) and that I checked against my everyday use.

## Links and licenses

Bug reports and suggestions: [Issues](https://github.com/fdrv161/nvda-paddle-ocr/issues).

Built on [NVDA](https://www.nvaccess.org/), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [ONNX Runtime](https://github.com/microsoft/onnxruntime), [OpenCV](https://opencv.org/) and [NumPy](https://numpy.org/).

The add-on code is distributed under [GPL v2 or later](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html), PaddleOCR models and OpenCV under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0), onnxruntime under [MIT](https://opensource.org/licenses/MIT), NumPy under [BSD](https://opensource.org/licenses/BSD-3-Clause).

Library versions included in the add-on: onnxruntime 1.30.0, NumPy 2.5.3, opencv-python-headless 5.0.0.93, from PyPI. Their files are not modified; only parts not needed for recognition, such as tests and type hints, are left out. Where each file was taken from is described in the Third-party components section below.

## For developers

### Building

Libraries and models are not stored in the repository; scripts download them.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy==2.5.3 opencv-python-headless==5.0.0.93 onnxruntime==1.30.0 onnx==1.23.2 pytest pillow markdown
.venv/Scripts/python tools/fetch_models.py det_v6_tiny rec_v6_small rec_cyrillic_v5
.venv/Scripts/python tools/fetch_libs.py
.venv/Scripts/python tools/build_addon.py
```

The add-on appears in the repository root as `paddleOcr-<version>.nvda-addon`. Tests (they need the models downloaded above):

```bash
.venv/Scripts/python -m pytest tests -q
```

### Releasing

1. Raise `version` in `addon/manifest.ini` and describe the changes in its `changelog` field: it becomes the release text.
2. Push a tag with the same version, for example `v1.1.0`.
3. GitHub Actions checks that the tag matches the version, runs the tests, builds the add-on and publishes the release.

Pushing to `main` without a tag publishes nothing.

### Third-party components

The add-on's `lib/` and `models/` folders are not stored in git. `tools/fetch_libs.py` and `tools/fetch_models.py` download them from the official sources below, with pinned versions, so running these two scripts rebuilds the same folders. Files are copied unchanged, except one model (see below); only parts not needed at runtime are left out: tests and test modules, type hints, C headers, NumPy's f2py, onnxruntime tools, OpenCV data files and the OpenCV ffmpeg video library. Licenses of the libraries go into the package with them.

* **onnxruntime 1.30.0** (MIT), PyPI wheel `onnxruntime-1.30.0-cp313-cp313-win_amd64.whl`. Binaries: `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, `onnxruntime_pybind11_state.pyd`. Licenses: `lib/onnxruntime/LICENSE`, `ThirdPartyNotices.txt`.
* **NumPy 2.5.3** (BSD-3-Clause), PyPI wheel `numpy-2.5.3-cp313-cp313-win_amd64.whl`. Binaries: the `.pyd` modules of NumPy and, in `numpy.libs`, OpenBLAS (`libscipy_openblas64_*.dll`) and the Microsoft C++ runtime (`msvcp140-*.dll`) that NumPy ships. Licenses: `lib/licenses/numpy/`.
* **opencv-python-headless 5.0.0.93** (Apache 2.0), PyPI wheel `opencv_python_headless-5.0.0.93-cp37-abi3-win_amd64.whl`. Binary: `cv2.pyd`. Licenses: `lib/cv2/LICENSE.txt`, `LICENSE-3RD-PARTY.txt`.
* **PaddleOCR models** (Apache 2.0) `det_v6_tiny`, `rec_v6_small`, `rec_cyrillic_v5`: ONNX files from the [PaddlePaddle organization on Hugging Face](https://huggingface.co/PaddlePaddle). Repository names and SHA-256 hashes are in `models/manifest.json`; the license is in `addon/licenses/PaddleOCR-models.txt`. `det_v6_tiny` and `rec_cyrillic_v5` are included unchanged. **`rec_v6_small` is modified:** `tools/slim_head.py` rebuilds its last layer so that it returns only the characters the add-on allows (about 760 of 18,700), computing the same probabilities in parts to use less memory; the rest of the model and its weights are unchanged.
