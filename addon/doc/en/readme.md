# Paddle OCR

Recognizes text on the screen with PaddleOCR models. It works just like the built-in Windows OCR (NVDA+R), but is usually more accurate, especially with small text and dark themes. Everything works offline.

Requires NVDA 2026.3 or later.

## Commands

There are no default gestures. Assign them in the NVDA menu, Preferences, Input gestures, category "Paddle OCR".

* **Recognizes the content of the current navigator object**. Opens a result window, like NVDA+R. Move through the text with the arrow keys, press Enter to click a word, Escape to close.
* **Cycles through the available languages**: Latin or Cyrillic.
* **Shows the Paddle OCR settings**.

The first recognition says "Loading OCR models". After that a full screen takes about 0.4 seconds.

## Languages

The models read alphabets rather than single languages.

* **Latin**: English, German, French, Spanish, Polish and other Latin-script languages. The best accuracy. Cyrillic is not read in this mode.
* **Cyrillic**: Russian, Ukrainian, Belarusian, Bulgarian, Serbian, Macedonian, Kazakh, Kyrgyz, Tajik, Mongolian, Tatar, Bashkir and other Cyrillic-script languages, plus English (slightly worse than in Latin mode).

By default the mode follows the NVDA interface language.

When NVDA is in Russian, Ukrainian, Belarusian or Bulgarian, the Cyrillic mode reads only the letters of these languages, so lookalike letters of other alphabets do not slip into the text. Letters of the other Cyrillic languages are read when NVDA itself runs in that language.

## Settings

NVDA menu, Preferences, Settings, category "Paddle OCR".

* **Recognition language**: same as NVDA, Latin or Cyrillic.
* **Periodically refresh recognized content**: the result window updates itself while open. Off by default.
* **Automatically say all when recognition finishes**. On by default.

## Screen Curtain

Like Windows OCR, the add-on recognizes with Screen Curtain on when Windows supports it. Otherwise NVDA asks you to turn Screen Curtain off temporarily.

## Licenses

Add-on code: GNU GPL version 2 or later. PaddleOCR models and OpenCV: Apache 2.0, onnxruntime: MIT, NumPy: BSD. The license texts are in the add-on folder.
