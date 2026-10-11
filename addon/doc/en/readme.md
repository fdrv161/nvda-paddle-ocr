# Paddle OCR

Recognizes text on the screen with PaddleOCR models. It works just like the built-in Windows OCR (NVDA+R), but is usually more accurate, especially with small text and dark themes. Everything works offline.

Requires NVDA 2026.3 or later.

## Commands

There are no default gestures. Assign them in the NVDA menu, Preferences, Input gestures, category "Paddle OCR".

* **Recognizes the content of the current navigator object**. Opens a result window, like NVDA+R. Move through the text with the arrow keys, press Enter to click a word, Escape to close.
* **Cycles through the available languages**: English or Russian.
* **Shows the Paddle OCR settings**.

The first recognition says "Loading OCR models". After that a full screen takes about 0.4 seconds.

## Languages

* **English**: the best accuracy for English and other Latin-script languages. Russian text is not read in this mode.
* **Russian**: reads both Russian and English text, but English slightly worse than in the English mode.

By default the language follows the NVDA interface language.

## Settings

NVDA menu, Preferences, Settings, category "Paddle OCR".

* **Recognition language**: same as NVDA, English or Russian.
* **Periodically refresh recognized content**: the result window updates itself while open. Off by default.
* **Automatically say all when recognition finishes**. On by default.

## Screen Curtain

Like Windows OCR, the add-on recognizes with Screen Curtain on when Windows supports it. Otherwise NVDA asks you to turn Screen Curtain off temporarily.

## Licenses

Add-on code: GNU GPL version 2 or later. PaddleOCR models and OpenCV: Apache 2.0, onnxruntime: MIT, NumPy: BSD. The license texts are in the add-on folder.
