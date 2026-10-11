# Paddle OCR для NVDA

Дополнение для NVDA, которое распознаёт текст на экране моделями [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). Работает так же, как встроенный Windows OCR (NVDA+R): то же окно результата, можно ходить по тексту и щёлкать по словам. Но читает обычно точнее, особенно мелкий текст, тёмные темы и интерфейсы программ вроде синтезаторов и DAW.

Всё работает без интернета: модели и библиотеки лежат внутри дополнения.

[English below](#english)

## Требования

- NVDA 2026.3 или новее (64 бит).
- Windows 10 или 11.

## Установка

1. Откройте страницу [Releases](../../releases) и скачайте последний файл `paddleOcr-x.y.z.nvda-addon`.
2. Откройте его (Enter в проводнике) и согласитесь на установку, потом перезапустите NVDA.
3. Назначьте жесты: меню NVDA, Параметры, Жесты ввода, категория «Paddle OCR». Жестов по умолчанию нет, чтобы не мешать вашим.

## Команды

- **Распознаёт содержимое текущего объекта навигатора.** Открывает окно с результатом. Стрелки — ходить по тексту, Enter — щёлкнуть по слову, Escape — закрыть.
- **Переключает язык распознавания**: английский или русский.
- **Открывает настройки Paddle OCR.**

При первом распознавании прозвучит «Загрузка моделей OCR»: модели загружаются один раз, дальше распознавание занимает около 0,4 секунды на экран.

## Языки

- **Английский** — самая высокая точность для английского и других языков на латинице. Русский в этом режиме не читается.
- **Русский** — читает и русский, и английский, но английский немного хуже.

По умолчанию язык берётся из языка NVDA. Если программа на английском, включите английский.

## Настройки

Меню NVDA, Параметры, Настройки, категория «Paddle OCR»: язык распознавания, периодическое обновление результата, чтение всего результата сразу после распознавания.

## Шторка экрана

Как и Windows OCR, дополнение распознаёт при включённой шторке, если это поддерживает Windows. Если нет, NVDA попросит временно отключить шторку.

## Для разработчиков

Код дополнения лежит в `addon/`, движок — `addon/globalPlugins/paddleOcr/engine.py` (не зависит от NVDA, его можно проверять обычным Python). Библиотеки и модели в репозиторий не входят, их скачивают скрипты.

Сборка у себя:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install numpy opencv-python-headless onnxruntime pytest pillow
```

```bash
.venv/Scripts/python tools/fetch_models.py det_v6_tiny rec_v6_small rec_eslav_v5
```

```bash
.venv/Scripts/python tools/fetch_libs.py
```

```bash
.venv/Scripts/python tools/build_addon.py
```

Проверки:

- `tools/accuracy.py` — процент ошибок на картинках с известным текстом;
- `tools/bench.py` — скорость и память на скриншотах;
- `pytest tests` — тесты движка.

### Выпуск новой версии

1. Поменять `version` в `addon/manifest.ini`, например на `1.1.0`.
2. Закоммитить и отправить тег с той же версией:

```bash
git tag v1.1.0
```

```bash
git push origin v1.1.0
```

GitHub Actions сам скачает модели и библиотеки, прогонит тесты, соберёт `.nvda-addon` и создаст релиз с этим файлом. Если версия в теге и в `manifest.ini` разная, сборка остановится.

## Лицензии

- Код дополнения — GNU GPL версии 2 или новее, текст в [LICENSE](LICENSE).
- Модели PaddleOCR (PP-OCRv6, PP-OCRv5 eslav) — Apache 2.0.
- onnxruntime — MIT, NumPy — BSD, OpenCV — Apache 2.0.

Полные тексты лежат в дополнении: папка `licenses` и рядом с каждой библиотекой в `lib`.

---

## English

An NVDA add-on that recognizes text on the screen with [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models. It works just like the built-in Windows OCR (NVDA+R): the same result window, text navigation and clicking on words, but it is usually more accurate. Everything works offline.

- Requires NVDA 2026.3 or later (64-bit).
- Download the latest `.nvda-addon` from [Releases](../../releases), install it, then assign gestures in Input gestures, category "Paddle OCR" (there are no default gestures).
- Commands: recognize the current navigator object, cycle the recognition language (English or Russian), open Paddle OCR settings.
- English mode gives the best accuracy for Latin-script text but does not read Russian. Russian mode reads both Russian and English.
- License: add-on code GNU GPL version 2 or later; bundled models and libraries keep their own licenses (Apache 2.0, MIT, BSD).
- Building: see the developer section above. Pushing a tag `vX.Y.Z` makes GitHub Actions build the add-on and publish a release.
