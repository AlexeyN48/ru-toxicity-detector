# ru-toxicity-detector

Детекция токсичных комментариев на русском языке: дообучение `ai-forever/ruBert-base`
для бинарной классификации `токсичный` / `нетоксичный`.

Модель помогает модераторам: отмечает подозрительные комментарии для проверки человеком,
а не банит пользователей автоматически. Модель регулярно переобучается на свежей разметке
модераторов, чтобы успевать за новым сленгом и способами обхода фильтра.

## Статус

| Этап | Содержание | Статус |
|---|---|---|
| ДЗ 1 | репозиторий и линтер, Model Card, BPMN жизненного цикла модели | готово |
| ДЗ 2 | BPMN retraining-цикла: дрифт, retraining, валидация, аукцион по AUC, деплой, Telegram | готово |
| ДЗ 3 | BPMN автоматического переобучения и смены модели: триггеры, процесс, критерии остановки | готово |
| Код | данные, дообучение, оценка и fairness-анализ, аугментация для fairness, тесты | готово |
| Дальше | мониторинг дрифта, аукцион champion/challenger, Telegram-бот | — |

## Модель и данные

- **Базовая модель:** [`ai-forever/ruBert-base`](https://huggingface.co/ai-forever/ruBert-base);
  baseline для сравнения — [`cointegrated/rubert-tiny2`](https://huggingface.co/cointegrated/rubert-tiny2).
- **Данные:** Russian Language Toxic Comments (Kaggle), подробности — [data/README.md](data/README.md).
- **Результаты:** метрики и fairness-анализ — в [Model Card](docs/MODEL_CARD.md), отчёты — в `reports/`.

## Как запустить

Зависимости ставятся через [uv](https://docs.astral.sh/uv/), точные версии зафиксированы в
`uv.lock`. Для обучения желательна видеокарта NVIDIA: на Windows и Linux uv сам поставит
PyTorch с CUDA 13.0. Всё работает и на CPU, только медленнее.

```bash
uv sync --all-extras                                             # окружение: обучение, демо, dev
```

```bash
uv run python -m toxicity.data prepare                           # скачать и разбить данные
uv run python -m toxicity.train --config configs/train_tiny.yaml # baseline, ~1 мин на GPU
uv run python -m toxicity.train --config configs/train_base.yaml # основная модель, ~3 мин на GPU
uv run python -m toxicity.augment                                # data v2 для fairness
uv run python -m toxicity.train --config configs/train_base_aug.yaml
uv run python -m toxicity.predict "Ты молодец, спасибо!" "Какой же ты идиот"
uv run pytest                                                    # тесты
```

После обучения модель сохраняется в `models/<run_name>/` (не в git), а метрики и
fairness-отчёт — в `reports/<run_name>/` (`train.json`, `metrics.json`, `fairness.json`).

### Веб-демо

```bash
uv run python -m toxicity.app   # страница откроется сама (обычно http://127.0.0.1:7860)
```

Вводим комментарий — модель показывает вердикт «токсичный / нормальный» и вероятность.
Демо работает только локально: сервер слушает `127.0.0.1`, публичная ссылка Gradio
выключена, телеметрия отключена, текст никуда не отправляется.

## Инструменты и стандарты

| Что | Чем | Зачем |
|---|---|---|
| Окружение и версии | `uv` + `uv.lock` + `.python-version` | у всей команды и в CI одинаковые версии библиотек |
| Форматтер | `ruff format` | единое оформление кода |
| Линтер | `ruff check` | неиспользуемые импорты, неизвестные имена, типичные ошибки |
| Проверки перед коммитом | `pre-commit` | ruff, пробелы, YAML/TOML, большие файлы, ключи, актуальность `uv.lock` |
| CI | GitHub Actions | те же проверки и тесты на каждый Pull Request |
| Тесты | `pytest` | данные, метрики, fairness, аугментация |
| Документация модели | [Model Card](docs/MODEL_CARD.md) | назначение, данные, качество, ограничения, fairness |

Рефакторинг под эти стандарты: код разложен по модулям с одной задачей каждый
(`src/toxicity/`), параметры экспериментов вынесены из кода в YAML-конфиги, данные
версионируются через манифест с sha256, логика метрик и fairness отделена от PyTorch и
покрыта тестами, результаты сохраняются в `reports/` в машиночитаемом виде.

## Структура репозитория

```
├── configs/          # конфиги обучения, аугментации и автоматического переобучения (YAML)
├── data/             # данные не хранятся в git: только README.md и manifest.json
├── docs/
│   ├── bpmn/         # BPMN-схемы (bpmn.io) и их PNG
│   └── fairness/     # словарь подгрупп для fairness-анализа
├── notebooks/        # EDA и эксперименты
├── reports/          # метрики и fairness-отчёты обученных моделей
├── src/toxicity/
│   ├── data.py       # скачивание, очистка, разбиение, manifest
│   ├── augment.py    # нейтральные фразы для fairness (data v2)
│   ├── train.py      # дообучение (Hugging Face Trainer)
│   ├── evaluate.py   # метрики на test, порог, fairness-отчёт
│   ├── metrics.py    # метрики качества и выбор порога
│   ├── fairness.py   # Subgroup / BPSN / BNSP AUC, шаблонный тест-сет
│   ├── inference.py  # загрузка модели и скоринг
│   └── predict.py    # предсказание из командной строки
├── tests/
├── pyproject.toml    # зависимости и настройки ruff / pytest / uv
├── uv.lock           # точные версии всех библиотек
└── .python-version   # версия Python для uv
```

## Документация

| Документ | Что внутри | Этап |
|---|---|---|
| [Model Card](docs/MODEL_CARD.md) | цель, метрики, ограничения, bias/fairness, версия данных | ДЗ 1 |
| [Жизненный цикл модели](docs/LIFECYCLE.md) | BPMN от запроса бизнеса до вывода из эксплуатации, workflow согласования ([схема](docs/bpmn/model_lifecycle.png)) | ДЗ 1 |
| [Эксплуатация модели](docs/MODERATION_PROCESS.md) | BPMN работы модели в продукте: модерация комментариев с очередью, жалобами, SLA и безопасным отказом ([схема 01](docs/bpmn/moderation_process.png)), подпроцесс оценки комментария ([схема 02](docs/bpmn/scoring_subprocess.png)) | ДЗ 1 |
| [Retraining-цикл](docs/RETRAINING.md) | BPMN переобучения: триггер по дрифту, валидация, аукцион по AUC, деплой, Telegram ([схема](docs/bpmn/retraining_cycle.png)) | ДЗ 2 |
| [Автоматическое переобучение](docs/AUTO_RETRAINING.md) | BPMN автоматического переобучения и смены модели: три триггера, критерии остановки, canary и откат ([схема](docs/bpmn/auto_retraining.png), [пороги](configs/auto_retraining.yaml)) | ДЗ 3 |
| [Правила работы](CONTRIBUTING.md) | окружение, ветки, коммиты, Pull Request, код-ревью | — |

## Команда

| Роль | Участник |
|---|---|
| Инфраструктура | Артём Клюкин |
| Данные | Владислав Савеня |
| Обучение | Алексей Нечкин |
| Оценка и fairness | Владислав Сусанов |
| Документация и процесс | Андрей Буравов |
