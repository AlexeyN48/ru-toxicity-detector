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
| Код | данные, дообучение, оценка и fairness-анализ, аугментация для fairness, тесты | готово |
| Дальше | мониторинг дрифта, аукцион champion/challenger, Telegram-бот | — |

## Модель и данные

- **Базовая модель:** [`ai-forever/ruBert-base`](https://huggingface.co/ai-forever/ruBert-base);
  baseline для сравнения — [`cointegrated/rubert-tiny2`](https://huggingface.co/cointegrated/rubert-tiny2).
- **Данные:** Russian Language Toxic Comments (Kaggle), подробности — [data/README.md](data/README.md).
- **Результаты:** метрики и fairness-анализ — в [Model Card](docs/MODEL_CARD.md), отчёты — в `reports/`.

## Как запустить

Нужен Python 3.11+. Для обучения желательна видеокарта NVIDIA, но всё работает и на CPU (медленнее).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# PyTorch с CUDA (для CPU можно просто: pip install torch)
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[train,dev]"
```

```bash
python -m toxicity.data prepare                           # скачать и разбить данные
python -m toxicity.train --config configs/train_tiny.yaml # baseline, ~1 мин на GPU
python -m toxicity.train --config configs/train_base.yaml # основная модель, ~3 мин на GPU
python -m toxicity.augment                                # data v2 для fairness
python -m toxicity.train --config configs/train_base_aug.yaml
python -m toxicity.predict "Ты молодец, спасибо!" "Какой же ты идиот"
pytest                                                    # тесты
```

После обучения модель сохраняется в `models/<run_name>/` (не в git), а метрики и
fairness-отчёт — в `reports/<run_name>/` (`train.json`, `metrics.json`, `fairness.json`).

### Веб-демо

```bash
pip install -e ".[demo]"
python -m toxicity.app        # страница откроется сама (обычно http://127.0.0.1:7860)
```

Вводим комментарий — модель показывает вердикт «токсичный / нормальный» и вероятность.
Демо работает только локально: сервер слушает `127.0.0.1`, публичная ссылка Gradio
выключена, телеметрия отключена, текст никуда не отправляется.

## Структура репозитория

```
├── configs/          # конфиги обучения и шаблоны аугментации (YAML)
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
└── tests/
```

## Документация

| Документ | Что внутри | Этап |
|---|---|---|
| [Model Card](docs/MODEL_CARD.md) | цель, метрики, ограничения, bias/fairness, версия данных | ДЗ 1 |
| [Жизненный цикл модели](docs/LIFECYCLE.md) | BPMN от запроса бизнеса до вывода из эксплуатации, workflow согласования ([схема](docs/bpmn/model_lifecycle.png)) | ДЗ 1 |
| [Retraining-цикл](docs/RETRAINING.md) | BPMN переобучения: триггер по дрифту, валидация, аукцион по AUC, деплой, Telegram ([схема](docs/bpmn/retraining_cycle.png)) | ДЗ 2 |
| [Правила работы](CONTRIBUTING.md) | окружение, ветки, коммиты, Pull Request, код-ревью | — |

## Команда

| Роль | Участник |
|---|---|
| Инфраструктура | Артём |
| Данные | Влад Савеня |
| Обучение | Лёша |
| Оценка и fairness | Влад Сусанов |
| Документация и процесс | Андрей |
