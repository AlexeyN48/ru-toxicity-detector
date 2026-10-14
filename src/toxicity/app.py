"""Local web demo: type a comment and see whether the model finds it toxic.

Usage:
    python -m toxicity.app            # opens http://127.0.0.1:7860 (or next free port), local only
"""

import json
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no usage telemetry

import gradio as gr

from toxicity import inference

MODELS = {
    "ruBert-base-aug — основная (с аугментацией для fairness)": "ruBert-base-aug",
    "ruBert-base — без аугментации": "ruBert-base",
    "rubert-tiny2 — маленький baseline": "rubert-tiny2",
}

EXAMPLES = [
    "Спасибо за подробный разбор, очень помогло!",
    "Ты вообще читать умеешь? Бред какой-то несёшь, идиот.",
    "Автор, ты молодец, но с выводами не согласен.",
    "Я мусульманин и работаю врачом.",
    "Мой сосед гей, отличный парень.",
    "Да пошёл ты, клоун.",
]


@lru_cache(maxsize=3)
def _load(run: str):
    tokenizer, model = inference.load(Path("models") / run)
    metrics_path = Path("reports") / run / "metrics.json"
    threshold = 0.5
    if metrics_path.exists():
        threshold = json.loads(metrics_path.read_text(encoding="utf-8"))["threshold"]
    return tokenizer, model, threshold


def classify(text: str, model_title: str):
    text = (text or "").strip()
    if not text:
        return {}, "Введите комментарий."
    run = MODELS[model_title]
    tokenizer, model, threshold = _load(run)
    p = float(inference.predict_proba([text], tokenizer, model)[0])
    if p >= threshold:
        verdict = f"### 🚩 Токсичный\nВероятность **{p:.1%}** ≥ порога {threshold:.1%} — отправить модератору."
    else:
        verdict = (
            f"### ✅ Нормальный\nВероятность токсичности **{p:.1%}** < порога {threshold:.1%}."
        )
    return {"токсичный": p, "нормальный": 1 - p}, verdict


def build() -> gr.Blocks:
    with gr.Blocks(title="ru-toxicity-detector") as demo:
        gr.Markdown(
            "# Детектор токсичных комментариев\n"
            "Дообученный ruBERT оценивает, насколько комментарий оскорбительный. "
            "Модель только **помогает модератору** — решение принимает человек."
        )
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Комментарий", lines=4, placeholder="Напишите комментарий на русском…"
                )
                model = gr.Dropdown(list(MODELS), value=next(iter(MODELS)), label="Модель")
                button = gr.Button("Проверить", variant="primary")
            with gr.Column(scale=2):
                verdict = gr.Markdown()
                scores = gr.Label(label="Вероятности", num_top_classes=2)
        gr.Examples(
            [[e] for e in EXAMPLES],
            inputs=text,
            outputs=[scores, verdict],
            fn=lambda t: classify(t, next(iter(MODELS))),
            run_on_click=True,
            cache_examples=False,
            label="Примеры (клик — сразу проверка основной моделью)",
        )
        gr.Markdown(
            "<small>Порог взят из `reports/<модель>/metrics.json` (подобран на validation под "
            "Recall ≥ 0.85). Известные ограничения и fairness-анализ — в "
            "`docs/MODEL_CARD.md`: модель всё ещё чаще ошибается на упоминаниях групп людей."
            "</small>"
        )
        button.click(classify, inputs=[text, model], outputs=[scores, verdict])
        text.submit(classify, inputs=[text, model], outputs=[scores, verdict])
    return demo


def main() -> None:
    build().launch(
        server_name="127.0.0.1",  # local only, not reachable from the network
        server_port=None,  # 7860, or the next free port if it is busy
        inbrowser=True,  # open the page automatically
        share=False,  # no public gradio.live link
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
