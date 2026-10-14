"""Score comments with a trained model.

Usage:
    python -m toxicity.predict "текст комментария" "ещё один"
"""

import argparse
import json
from pathlib import Path

from toxicity import inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="+")
    parser.add_argument("--model", type=Path, default=Path("models/ruBert-base"))
    args = parser.parse_args()

    metrics_path = Path("reports") / args.model.name / "metrics.json"
    threshold = 0.5
    if metrics_path.exists():
        threshold = json.loads(metrics_path.read_text(encoding="utf-8"))["threshold"]

    tokenizer, model = inference.load(args.model)
    scores = inference.predict_proba(args.texts, tokenizer, model)
    for text, score in zip(args.texts, scores, strict=True):
        label = "ТОКСИЧНЫЙ" if score >= threshold else "нормальный"
        print(f"{score:.3f}  {label:<10}  {text}")
    print(f"(порог {threshold:.3f}, модель {args.model})")


if __name__ == "__main__":
    main()
