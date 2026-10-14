"""Evaluate a trained model: metrics on test, threshold from validation, fairness report.

Usage:
    python -m toxicity.evaluate --model models/ruBert-base
"""

import argparse
import json
from pathlib import Path

import numpy as np

from toxicity import fairness, inference
from toxicity.data import load_split
from toxicity.metrics import check_targets, choose_threshold, classification_report


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")


def evaluate_model(model_dir: Path, report_dir: Path, max_length: int = 128) -> dict:
    tokenizer, model = inference.load(model_dir)
    val, test = load_split("val"), load_split("test")

    def score(texts: list[str]) -> np.ndarray:
        return inference.predict_proba(texts, tokenizer, model, max_length=max_length)

    threshold = choose_threshold(val["label"].to_numpy(), score(val["text"].tolist()))
    test_texts, y_test = test["text"].tolist(), test["label"].to_numpy()
    test_scores = score(test_texts)

    metrics = classification_report(y_test, test_scores, threshold)
    metrics["latency_ms"] = inference.latency_ms(
        test_texts, tokenizer, model, max_length=max_length
    )
    metrics["targets_met"] = check_targets(metrics)

    terms = fairness.load_terms()
    sentences = fairness.template_sentences(terms["categories"], terms["templates"])
    sent_scores = score([s for _, s in sentences])
    template_scores = {
        name: sent_scores[[i for i, (cat, _) in enumerate(sentences) if cat == name]]
        for name in terms["categories"]
    }
    fair = fairness.analyze(test_texts, y_test, test_scores, threshold, template_scores, terms)

    report_dir.mkdir(parents=True, exist_ok=True)
    _write(report_dir / "metrics.json", metrics)
    _write(report_dir / "fairness.json", fair)

    print(json.dumps({k: v for k, v in metrics.items() if k != "targets_met"}, indent=2))
    print("Targets met:", metrics["targets_met"])
    print("Fairness passed:", fair["passed"], fair["failures"])
    return {"metrics": metrics, "fairness": fair}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()
    evaluate_model(args.model, Path("reports") / args.model.name, args.max_length)


if __name__ == "__main__":
    main()
