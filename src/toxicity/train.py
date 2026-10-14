"""Fine-tune a BERT model for toxic comment classification.

Usage:
    python -m toxicity.train --config configs/train_base.yaml
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from toxicity.data import load_split
from toxicity.evaluate import evaluate_model


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(texts, truncation=True, max_length=max_length)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    scores = torch.softmax(logits, dim=-1)[:, 1].numpy()
    return {
        "roc_auc": roc_auc_score(labels, scores),
        "pr_auc": average_precision_score(labels, scores),
    }


def train(config: dict) -> Path:
    set_seed(config["seed"])
    run = config["run_name"]
    model_dir = Path("models") / run
    report_dir = Path("reports") / run

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=2)

    datasets = {}
    splits = {"train": config.get("train_split", "train"), "val": "val"}
    for name, split_name in splits.items():
        df = load_split(split_name)
        datasets[name] = EncodedDataset(
            df["text"].tolist(), df["label"].tolist(), tokenizer, config["max_length"]
        )

    args = TrainingArguments(
        output_dir=str(Path("outputs") / run),
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"] * 2,
        weight_decay=config["weight_decay"],
        warmup_steps=config["warmup_ratio"],
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        fp16=config["fp16"] and torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=config["seed"],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])
        ],
    )

    start = time.perf_counter()
    trainer.train()
    train_minutes = (time.perf_counter() - start) / 60

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    history = [
        {k: v for k, v in rec.items() if k in ("epoch", "eval_roc_auc", "eval_pr_auc", "eval_loss")}
        for rec in trainer.state.log_history
        if "eval_roc_auc" in rec
    ]
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": config,
        "train_minutes": round(train_minutes, 2),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "best_val_roc_auc": float(np.max([h["eval_roc_auc"] for h in history])),
        "epochs_run": len(history),
        "history": history,
    }
    (report_dir / "train.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(f"Training finished in {train_minutes:.1f} min, model saved to {model_dir}")
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--no-eval", action="store_true", help="skip evaluation after training")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_dir = train(config)
    if not args.no_eval:
        evaluate_model(model_dir, Path("reports") / config["run_name"], config["max_length"])


if __name__ == "__main__":
    main()
