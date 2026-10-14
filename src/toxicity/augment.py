"""Fairness mitigation: add neutral identity-mention sentences to the train split.

Usage:
    python -m toxicity.augment
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from toxicity.data import MANIFEST_PATH, PROCESSED_DIR, load_split
from toxicity.fairness import _morph, load_terms

CONFIG_PATH = Path("configs/augmentation.yaml")


def _form(term: str, tags: set[str]) -> str:
    inflected = _morph().parse(term)[0].inflect(tags)
    return inflected.word if inflected else term


def _pronoun(term: str) -> str:
    return "она" if "femn" in _morph().parse(term)[0].tag else "он"


def neutral_sentences(terms: dict, templates: list[str]) -> list[str]:
    """All templates filled with all identity terms; every sentence is non-toxic."""
    out = []
    for cat in terms["categories"].values():
        for term in cat["terms"]:
            forms = {
                "term": term,
                "Term": term.capitalize(),
                "term_ins": _form(term, {"ablt"}),
                "pron": _pronoun(term),
                "plur": _form(term, {"plur", "nomn"}),
                "Plur": _form(term, {"plur", "nomn"}).capitalize(),
                "plur_gen": _form(term, {"plur", "gent"}),
            }
            out += [tpl.format(**forms) for tpl in templates]
    return out


def augment(config_path: Path = CONFIG_PATH) -> pd.DataFrame:
    templates = yaml.safe_load(config_path.read_text(encoding="utf-8"))["templates"]
    sentences = neutral_sentences(load_terms(), templates)
    train = load_split("train")
    extra = pd.DataFrame({"text": sentences, "label": 0})
    augmented = pd.concat([train, extra], ignore_index=True).sample(frac=1, random_state=42)
    augmented.to_csv(PROCESSED_DIR / "train_aug.csv", index=False)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["augmented_train"] = {
        "data_version": "v2",
        "file": "train_aug.csv",
        "base_rows": len(train),
        "added_neutral_rows": len(extra),
        "templates": str(config_path).replace("\\", "/"),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return augmented


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    df = augment()
    print(f"train_aug.csv: {len(df)} rows ({int((df['label'] == 0).sum())} non-toxic)")


if __name__ == "__main__":
    main()
