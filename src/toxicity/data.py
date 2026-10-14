"""Dataset download, cleaning, stratified split and version manifest.

Usage:
    python -m toxicity.data prepare
"""

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_NAME = "Russian Language Toxic Comments"
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/blackmoon/russian-language-toxic-comments"
)
DATASET_PAGE = "https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments"
DATASET_LICENSE = "GPL-2.0"
RAW_FILE = "labeled.csv"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = DATA_DIR / "manifest.json"

SPLITS = ("train", "val", "test")


def download(raw_dir: Path = RAW_DIR) -> Path:
    """Download the Kaggle archive and extract the CSV (skipped if already present)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / RAW_FILE
    if target.exists():
        return target
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with archive.open(RAW_FILE) as src:
        target.write_bytes(src.read())
    return target


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV into columns `text` (str) and `label` (0/1)."""
    df = pd.read_csv(path)
    return pd.DataFrame({"text": df["comment"].astype(str), "label": df["toxic"].astype(int)})


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, drop empty texts and exact duplicates.

    Case, punctuation, obscenities and emoji are kept on purpose: they carry toxicity signal.
    """
    df = df.assign(text=df["text"].str.strip())
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="text")
    return df.reset_index(drop=True)


def split(
    df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Stratified train / val / test split."""
    train, rest = train_test_split(
        df, test_size=val_size + test_size, stratify=df["label"], random_state=seed
    )
    val, test = train_test_split(
        rest,
        test_size=test_size / (val_size + test_size),
        stratify=rest["label"],
        random_state=seed,
    )
    return {
        name: part.reset_index(drop=True)
        for name, part in zip(SPLITS, (train, val, test), strict=True)
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(raw_path: Path, parts: dict[str, pd.DataFrame], seed: int) -> dict:
    total = sum(len(p) for p in parts.values())
    toxic = sum(int(p["label"].sum()) for p in parts.values())
    return {
        "dataset": DATASET_NAME,
        "source": DATASET_PAGE,
        "license": DATASET_LICENSE,
        "data_version": "v1",
        "raw_file": RAW_FILE,
        "raw_sha256": sha256(raw_path),
        "rows_after_cleaning": total,
        "toxic_share": round(toxic / total, 4),
        "seed": seed,
        "splits": {
            name: {"rows": len(part), "toxic": int(part["label"].sum())}
            for name, part in parts.items()
        },
        "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def prepare(seed: int = 42) -> dict:
    raw_path = download()
    parts = split(clean(load_raw(raw_path)), seed=seed)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    manifest = build_manifest(raw_path, parts, seed)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return manifest


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / f"{name}.csv", keep_default_na=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
