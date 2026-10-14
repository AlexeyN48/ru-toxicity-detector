import pandas as pd

from toxicity.data import build_manifest, clean, sha256, split


def make_df(n: int = 200) -> pd.DataFrame:
    return pd.DataFrame(
        {"text": [f"комментарий {i}" for i in range(n)], "label": [i % 3 == 0 for i in range(n)]}
    ).astype({"label": int})


def test_clean_strips_and_drops_empty_and_duplicates():
    df = pd.DataFrame({"text": ["  привет ", "привет", "   ", "ПРИВЕТ"], "label": [0, 0, 0, 1]})
    out = clean(df)
    assert out["text"].tolist() == ["привет", "ПРИВЕТ"]  # case is kept on purpose


def test_split_sizes_and_stratification():
    df = make_df(1000)
    parts = split(df, seed=42)
    assert sum(len(p) for p in parts.values()) == len(df)
    assert abs(len(parts["train"]) - 700) <= 1
    share = df["label"].mean()
    for part in parts.values():
        assert abs(part["label"].mean() - share) < 0.01


def test_split_is_deterministic():
    df = make_df(500)
    assert split(df, seed=1)["test"].equals(split(df, seed=1)["test"])


def test_sha256_and_manifest(tmp_path):
    raw = tmp_path / "labeled.csv"
    raw.write_text("comment,toxic\nпривет,0.0\n", encoding="utf-8")
    parts = split(make_df(100), seed=42)
    manifest = build_manifest(raw, parts, seed=42)
    assert manifest["raw_sha256"] == sha256(raw)
    assert len(manifest["raw_sha256"]) == 64
    assert manifest["rows_after_cleaning"] == 100
    assert set(manifest["splits"]) == {"train", "val", "test"}
