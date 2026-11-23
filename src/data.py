"""Step 1: assemble a large labelled human-vs-AI dataset.
Run:  python -m src.data
Source: HuggingFace 'artem9k/ai-text-detection-pile' (~1.4M human + AI texts).
"""
import re
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

DATASET = "artem9k/ai-text-detection-pile"
POOL = 400_000          # rows pulled from the 1.4M pool
PER_CLASS = 60_000      # final balanced size per class (120k total)
MIN_CHARS, MAX_CHARS = 300, 1500
SEED = 42


def clean(t: str) -> str:
    return re.sub(r"\s+", " ", str(t)).strip()


def main():
    ds = load_dataset(DATASET, split="train").shuffle(seed=SEED).select(range(POOL))
    df = ds.to_pandas()
    print("Columns:", list(df.columns))
    print(df["source"].value_counts())            # check label names here

    df["label"] = (df["source"].str.lower() != "human").astype(int)  # 1 = AI
    df["text"] = df["text"].map(clean)
    df = df[df["text"].str.len() >= MIN_CHARS]
    df["text"] = df["text"].str.slice(0, MAX_CHARS)
    df = df.drop_duplicates("text")

    n = min(PER_CLASS, df["label"].value_counts().min())
    df = pd.concat([g.sample(n, random_state=SEED) for _, g in df.groupby("label")])
    df = df[["text", "label", "source"]].reset_index(drop=True)

    train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)
    Path("data").mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(f"data/{name}.parquet", index=False)
        print(name, len(part), round(part["label"].mean(), 3))


if __name__ == "__main__":
    main()
