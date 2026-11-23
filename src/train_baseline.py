"""Step 2a: fast classical baseline (TF-IDF + Logistic Regression)."""
from pathlib import Path

import joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

train = pd.read_parquet("data/train.parquet")
pipe = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_features=300_000, sublinear_tf=True, min_df=3),
    LogisticRegression(max_iter=1000, C=4.0),
)
pipe.fit(train["text"], train["label"])
Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, "models/baseline.joblib")
print("Saved models/baseline.joblib")
