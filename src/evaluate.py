"""Steps 3-4: evaluation, failure analysis and robustness (evasion) tests.
Run:  python -m src.evaluate
Writes report/results.json, report/confusion_*.png, report/failures.csv
"""
import json, random, re
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from src.detector import Detector

Path("report").mkdir(parents=True, exist_ok=True)
test = pd.read_parquet("data/test.parquet").sample(6000, random_state=1).reset_index(drop=True)
y = test["label"].values
base = joblib.load("models/baseline.joblib")
det = Detector()


def summarize(y, p):
    pred = (p >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred), "auc": roc_auc_score(y, p),
            "false_positive_rate": float(((pred == 1) & (y == 0)).sum() / (y == 0).sum())}


def cm_plot(y, p, title, path):
    cm = confusion_matrix(y, (p >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=13)
    ax.set_xticks([0, 1], ["Human", "AI"]); ax.set_yticks([0, 1], ["Human", "AI"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


p_base = base.predict_proba(test["text"])[:, 1]
p_tr = det.predict(test["text"].tolist())
res = {"n_test": len(test), "baseline": summarize(y, p_base), "transformer": summarize(y, p_tr)}
cm_plot(y, p_base, "TF-IDF + LogReg", "report/confusion_baseline.png")
cm_plot(y, p_tr, "DistilRoBERTa", "report/confusion_transformer.png")

# Failure cases (most confident mistakes)
test["p_ai"] = p_tr
fail = test[(p_tr >= 0.5).astype(int) != y].copy()
fail["confidence"] = np.abs(fail["p_ai"] - 0.5)
fail.sort_values("confidence", ascending=False).head(200)[["label", "p_ai", "text"]] \
    .to_csv("report/failures.csv", index=False)

# Robustness: how does AI-text recall drop after simple evasion edits?
random.seed(0)
ai_texts = test.loc[test.label == 1, "text"].head(800).tolist()


def typos(t, rate=0.04):
    c = list(t)
    for i in range(len(c) - 1):
        if c[i].isalpha() and random.random() < rate:
            c[i], c[i + 1] = c[i + 1], c[i]
    return "".join(c)


def no_punct_lower(t): return re.sub(r"[^\w\s]", "", t.lower())


def shuffle_sents(t):
    s = re.split(r"(?<=[.!?])\s+", t); random.shuffle(s); return " ".join(s)


def casual(t):
    for a, b in [("Moreover,", "Also,"), ("Furthermore,", "Plus,"), ("In conclusion,", "So,"),
                 ("However,", "But"), ("utilize", "use"), ("Additionally,", "And")]:
        t = t.replace(a, b)
    return t


attacks = {"none": lambda t: t, "typos (4%)": typos, "lowercase+no punctuation": no_punct_lower,
           "shuffled sentences": shuffle_sents, "casual word swaps": casual}
res["robustness_ai_recall"] = {k: float((det.predict([f(t) for t in ai_texts]) >= 0.5).mean())
                               for k, f in attacks.items()}
json.dump(res, open("report/results.json", "w"), indent=2)
print(json.dumps(res, indent=2))
