"""Inference wrapper used by the evaluation script and the web app."""
import re
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Detector:
    def __init__(self, path="models/transformer", max_len=256):
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.max_len = max_len

    @torch.no_grad()
    def predict(self, texts, batch_size=32):
        """Return P(AI) for each text."""
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(list(texts[i:i + batch_size]), truncation=True, max_length=self.max_len,
                           padding=True, return_tensors="pt").to(self.device)
            out.append(torch.softmax(self.model(**enc).logits, -1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.array([])

    def analyze(self, text):
        """Overall score + per-sentence scores (sentence scores are noisy)."""
        sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
        overall = float(self.predict([text])[0])
        per_sent = self.predict(sents).tolist() if len(sents) > 1 else [overall]
        return overall, sents, per_sent
