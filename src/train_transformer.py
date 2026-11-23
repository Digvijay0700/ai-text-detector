"""Step 2b: fine-tune DistilRoBERTa (uses Apple-silicon GPU via MPS automatically).
Lower N_TRAIN if it is too slow.
"""
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

MODEL, MAX_LEN, N_TRAIN, N_VAL = "distilroberta-base", 256, 40_000, 4_000
tok = AutoTokenizer.from_pretrained(MODEL)


def load(name, n):
    df = pd.read_parquet(f"data/{name}.parquet").sample(frac=1, random_state=0).head(n)
    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN),
                  batched=True, remove_columns=["text"])


def metrics(p):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds), "f1": f1_score(p.label_ids, preds)}


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=2, id2label={0: "human", 1: "ai"}, label2id={"human": 0, "ai": 1})

args = TrainingArguments(
    output_dir="models/checkpoints", num_train_epochs=1,
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    learning_rate=3e-5, warmup_steps=150, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="no", logging_steps=100, report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=load("train", N_TRAIN),
                  eval_dataset=load("val", N_VAL), data_collator=DataCollatorWithPadding(tok),
                  compute_metrics=metrics)
trainer.train()
trainer.save_model("models/transformer")
tok.save_pretrained("models/transformer")
print("Saved models/transformer")
