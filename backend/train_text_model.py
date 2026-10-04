import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

ID2LABEL = {0: "Human", 1: "AI-Generated"}
LABEL2ID = {"Human": 0, "AI-Generated": 1}


def load_records(path: str):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".json") or path.endswith(".jsonl"):
        df = pd.read_json(path, lines=path.endswith(".jsonl"))
    else:
        raise ValueError(f"unsupported file type: {path}")

    assert "text" in df.columns and "label" in df.columns, "file must have 'text' and 'label' columns"
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    assert set(df["label"].unique()) <= {0, 1}, "labels must be 0 (Human) or 1 (AI-Generated)"
    return df["text"].astype(str).tolist(), df["label"].tolist()


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length", max_length=max_length
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    ai_probs = probs[:, LABEL2ID["AI-Generated"]]
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, ai_probs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_file", help="CSV/JSON with 'text','label' columns")
    ap.add_argument("--val_file", default=None, help="optional separate validation file")
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--output_dir", default="app/models/artifacts/text_transformer")
    ap.add_argument("--max_length", default=512, type=int)
    ap.add_argument("--epochs", default=4, type=int)
    ap.add_argument("--lr", default=2e-5, type=float)
    ap.add_argument("--batch_size", default=8, type=int)
    ap.add_argument("--val_split", default=0.15, type=float)
    args = ap.parse_args()

    texts, labels = load_records(args.train_file)

    if args.val_file:
        val_texts, val_labels = load_records(args.val_file)
        train_texts, train_labels = texts, labels
    else:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=args.val_split, stratify=labels, random_state=42
        )

    print(f"[INFO] train={len(train_texts)} val={len(val_texts)}")
    print(f"[INFO] train class balance: {np.bincount(train_labels)}")
    print(f"[INFO] val class balance  : {np.bincount(val_labels)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    # belt-and-suspenders: force mapping explicitly, some architectures reset config on load
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID

    train_ds = TextDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=int(0.1 * (len(train_ds) / args.batch_size) * args.epochs),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    print("[INFO] final eval metrics:")
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # re-assert label mapping survives save/reload round trip
    saved_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    print(f"[VERIFY] reloaded id2label: {saved_model.config.id2label}")
    assert saved_model.config.id2label[1] == "AI-Generated", "[FAIL] label mapping did not persist correctly"

    print(f"[SUCCESS] model + tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()
