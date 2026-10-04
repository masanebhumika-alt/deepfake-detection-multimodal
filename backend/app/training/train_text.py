"""
Trains a real classifier for the text detector on labeled data.

Expected input CSV columns: text,label   (label: 1 = AI-generated, 0 = human)

To use with HC3 (after downloading via load_unified_dataset.py), first
flatten it into this format, e.g.:

    import pandas as pd
    from datasets import load_dataset
    hc3 = load_dataset("Hello-SimpleAI/HC3", "all")["train"]
    rows = []
    for r in hc3:
        for h in r["human_answers"]:
            rows.append({"text": h, "label": 0})
        for c in r["chatgpt_answers"]:
            rows.append({"text": c, "label": 1})
    pd.DataFrame(rows).to_csv("hc3_flat.csv", index=False)

Then run:
    python -m app.training.train_text hc3_flat.csv
"""
import os
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.models import text_detector

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts", "text_model.joblib")
CUSTOM_FEATURE_NAMES = [
    "avg_sentence_len", "sentence_len_burstiness", "avg_word_len",
    "type_token_ratio", "punctuation_diversity", "local_perplexity",
]
SAMPLES_PER_CLASS = 3000
SAMPLING_RANDOM_STATE = 42


def chunk_text_for_short_samples(text, max_sentences=3):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    for index in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[index:index + max_sentences]).strip()
        if len(chunk.split()) >= 10:
            chunks.append(chunk)
    return chunks or ([text.strip()] if len(text.split()) >= 1 else [])


def custom_feature_matrix(texts, stage):
    rows = []
    for index, text in enumerate(texts, start=1):
        features = text_detector.extract_features(text)
        rows.append([features[name] for name in CUSTOM_FEATURE_NAMES])
        if index % 500 == 0 or index == len(texts):
            print(f"[INFO] {stage} perplexity extraction: {index}/{len(texts)} chunks", flush=True)
    return np.asarray(rows, dtype=float)


def main(csv_path: str):
    df = pd.read_csv(csv_path)
    assert {"text", "label"}.issubset(df.columns), "CSV must have 'text' and 'label' columns"

    expanded_texts, expanded_labels = [], []
    for text, label in zip(df["text"].astype(str), df["label"].values):
        for chunk in chunk_text_for_short_samples(text):
            expanded_texts.append(chunk)
            expanded_labels.append(label)

    expanded_texts = np.asarray(expanded_texts, dtype=object)
    expanded_labels = np.asarray(expanded_labels)
    class_indices = {
        label: np.flatnonzero(expanded_labels == label)
        for label in (0, 1)
    }
    samples_per_class = min(SAMPLES_PER_CLASS, *(len(indices) for indices in class_indices.values()))
    if samples_per_class == 0:
        raise ValueError("Training CSV must contain examples for both labels 0 and 1")

    rng = np.random.default_rng(SAMPLING_RANDOM_STATE)
    selected_indices = np.concatenate([
        rng.choice(indices, size=samples_per_class, replace=False)
        for indices in class_indices.values()
    ])
    rng.shuffle(selected_indices)
    expanded_texts = expanded_texts[selected_indices].tolist()
    expanded_labels = expanded_labels[selected_indices]

    print(f"[INFO] expanded samples before cap: {len(class_indices[0]) + len(class_indices[1])}")
    print(f"[INFO] balanced sampled examples: {samples_per_class} per class")

    texts_train, texts_test, y_train, y_test = train_test_split(
        expanded_texts, expanded_labels, test_size=0.2,
        random_state=42, stratify=expanded_labels,
    )

    word_tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, analyzer="word")
    char_tfidf = TfidfVectorizer(ngram_range=(3, 5), max_features=1000, analyzer="char")
    word_train = word_tfidf.fit_transform(texts_train)
    char_train = char_tfidf.fit_transform(texts_train)
    word_test = word_tfidf.transform(texts_test)
    char_test = char_tfidf.transform(texts_test)

    custom_scaler = StandardScaler()
    custom_train = custom_scaler.fit_transform(custom_feature_matrix(texts_train, "train"))
    custom_test = custom_scaler.transform(custom_feature_matrix(texts_test, "test"))
    X_train = hstack([csr_matrix(custom_train), word_train, char_train])
    X_test = hstack([csr_matrix(custom_test), word_test, char_test])

    print(f"[INFO] expanded samples: {len(expanded_texts)}")
    print(f"[INFO] hybrid feature shape: {X_train.shape}")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    print(classification_report(y_test, clf.predict(X_test), target_names=["Human", "AI"]))

    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    joblib.dump({
        "classifier": clf,
        "word_tfidf": word_tfidf,
        "char_tfidf": char_tfidf,
        "custom_scaler": custom_scaler,
        "custom_feature_names": CUSTOM_FEATURE_NAMES,
    }, ARTIFACT_PATH)
    print(f"Saved trained model to {ARTIFACT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m app.training.train_text <path_to_labeled_csv>")
        sys.exit(1)
    main(sys.argv[1])
