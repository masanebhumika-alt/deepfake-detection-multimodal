import json
import hashlib
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix, hstack, vstack

from app.models import audio_detector, image_detector, text_detector
from app.training import train_audio, train_text


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "app" / "models" / "artifacts"
CACHE = ROOT / "training_logs" / "cache"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _fingerprint(items, labels):
    digest = hashlib.sha256()
    for item, label in zip(items, labels):
        if isinstance(item, Path):
            stat = item.stat()
            value = f"{item.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        else:
            value = repr(item)
        digest.update(value.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(str(label).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_cache(path, fingerprint):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["fingerprint"].item()) != fingerprint:
                return None
            return data["X"], data["y"]
    except (OSError, ValueError, KeyError):
        return None


def _write_cache(path, features, labels, fingerprint):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as cache_file:
        np.savez_compressed(
            cache_file,
            X=np.asarray(features),
            y=np.asarray(labels),
            fingerprint=np.asarray(fingerprint),
        )
    temporary.replace(path)


def cached(name, items, labels, compute_fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)
    fingerprint = _fingerprint(items, labels)
    path = CACHE / f"{name}.npz"
    result = _read_cache(path, fingerprint)
    if result is not None:
        print(f"[CACHE] loaded {name}: {len(result[1])} rows", flush=True)
        return result

    features, computed_labels = compute_fn()
    features = np.asarray(features)
    computed_labels = np.asarray(computed_labels)
    if len(features) != len(labels) or not np.array_equal(computed_labels, labels):
        raise ValueError(f"Computed features/labels do not match the {name} cache inputs")
    _write_cache(path, features, computed_labels, fingerprint)
    print(f"[CACHE] saved {name}: {len(computed_labels)} rows", flush=True)
    return features, computed_labels


def chunked(name, items, labels, compute_fn, size=500):
    CACHE.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels)
    parts = []
    for start in range(0, len(items), size):
        stop = min(start + size, len(items))
        chunk_items = items[start:stop]
        chunk_labels = labels[start:stop]
        fingerprint = _fingerprint(chunk_items, chunk_labels)
        path = CACHE / f"{name}_{start:06d}_{fingerprint[:16]}.npz"
        result = _read_cache(path, fingerprint)
        if result is not None and np.array_equal(result[1], chunk_labels):
            features = result[0]
            print(f"[CACHE] loaded {name} rows {start}:{stop}", flush=True)
        else:
            features = np.asarray(compute_fn(chunk_items))
            if len(features) != len(chunk_labels):
                raise ValueError(f"Feature count mismatch in {name} rows {start}:{stop}")
            _write_cache(path, features, chunk_labels, fingerprint)
            print(f"[CACHE] saved {name} rows {start}:{stop}", flush=True)
        parts.append(features)
    if not parts:
        return np.empty((0, 0)), labels
    return np.vstack(parts), labels


def split_indices(labels):
    indices = np.arange(len(labels))
    return train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def dataset_counts(labels, train_indices, test_indices):
    labels = np.asarray(labels)
    return {
        "total": int(len(labels)),
        "train": int(len(train_indices)),
        "test": int(len(test_indices)),
        "train_class_balance": {
            str(label): int(np.count_nonzero(labels[train_indices] == label))
            for label in np.unique(labels)
        },
        "test_class_balance": {
            str(label): int(np.count_nonzero(labels[test_indices] == label))
            for label in np.unique(labels)
        },
    }


def split_dataset_counts(train_labels, test_labels):
    train_labels = np.asarray(train_labels)
    test_labels = np.asarray(test_labels)
    return {
        "train": int(len(train_labels)),
        "test": int(len(test_labels)),
        "train_class_balance": {
            str(label): int(np.count_nonzero(train_labels == label))
            for label in np.unique(train_labels)
        },
        "test_class_balance": {
            str(label): int(np.count_nonzero(test_labels == label))
            for label in np.unique(test_labels)
        },
    }


def wilson_interval(correct, total, z=1.96):
    if total <= 0:
        raise ValueError("Wilson interval requires a positive test-set size")
    proportion = correct / total
    denominator = 1 + z ** 2 / total
    center = (proportion + z ** 2 / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z ** 2 / (4 * total ** 2)
    ) / denominator
    return {
        "lower": round(center - margin, 6),
        "upper": round(center + margin, 6),
        "n": int(total),
    }


def metrics_for(estimator, features, labels, test_indices, include_auc):
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    predictions = estimator.predict(test_features)
    correct = int(np.count_nonzero(predictions == test_labels))
    metrics = {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "precision": float(precision_score(test_labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(test_labels, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(test_labels, predictions, pos_label=1, zero_division=0)),
    }
    if include_auc:
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(test_features)
            positive_index = list(estimator.classes_).index(1)
            positive_scores = probabilities[:, positive_index]
        else:
            positive_scores = estimator.decision_function(test_features)
            if estimator.classes_[1] != 1:
                positive_scores = -positive_scores
        metrics["auc"] = float(roc_auc_score(test_labels, positive_scores))
    rounded = {name: round(value, 6) for name, value in metrics.items()}
    rounded["accuracy_ci_95"] = wilson_interval(correct, len(test_labels))
    return rounded


def evaluate_models(modality, features, labels, train_indices, test_indices, include_auc):
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ("Random Forest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
        ("Linear SVM", LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000)),
    ]
    results = {}
    for name, estimator in models:
        print(f"[FIT] {modality}: {name} ({type(estimator).__name__})", flush=True)
        estimator.fit(features[train_indices], labels[train_indices])
        results[name] = metrics_for(estimator, features, labels, test_indices, include_auc)
    return results


def evaluate_frozen_model(estimator, features, labels):
    return metrics_for(
        estimator,
        features,
        labels,
        np.arange(len(labels)),
        include_auc=True,
    )


def batched_perplexities(texts, batch_size=8):
    tokenizer, model = text_detector._load_ppl_model()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    perplexities = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        if input_ids.shape[1] < 2:
            perplexities.extend([0.0] * len(batch))
            continue

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        token_losses = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.shape[-1]),
            shifted_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(len(batch), -1)
        token_counts = shifted_labels.ne(-100).sum(dim=1)
        mean_losses = token_losses.sum(dim=1) / token_counts.clamp(min=1)
        batch_perplexities = torch.exp(mean_losses)
        batch_perplexities[token_counts == 0] = 0.0
        perplexities.extend(batch_perplexities.cpu().tolist())

        processed = min(start + len(batch), len(texts))
        if processed % 500 < batch_size or processed == len(texts):
            print(f"[INFO] benchmark batched perplexity: {processed}/{len(texts)} chunks", flush=True)

    return np.asarray(perplexities, dtype=float)


def extract_text_feature_chunk(texts):
    original_perplexity = text_detector._compute_perplexity
    text_detector._compute_perplexity = lambda text, max_length=512: 0.0
    try:
        rows = [text_detector.extract_features(text) for text in texts]
    finally:
        text_detector._compute_perplexity = original_perplexity

    local_perplexities = batched_perplexities(texts)
    return np.asarray([
        [
            row["avg_sentence_len"],
            row["sentence_len_burstiness"],
            row["avg_word_len"],
            row["type_token_ratio"],
            row["punctuation_diversity"],
            float(perplexity),
        ]
        for row, perplexity in zip(rows, local_perplexities)
    ], dtype=float)


def sample_text_chunks():
    csv_path = ROOT / "hc3_flat.csv"
    frame = pd.read_csv(csv_path)
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"Expected text and label columns in {csv_path}")

    texts = []
    labels = []
    groups = []
    for source_answer_id, (text, label) in enumerate(
        zip(frame["text"].astype(str), frame["label"].astype(int))
    ):
        if label not in (0, 1):
            continue
        chunks = train_text.chunk_text_for_short_samples(text)
        texts.extend(chunks)
        labels.extend([label] * len(chunks))
        groups.extend([source_answer_id] * len(chunks))

    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=int)
    class_indices = {label: np.flatnonzero(labels == label) for label in (0, 1)}
    samples_per_class = min(train_text.SAMPLES_PER_CLASS, *(len(indices) for indices in class_indices.values()))
    if samples_per_class == 0:
        raise ValueError("HC3 data must contain examples for both labels 0 and 1")

    rng = np.random.default_rng(train_text.SAMPLING_RANDOM_STATE)
    selected = np.concatenate([
        rng.choice(class_indices[label], size=samples_per_class, replace=False)
        for label in (0, 1)
    ])
    rng.shuffle(selected)
    texts = np.asarray(texts, dtype=object)[selected]
    labels = labels[selected]
    groups = groups[selected]
    return texts, labels, groups, samples_per_class


def cached_text_features(name, texts, labels):
    return cached(
        name,
        texts.tolist(),
        labels,
        lambda: chunked(
            f"{name}_chunk",
            texts.tolist(),
            labels,
            extract_text_feature_chunk,
        ),
    )


def prepare_text_data():
    texts, labels, groups, samples_per_class = sample_text_chunks()
    custom_features, labels = cached(
        "text_gpt2_original_sample",
        texts.tolist(),
        labels,
        lambda: chunked(
            "text_gpt2_original_sample_chunk",
            texts.tolist(),
            labels,
            extract_text_feature_chunk,
        ),
    )
    grouped_split = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_indices, test_indices = next(grouped_split.split(texts, labels, groups))
    if len(np.unique(labels[train_indices])) != 2 or len(np.unique(labels[test_indices])) != 2:
        raise ValueError("Grouped text split did not retain both classes in train and test")

    train_texts = [texts[index] for index in train_indices]
    test_texts = [texts[index] for index in test_indices]
    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000, analyzer="word")
    char_vectorizer = TfidfVectorizer(ngram_range=(3, 5), max_features=1000, analyzer="char")
    word_train = word_vectorizer.fit_transform(train_texts)
    word_test = word_vectorizer.transform(test_texts)
    char_train = char_vectorizer.fit_transform(train_texts)
    char_test = char_vectorizer.transform(test_texts)

    custom_train = custom_features[train_indices]
    custom_test = custom_features[test_indices]
    scaler = StandardScaler()
    custom_train = scaler.fit_transform(custom_train)
    custom_test = scaler.transform(custom_test)

    train_features = hstack([csr_matrix(custom_train), word_train, char_train]).tocsr()
    test_features = hstack([csr_matrix(custom_test), word_test, char_test]).tocsr()
    features = vstack([train_features, test_features]).tocsr()
    labels = np.concatenate([labels[train_indices], labels[test_indices]])
    ordered_train_indices = np.arange(len(train_indices))
    ordered_test_indices = np.arange(len(train_indices), len(labels))
    split_info = {
        "method": "GroupShuffleSplit",
        "description": "source-grouped; leakage-reduced text baseline split",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "sampling_seed": train_text.SAMPLING_RANDOM_STATE,
        "selected_per_class": int(samples_per_class),
        "source_group_key": "zero-based HC3 CSV answer-row index",
        "train_source_answers": int(len(np.unique(groups[train_indices]))),
        "test_source_answers": int(len(np.unique(groups[test_indices]))),
        "source_answers_in_both_splits": int(len(
            set(groups[train_indices]) & set(groups[test_indices])
        )),
    }
    return features, labels, ordered_train_indices, ordered_test_indices, split_info


def _transform_saved_text_artifact(artifact, texts, custom_features):
    custom_indices = [
        train_text.CUSTOM_FEATURE_NAMES.index(name)
        for name in artifact["custom_feature_names"]
    ]
    custom = csr_matrix(artifact["custom_scaler"].transform(custom_features[:, custom_indices]))
    word = artifact["word_tfidf"].transform(texts)
    char = artifact["char_tfidf"].transform(texts)
    return hstack([custom, word, char]).tocsr()


def prepare_original_text_holdout(text_artifact):
    texts, labels, _, samples_per_class = sample_text_chunks()
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    all_custom_features, cached_labels = cached_text_features(
        "text_gpt2_original_sample",
        texts,
        labels,
    )
    if not np.array_equal(cached_labels, labels):
        raise ValueError("Original text cache labels do not match selected samples")

    all_production_features = _transform_saved_text_artifact(
        text_artifact,
        texts.tolist(),
        all_custom_features,
    )
    test_features = all_production_features[test_indices]
    test_labels = labels[test_indices]
    return test_features, test_labels, {
        **split_dataset_counts(labels[train_indices], test_labels),
        "method": "train_test_split(test_size=0.2, stratified)",
        "description": "original split (chunk-level, leakage-prone)",
        "random_state": RANDOM_STATE,
        "sampling_seed": train_text.SAMPLING_RANDOM_STATE,
        "selected_per_class": int(samples_per_class),
        "source_grouping": False,
    }


def cached_image_embeddings(name, paths, labels):
    image_paths = [Path(path) for path in paths]

    def extract(chunk_paths):
        return np.asarray([
            image_detector.extract_resnet18_embedding(path)
            for path in chunk_paths
        ])

    return cached(
        name,
        image_paths,
        labels,
        lambda: chunked(f"{name}_chunk", image_paths, labels, extract),
    )


def prepare_image_data():
    torch.set_num_threads(2)
    image_root = ROOT / "data" / "faces" / "real_vs_fake" / "real-vs-fake" / "train"
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    samples = []
    rng = np.random.default_rng(RANDOM_STATE)
    for label, class_name in ((0, "real"), (1, "fake")):
        class_dir = image_root / class_name
        candidates = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        )
        if len(candidates) < 5000:
            raise ValueError(f"Expected at least 5,000 {class_name} images in {class_dir}; found {len(candidates)}")
        selected = rng.choice(len(candidates), size=5000, replace=False)
        samples.extend((candidates[int(index)], label) for index in selected)

    paths = [str(path) for path, _ in samples]
    labels = np.asarray([label for _, label in samples], dtype=int)
    train_indices, test_indices = split_indices(labels)

    embeddings, labels = cached_image_embeddings("image_resnet18", paths, labels)
    train_indices, test_indices = split_indices(labels)
    return embeddings, labels, train_indices, test_indices


def prepare_deployed_image_holdout():
    test_root = ROOT / "data" / "faces" / "real_vs_fake" / "real-vs-fake" / "test"
    real_paths = sorted(str(path) for path in (test_root / "real").glob("*.jpg"))
    fake_paths = sorted(str(path) for path in (test_root / "fake").glob("*.jpg"))
    samples_per_class = min(500, len(real_paths), len(fake_paths))
    if samples_per_class == 0:
        raise ValueError(f"No usable real/fake image test files found under {test_root}")

    test_rng = np.random.default_rng(RANDOM_STATE + 1)
    selected_real = test_rng.choice(real_paths, size=samples_per_class, replace=False).tolist()
    selected_fake = test_rng.choice(fake_paths, size=samples_per_class, replace=False).tolist()
    paths = selected_real + selected_fake
    labels = np.asarray([0] * samples_per_class + [1] * samples_per_class, dtype=int)
    embeddings, labels = cached_image_embeddings("image_deployed_holdout_resnet18", paths, labels)
    return embeddings, labels, {
        "method": "separate image test folder; per-class sample seeded with 43",
        "train": "not recorded in image_model.joblib",
        "test": int(len(labels)),
        "train_class_balance": "not recorded in image_model.joblib",
        "test_class_balance": {
            str(label): int(np.count_nonzero(labels == label))
            for label in np.unique(labels)
        },
        "model_training_count": "not recorded in image_model.joblib",
    }


def prepare_audio_data():
    protocol = ROOT / "data" / "audio" / "mini_asv" / "mini_protocol.txt"
    audio_dir = ROOT / "data" / "audio" / "mini_asv" / "flac"
    rows = train_audio.parse_protocol(str(protocol), str(audio_dir))
    if not rows:
        raise ValueError(f"No usable audio rows found from {protocol}")
    paths = [Path(path) for path, _ in rows]
    labels = np.asarray([label for _, label in rows], dtype=int)

    def extract_audio_chunk(chunk_paths):
        return np.asarray([
            audio_detector.extract_features(str(path))
            for path in chunk_paths
        ])

    features, labels = cached(
        "audio",
        paths,
        labels,
        lambda: chunked("audio_chunk", paths, labels, extract_audio_chunk),
    )
    train_indices, test_indices = split_indices(labels)
    return features, labels.astype(int), train_indices, test_indices


def prepare_deployed_audio_holdout():
    manifest_path = ARTIFACTS / "audio_model.split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["test"]
    paths = [Path(entry["file"]) for entry in entries]
    labels = np.asarray([entry["label"] for entry in entries], dtype=int)

    def extract(chunk_paths):
        return np.asarray([
            audio_detector.extract_features(str(path))
            for path in chunk_paths
        ])

    features, labels = cached(
        "audio_deployed_holdout",
        paths,
        labels,
        lambda: chunked("audio_deployed_holdout_chunk", paths, labels, extract),
    )
    counts = {
        "method": "file list in audio_model.split.json",
        "train": int(len(manifest["train"])),
        "test": int(len(manifest["test"])),
        "train_class_balance": {
            str(label): sum(entry["label"] == label for entry in manifest["train"])
            for label in sorted({entry["label"] for entry in manifest["train"]})
        },
        "test_class_balance": {
            str(label): int(np.count_nonzero(labels == label))
            for label in np.unique(labels)
        },
        "random_state": manifest.get("random_state"),
        "test_size": manifest.get("test_size"),
    }
    return features, labels, counts


def print_summary(results, dataset_sizes):
    print("\nDataset sizes (class labels: text 0=human/1=AI, image 0=real/1=fake, audio 0=bonafide/1=spoof)")
    for modality, splits in dataset_sizes.items():
        for split_name, counts in splits.items():
            print(
                f"{modality:<8} {split_name:<24} "
                f"train={counts.get('train', 'not recorded')} "
                f"test={counts.get('test', 'not recorded')} "
                f"train_classes={counts.get('train_class_balance', 'not recorded')} "
                f"test_classes={counts.get('test_class_balance', 'not recorded')}"
            )

    print("\nEvaluation metrics (positive class = 1)")
    print(f"{'Modality':<8} {'Model':<30} {'Accuracy':>9} {'95% CI (n)':<25} {'Precision':>10} {'Recall':>9} {'F1':>9} {'AUC':>9}")
    for modality, model_results in results.items():
        for model_name, metrics in model_results.items():
            auc = f"{metrics['auc']:.4f}" if "auc" in metrics else "N/A"
            interval = metrics["accuracy_ci_95"]
            ci_text = f"[{interval['lower']:.4f}, {interval['upper']:.4f}] ({interval['n']})"
            print(
                f"{modality:<8} {model_name:<30} "
                f"{metrics['accuracy']:>9.4f} {ci_text:<25} {metrics['precision']:>10.4f} "
                f"{metrics['recall']:>9.4f} {metrics['f1']:>9.4f} {auc:>9}"
            )


def main():
    text_artifact = joblib.load(ARTIFACTS / "text_model.joblib")
    image_artifact = joblib.load(ARTIFACTS / "image_model.joblib")
    audio_artifact = joblib.load(ARTIFACTS / "audio_model.joblib")
    production_models = {
        "text": text_artifact["classifier"],
        "image": image_artifact["classifier"],
        "audio": audio_artifact,
    }
    print({
        modality: type(model).__name__
        for modality, model in production_models.items()
    }, flush=True)

    results = {}
    dataset_sizes = {}
    for modality, prepare in (
        ("text", prepare_text_data),
        ("image", prepare_image_data),
        ("audio", prepare_audio_data),
    ):
        print(f"\n[DATA] Preparing {modality} features", flush=True)
        prepared = prepare()
        if modality == "text":
            features, labels, train_indices, test_indices, split_info = prepared
        else:
            features, labels, train_indices, test_indices = prepared
            split_info = None
        dataset_sizes[modality] = {
            "benchmark_split": dataset_counts(labels, train_indices, test_indices)
        }
        if split_info is not None:
            dataset_sizes[modality]["benchmark_split"].update(split_info)
        results[modality] = evaluate_models(
            modality,
            features,
            labels,
            train_indices,
            test_indices,
            include_auc=True,
        )

    original_text_features, original_text_labels, original_text_counts = (
        prepare_original_text_holdout(text_artifact)
    )
    results["text"][
        "Deployed model (frozen) - original split (chunk-level, leakage-prone)"
    ] = evaluate_frozen_model(
        production_models["text"],
        original_text_features,
        original_text_labels,
    )
    dataset_sizes["text"]["deployed_holdout"] = original_text_counts
    dataset_sizes["text"]["grouped_frozen_score"] = {
        "status": "not reported",
        "reason": "Frozen model was trained on a chunk-level split; grouped-split metrics come from refit baselines.",
    }

    image_holdout_features, image_holdout_labels, image_holdout_counts = (
        prepare_deployed_image_holdout()
    )
    results["image"]["Deployed model (frozen)"] = evaluate_frozen_model(
        production_models["image"],
        image_holdout_features,
        image_holdout_labels,
    )
    dataset_sizes["image"]["deployed_holdout"] = image_holdout_counts

    audio_holdout_features, audio_holdout_labels, audio_holdout_counts = (
        prepare_deployed_audio_holdout()
    )
    results["audio"]["Deployed model (frozen)"] = evaluate_frozen_model(
        production_models["audio"],
        audio_holdout_features,
        audio_holdout_labels,
    )
    dataset_sizes["audio"]["deployed_holdout"] = audio_holdout_counts

    seeds = {
        "benchmark_split": RANDOM_STATE,
        "text_balanced_sampling": train_text.SAMPLING_RANDOM_STATE,
        "text_grouped_baseline_split": RANDOM_STATE,
        "text_original_split": RANDOM_STATE,
        "image_benchmark_subsample": RANDOM_STATE,
        "image_deployed_test_sample": RANDOM_STATE + 1,
        "audio_benchmark_split": RANDOM_STATE,
        "audio_deployed_split": audio_holdout_counts["random_state"],
    }
    summary = {
        **results,
        "dataset_sizes": dataset_sizes,
        "seeds": seeds,
    }
    output_dir = ROOT / "training_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results_summary.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")

    print_summary(results, dataset_sizes)
    print(f"\n[SAVED] {output_path}")


if __name__ == "__main__":
    main()