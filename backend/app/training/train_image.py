"""
Trains a real classifier for the image detector on labeled data.

Expected folder layout (matches the 140k Real and Fake Faces Kaggle
dataset after --unzip):
    <root>/real/*.jpg
    <root>/fake/*.jpg

Run:
    python -m app.training.train_image /path/to/real_vs_fake/train
"""
import sys
import os
import glob
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from app.models import image_detector

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "artifacts", "image_model.joblib")
RANDOM_STATE = 42
BATCH_SIZE = 2


class ImagePathDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[index]


def extract_embeddings(paths, labels, model, transform, stage):
    loader = DataLoader(
        ImagePathDataset(paths, labels, transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    embeddings = []
    extracted_labels = []
    with torch.inference_mode():
        for batch_index, (images, batch_labels) in enumerate(loader, 1):
            embeddings.append(model(images).cpu().numpy())
            extracted_labels.extend(batch_labels.numpy().tolist())
            processed = min(batch_index * BATCH_SIZE, len(paths))
            if processed % 1000 == 0 or processed == len(paths):
                print(f"[INFO] {stage} embeddings: {processed}/{len(paths)}", flush=True)
    return np.concatenate(embeddings), np.asarray(extracted_labels)


def main(root: str, max_per_class: int = 15000):
    torch.set_num_threads(2)
    real_candidates = sorted(glob.glob(os.path.join(root, "real", "*.jpg")))
    fake_candidates = sorted(glob.glob(os.path.join(root, "fake", "*.jpg")))
    assert real_candidates and fake_candidates, f"Expected real/ and fake/ subfolders under {root}"

    samples_per_class = min(max_per_class, len(real_candidates), len(fake_candidates))
    rng = np.random.default_rng(RANDOM_STATE)
    real_files = rng.choice(real_candidates, size=samples_per_class, replace=False).tolist()
    fake_files = rng.choice(fake_candidates, size=samples_per_class, replace=False).tolist()
    print(f"[INFO] sampling {samples_per_class} real and {samples_per_class} fake images", flush=True)

    model, transform = image_detector.load_resnet18_feature_extractor()
    train_paths = real_files + fake_files
    train_labels = [0] * len(real_files) + [1] * len(fake_files)
    X, y = extract_embeddings(train_paths, train_labels, model, transform, "train")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    print("[INFO] balanced training-sample validation:")
    print(classification_report(y_test, clf.predict(X_test), target_names=["Real", "Fake"]))

    clf.fit(X, y)
    test_root = os.path.join(os.path.dirname(os.path.abspath(root)), "test")
    real_test = sorted(glob.glob(os.path.join(test_root, "real", "*.jpg")))
    fake_test = sorted(glob.glob(os.path.join(test_root, "fake", "*.jpg")))
    if real_test and fake_test:
        test_per_class = min(500, len(real_test), len(fake_test))
        test_rng = np.random.default_rng(RANDOM_STATE + 1)
        real_test = test_rng.choice(real_test, size=test_per_class, replace=False).tolist()
        fake_test = test_rng.choice(fake_test, size=test_per_class, replace=False).tolist()
        test_paths = real_test + fake_test
        test_labels = [0] * len(real_test) + [1] * len(fake_test)
        X_external, y_external = extract_embeddings(
            test_paths, test_labels, model, transform, "held-out test"
        )
        print("[INFO] separate test-folder evaluation:")
        print(classification_report(
            y_external, clf.predict(X_external), target_names=["Real", "Fake"]
        ))

    os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
    joblib.dump({
        "classifier": clf,
        "feature_extractor": "resnet18_imagenet1k_v1",
        "embedding_dimensions": 512,
    }, ARTIFACT_PATH)
    print(f"Saved trained model to {ARTIFACT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m app.training.train_image /path/to/real_vs_fake/train")
        sys.exit(1)
    main(sys.argv[1])
