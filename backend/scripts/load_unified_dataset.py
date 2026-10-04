# ================================================================
# Unified Multimodal AI-Generated Content Detection — Dataset Loader
# ================================================================
# Run this in Google Colab or a Kaggle Notebook (both have open internet
# access). It will NOT run inside a restricted/offline sandbox.
#
# Pulls one REAL public dataset per modality:
#   TEXT  -> HC3 (Human ChatGPT Comparison Corpus)      [Hugging Face]
#   IMAGE -> 140k Real and Fake Faces                   [Kaggle]
#   AUDIO -> ASVspoof 2019 (Logical Access)              [Kaggle]
#
# Kaggle datasets need a free Kaggle account + API token:
#   1. Go to kaggle.com -> Account -> Create New API Token
#   2. This downloads a "kaggle.json" file
#   3. Upload it when prompted below (Colab) or place it at
#      ~/.kaggle/kaggle.json (local/other notebook)
#
# After downloading, flatten each dataset into the format expected by
# app/training/train_text.py, train_image.py, and train_audio.py
# (see the docstring at the top of each of those scripts).
# ================================================================

# ---------------------------------------------------------------
# 1) TEXT — HC3 (48,644 rows: human vs ChatGPT Q&A pairs, ~147MB)
# ---------------------------------------------------------------
!pip install -q datasets
from datasets import load_dataset

hc3 = load_dataset("Hello-SimpleAI/HC3", "all")
print("TEXT (HC3) rows:", len(hc3["train"]))
print("Sample record keys:", hc3["train"][0].keys())
print("Example question:", hc3["train"][0]["question"][:120], "...")

# Flatten to text,label CSV for train_text.py
import pandas as pd
rows = []
for r in hc3["train"]:
    for h in r["human_answers"]:
        rows.append({"text": h, "label": 0})
    for c in r["chatgpt_answers"]:
        rows.append({"text": c, "label": 1})
pd.DataFrame(rows).to_csv("hc3_flat.csv", index=False)
print(f"Wrote hc3_flat.csv with {len(rows)} rows")

# ---------------------------------------------------------------
# 2) IMAGE — 140k Real and Fake Faces (70k real + 70k GAN-fake, ~4GB)
# ---------------------------------------------------------------
!pip install -q kaggle

try:
    from google.colab import files
    print("Upload your kaggle.json now:")
    uploaded = files.upload()
    import os
    os.makedirs("/root/.kaggle", exist_ok=True)
    for fn in uploaded:
        os.rename(fn, "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
except ImportError:
    pass  # not running in Colab; assumes kaggle.json already in place

import os
os.makedirs("./data/faces", exist_ok=True)
!kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p ./data/faces --unzip

real_dir = "./data/faces/real_vs_fake/real-vs-fake/train/real"
fake_dir = "./data/faces/real_vs_fake/real-vs-fake/train/fake"
n_real = len(os.listdir(real_dir)) if os.path.exists(real_dir) else 0
n_fake = len(os.listdir(fake_dir)) if os.path.exists(fake_dir) else 0
print(f"IMAGE (140k Faces) — real: {n_real}, fake: {n_fake}")
print("Feed this folder directly to: python -m app.training.train_image "
      "./data/faces/real_vs_fake/real-vs-fake/train")

# ---------------------------------------------------------------
# 3) AUDIO — ASVspoof 2019 (Logical Access), ~7GB
# ---------------------------------------------------------------
os.makedirs("./data/audio", exist_ok=True)
!kaggle datasets download -d awsaf49/asvpoof-2019-dataset -p ./data/audio --unzip

protocol_path = "./data/audio/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
flac_dir = "./data/audio/LA/ASVspoof2019_LA_train/flac"
if os.path.exists(protocol_path):
    protocol = pd.read_csv(protocol_path, sep=" ", header=None,
                            names=["speaker", "filename", "codec", "source", "label"])
    print("AUDIO (ASVspoof2019 LA) label counts:")
    print(protocol["label"].value_counts())
    print(f"Feed this to: python -m app.training.train_audio {protocol_path} {flac_dir}")
else:
    print("Audio protocol file not found at expected path — check extracted folder structure.")

# ---------------------------------------------------------------
# COMBINED SUMMARY
# ---------------------------------------------------------------
print("\n=== UNIFIED MULTIMODAL DATASET SUMMARY ===")
print(f"Text  (HC3):              {len(rows):>7} labeled text samples -> hc3_flat.csv")
print(f"Image (140k Faces):       {n_real + n_fake:>7} images ({n_real} real / {n_fake} fake)")
if os.path.exists(protocol_path):
    print(f"Audio (ASVspoof2019 LA):  {len(protocol):>7} utterances")
print("=" * 50)
print("Next step: run the three train_*.py scripts in app/training/ using the")
print("paths printed above, then restart the API — it will auto-load the")
print("trained models from app/models/artifacts/.")
