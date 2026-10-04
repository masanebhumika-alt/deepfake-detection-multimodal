# Multimodal AI-Generated Content Detection Backend

FastAPI service for text, image, audio, and multimodal prediction. Each
detector returns an AI-generated confidence from 0 to 100 and a label.

## Measured Status

These are local evaluation results, not guarantees on other datasets or
generators.

| Modality | Current model | Evaluation result | Status |
|---|---|---|---|
| Text | Hybrid TF-IDF/stylometric logistic classifier with scaled custom features | 95% on HC3 chunk-level validation; fresh endpoint probes classified only 1 of 3 AI-style examples as AI | Runs, but generalization to novel phrasing is limited; chunks from 59 source answers crossed the train/test split |
| Image | Frozen ImageNet ResNet18 embeddings plus logistic regression | 79% on a separate test-folder sample of 500 real and 500 fake images | Improved over the handcrafted baseline; broader test coverage is needed |
| Audio | Random Forest over three spectral/MFCC features | 75.25% on the 400-item mini-ASV holdout (200 bonafide, 200 spoof) | Moderate result; holdout is from the same 2,000-clip mini dataset, not an external domain |

The image model uses 512-dimensional ResNet18 embeddings. Its pretrained
weights are loaded from the Torch cache (downloaded on first use if absent).
The audio model uses a random split of `mini_asv`; the earlier 100% small
sample result was contaminated by training overlap. The verified holdout
result is 75.25%.

## Run the API

From the `backend` directory:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

Swagger UI is available at `http://127.0.0.1:8000/docs`.

```powershell
curl.exe -X POST http://127.0.0.1:8000/detect/text `
  -H "Content-Type: application/json" `
  -d '{"text":"Example text to analyze."}'

curl.exe -X POST http://127.0.0.1:8000/detect/image -F "file=@photo.jpg"
curl.exe -X POST http://127.0.0.1:8000/detect/audio -F "file=@clip.flac"
curl.exe -X POST "http://127.0.0.1:8000/detect/multimodal?text=Example" `
  -F "image=@photo.jpg" -F "audio=@clip.flac"
```

## Training and Provenance

Training scripts are in `app/training/`. Image training uses up to 15,000
images per class from the supplied `train/` folders; evaluation samples from
the separate `test/` folders. Audio training writes exact successful
train/test file lists, labels, split seed, and protocol hash to
`app/models/artifacts/audio_model.split.json` alongside the estimator.
The current audio accuracy was verified by reproducing the existing seed-42
split; no retraining was performed for that check.

## Limitations

- Text detection does not reliably generalize from HC3-style data to novel
  writing styles. The chunk-level random split allows some source-answer
  overlap; use source-grouped and external test sets before relying on the
  reported validation score.
- Image results are from a sampled test-folder evaluation and may vary on
  other image sources or generators.
- Audio results use a mini ASVspoof subset and do not establish
  cross-corpus performance.
- The service processes single text inputs and uploaded files; it is not
  adversarially robust.