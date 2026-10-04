"""
Multimodal AI-Generated Content Detection — Backend API

Implements the architecture proposed in Experiment 4:
  Input -> Preprocessing -> Feature Extraction -> Per-Modality Classifier
        -> Confidence-Calibrated Fusion -> Output Report

Run locally:
    uvicorn app.main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
import shutil
import tempfile
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.models import text_detector, image_detector, audio_detector
from app import fusion

app = FastAPI(
    title="Multimodal AI-Generated Content Detection API",
    description="Detects likely AI-generated text, images, and audio, "
                "with a confidence-calibrated multimodal fusion endpoint.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Multimodal AI-Generated Content Detection API",
        "endpoints": ["/detect/text", "/detect/image", "/detect/audio", "/detect/multimodal", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/detect/text")
def detect_text(payload: TextRequest):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text field must not be empty.")
    return text_detector.predict(payload.text)


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = image_detector.predict(tmp_path)
    finally:
        os.remove(tmp_path)
    return result


@app.post("/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        result = audio_detector.predict(tmp_path)
    finally:
        os.remove(tmp_path)
    return result


@app.post("/detect/multimodal")
async def detect_multimodal(
    text: str = None,
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
):
    results = []

    if text and text.strip():
        results.append(text_detector.predict(text))

    if image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as tmp:
            shutil.copyfileobj(image.file, tmp)
            tmp_path = tmp.name
        try:
            results.append(image_detector.predict(tmp_path))
        finally:
            os.remove(tmp_path)

    if audio is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio.filename)[1] or ".wav") as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name
        try:
            results.append(audio_detector.predict(tmp_path))
        finally:
            os.remove(tmp_path)

    if not results:
        raise HTTPException(status_code=400, detail="Provide at least one of: text, image, audio.")

    return fusion.fuse(results)
