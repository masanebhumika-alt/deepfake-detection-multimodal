"""Tests for the FastAPI endpoints that need no model inference."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_root_lists_endpoints():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "/detect/multimodal" in body["endpoints"]


def test_blank_text_returns_400():
    r = client.post("/detect/text", json={"text": "   "})
    assert r.status_code == 400


def test_empty_text_returns_400():
    r = client.post("/detect/text", json={"text": ""})
    assert r.status_code == 400


def test_multimodal_with_no_input_returns_400():
    r = client.post("/detect/multimodal")
    assert r.status_code == 400


def test_image_endpoint_rejects_non_image():
    r = client.post(
        "/detect/image",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
