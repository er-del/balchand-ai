"""Tests for the PIXEL web API."""

from fastapi.testclient import TestClient

from configs.base import ModelConfig
from core.checkpoint import CheckpointInspection
import web.app as web_app


def test_health_and_modes_endpoints() -> None:
    client = TestClient(web_app.app)
    health = client.get("/api/health")
    assert health.status_code == 200
    modes = client.get("/api/ui-modes")
    assert modes.status_code == 200
    assert "chat" in modes.json()["modes"]


def test_generate_endpoint_with_fake_generator(monkeypatch) -> None:
    class FakeGenerator:
        checkpoint_loaded = True
        config_source = "checkpoint"

        def generate(self, request):
            from core.types import GenerationResponse

            return GenerationResponse(output=f"echo:{request.prompt}", tokens_generated=1, model_name="fake", used_checkpoint=True)

        def stream(self, request):
            yield "echo:"
            yield request.prompt

        def describe(self):
            return {
                "requested_model": {"name": "pixel_1b"},
                "resolved_model": {"name": "pixel_100m", "hidden_size": 768, "vocab_size": 1262},
                "config_source": "checkpoint",
                "checkpoint_loaded": True,
                "checkpoint_path": "checkpoints/pixel_100m/latest.pt",
                "checkpoint_step": 10,
                "hardware": {"device": "cpu"},
            }

    monkeypatch.setattr(web_app, "_build_generator", lambda size, model_path: FakeGenerator())
    client = TestClient(web_app.app)
    response = client.post("/api/generate", json={"prompt": "hi", "size": "100m"})
    assert response.status_code == 200
    assert response.json()["output"] == "echo:hi"
    assert response.json()["config_source"] == "checkpoint"
    assert response.json()["resolved_model"]["name"] == "pixel_100m"


def test_models_endpoint_exposes_latest_checkpoint_metadata(monkeypatch) -> None:
    inspection = CheckpointInspection(
        path="checkpoints/pixel_100m/latest.pt",
        step=10,
        model_config=ModelConfig(
            name="pixel_100m",
            vocab_size=1262,
            context_length=1024,
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            num_key_value_heads=4,
            intermediate_size=2048,
        ),
        training_config=None,
        metadata={"model": {"name": "pixel_100m"}},
    )
    monkeypatch.setattr(web_app, "_latest_checkpoint", lambda: "checkpoints/pixel_100m/latest.pt")
    monkeypatch.setattr(web_app, "_inspect_checkpoint", lambda checkpoint: inspection)
    client = TestClient(web_app.app)
    response = client.get("/api/models")
    assert response.status_code == 200
    assert response.json()["latest"] == "checkpoints/pixel_100m/latest.pt"
    assert response.json()["latest_details"]["model"]["name"] == "pixel_100m"


def test_generate_endpoint_invalid_preset_returns_400() -> None:
    client = TestClient(web_app.app)
    response = client.post("/api/generate", json={"prompt": "hi", "size": "invalid"})
    assert response.status_code == 400
    assert "Unknown PIXEL preset" in response.json()["detail"]
