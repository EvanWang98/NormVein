from pathlib import Path

import pytest

from normvein.checkpoints import load_artifact, load_manifest
from normvein.models import MODEL_SPECS, build_model


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = load_manifest(ROOT / "weights" / "manifest.json")


def test_manifest_covers_registry():
    manifest_ids = {record["model_id"] for record in MANIFEST["models"]}
    assert manifest_ids == set(MODEL_SPECS)


@pytest.mark.parametrize("record", MANIFEST["models"], ids=lambda item: item["model_id"])
def test_artifact_schema(record):
    path = ROOT / "weights" / record["relative_path"]
    if not path.is_file():
        pytest.skip("release assets are not checked into Git")
    artifact = load_artifact(path)
    assert artifact["model_id"] == record["model_id"]
    assert artifact["metadata"]["source_epoch"] == record["source_epoch"]


@pytest.mark.parametrize("record", MANIFEST["models"], ids=lambda item: item["model_id"])
def test_strict_model_load(record):
    if record["model_id"] == "rec_convnext_small":
        pytest.importorskip("timm")
    path = ROOT / "weights" / record["relative_path"]
    if not path.is_file():
        pytest.skip("release assets are not checked into Git")
    artifact = load_artifact(path)
    model = build_model(record["model_id"])
    model.load_state_dict(artifact["state_dict"], strict=True)

