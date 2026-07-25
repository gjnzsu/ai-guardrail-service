import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ai_guardrail.detectors.ner import NerDetector
from ai_guardrail.domain import EntityType
from ai_guardrail.ner.labels import LABEL_TO_ID
from ai_guardrail.ner.manifest import (
    ARTIFACT_NAME,
    BASE_CHECKPOINT,
    GENERATOR_VERSION,
    sha256_file,
)


class FakeBatch(dict):
    def to(self, device: str) -> "FakeBatch":
        return self


class FakeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> FakeBatch:
        return FakeBatch(
            input_ids=torch.tensor([[101, 1, 2, 102]]),
            attention_mask=torch.tensor([[1, 1, 1, 1]]),
            offset_mapping=torch.tensor([[[0, 0], [0, 4], [5, 11], [0, 0]]]),
        )


class FakeOutput:
    def __init__(self) -> None:
        logits = torch.zeros((1, 4, 13))
        logits[0, 1, LABEL_TO_ID["B-PERSON"]] = 8
        logits[0, 2, LABEL_TO_ID["I-PERSON"]] = 8
        self.logits = logits


class FakeModel:
    def eval(self) -> "FakeModel":
        return self

    def __call__(self, **kwargs: object) -> FakeOutput:
        return FakeOutput()


@pytest.mark.asyncio
async def test_ner_detector_decodes_model_logits() -> None:
    detector = NerDetector(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        model_version="ai-guardrail-ner-en-v1",
        threshold=0.5,
    )

    output = await detector.detect("Jane Cooper")

    assert output.status == "success"
    assert len(output.candidates) == 1
    assert output.candidates[0].type == EntityType.PERSON
    assert output.candidates[0].start == 0
    assert output.candidates[0].end == 11


@pytest.mark.parametrize("threshold", [-0.01, 1.01, math.nan])
def test_ner_detector_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        NerDetector(
            tokenizer=FakeTokenizer(),
            model=FakeModel(),
            model_version="ai-guardrail-ner-en-v1",
            threshold=threshold,
        )


@pytest.mark.asyncio
async def test_ner_detector_filters_candidates_below_threshold() -> None:
    detector = NerDetector(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        model_version="ai-guardrail-ner-en-v1",
        threshold=1.0,
    )

    output = await detector.detect("Jane Cooper")

    assert output.status == "success"
    assert output.candidates == []


class RaisingTokenizer:
    def __call__(self, text: str, **kwargs: object) -> FakeBatch:
        raise RuntimeError(f"inference failed for {text}")


@pytest.mark.asyncio
async def test_ner_detector_returns_privacy_safe_error() -> None:
    private_text = "secret-user@example.test"
    detector = NerDetector(
        tokenizer=RaisingTokenizer(),
        model=FakeModel(),
        model_version="ai-guardrail-ner-en-v1",
        threshold=0.5,
    )

    output = await detector.detect(private_text)

    assert output.status == "error"
    assert output.candidates == []
    assert output.error_code == "ner_inference_failed"
    assert private_text not in output.model_dump_json()


class WrongBatchOutput:
    def __init__(self) -> None:
        self.logits = torch.zeros((2, 4, 13))


class WrongBatchModel(FakeModel):
    def __call__(self, **kwargs: object) -> WrongBatchOutput:
        return WrongBatchOutput()


@pytest.mark.asyncio
async def test_ner_detector_rejects_unexpected_logits_shape() -> None:
    detector = NerDetector(
        tokenizer=FakeTokenizer(),
        model=WrongBatchModel(),
        model_version="ai-guardrail-ner-en-v1",
        threshold=0.5,
    )

    output = await detector.detect("Jane Cooper")

    assert output.status == "error"
    assert output.error_code == "ner_inference_failed"
    assert output.candidates == []


class WrongOffsetRankTokenizer:
    def __call__(self, text: str, **kwargs: object) -> FakeBatch:
        return FakeBatch(
            input_ids=torch.tensor([[101, 1, 2, 102]]),
            attention_mask=torch.tensor([[1, 1, 1, 1]]),
            offset_mapping=torch.tensor([[0, 0, 4, 11]]),
        )


@pytest.mark.asyncio
async def test_ner_detector_rejects_unexpected_offset_rank() -> None:
    detector = NerDetector(
        tokenizer=WrongOffsetRankTokenizer(),
        model=FakeModel(),
        model_version=ARTIFACT_NAME,
        threshold=0.5,
    )

    output = await detector.detect("Jane Cooper")

    assert output.status == "error"
    assert output.error_code == "ner_inference_failed"


class WrongSequenceOutput:
    def __init__(self) -> None:
        self.logits = torch.zeros((1, 3, len(LABEL_TO_ID)))


class WrongSequenceModel(FakeModel):
    def __call__(self, **kwargs: object) -> WrongSequenceOutput:
        return WrongSequenceOutput()


@pytest.mark.asyncio
async def test_ner_detector_rejects_mismatched_sequence_dimension() -> None:
    detector = NerDetector(
        tokenizer=FakeTokenizer(),
        model=WrongSequenceModel(),
        model_version=ARTIFACT_NAME,
        threshold=0.5,
    )

    output = await detector.detect("Jane Cooper")

    assert output.status == "error"
    assert output.error_code == "ner_inference_failed"


class WrongLabelDimensionOutput:
    def __init__(self) -> None:
        self.logits = torch.zeros((1, 4, len(LABEL_TO_ID) - 1))


class WrongLabelDimensionModel(FakeModel):
    def __call__(self, **kwargs: object) -> WrongLabelDimensionOutput:
        return WrongLabelDimensionOutput()


@pytest.mark.asyncio
async def test_ner_detector_rejects_mismatched_label_dimension() -> None:
    detector = NerDetector(
        tokenizer=FakeTokenizer(),
        model=WrongLabelDimensionModel(),
        model_version=ARTIFACT_NAME,
        threshold=0.5,
    )

    output = await detector.detect("Jane Cooper")

    assert output.status == "error"
    assert output.error_code == "ner_inference_failed"


class UnicodeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> FakeBatch:
        return FakeBatch(
            input_ids=torch.tensor([[101, 1, 2, 102]]),
            attention_mask=torch.tensor([[1, 1, 1, 1]]),
            offset_mapping=torch.tensor([[[0, 0], [5, 9], [10, 16], [0, 0]]]),
        )


@pytest.mark.asyncio
async def test_ner_detector_preserves_unicode_code_point_offsets() -> None:
    text = "前缀 😀 Jane Cooper"
    detector = NerDetector(
        tokenizer=UnicodeTokenizer(),
        model=FakeModel(),
        model_version=ARTIFACT_NAME,
        threshold=0.5,
    )

    output = await detector.detect(text)

    assert output.status == "success"
    assert len(output.candidates) == 1
    candidate = output.candidates[0]
    assert (candidate.start, candidate.end) == (5, 16)
    assert text[candidate.start : candidate.end] == "Jane Cooper"


class RuntimeTrackingModel(FakeModel):
    def __init__(self) -> None:
        self.eval_called = False
        self.inference_mode_seen = False

    def eval(self) -> "RuntimeTrackingModel":
        self.eval_called = True
        return self

    def __call__(self, **kwargs: object) -> FakeOutput:
        self.inference_mode_seen = torch.is_inference_mode_enabled()
        return FakeOutput()


@pytest.mark.asyncio
async def test_ner_detector_evaluates_model_in_inference_mode() -> None:
    model = RuntimeTrackingModel()
    detector = NerDetector(
        tokenizer=FakeTokenizer(),
        model=model,
        model_version=ARTIFACT_NAME,
        threshold=0.5,
    )

    output = await detector.detect("Jane Cooper")

    assert output.status == "success"
    assert model.eval_called
    assert model.inference_mode_seen


class LoadableFakeModel(FakeModel):
    def __init__(self) -> None:
        self.devices: list[str] = []
        self.config = SimpleNamespace(
            label2id=LABEL_TO_ID,
            id2label={value: key for key, value in LABEL_TO_ID.items()},
        )

    def to(self, device: str) -> "LoadableFakeModel":
        self.devices.append(device)
        return self


def write_valid_manifest(model_path: Path, **overrides: object) -> None:
    config = {
        "label2id": LABEL_TO_ID,
        "id2label": {str(value): key for key, value in LABEL_TO_ID.items()},
    }
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (model_path / "model.safetensors").write_bytes(b"weights")
    (model_path / "tokenizer.json").write_bytes(b"tokenizer")
    manifest: dict[str, object] = {
        "manifest_schema_version": 2,
        "artifact_name": ARTIFACT_NAME,
        "base_checkpoint": BASE_CHECKPOINT,
        "base_revision": "immutable-revision",
        "dataset_version": "v1",
        "generator_version": GENERATOR_VERSION,
        "label_mapping": LABEL_TO_ID,
        "seed": 7,
        "threshold": None,
        "metrics": {"eval_loss": 0.25},
        "hyperparameters": {"epochs": 3},
        "python_version": "3.12.0",
        "library_versions": {
            "accelerate": "1.3.0",
            "torch": "2.5.0",
            "transformers": "4.49.0",
        },
        "artifact_checksums": {
            name: sha256_file(model_path / name)
            for name in ("config.json", "model.safetensors", "tokenizer.json")
        },
        "datasets": {
            "train": {
                "sha256": "1" * 64,
                "record_count": 1,
                "record_id_hashes": ["2" * 64],
                "template_families": ["train-family"],
                "content_hashes": ["3" * 64],
            },
            "validation": {
                "sha256": "4" * 64,
                "record_count": 1,
                "record_id_hashes": ["5" * 64],
                "template_families": ["validation-family"],
                "content_hashes": ["6" * 64],
            },
        },
    }
    manifest.update(overrides)
    (model_path / "training-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_ner_detector_loads_local_artifact_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / ARTIFACT_NAME
    model_path.mkdir()
    write_valid_manifest(model_path)
    tokenizer = FakeTokenizer()
    model = LoadableFakeModel()
    tokenizer_calls: list[tuple[Path, dict[str, object]]] = []
    model_calls: list[tuple[Path, dict[str, object]]] = []

    def load_tokenizer(path: Path, **kwargs: object) -> FakeTokenizer:
        tokenizer_calls.append((path, kwargs))
        return tokenizer

    def load_model(path: Path, **kwargs: object) -> LoadableFakeModel:
        model_calls.append((path, kwargs))
        return model

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(AutoModelForTokenClassification, "from_pretrained", load_model)

    detector = NerDetector.load(model_path, threshold=0.5)

    assert tokenizer_calls == [
        (model_path, {"use_fast": True, "local_files_only": True})
    ]
    assert model_calls == [(model_path, {"local_files_only": True})]
    assert model.devices == ["cpu"]
    assert detector.model_version == ARTIFACT_NAME


def test_ner_detector_rejects_renamed_artifact_before_ml_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "private-renamed-artifact"
    model_path.mkdir()
    write_valid_manifest(model_path)
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("tokenizer must not load"),
    )
    monkeypatch.setattr(
        AutoModelForTokenClassification,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("model must not load"),
    )

    with pytest.raises(ValueError, match="invalid NER model artifact") as exc_info:
        NerDetector.load(model_path, threshold=0.5)

    assert "private-renamed-artifact" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "private_value"),
    [
        ("artifact_name", "private-artifact-v2"),
        ("base_checkpoint", "private/checkpoint"),
        ("generator_version", "private-generator-v2"),
        ("label_mapping", {"O": 99}),
        ("label_mapping", {**LABEL_TO_ID, "B-PERSON": True}),
    ],
)
def test_ner_detector_rejects_mismatched_manifest_before_ml_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    private_value: object,
) -> None:
    model_path = tmp_path / ARTIFACT_NAME
    model_path.mkdir()
    write_valid_manifest(model_path, **{field: private_value})
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("tokenizer must not load"),
    )
    monkeypatch.setattr(
        AutoModelForTokenClassification,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("model must not load"),
    )

    with pytest.raises(ValueError, match="invalid NER model artifact") as exc_info:
        NerDetector.load(model_path, threshold=0.5)

    assert str(private_value) not in str(exc_info.value)


def test_ner_detector_rejects_loaded_model_label_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / ARTIFACT_NAME
    model_path.mkdir()
    write_valid_manifest(model_path)
    model = LoadableFakeModel()
    model.config.label2id = {**LABEL_TO_ID, "O": 99}
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        AutoModelForTokenClassification,
        "from_pretrained",
        lambda *args, **kwargs: model,
    )

    with pytest.raises(ValueError, match="invalid NER model artifact") as exc_info:
        NerDetector.load(model_path, threshold=0.5)

    assert "99" not in str(exc_info.value)
