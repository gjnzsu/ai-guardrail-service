import math
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ai_guardrail.detectors.ner import NerDetector
from ai_guardrail.domain import EntityType
from ai_guardrail.ner.labels import LABEL_TO_ID


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


class LoadableFakeModel(FakeModel):
    def __init__(self) -> None:
        self.devices: list[str] = []

    def to(self, device: str) -> "LoadableFakeModel":
        self.devices.append(device)
        return self


def test_ner_detector_loads_local_artifact_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "ai-guardrail-ner-en-v1"
    model_path.mkdir()
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
    assert detector.model_version == "ai-guardrail-ner-en-v1"
