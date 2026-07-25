from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EntityType(StrEnum):
    PERSON = "PERSON"
    ADDRESS = "ADDRESS"
    EMAIL = "EMAIL"
    API_KEY = "API_KEY"
    CUSTOMER_ID = "CUSTOMER_ID"
    INTERNAL_PROJECT = "INTERNAL_PROJECT"


class DetectionSource(StrEnum):
    REGEX = "regex"
    NER = "ner"
    QWEN = "qwen"


DatasetSplit = Literal["train", "validation", "challenge"]


class EntitySpan(BaseModel):
    type: EntityType
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> EntitySpan:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class LabeledExample(BaseModel):
    id: str = Field(min_length=1)
    language: Literal["en"]
    text: str = Field(min_length=1)
    entities: list[EntitySpan]
    template_family: str = Field(min_length=1)
    generator_version: Literal["v1"]
    split: DatasetSplit

    @model_validator(mode="after")
    def validate_entity_bounds(self) -> LabeledExample:
        ordered = sorted(self.entities, key=lambda entity: (entity.start, entity.end))
        previous_end = 0
        for entity in ordered:
            if entity.end > len(self.text):
                raise ValueError("entity span exceeds text length")
            if entity.start < previous_end:
                raise ValueError("gold entity spans must not overlap")
            previous_end = entity.end
        return self


class CandidateDetection(EntitySpan):
    message_index: int = Field(ge=0)
    source: DetectionSource
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class DetectorOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    detector: str
    model_version: str
    status: Literal["success", "error", "timeout"]
    candidates: list[CandidateDetection] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)
    invalid_candidate_count: int = Field(default=0, ge=0)
    error_code: str | None = None
