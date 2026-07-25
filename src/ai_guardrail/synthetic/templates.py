from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Template:
    family: str
    split: Literal["train", "validation", "challenge"]
    text: str


TEMPLATES = (
    Template(
        "record-transfer-train",
        "train",
        "Send {PERSON}'s {CUSTOMER_ID} record to {INTERNAL_PROJECT}.",
    ),
    Template(
        "contact-update-train",
        "train",
        "Update {PERSON} at {EMAIL} and note the address {ADDRESS}.",
    ),
    Template(
        "credential-copy-train",
        "train",
        "Copy {API_KEY} into the deployment note for {INTERNAL_PROJECT}.",
    ),
    Template(
        "ordinary-summary-negative-train",
        "train",
        "Summarize the quarterly report without changing its wording.",
    ),
    Template(
        "falcon-animal-negative-train",
        "train",
        "The peregrine falcon is one of the fastest birds.",
    ),
    Template(
        "meeting-summary-validation",
        "validation",
        "The owner is {PERSON}; follow up at {EMAIL} about {INTERNAL_PROJECT}.",
    ),
    Template(
        "account-review-validation",
        "validation",
        "Review customer {CUSTOMER_ID}, located at {ADDRESS}.",
    ),
    Template(
        "secure-config-validation",
        "validation",
        "The temporary credential for testing is {API_KEY}.",
    ),
    Template(
        "orion-constellation-negative-validation",
        "validation",
        "Orion is visible in the winter night sky.",
    ),
    Template(
        "codename-context-challenge",
        "challenge",
        "Use the codename {INTERNAL_PROJECT} in the briefing for {PERSON}.",
    ),
    Template(
        "mixed-contact-challenge",
        "challenge",
        "Mail {PERSON} at {ADDRESS}, or use {EMAIL}.",
    ),
    Template(
        "lantern-object-negative-challenge",
        "challenge",
        "The lantern on the porch needs a new battery.",
    ),
)
