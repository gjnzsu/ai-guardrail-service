from ai_guardrail.domain import EntityType

LABELS = ["O"]
for entity_type in EntityType:
    LABELS.extend((f"B-{entity_type.value}", f"I-{entity_type.value}"))

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

assert len(LABELS) == 13
