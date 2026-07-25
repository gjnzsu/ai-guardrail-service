from ai_guardrail.domain import EntityType

CATALOG: dict[str, dict[EntityType, tuple[str, ...]]] = {
    "train": {
        EntityType.PERSON: ("Jane Cooper", "Marcus Reed", "jane cooper"),
        EntityType.ADDRESS: (
            "18 Willow Lane, Northbridge",
            "440 Cedar Avenue, Lakeview",
        ),
        EntityType.EMAIL: (
            "jane.cooper@example.test",
            "marcus.reed@example.test",
        ),
        EntityType.API_KEY: (
            "sk-test-A1B2C3D4E5F6G7H8",
            "sk test A1B2 C3D4 E5F6 G7H8",
        ),
        EntityType.CUSTOMER_ID: ("CUST-93821", "CUST-10482"),
        EntityType.INTERNAL_PROJECT: (
            "Project Orion",
            "project falcon",
        ),
    },
    "validation": {
        EntityType.PERSON: ("Elena Brooks", "Noah Bennett"),
        EntityType.ADDRESS: (
            "72 Juniper Road, Westhaven",
            "9 Harbor Street, Eastford",
        ),
        EntityType.EMAIL: (
            "elena.brooks@example.test",
            "noah.bennett@example.test",
        ),
        EntityType.API_KEY: (
            "api_test_Z9Y8X7W6V5U4T3S2",
            "pk-test-1122334455667788",
        ),
        EntityType.CUSTOMER_ID: ("CUST-77190", "CUST-55014"),
        EntityType.INTERNAL_PROJECT: (
            "Project Lantern",
            "Project Northstar",
        ),
    },
    "challenge": {
        EntityType.PERSON: ("Priya Wallace", "Owen Parker"),
        EntityType.ADDRESS: (
            "31 Maple Crescent, Stonehaven",
            "6 Orchard Way, Clearford",
        ),
        EntityType.EMAIL: (
            "priya.wallace@example.test",
            "owen.parker@example.test",
        ),
        EntityType.API_KEY: (
            "sk-test-Q1W2E3R4T5Y6U7I8",
            "api_test_M9N8B7V6C5X4Z3L2",
        ),
        EntityType.CUSTOMER_ID: ("CUST-66241", "CUST-88370"),
        EntityType.INTERNAL_PROJECT: (
            "Project Meridian",
            "Project Ember",
        ),
    },
}
