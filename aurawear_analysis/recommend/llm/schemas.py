# aurawear_analysis/recommend/llm/schemas.py

INTENT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "query_text": {"type": "string"},
        "must_have": {
            "type": "array",
            "items": {"type": "string"}
        },
        "avoid": {
            "type": "array",
            "items": {"type": "string"}
        },
        "style_tags": {
            "type": "array",
            "items": {"type": "string"}
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0
        },
    },
    "required": [
        "query_text",
        "must_have",
        "avoid",
        "style_tags",
        "confidence",
    ],
}


DISLIKE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "avoid": {
            "type": "array",
            "items": {"type": "string"}
        },
    },
    "required": [
        "avoid",
    ],
}


EXPLANATION_BATCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "explanations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "item_id": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["item_id", "explanation"],
            },
        },
    },
    "required": ["explanations"],
}

