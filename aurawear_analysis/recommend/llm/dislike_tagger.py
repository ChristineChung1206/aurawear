# aurawear_analysis/recommend/llm/dislike_tagger.py

from typing import Dict, Any
from .client import LLMClient
from .prompts import DISLIKE_INSTRUCTIONS
from .schemas import DISLIKE_SCHEMA


def tag_dislike(
    llm: LLMClient,
    item_id: str,
    item_meta: Dict[str, Any],
    *,
    critique_tags: list | None = None,
    reasons: list | None = None,
    free_text: str = "",
):
    """Extract avoid terms grounded in the user's actual feedback."""
    parts = [f"Item ID: {item_id}", f"Category: {item_meta.get('category', 'unknown')}"]
    if critique_tags:
        parts.append(f"User's quick critique tags: {', '.join(critique_tags)}")
    if reasons:
        parts.append(f"User's reasons: {', '.join(reasons)}")
    if free_text and free_text.strip():
        parts.append(f"User's free-text comment: {free_text.strip()}")
    if not critique_tags and not reasons and not free_text:
        parts.append("User did not provide specific feedback.")
    text = "\n".join(parts)
    return llm.json_call(
        instructions=DISLIKE_INSTRUCTIONS,
        input_text=text,
        schema=DISLIKE_SCHEMA,
    )