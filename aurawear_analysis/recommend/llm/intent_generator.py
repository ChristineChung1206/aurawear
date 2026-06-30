from typing import Any, Dict, List, Optional
import json

from .client import LLMClient
from .schemas import INTENT_SCHEMA
from aurawear_analysis.recommend.llm.prompts import INTENT_PROMPT
from aurawear_analysis.recommend.schemas import UserTextPayload


def _extract_palette_terms(palette_selected: List[Dict[str, str]]) -> List[str]:
    """
    Prefer human-readable color names for LLM stability.
    Fallback to hex only if name is missing.
    """
    terms: List[str] = []
    for c in palette_selected:
        name = (c.get("name") or "").strip()
        if name:
            terms.append(name.lower())
            continue
        hx = (c.get("hex") or "").strip()
        if hx:
            terms.append(hx.lower())
    return terms


def generate_intent(
    llm: LLMClient,
    *,
    gender: str,
    style: str,
    palette_selected: List[Dict[str, str]],
    user_text: Optional[UserTextPayload] = None,
    existing_avoid_terms: Optional[List[str]] = None,
) -> Dict[str, Any]:

    palette_terms = _extract_palette_terms(palette_selected)
    palette_phrase = ", ".join(palette_terms[:8]) if palette_terms else ""

    existing_avoid_terms = existing_avoid_terms or []
    existing_avoid_terms = [str(x).strip().lower() for x in existing_avoid_terms if str(x).strip()]

    # Extract raw text and chosen patch from user_text payload
    raw_text = ""
    chosen_patch: Dict = {}
    if user_text is not None:
        raw_text = user_text.raw or ""
        if user_text.choice and user_text.options:
            for opt in user_text.options:
                if opt.id == user_text.choice:
                    chosen_patch = opt.intent_patch or {}
                    break

    prompt = INTENT_PROMPT.format(
        gender=gender,
        style=style,
        palette_phrase=palette_phrase,
        existing_avoid_terms=json.dumps(existing_avoid_terms, ensure_ascii=False),
        user_text=raw_text,
        chosen_patch=json.dumps(chosen_patch, ensure_ascii=False),
    )


    intent = llm.json_call(
        instructions=prompt,
        input_text="",
        schema=INTENT_SCHEMA,
    )

    intent["query_text"] = (intent.get("query_text") or "").strip()[:240]
    return intent
