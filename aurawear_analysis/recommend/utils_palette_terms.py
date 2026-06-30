# aurawear_analysis/recommend/utils_palette_terms.py
from __future__ import annotations

from typing import List, Tuple
import colorsys


def _hex_to_rgb01(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex: {hex_color}")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return r, g, b


def hex_to_color_terms(hex_color: str) -> List[str]:
    """
    Map hex -> short color terms that CLIP tends to understand.
    Keep it simple and stable (no fancy naming model).
    """
    r, g, b = _hex_to_rgb01(hex_color)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # tone adjectives
    tone = []
    if v >= 0.78:
        tone.append("light")
    elif v <= 0.35:
        tone.append("dark")

    if s <= 0.22:
        tone.append("muted")
    elif s >= 0.65:
        tone.append("vivid")

    # hue bucket
    deg = h * 360.0
    if s <= 0.10:
        # near gray
        base = "warm gray" if r > b else "cool gray"
        return tone + [base]

    if deg < 15 or deg >= 345:
        base = "red"
    elif deg < 45:
        base = "orange"
    elif deg < 70:
        base = "yellow"
    elif deg < 160:
        base = "green"
    elif deg < 200:
        base = "teal"
    elif deg < 255:
        base = "blue"
    elif deg < 290:
        base = "purple"
    elif deg < 345:
        base = "pink"
    else:
        base = "red"

    # warm/cool nuance
    warm = (r >= b) and (deg < 200)
    nuance = "warm" if warm else "cool"

    # special cases: olive/camel/beige-ish
    # (simple heuristics to improve palette realism)
    if base == "yellow" and s <= 0.45:
        base = "beige"
    if base == "orange" and s <= 0.45 and v <= 0.75:
        base = "camel"
    if base == "green" and (deg > 70 and deg < 110) and s <= 0.55:
        base = "olive green"

    return tone + [f"{nuance} {base}"]


def build_palette_phrase(selected_hexes: List[str], max_terms: int = 8) -> str:
    """
    Returns a stable palette phrase to inject into CLIP query.
    Example: "muted warm olive green, light warm beige, warm camel, cool blue ..."
    """
    terms: List[str] = []
    for hx in selected_hexes[:max_terms]:
        try:
            t = " ".join(hex_to_color_terms(hx))
            terms.append(t)
        except Exception:
            continue
    return ", ".join(terms) if terms else ""
