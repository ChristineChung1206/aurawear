"""Session summary builder for the Session Memory panel.

Produces a compact, UI-ready snapshot of current session state
so that Gradio callbacks never need to reach into scattered internals.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ── Interaction Trace Entry ──────────────────────────────────────────────────
@dataclass
class TraceEntry:
    """One row in the interaction trace."""
    ts: float                       # epoch seconds
    icon: str                       # emoji
    text: str                       # human-readable description

    def to_markdown(self) -> str:
        return f"{self.icon} {self.text}"


# ── Session Summary (UI-ready) ──────────────────────────────────────────────
@dataclass
class SessionSummary:
    """Compact snapshot of the session for the Session Memory panel."""

    # liked / carted items: list of (item_id, pretty_name, category, tags)
    liked_items: List[Dict[str, Any]] = field(default_factory=list)
    carted_items: List[Dict[str, Any]] = field(default_factory=list)

    # inferred style tags from liked/carted item metadata
    liked_style_tags: List[str] = field(default_factory=list)

    # avoid terms (from LLM dislike tagger + quick critique)
    avoid_terms: List[str] = field(default_factory=list)

    # critique tags (from quick critique chips)
    critique_tags: List[str] = field(default_factory=list)

    # active palette: list of {hex, name, id}
    active_palette: List[Dict[str, str]] = field(default_factory=list)

    # intent info (from chosen A/B interpretation)
    intent_style_tags: List[str] = field(default_factory=list)
    intent_must_have: List[str] = field(default_factory=list)
    intent_avoid: List[str] = field(default_factory=list)
    intent_raw_text: str = ""

    # active reco mode
    active_mode: str = "full"

    # counts
    n_liked: int = 0
    n_disliked: int = 0
    n_carted: int = 0
    n_seen: int = 0

    # interaction trace (most recent first, capped)
    trace: List[TraceEntry] = field(default_factory=list)


def build_session_summary(
    *,
    state: Any,                  # SessionState
    palette: List[Dict[str, Any]],
    selected_ids: List[str],
    user_text_payload: Any,      # UserTextPayload | None
    chosen_option: Optional[str],
    active_mode: str = "full",
    item_lookup: Optional[Any] = None,  # ItemIndexProtocol
    pretty_id_fn=None,
    trace_entries: Optional[List[TraceEntry]] = None,
) -> SessionSummary:
    """Build a SessionSummary from current state + context."""

    sel_set = set(selected_ids or [])
    active_pal = [
        {"hex": p.get("hex", ""), "name": p.get("name", ""), "id": p.get("id", "")}
        for p in (palette or []) if p.get("id") in sel_set
    ]

    # Liked / carted items with metadata
    liked_items: List[Dict[str, Any]] = []
    carted_items: List[Dict[str, Any]] = []
    liked_tag_counter: Dict[str, int] = {}

    def _item_meta(item_id: str) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"item_id": item_id}
        if pretty_id_fn:
            meta["name"] = pretty_id_fn(item_id)
        if item_lookup:
            it = item_lookup.get(item_id)
            if it:
                meta["category"] = getattr(it, "category", "")
                meta["tags"] = list(getattr(it, "tags", []) or [])
                meta["style"] = getattr(it, "style", "")
                for t in meta["tags"]:
                    if t and isinstance(t, str):
                        liked_tag_counter[t.strip().lower()] = liked_tag_counter.get(t.strip().lower(), 0) + 1
                if meta.get("style"):
                    s = meta["style"].strip().lower()
                    liked_tag_counter[s] = liked_tag_counter.get(s, 0) + 1
        return meta

    if state:
        for iid in (getattr(state, "liked", set()) or set()):
            liked_items.append(_item_meta(iid))
        for iid in (getattr(state, "cart", set()) or set()):
            carted_items.append(_item_meta(iid))

    # Top style tags from liked/carted items
    liked_style_tags = sorted(liked_tag_counter.keys(), key=lambda k: -liked_tag_counter[k])[:8]

    # Avoid terms
    avoid = sorted(getattr(state, "avoid_terms", set()) or set()) if state else []

    # Critique tags
    critique = sorted(getattr(state, "critique_tags", set()) or set()) if state else []

    # Intent info
    intent_style_tags: List[str] = []
    intent_must_have: List[str] = []
    intent_avoid: List[str] = []
    intent_raw = ""
    if user_text_payload:
        intent_raw = getattr(user_text_payload, "raw", "") or ""
        opts = getattr(user_text_payload, "options", None)
        if opts and chosen_option:
            for opt in opts:
                if getattr(opt, "id", "") == chosen_option:
                    patch = getattr(opt, "intent_patch", {}) or {}
                    intent_style_tags = patch.get("style_tags", [])
                    intent_must_have = patch.get("must_have", [])
                    intent_avoid = patch.get("avoid", [])
                    break

    return SessionSummary(
        liked_items=liked_items,
        carted_items=carted_items,
        liked_style_tags=liked_style_tags,
        avoid_terms=avoid,
        critique_tags=critique,
        active_palette=active_pal,
        intent_style_tags=intent_style_tags,
        intent_must_have=intent_must_have,
        intent_avoid=intent_avoid,
        intent_raw_text=intent_raw,
        active_mode=active_mode,
        n_liked=len(getattr(state, "liked", set()) or set()) if state else 0,
        n_disliked=len(getattr(state, "disliked", set()) or set()) if state else 0,
        n_carted=len(getattr(state, "cart", set()) or set()) if state else 0,
        n_seen=len(getattr(state, "seen", set()) or set()) if state else 0,
        trace=list(trace_entries or []),
    )


# ── Render helpers ───────────────────────────────────────────────────────────

_MODE_LABELS = {
    "full": "Full AuraWear",
    "color_only": "Color-only",
    "text_only": "Text-only",
    "color_text": "Color + Text",
}

_MODE_DESCRIPTIONS = {
    "full": "All signals: color + intent + preference memory + negative suppression + avoid terms + diversity",
    "color_only": "Color compatibility only — no preference memory, no intent, no feedback effects",
    "text_only": "Text/intent similarity only — minimal color gating, no preference memory",
    "color_text": "Color + intent similarity — no learned preference memory, no negative suppression",
}

_MODE_SIGNALS = {
    "full":       {"active": ["color", "intent", "preference", "neg_suppress", "avoid_terms", "diversity", "novelty"],
                   "disabled": []},
    "color_only": {"active": ["color"],
                   "disabled": ["intent", "preference", "neg_suppress", "avoid_terms", "diversity", "novelty"]},
    "text_only":  {"active": ["intent"],
                   "disabled": ["preference", "neg_suppress", "avoid_terms", "diversity", "novelty"]},
    "color_text": {"active": ["color", "intent", "diversity"],
                   "disabled": ["preference", "neg_suppress", "avoid_terms", "novelty"]},
}


def render_session_memory_html(summary: SessionSummary) -> str:
    """Render the Session Memory panel as user-centered HTML.

    Layout (user-centered, top-to-bottom):
      1. What I'm Looking For (intent)
      2. My Color Palette
      3. What I Like / What I Don't Like (preference + avoid)
      4. Session Activity (compact stats bar)
      5. Details (mode/signals/trace — collapsed by default)
    """
    parts: List[str] = []

    # ── Section helper ──
    def _section(title: str, icon: str, content: str, *, mb: int = 10) -> str:
        icon_html = f'<span style="font-size:14px;">{icon}</span> ' if icon else ""
        return (
            f'<div style="margin-bottom:{mb}px;">'
            f'<div style="font-size:12px;font-weight:700;margin-bottom:4px;'
            f'display:flex;align-items:center;gap:5px;">'
            f'{icon_html}{title}</div>'
            f'{content}</div>'
        )

    def _chip(text: str, bg: str, fg: str, *, border: str = "") -> str:
        bdr = f'border:1px solid {border};' if border else ''
        return (
            f'<span style="display:inline-block;background:{bg};color:{fg};'
            f'border-radius:12px;padding:2px 10px;font-size:11px;margin:2px;{bdr}">{text}</span>'
        )

    def _empty_hint(text: str) -> str:
        return (
            f'<div style="font-size:11px;color:var(--body-text-color-subdued,#aaa);'
            f'font-style:italic;padding:2px 0;">{text}</div>'
        )

    # ── 1) What I'm Looking For ──
    intent_parts: List[str] = []
    if summary.intent_style_tags:
        intent_parts.append(
            '<div style="display:flex;flex-wrap:wrap;gap:0;">'
            + "".join(_chip(t, "rgba(0,150,136,.12)", "#00695c") for t in summary.intent_style_tags)
            + '</div>'
        )
        if summary.intent_must_have:
            intent_parts.append(
                '<div style="margin-top:3px;display:flex;flex-wrap:wrap;gap:0;">'
                + "".join(_chip(f"✓ {t}", "rgba(76,175,80,.12)", "#2e7d32") for t in summary.intent_must_have)
                + '</div>'
            )
        if summary.intent_avoid:
            intent_parts.append(
                '<div style="margin-top:3px;display:flex;flex-wrap:wrap;gap:0;">'
                + "".join(_chip(f"✗ {t}", "rgba(244,67,54,.10)", "#c62828") for t in summary.intent_avoid)
                + '</div>'
            )
    elif summary.intent_raw_text:
        intent_parts.append(
            f'<div style="font-size:11px;color:var(--body-text-color-subdued,#888);'
            f'font-style:italic;padding:4px 8px;background:rgba(0,0,0,.03);border-radius:6px;">'
            f'"{summary.intent_raw_text[:100]}"</div>'
        )
    else:
        intent_parts.append(_empty_hint("Tell me what style you're looking for — e.g. \"casual weekend brunch outfit\""))

    parts.append(_section("What I'm Looking For", "", "".join(intent_parts)))

    # ── 2) My Color Palette ──
    if summary.active_palette:
        swatches = "".join(
            f'<div title="{p["name"]}" style="width:24px;height:24px;border-radius:6px;'
            f'background:{p["hex"]};border:1px solid rgba(0,0,0,.12);'
            f'box-shadow:0 1px 3px rgba(0,0,0,.08);cursor:default;"></div>'
            for p in summary.active_palette
        )
        pal_html = (
            f'<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;">'
            f'{swatches}'
            f'<span style="font-size:10px;color:var(--body-text-color-subdued,#999);'
            f'margin-left:4px;">{len(summary.active_palette)} colors selected</span></div>'
        )
        parts.append(_section("My Color Palette", "", pal_html))

    # ── 3a) What I Like ──
    like_parts: List[str] = []
    if summary.liked_style_tags:
        like_parts.append(
            '<div style="display:flex;flex-wrap:wrap;gap:0;">'
            + "".join(_chip(t, "rgba(33,150,243,.10)", "#1565c0") for t in summary.liked_style_tags)
            + '</div>'
        )
    if summary.n_liked or summary.n_carted:
        counts = []
        if summary.n_liked:
            counts.append(f"{summary.n_liked} liked")
        if summary.n_carted:
            counts.append(f"{summary.n_carted} in cart")
        like_parts.append(
            f'<div style="font-size:10px;color:var(--body-text-color-subdued,#888);margin-top:3px;">'
            f'{" · ".join(counts)}</div>'
        )
    if not like_parts:
        like_parts.append(_empty_hint("Like or add items to cart — I'll learn your taste"))

    parts.append(_section("What I Like", "", "".join(like_parts)))

    # ── 3b) What I Don't Like ──
    # avoid_terms is a superset of critique_tags (chips write to both),
    # so we only render avoid_terms to avoid duplicates.
    dislike_parts: List[str] = []
    if summary.avoid_terms:
        # Separate user-selected chips from AI-inferred expansions
        critique_set = set(summary.critique_tags)
        user_chips = [t for t in summary.avoid_terms if t in critique_set]
        ai_terms   = [t for t in summary.avoid_terms if t not in critique_set]
        if user_chips:
            dislike_parts.append(
                '<div style="display:flex;flex-wrap:wrap;gap:0;">'
                + "".join(
                    _chip(t, "rgba(244,67,54,.08)", "#c62828", border="rgba(244,67,54,.20)")
                    for t in user_chips
                )
                + '</div>'
            )
        if ai_terms:
            dislike_parts.append(
                '<div style="margin-top:3px;display:flex;flex-wrap:wrap;gap:0;">'
                + "".join(
                    _chip(f"↪ {t}", "rgba(255,152,0,.10)", "#e65100", border="rgba(255,152,0,.18)")
                    for t in ai_terms
                )
                + '</div>'
            )
    if summary.n_disliked:
        dislike_parts.append(
            f'<div style="font-size:10px;color:var(--body-text-color-subdued,#888);margin-top:3px;">'
            f'{summary.n_disliked} items disliked</div>'
        )
    if not dislike_parts:
        dislike_parts.append(_empty_hint("Dislike items to help me filter out things you don't want"))

    parts.append(_section("What I Don't Like", "", "".join(dislike_parts)))

    # ── 4) Session Activity bar ──
    parts.append(
        f'<div style="display:flex;gap:12px;padding:6px 10px;background:var(--background-fill-secondary,#f7f7f8);'
        f'border-radius:8px;margin-bottom:8px;font-size:11px;color:var(--body-text-color-subdued,#777);">'
        f'<span>{summary.n_seen} seen</span>'
        f'<span>liked: {summary.n_liked}</span>'
        f'<span>disliked: {summary.n_disliked}</span>'
        f'<span>in cart: {summary.n_carted}</span>'
        f'</div>'
    )

    # ── 5) Details (mode + trace) — collapsible ──
    detail_parts: List[str] = []
    mode_label = _MODE_LABELS.get(summary.active_mode, summary.active_mode)
    mode_desc = _MODE_DESCRIPTIONS.get(summary.active_mode, "")
    sig = _MODE_SIGNALS.get(summary.active_mode, {})
    active_sig = ", ".join(sig.get("active", []))
    disabled_sig = ", ".join(sig.get("disabled", [])) or "none"
    detail_parts.append(
        f'<div style="font-size:11px;margin-bottom:4px;">'
        f'<b>{mode_label}</b> — '
        f'<span style="color:var(--body-text-color-subdued,#777);">{mode_desc}</span></div>'
        f'<div style="font-size:10px;">'
        f'Active: <span style="color:#2e7d32;">{active_sig}</span>'
    )
    if disabled_sig != "none":
        detail_parts.append(
            f' · Off: <span style="color:#999;">{disabled_sig}</span>'
        )
    detail_parts.append('</div>')

    if summary.trace:
        detail_parts.append(
            '<div style="margin-top:6px;border-top:1px solid var(--border-color-primary,#eee);padding-top:4px;">'
            '<div style="font-size:10px;font-weight:600;margin-bottom:2px;">Recent Activity</div>'
        )
        for entry in summary.trace[-6:]:
            detail_parts.append(
                f'<div style="font-size:10px;color:var(--body-text-color-subdued,#777);'
                f'padding:1px 0;line-height:1.4;">{entry.to_markdown()}</div>'
            )
        detail_parts.append('</div>')

    parts.append(
        f'<details style="margin-top:2px;">'
        f'<summary style="font-size:11px;color:var(--body-text-color-subdued,#888);cursor:pointer;'
        f'user-select:none;">Technical Details</summary>'
        f'<div style="padding:6px 0;">{"".join(detail_parts)}</div>'
        f'</details>'
    )

    return f'<div style="font-size:11px;line-height:1.6;">{"".join(parts)}</div>'


def compute_dominant_signal(debug: Dict[str, float], mode: str = "full") -> str:
    """Determine which scoring signal dominated for an item."""
    if not debug:
        return "mixed"

    cs = abs(debug.get("color_score", 0))
    ps = abs(debug.get("pref_sim", 0))
    intent = abs(debug.get("intent_sim", 0))

    # Weight by config importance
    signals = {
        "color": cs * 1.0,
        "intent": intent * 0.3,
        "preference": ps * 0.6,
    }

    if mode == "color_only":
        return "color"
    if mode == "text_only":
        return "intent"

    top = max(signals, key=signals.get)  # type: ignore
    top_val = signals[top]

    # If top is much larger than second, it's dominant
    sorted_vals = sorted(signals.values(), reverse=True)
    if len(sorted_vals) >= 2 and sorted_vals[0] > 1.5 * sorted_vals[1] and top_val > 0.1:
        return top
    return "mixed"


_SIGNAL_LABEL = {
    "color": "Color-led",
    "intent": "Intent-led",
    "preference": "Preference-led",
    "mixed": "Mixed",
}


def render_item_detail_html(
    item_id: str,
    category: str,
    score: float,
    explanation: str,
    debug: Dict[str, float],
    dominant: str,
    avoid_terms: List[str],
    mode: str = "full",
    pretty_name: str = "",
) -> str:
    """Render the enhanced explanation panel for a selected item."""
    parts: List[str] = []

    name = pretty_name or item_id
    dom_label = _SIGNAL_LABEL.get(dominant, "Mixed")

    parts.append(
        f'<div style="font-size:14px;font-weight:700;margin-bottom:4px;">{name}</div>'
        f'<div style="font-size:11px;color:var(--body-text-color-subdued,#777);margin-bottom:6px;">'
        f'{category} · score: {score:.3f} · {dom_label}</div>'
    )

    if explanation:
        parts.append(
            f'<div style="background:rgba(0,150,136,.06);border-radius:6px;padding:6px 10px;'
            f'margin-bottom:8px;font-size:12px;">{explanation}</div>'
        )

    # Score breakdown
    if debug:
        cs = debug.get("color_score", 0)
        ps = debug.get("pref_sim", 0)
        ins = debug.get("intent_sim", 0)
        neg = debug.get("neg_pen", 0)
        dup = debug.get("dup_pen", 0)
        nov = debug.get("novelty_pen", 0)
        rule = debug.get("dislike_rule_pen", 0)

        def _bar(val, max_val=1.0, color="#009688"):
            pct = min(100, max(0, abs(val) / max(max_val, 0.01) * 100))
            return (
                f'<div style="background:var(--background-fill-secondary,#f0f0f0);border-radius:3px;height:8px;width:80px;display:inline-block;vertical-align:middle;">'
                f'<div style="background:{color};height:100%;border-radius:3px;width:{pct:.0f}%;"></div></div>'
            )

        rows = []
        rows.append(f'Color match: {cs:.3f} {_bar(cs, 1.0, "#009688")}')
        if mode in ("full", "color_text"):
            rows.append(f'Intent match: {ins:.3f} {_bar(ins, 1.0, "#1976d2")}')
        if mode == "full":
            rows.append(f'Preference: {ps:.3f} {_bar(ps, 1.0, "#7b1fa2")}')
            if neg > 0.001:
                rows.append(f'Neg suppression: -{neg:.3f} {_bar(neg, 0.5, "#c62828")}')
            if rule > 0.001:
                rows.append(f'Avoid-term penalty: -{rule:.3f} {_bar(rule, 1.0, "#e65100")}')
        if nov > 0.001:
            rows.append(f'Novelty penalty: -{nov:.3f} {_bar(nov, 0.5, "#795548")}')
        if dup > 0.001:
            rows.append(f'Diversity penalty: -{dup:.3f} {_bar(dup, 0.5, "#546e7a")}')

        parts.append(
            '<div style="font-size:10px;line-height:1.8;margin-top:4px;">'
            + '<br>'.join(rows)
            + '</div>'
        )

        # Penalization notes
        penalties: List[str] = []
        if rule > 0.05:
            matched = [t for t in avoid_terms] if avoid_terms else []
            if matched:
                penalties.append(f"Penalized by avoid terms: {', '.join(matched[:3])}")
        if neg > 0.05:
            penalties.append("Suppressed by similarity to disliked items")
        if nov > 0:
            penalties.append("Previously seen item (novelty penalty applied)")
        if dup > 0.05:
            penalties.append("Diversity penalty to ensure variety")

        if penalties:
            parts.append(
                '<div style="margin-top:4px;font-size:10px;color:var(--body-text-color-subdued,#888);">'
                + '<br>'.join(f'Note: {p}' for p in penalties)
                + '</div>'
            )

    return ''.join(parts)
