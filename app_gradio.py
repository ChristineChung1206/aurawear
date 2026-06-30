from __future__ import annotations

import os
import re
import json
import uuid
import time
import random
import string
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Optional .env loading (pip install python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in cwd (or parent dirs)
except ImportError:
    pass

import gradio as gr

from aurawear_analysis.color_analysis import ColorAnalysisPipeline
from aurawear_analysis.recommend.recommender import AuraWearRecommender
from aurawear_analysis.recommend.schemas import (
    Item,
    SessionState,
    GenerateRequest,
    Filters,
    FeedbackEvent,
    UserTextPayload,
    UserTextOption,
)
from aurawear_analysis.config import RecoConfig
from aurawear_analysis.recommend.llm.client import LLMClient
from aurawear_analysis.recommend.utils_palette_terms import build_palette_phrase
from aurawear_analysis.interaction_logger import log_event, Timer
from aurawear_analysis.recommend.session_summary import (
    TraceEntry,
    build_session_summary, render_session_memory_html,
    compute_dominant_signal, render_item_detail_html,
    _MODE_DESCRIPTIONS,
)


# =============================================================================
# CONFIG — paths relative to this repo
# =============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parent
_PKG_DIR = _PROJECT_ROOT / "aurawear_analysis"

DATA_DIR = str(_PKG_DIR / "data")
_default_csv = str(_PROJECT_ROOT / "data" / "sample_catalog.csv")
_default_json = os.path.join(DATA_DIR, "items_deepfashion.json")
# Catalog fallback order: CATALOG_PATH env → sample_catalog.csv → items_deepfashion.json
if os.getenv("CATALOG_PATH"):
    ITEMS_JSON_PATH = os.getenv("CATALOG_PATH")
elif os.path.exists(_default_csv):
    ITEMS_JSON_PATH = _default_csv
else:
    ITEMS_JSON_PATH = _default_json
IMAGES_DIR = os.getenv("IMAGES_DIR") or os.path.join(DATA_DIR, "products", "images")
_PLACEHOLDER_IMG = str(_PROJECT_ROOT / "assets" / "synthetic_demo_images" / "placeholder.png")
PALETTE18_PATH = str(_PKG_DIR / "assets" / "palette18.json")
DEMO_SELFIE_PATH = str(_PKG_DIR / "assets" / "demo_selfie.jpg")  # optional bundled sample

DEFAULT_K = 50

STYLE_OPTIONS = [
    "Polished",
    "Relaxed",
    "Minimal",
    "Feminine",
    "Structured",
    "Soft",
    "Bold",
    "Casual",
    "Classic",
    "Trendy",
]

# Single unified chip list — all items write directly to avoid_terms + feed LLM
DISLIKE_CHIPS = [
    "too formal",
    "too casual",
    "too loose",
    "too tight",
    "too boxy",
    "too short",
    "too long",
    "too bright",
    "too dark",
    "too plain",
    "too flashy",
    "too feminine",
    "too masculine",
]

CATEGORY_OPTIONS = ["top", "pants", "dress", "outer", "rompers", "skirt", "leggings"]

RECO_MODE_CHOICES = [
    ("Full AuraWear", "full"),
    ("Color-only", "color_only"),
    ("Text-only", "text_only"),
    ("Color + Text", "color_text"),
]



def _initial_chat_history() -> list:
    """Return an empty chat — session history only shows actual user/dialog interactions."""
    return []


# =============================================================================
# CSV catalog loader for synthetic demo mode
# =============================================================================
def _load_csv_catalog(csv_path: str) -> list:
    """Load synthetic demo catalog from CSV.

    Generates a deterministic 512-dim unit-normalised pseudo-embedding per item
    using SHA-256 of item_id as the RNG seed, so recommendations are reproducible
    across runs without any external embedding service or CLIP model call.
    """
    import csv
    import hashlib
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seed = int(hashlib.sha256(row["item_id"].encode()).hexdigest(), 16) % (2 ** 32)
            rng = np.random.default_rng(seed)
            emb = rng.standard_normal(512).astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-8
            tags = [t.strip() for t in row.get("tags", "").split(";") if t.strip()]
            rows.append({
                "item_id": row["item_id"],
                "image_uri": row.get("image_uri", ""),
                "category": row.get("category", ""),
                "dominant_hex": [row.get("dominant_hex", "#888888")],
                "emb": emb.tolist(),
                "title": row.get("title", ""),
                "style": row.get("style", ""),
                "brand": row.get("brand", ""),
                "tags": tags,
                "meta_text": row.get("meta_text", ""),
            })
    return rows


# =============================================================================
# Minimal ItemIndex + SessionStore (in-memory)
# =============================================================================
class SimpleItemIndex:
    def __init__(self, items_json_path: str):
        if not os.path.exists(items_json_path):
            raise FileNotFoundError(f"Catalog not found: {items_json_path}")
        if items_json_path.endswith(".csv"):
            raw = _load_csv_catalog(items_json_path)
        else:
            with open(items_json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        self._items: List[Item] = [Item.from_dict(d) for d in raw]
        self._map: Dict[str, Item] = {it.item_id: it for it in self._items}
        self._emb_dim: int = self._items[0].emb.shape[0] if self._items else 512

    @property
    def items(self) -> List[Item]:
        return self._items

    @property
    def emb_dim(self) -> int:
        return self._emb_dim

    def get(self, item_id: str) -> Optional[Item]:
        return self._map.get(item_id)


class SimpleSessionStore:
    def __init__(self):
        self._db: Dict[str, SessionState] = {}
        self._friend_ctx: Dict[str, Dict] = {}   # code → user context
        self._friend_ans: Dict[str, str] = {}    # code → friend suggestion text

    def get_or_create(self, session_id: str) -> SessionState:
        if session_id not in self._db:
            self._db[session_id] = SessionState(session_id=session_id)
        return self._db[session_id]

    def save(self, state: SessionState) -> None:
        self._db[state.session_id] = state

    # --- Friend invite helpers ---
    def create_friend_invite(self, code: str, ctx: Dict) -> None:
        """Store user context under code for friend to view."""
        self._friend_ctx[code] = ctx

    def get_friend_context(self, code: str) -> Optional[Dict]:
        return self._friend_ctx.get(code)

    def submit_friend_input(self, code: str, suggestion: str) -> bool:
        """Store friend's suggestion. Returns False if code unknown."""
        if code not in self._friend_ctx:
            return False
        self._friend_ans[code] = suggestion
        return True

    def poll_friend_input(self, code: str) -> Optional[str]:
        """Return friend suggestion if submitted, else None."""
        return self._friend_ans.get(code)

    def consume_friend_input(self, code: str) -> Optional[str]:
        """Return and clear friend suggestion (consume-once). Returns None if not yet submitted."""
        return self._friend_ans.pop(code, None)

    def update_friend_context_intent(self, code: str, user_intent: str) -> None:
        """Update user_intent in-place so friend sees the latest round's intent."""
        if code in self._friend_ctx:
            self._friend_ctx[code]["user_intent"] = user_intent


# =============================================================================
# Pretty item-name helper
# =============================================================================

def _pretty_id(item_id: str) -> str:
    """Convert ugly IDs like 'df_MEN-Jackets_Vests-id_00006912-09_4_full'
    into human-friendly labels like 'Jackets & Vests #6912'."""
    m = re.match(
        r"df_(?:MEN|WOMEN)-([A-Za-z_]+)-id_(\d+)", item_id or ""
    )
    if not m:
        return item_id  # fallback: return raw id
    subcat = m.group(1).replace("_", " & ", 1).replace("_", " ")
    num = str(int(m.group(2)))  # strip leading zeros
    return f"{subcat} #{num}"


def _step_indicator_html(active: int) -> str:
    """Render a horizontal step-progress bar (1\u20135)."""
    labels = ["Selfie", "Style", "Analyzing", "Palette", "Recommend"]
    parts = []
    for i, lbl in enumerate(labels, 1):
        if i < active:
            circ = (
                '<div style="width:24px;height:24px;border-radius:50%;'
                'background:#4CAF50;color:#fff;font-size:11px;display:flex;'
                'align-items:center;justify-content:center;">\u2713</div>'
            )
            clr = "#4CAF50"
        elif i == active:
            circ = (
                f'<div style="width:24px;height:24px;border-radius:50%;'
                f'background:#009688;color:#fff;font-size:12px;font-weight:700;'
                f'display:flex;align-items:center;justify-content:center;">{i}</div>'
            )
            clr = "#009688"
        else:
            circ = (
                f'<div style="width:24px;height:24px;border-radius:50%;'
                f'background:transparent;border:2px solid #ccc;color:#aaa;'
                f'font-size:11px;display:flex;align-items:center;'
                f'justify-content:center;">{i}</div>'
            )
            clr = "#aaa"
        parts.append(
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:2px;">'
            f'{circ}<span style="font-size:10px;color:{clr};white-space:nowrap;">{lbl}</span></div>'
        )
    joined = []
    for idx, p in enumerate(parts):
        joined.append(p)
        if idx < len(parts) - 1:
            lc = "#4CAF50" if (idx + 1) < active else "#ddd"
            joined.append(
                f'<div style="flex:1;height:2px;background:{lc};align-self:center;'
                f'margin:0 4px;margin-bottom:16px;"></div>'
            )
    return (
        '<div style="display:flex;align-items:flex-start;justify-content:center;'
        'padding:8px 0 4px;max-width:480px;margin:0 auto;">'
        + ''.join(joined) + '</div>'
    )


# =============================================================================
# HTML Helpers
# =============================================================================
def render_clickable_palette_html(
    palette: List[Dict[str, Any]],
    selected_ids: Optional[List[str]] = None,
    prefix: str = "step4",
) -> str:
    """Render palette as a 6-column visual grid (selection via CheckboxGroup)."""
    if not palette:
        return '<div style="color:var(--body-text-color-subdued,#999);">No palette available.</div>'
    sel = set(selected_ids or [])
    # sidebar = smaller swatches to fit 1/3 column
    is_sidebar = prefix == "sidebar"
    sw = 36 if is_sidebar else 48
    br = 8 if is_sidebar else 10
    fs = 8 if is_sidebar else 10
    gap = 6 if is_sidebar else 10
    html = (
        f'<div id="pal-grid-{prefix}" style="display:grid; '
        f'grid-template-columns:repeat(6,1fr); gap:{gap}px; padding:6px 0;">'
    )
    for p in palette:
        hexv = p.get("hex", "#000000")
        pid = p.get("id", "")
        name = p.get("name", pid)
        is_sel = pid in sel
        border = "3px solid #4CAF50" if is_sel else "2px solid var(--border-color-primary, rgba(128,128,128,0.3))"
        check = "✓" if is_sel else ""
        opa = "1" if is_sel else "0.55"
        html += (
            f'<div data-pid="{pid}" '
            f'style="text-align:center; padding:2px; '
            f'opacity:{opa}; transition:all .15s;">'
            f'<div style="width:{sw}px; height:{sw}px; border-radius:{br}px; '
            f'background:{hexv}; border:{border}; margin:0 auto; '
            f'position:relative; box-shadow:0 1px 4px rgba(0,0,0,.25); '
            f'transition:border .15s;">'
            f'<span style="color:#fff; font-size:{sw//3}px; font-weight:700; '
            f'text-shadow:0 0 3px #000; position:absolute; top:50%; left:50%; '
            f'transform:translate(-50%,-50%);">{check}</span></div>'
            f'<div style="font-size:{fs}px; margin-top:2px; '
            f'color:var(--body-text-color-subdued,#888); '
            f'line-height:1.1; max-width:{sw+14}px; margin-left:auto; '
            f'margin-right:auto; white-space:nowrap; overflow:hidden; '
            f'text-overflow:ellipsis;">{name}</div></div>'
        )
    html += "</div>"
    return html


def render_analysis_html(
    skin_hex: str, hair_hex: str,
    eye_color: str, eye_hex: str,
    season: str, conf: float, undertone: str,
) -> str:
    """Render color analysis results as compact HTML for sidebar."""
    def swatch(hex_val, label):
        return (
            f'<div style="display:flex; align-items:center; gap:8px; margin:3px 0;">'
            f'<div style="width:18px; height:18px; border-radius:5px; '
            f'background:{hex_val}; border:1px solid var(--border-color-primary,#999);"></div>'
            f'<span style="font-size:12px; color:var(--body-text-color,#333);">{label}: {hex_val}</span></div>'
        )
    return (
        f'<div style="padding:6px 0;">'
        f'<div style="font-size:15px; font-weight:700; color:var(--body-text-color,#222); margin-bottom:6px;">'
        f'{season} <span style="font-size:11px; color:var(--body-text-color-subdued,#777);">(conf: {conf:.2f})</span></div>'
        f'{swatch(skin_hex, "Skin")}'
        f'{swatch(hair_hex, "Hair")}'
        f'{swatch(eye_hex, "Eyes")}'
        #f'<div style="font-size:11px; color:var(--body-text-color-subdued,#777); margin-top:4px;">Undertone: {undertone}</div>'
        f'</div>'
    )


def palette_selected_dicts(palette, selected_ids):
    sid = set(selected_ids or [])
    return [p for p in (palette or []) if p.get("id") in sid]


# =============================================================================
# Session Memory builder (UI helper)
# =============================================================================
def _build_memory_html(
    session_id, palette, selected_ids,
    user_text_payload, chosen_option,
    reco_mode, trace_entries,
):
    """Build the Session Memory HTML from current state + context."""
    state = store.get_or_create(session_id) if session_id else None
    summary = build_session_summary(
        state=state,
        palette=palette or [],
        selected_ids=selected_ids or [],
        user_text_payload=user_text_payload,
        chosen_option=chosen_option,
        active_mode=reco_mode or "full",
        item_lookup=index,
        pretty_id_fn=_pretty_id,
        trace_entries=trace_entries or [],
    )
    return render_session_memory_html(summary)


def _add_trace(trace_entries, icon, text):
    """Append a trace entry and return the updated list."""
    entries = list(trace_entries or [])
    entries.append(TraceEntry(ts=time.time(), icon=icon, text=text))
    # keep last 20
    return entries[-20:]


# =============================================================================
# Friend Invite helpers
# =============================================================================

def _gen_invite_code() -> str:
    """Generate a random 6-character alphanumeric code (uppercase)."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def on_invite_friend(session_id, palette, selected_ids, analysis, user_text_payload, chat_input_text, existing_code):
    """
    Create (or refresh) a friend invite.
    - If existing_code is set, reuse it and overwrite context (no new code).
    - Otherwise generate a new code.
    Returns: (code_display_md, invite_code_state, invite_btn_update, check_btn_update)
    """
    season = (analysis or {}).get("season", "Unknown")
    intent_raw = (chat_input_text or "").strip() or (user_text_payload.raw if user_text_payload else "")
    hex_list = [p.get("hex", "") for p in (palette or []) if p.get("id") in set(selected_ids or [])]
    ctx = {
        "session_id": session_id,
        "season": season,
        "palette_hexes": hex_list[:6],
        "user_intent": intent_raw,
    }
    code = (existing_code or "").strip() or _gen_invite_code()
    store.create_friend_invite(code, ctx)  # overwrites if exists (refresh)
    md = (
        f"**Invite code: `{code}`**  "
        f"<small style='color:var(--body-text-color-subdued,#888);'>"
        f"Share this with your friend → they open AuraWear, click **Open Friend Mode**, "
        f"enter the code, and send a suggestion. "
        f"The context auto-updates each time you press **Send**.</small>"
    )
    return md, code, gr.update(visible=False), gr.update(visible=True)


def on_check_friend_input(invite_code):
    """
    Poll the store for the friend's suggestion (consume-once: clears after reading).
    Returns: (status_md, friend_suggestion_text)
    """
    if not invite_code:
        return "⚠️ No active invite. Click **Invite Friend** first.", ""
    suggestion = store.consume_friend_input(invite_code)  # consume-once
    if suggestion is None:
        return "⏳ Friend hasn't submitted yet. Try again in a moment.", ""
    return f"✅ Friend's suggestion received!", suggestion


# =============================================================================
# LLM: option generation (A/B)
# =============================================================================
OPTIONS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "occasion":    {"type": "string"},
                "style_goal":  {"type": "string"},
                "constraints": {"type": "string"},
            },
            "required": ["occasion", "style_goal", "constraints"],
        },
        "options": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "id": {"type": "string", "enum": ["A", "B"]},
                    "interpretation": {"type": "string"},
                    "intent_patch": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "style_tags": {"type": "array", "items": {"type": "string"}},
                            "must_have": {"type": "array", "items": {"type": "string"}},
                            "avoid": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["style_tags", "must_have", "avoid"],
                    },
                },
                "required": ["id", "interpretation", "intent_patch"],
            },
        },
    },
    "required": ["summary", "options"],
}


def llm_interpret_options(
    llm: LLMClient, *, user_text_raw: str,
    gender: str, style_hint: str,
    palette_selected: List[Dict[str, Any]],
    friend_suggestion: Optional[str] = None,
) -> tuple:
    """Return (summary_dict, List[UserTextOption])."""
    hex_list = [c.get("hex", "") for c in palette_selected if c.get("hex")]
    palette_phrase = build_palette_phrase(hex_list, max_terms=8)
    friend_line = (
        f"- a friend suggested: {friend_suggestion.strip()}\n"
        if friend_suggestion and friend_suggestion.strip()
        else ""
    )
    b_instruction = (
        "B should explore a plausible alternative more shaped by the friend's suggestion, "
        "while still respecting palette and gender."
        if friend_line
        else "B should be a plausible alternative interpretation of the user's request."
    )
    prompt = (
        "You are an assistant helping interpret a user's outfit request "
        "into two alternative structured intents.\n"
        "Return a JSON object with:\n"
        "  1. 'summary': occasion (where/when), style_goal (the vibe/look), "
        "constraints (must-haves or avoid, or 'none'). Keep each field to 1 sentence.\n"
        "  2. 'options': EXACTLY TWO items A and B.\n\n"
        f"Context:\n- gender: {gender or 'unknown/any'}\n"
        f"- style hint: {style_hint or 'none'}\n"
        f"- palette constraint: {palette_phrase}\n"
        f"{friend_line}\n"
        f"User input: {user_text_raw}\n\n"
        "Each option: id ('A'/'B'), interpretation (English, short), "
        "intent_patch (style_tags required, must_have optional, avoid optional).\n"
        f"A should remain closer to the user's original intent. {b_instruction}\n"
        "Keep intent_patch lightweight."
    )
    out = llm.json_call(instructions=prompt, input_text="", schema=OPTIONS_SCHEMA)
    summary = out.get("summary") or {"occasion": "", "style_goal": "", "constraints": ""}
    opts_raw = sorted(out.get("options", []) or [], key=lambda x: x.get("id", "Z"))
    options = [
        UserTextOption(
            id=str(o.get("id")),
            interpretation=str(o.get("interpretation", "")).strip(),
            intent_patch=o.get("intent_patch") or {},
        )
        for o in opts_raw
    ]
    return summary, options


# =============================================================================
# Global objects (loaded once)
# =============================================================================
def load_palette18(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"palette18.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def palette_for_season(all_palettes, season_12):
    return [p for p in all_palettes if p.get("season") == season_12]


palette18_all = load_palette18(PALETTE18_PATH)
index = SimpleItemIndex(ITEMS_JSON_PATH)
store = SimpleSessionStore()
pipeline = ColorAnalysisPipeline()

# LLM client: graceful fallback when API key is missing
if os.getenv("OPENAI_API_KEY"):
    llm_client = LLMClient(model="gpt-4.1-mini")
    print("[INIT] LLM enabled (gpt-4.1-mini)")
else:
    llm_client = None
    print("[INIT] OPENAI_API_KEY not set — LLM features disabled (rule-based fallback)")

reco = AuraWearRecommender(index=index, store=store, llm=llm_client, cfg=RecoConfig())


# =============================================================================
# Chat helpers
# =============================================================================
_CONFIRM_PREFIX = "✅ Selected option"


def _msg_text(msg) -> str:
    """Safely extract text content from a Gradio chatbot message.

    After Gradio Chatbot round-trip (postprocess → preprocess), content becomes
    a *list* of content dicts like ``[{"text": "...", "type": "text"}]`` rather
    than a plain string.  Handle both formats.
    """
    if isinstance(msg, dict):
        c = msg.get("content", "")
    else:
        c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, (list, tuple)):
        parts: list[str] = []
        for p in c:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
        return "".join(parts)
    return str(c) if c else ""


# =============================================================================
# CALLBACKS
# =============================================================================
def on_init_session():
    return f"gr_{uuid.uuid4().hex[:10]}"


# ── Step 1 → Step 2 ──
def go_step1_to_step2(selfie_path):
    if not selfie_path:
        gr.Warning("Please upload a selfie photo before continuing.")
        return (gr.update(), gr.update(), gr.update())
    return (
        selfie_path,               # st_selfie_path
        gr.update(visible=False),  # page1
        gr.update(visible=True),   # page2
    )


# ── Step 2 → Step 3 (show loading) ──
def go_step2_to_step3(selected_styles, gender_val):
    if not selected_styles:
        gr.Warning("Please select at least one style preference before continuing.")
        return (gr.update(), gr.update(), gr.update(), gr.update())
    style_hint = ", ".join(selected_styles)
    return (
        gender_val or "",          # st_gender
        style_hint,                # st_style_hint
        gr.update(visible=False),  # page2
        gr.update(visible=True),   # page3
    )


# ── Step 3: Run analysis → auto-advance to Step 4 ──
def run_analysis(selfie_path):
    if not selfie_path:
        raise gr.Error("No selfie found.")

    import traceback as _tb
    try:
        diagnosis = pipeline.diagnose(selfie_path)
    except Exception as _exc:
        _tb.print_exc()
        raise gr.Error(f"Analysis failed: {_exc}")
    if diagnosis is None:
        raise gr.Error(
            "No face detected. Please go back and try another selfie "
            "(better lighting, face centered)."
        )

    season = diagnosis.season_12
    conf = float(diagnosis.season_confidence)
    undertone = "neutral"
    if isinstance(diagnosis.diagnostics, dict):
        undertone = diagnosis.diagnostics.get("undertone_type", "neutral")

    skin_hex = diagnosis.skin_color_hex
    hair_hex = diagnosis.hair_color_hex
    eye_color = diagnosis.eye_color
    eye_hex = diagnosis.eye_color_hex

    palette = palette_for_season(palette18_all, season)
    default_ids = [p["id"] for p in palette[: min(6, len(palette))]]
    palette_grid = render_clickable_palette_html(palette, default_ids, prefix="step4")
    choices = [(p["name"], p["id"]) for p in palette]

    analysis = {
        "season": season, "conf": conf, "undertone": undertone,
        "skin_hex": skin_hex, "hair_hex": hair_hex,
        "eye_color": eye_color, "eye_hex": eye_hex,
    }

    return (
        gr.update(visible=False),   # page3
        gr.update(visible=True),    # page4
        palette_grid,               # step4_palette_html
        gr.update(choices=choices, value=default_ids),  # step4_palette_cb
        default_ids,                # st_selected_ids_step4
        palette,                    # st_palette
        analysis,                   # st_analysis
        f"Your season: **{season}** (confidence: {conf:.2f})",
    )


# ── Step 4 / Sidebar: palette checkbox changed ──
def on_palette_change(selected_ids, palette, prefix="step4"):
    """Called when any palette CheckboxGroup selection changes."""
    html = render_clickable_palette_html(palette, selected_ids or [], prefix=prefix)
    return html, selected_ids or []


# ── Step 4 → Step 5 ──
def go_step4_to_step5(selected_ids, palette, analysis, selfie_path, style_hint):
    if not selected_ids:
        gr.Warning("Please select at least 1 color from your palette before continuing.")
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    a = analysis or {}
    analysis_html = render_analysis_html(
        a.get("skin_hex", "#000"), a.get("hair_hex", "#000"),
        a.get("eye_color", "unknown"), a.get("eye_hex", "#808080"),
        a.get("season", "Unknown"), a.get("conf", 0.0),
        a.get("undertone", "neutral"),
    )
    sidebar_pal_html = render_clickable_palette_html(palette, selected_ids, prefix="sidebar")
    choices = [(p["name"], p["id"]) for p in (palette or [])]

    return (
        gr.update(visible=False),  # page4
        gr.update(visible=True),   # page5
        selfie_path,               # sidebar_selfie
        analysis_html,             # sidebar_analysis
        sidebar_pal_html,          # sidebar_palette_html
        gr.update(choices=choices, value=selected_ids),  # sidebar_palette_cb
        selected_ids,              # st_selected_ids
        _initial_chat_history(),   # chatbot
        gr.update(visible=True),   # style_goal_fab_btn
    )


# ── Chat submit ──

# Phase 1: instantly show user message + typing indicator
def on_chat_submit_show(user_msg, chat_history, friend_suggestion_raw):
    """Immediately show user message + typing indicator, disable Send.
    Captures friend_suggestion_raw into State before clearing the UI field.
    """
    if not user_msg or not user_msg.strip():
        gr.Warning("Please describe your style request before sending.")
        return (
            chat_history,
            gr.update(),  # chat_input
            gr.update(),  # chat_send_btn
            gr.update(), gr.update(), gr.update(), gr.update(),  # sc_btns
            gr.update(),  # invite_btn
            gr.update(),  # check_friend_btn
            gr.update(),  # friend_input
            gr.update(),  # check_friend_status
            (friend_suggestion_raw or "").strip(),
        )
    chat_history = list(chat_history or [])
    # Add a visual separator between rounds
    if chat_history:
        chat_history.append({
            "role": "assistant",
            "content": "---\n*── Previous round above ──*",
        })
    chat_history.append({"role": "user", "content": user_msg.strip()})
    chat_history.append({"role": "assistant", "content": "⏳ *Thinking...*"})
    return (
        chat_history,                    # chatbot
        gr.update(value="", interactive=False),  # chat_input (clear + lock)
        gr.update(interactive=False),    # chat_send_btn
        gr.update(interactive=False),    # sc_btn_1
        gr.update(interactive=False),    # sc_btn_2
        gr.update(interactive=False),    # sc_btn_3
        gr.update(interactive=False),    # sc_btn_4
        gr.update(interactive=False),    # invite_btn
        gr.update(interactive=False),    # check_friend_btn
        gr.update(value=""),             # friend_input  (clear for new round)
        gr.update(value=""),             # check_friend_status (clear '✅ filled in...')
        (friend_suggestion_raw or "").strip(),   # st_friend_suggestion (capture before clear)
    )


# Phase 2: run LLM + replace typing indicator with real response
def on_chat_submit_llm(chat_history, palette, selected_ids, gender, style_hint, friend_suggestion, invite_code):
    """Run LLM interpretation, replace typing indicator.
    Also auto-updates the friend invite context with the latest user intent.
    """
    if not palette:
        gr.Warning("Palette not ready — please complete the color analysis first.")
        return (chat_history, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    if not selected_ids:
        gr.Warning("Please select at least 1 palette color in the sidebar before sending.")
        return (chat_history, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    # Extract user message — search backwards for the last 'user' role message
    user_msg = ""
    for msg in reversed(chat_history or []):
        role = ""
        if isinstance(msg, dict):
            role = msg.get("role", "")
        else:
            role = getattr(msg, "role", "")
        if role == "user":
            user_msg = _msg_text(msg)
            break
    if not user_msg:
        gr.Warning("Could not find your message. Please type your request and try again.")
        return (chat_history, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    pal_sel = palette_selected_dicts(palette, selected_ids)

    # --- Auto-update friend context with latest intent ---
    if invite_code:
        store.update_friend_context_intent(invite_code, user_msg)

    # --- LLM A/B interpretation (graceful fallback when LLM disabled) ---
    if llm_client is not None:
        summary, options = llm_interpret_options(
            llm=llm_client, user_text_raw=user_msg,
            gender=(gender or ""), style_hint=(style_hint or ""),
            palette_selected=pal_sel,
            friend_suggestion=(friend_suggestion or None),
        )

        lines = ["I generated two interpretations:\n"]
        for opt in sorted(options, key=lambda x: x.id):
            tags = ", ".join(opt.intent_patch.get("style_tags", []))
            must = ", ".join(opt.intent_patch.get("must_have", []))
            avoid_str = ", ".join(opt.intent_patch.get("avoid", []))
            lines.append(f"**{opt.id}. {opt.interpretation}**")
            if tags:
                lines.append(f"  - style: {tags}")
            if must:
                lines.append(f"  - must have: {must}")
            if avoid_str:
                lines.append(f"  - avoid: {avoid_str}")
            lines.append("")

        lines.append("Choose **A** or **B** below, then click **Recommend**")

        chat_history = list(chat_history or [])
        if chat_history and "Thinking" in _msg_text(chat_history[-1]):
            chat_history[-1] = {"role": "assistant", "content": "\n".join(lines)}
        else:
            chat_history.append({"role": "assistant", "content": "\n".join(lines)})

        payload = UserTextPayload(raw=user_msg, choice=None, options=options)
        return (
            chat_history,                    # chatbot
            gr.update(visible=True),         # option_btn_row — show A/B buttons inside accordion
            payload,                         # st_user_text_payload
            None,                            # st_chosen_option
            gr.update(interactive=True),     # choose_a_btn
            gr.update(interactive=True),     # choose_b_btn
            gr.update(interactive=True),     # chat_input (re-enable)
            gr.update(interactive=True),     # chat_send_btn (re-enable)
            gr.update(interactive=True),     # sc_btn_1
            gr.update(interactive=True),     # sc_btn_2
            gr.update(interactive=True),     # sc_btn_3
            gr.update(interactive=True),     # sc_btn_4
            gr.update(interactive=True),     # invite_btn
            gr.update(interactive=True),     # check_friend_btn
        )
    else:
        # LLM disabled — pass raw text directly, skip A/B
        chat_history = list(chat_history or [])
        fallback_msg = (
            f"✅ Got it: *\"{user_msg}\"*\n\n"
            "*(LLM disabled — using your text directly. Click **Recommend** to generate results.)*"
        )
        if chat_history and "Thinking" in _msg_text(chat_history[-1]):
            chat_history[-1] = {"role": "assistant", "content": fallback_msg}
        else:
            chat_history.append({"role": "assistant", "content": fallback_msg})

        payload = UserTextPayload(raw=user_msg, choice=None, options=None)
        return (
            chat_history,                    # chatbot
            gr.update(visible=False),        # option_btn_row
            payload,                         # st_user_text_payload
            None,                            # st_chosen_option
            gr.update(interactive=False),    # choose_a_btn
            gr.update(interactive=False),    # choose_b_btn
            gr.update(interactive=True),     # chat_input (re-enable)
            gr.update(interactive=True),     # chat_send_btn (re-enable)
            gr.update(interactive=True),     # sc_btn_1
            gr.update(interactive=True),     # sc_btn_2
            gr.update(interactive=True),     # sc_btn_3
            gr.update(interactive=True),     # sc_btn_4
            gr.update(interactive=True),     # invite_btn
            gr.update(interactive=True),     # check_friend_btn
        )


# ── Choose A / B ──
def on_choose_option(choice_value, chat_history, payload):
    if not choice_value or not payload or not payload.options:
        return chat_history, payload, None, gr.update(), gr.update(), gr.update()

    interp = choice_value
    for opt in payload.options:
        if opt.id == choice_value:
            interp = opt.interpretation
            break

    new_payload = UserTextPayload(
        raw=payload.raw, choice=choice_value, options=payload.options,
    )

    chat_history = list(chat_history or [])

    chat_history.append({
        "role": "user",
        "content": f"Option {choice_value}",
    })
    return (
        chat_history, new_payload, choice_value,
        gr.update(interactive=False),  # choose_a_btn
        gr.update(interactive=False),  # choose_b_btn
        gr.update(interactive=False),  # chat_send_btn
    )


# =============================================================================
# STYLE GOAL DIALOG CALLBACKS
# =============================================================================

def _build_simple_log_html(log: dict) -> str:
    """Render the Style Goal Log panel: user input → summary → friend → A/B → choice."""
    if not log:
        return (
            '<div style="font-size:12px;color:var(--body-text-color-subdued,#aaa);'
            'text-align:center;padding:12px 0;">'
            'Use the \u2726 Style Goal button to describe your outfit idea. '
            'Your goal, summary, and choice will appear here.</div>'
        )
    parts = ['<div style="font-size:13px;line-height:1.6;">']

    # User input
    if log.get("user_input"):
        parts.append(
            f'<div style="margin-bottom:10px;">'
            f'<div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;'
            f'letter-spacing:.04em;margin-bottom:3px;">Your Request</div>'
            f'<div style="background:#f0f4ff;border-radius:8px;padding:8px 12px;">'
            f'{log["user_input"]}</div></div>'
        )

    # AuraWear summary
    if log.get("summary"):
        s = log["summary"]
        rows = []
        if s.get("occasion"):
            rows.append(f'<b>Occasion:</b> {s["occasion"]}')
        if s.get("style_goal"):
            rows.append(f'<b>Style Goal:</b> {s["style_goal"]}')
        if s.get("constraints"):
            rows.append(f'<b>Constraints:</b> {s["constraints"]}')
        if rows:
            parts.append(
                f'<div style="margin-bottom:10px;">'
                f'<div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;'
                f'letter-spacing:.04em;margin-bottom:3px;">AuraWear Summary</div>'
                f'<div style="background:#f0faf8;border-radius:8px;padding:8px 12px;">'
                + '<br>'.join(rows) +
                f'</div></div>'
            )

    # Friend input
    if log.get("friend_suggestion"):
        parts.append(
            f'<div style="margin-bottom:10px;">'
            f'<div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;'
            f'letter-spacing:.04em;margin-bottom:3px;">Friend\'s Suggestion</div>'
            f'<div style="background:#fff8e1;border-radius:8px;padding:8px 12px;">'
            f'\u201c{log["friend_suggestion"]}\u201d</div></div>'
        )

    # A/B options
    if log.get("option_a") or log.get("option_b"):
        opt_parts = []
        for key, label in [("option_a", "A"), ("option_b", "B")]:
            opt = log.get(key)
            if opt:
                interp = opt.get("interpretation", "")
                tags = ", ".join(opt.get("style_tags", []))
                opt_parts.append(
                    f'<div style="flex:1;background:#f7f7f8;border-radius:8px;padding:8px 12px;">'
                    f'<b>{label}.</b> {interp}'
                    + (f'<br><span style="font-size:11px;color:#666;">{tags}</span>' if tags else "")
                    + '</div>'
                )
        parts.append(
            f'<div style="margin-bottom:10px;">'
            f'<div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;'
            f'letter-spacing:.04em;margin-bottom:3px;">Options</div>'
            f'<div style="display:flex;gap:8px;">' + "".join(opt_parts) + '</div></div>'
        )

    # Chosen
    if log.get("chosen"):
        chosen = log["chosen"]
        chosen_opt = log.get(f"option_{chosen.lower()}", {})
        interp = chosen_opt.get("interpretation", "")
        parts.append(
            f'<div style="margin-bottom:6px;">'
            f'<div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;'
            f'letter-spacing:.04em;margin-bottom:3px;">Your Choice</div>'
            f'<div style="background:#e0f2f1;border-radius:8px;padding:8px 12px;">'
            f'\u2714 <b>Direction {chosen}</b>'
            + (f' \u2014 {interp}' if interp else "")
            + '</div></div>'
        )

    parts.append('</div>')
    return "".join(parts)


def _build_session_history_html(session_id, trace_entries, feedback_done, last_items):
    """Render how the system has learned the user's preferences this session.

    Shows:  Preference Signals (liked/disliked items, avoid terms, critique tags, cart)
            Activity Log (reverse-chronological trace timeline)
    """
    state = store.get_or_create(session_id) if session_id else None

    liked_ids    = sorted(state.liked)       if state else []
    disliked_ids = sorted(state.disliked)    if state else []
    carted_ids   = sorted(state.cart)        if state else []
    avoid_terms  = sorted(state.avoid_terms) if state else []

    # Build pretty-name lookup from last_items (item_id → display name)
    id_to_name = {_get(it, "item_id", ""): _pretty_id(_get(it, "item_id", ""))
                  for it in (last_items or []) if _get(it, "item_id")}

    def pname(iid):
        return id_to_name.get(iid, _pretty_id(iid))

    # Task context (for header badge)
    current_task_id   = state.current_task_id if state else None
    archived_task_cnt = len(state.task_pref_vecs) if state else 0
    task_number       = archived_task_cnt + 1 if current_task_id else None

    _EMPTY = (
        '<div style="font-size:12px;color:var(--body-text-color-subdued,#aaa);'
        'text-align:center;padding:16px 0;">'
        'Generate recommendations and give feedback (👍/👎/🛒) to see how the system learns.<br>'
        '<small>Liked/disliked patterns, avoid signals, and your activity log will appear here.</small>'
        '</div>'
    )

    has_signals = liked_ids or disliked_ids or carted_ids or avoid_terms
    has_trace   = bool(trace_entries)
    if not has_signals and not has_trace:
        return _EMPTY

    # Task badge shown at top when a task is active
    task_badge = ""
    if task_number is not None:
        past_note = (
            f' <span style="font-size:10px;color:#999;font-weight:400;">'
            f'(blending {archived_task_cnt} past task{"s" if archived_task_cnt != 1 else ""} @ 30%)</span>'
            if archived_task_cnt > 0 else ""
        )
        task_badge = (
            f'<div style="margin-bottom:10px;background:#e8f4fd;border-radius:8px;'
            f'padding:5px 11px;font-size:11px;color:#1565c0;font-weight:600;">'
            f'🎯 Outfit Goal #{task_number}{past_note}'
            f'</div>'
        )

    parts = [f'<div style="font-size:13px;line-height:1.6;padding:2px 0;">{task_badge}']

    # ── Preference Signals ──────────────────────────────────────────────────
    if has_signals:
        parts.append(
            '<div style="margin-bottom:14px;">'
            '<div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;'
            'letter-spacing:.06em;margin-bottom:8px;">What the System Learned</div>'
        )
        if liked_ids:
            items_html = ", ".join(f'<b>{pname(i)}</b>' for i in liked_ids)
            parts.append(
                f'<div style="margin-bottom:7px;background:#f1f8f1;border-radius:8px;padding:7px 11px;">'
                f'<span style="font-size:11px;color:#2e7d32;font-weight:700;">👍 Liked ({len(liked_ids)})</span>'
                f'<div style="margin-top:3px;font-size:12px;">{items_html}</div></div>'
            )
        if disliked_ids:
            items_html = ", ".join(f'<b>{pname(i)}</b>' for i in disliked_ids)
            parts.append(
                f'<div style="margin-bottom:7px;background:#fdf2f2;border-radius:8px;padding:7px 11px;">'
                f'<span style="font-size:11px;color:#c62828;font-weight:700;">👎 Disliked ({len(disliked_ids)})</span>'
                f'<div style="margin-top:3px;font-size:12px;">{items_html}</div></div>'
            )
        if avoid_terms:
            tags_html = "".join(
                f'<span style="background:#fff3e0;border:1px solid #ffe0b2;border-radius:10px;'
                f'padding:2px 9px;font-size:11px;margin:2px 3px 0 0;display:inline-block;'
                f'color:#e65100;">🚫 {t}</span>'
                for t in avoid_terms
            )
            parts.append(
                f'<div style="margin-bottom:7px;">'
                f'<div style="font-size:11px;color:#888;margin-bottom:4px;">'
                f'Avoid signals learned ({len(avoid_terms)}):</div>'
                f'<div style="line-height:2;">{tags_html}</div></div>'
            )
        if carted_ids:
            items_html = ", ".join(f'<b>{pname(i)}</b>' for i in carted_ids)
            parts.append(
                f'<div style="margin-bottom:7px;background:#fff8e1;border-radius:8px;padding:7px 11px;">'
                f'<span style="font-size:11px;color:#f57f17;font-weight:700;">🛒 Saved ({len(carted_ids)})</span>'
                f'<div style="margin-top:3px;font-size:12px;">{items_html}</div></div>'
            )
        parts.append('</div>')

    # ── Activity Log ─────────────────────────────────────────────────────────
    if has_trace:
        parts.append(
            '<div>'
            '<div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;'
            'letter-spacing:.06em;margin-bottom:8px;">Activity Log</div>'
            '<div style="border-left:2px solid #eeeeee;padding-left:12px;">'
        )
        recent = list(trace_entries)[-12:]
        recent.reverse()  # newest first
        for entry in recent:
            if isinstance(entry, dict):
                icon = entry.get("icon", "") or ""
                text = entry.get("text", "") or ""
                ts   = entry.get("ts")
            else:
                icon = getattr(entry, "icon", "") or ""
                text = getattr(entry, "text", "") or ""
                ts   = getattr(entry, "ts", None)
            t_str = datetime.datetime.fromtimestamp(ts).strftime("%H:%M") if ts else ""
            parts.append(
                f'<div style="margin-bottom:5px;font-size:12px;color:#555;'
                f'display:flex;gap:6px;align-items:flex-start;">'
                f'<span style="color:#bbb;white-space:nowrap;font-size:11px;min-width:36px;">{t_str}</span>'
                f'<span>{icon} {text}</span></div>'
            )
        parts.append('</div></div>')

    parts.append('</div>')
    return "".join(parts)


def _refresh_session_history(session_id, trace_entries, feedback_done, last_items):
    """Standalone callback for `.then()` chains that need to refresh Session History."""
    return _build_session_history_html(session_id, trace_entries, feedback_done, last_items)


def _opt_to_html_card(opt) -> str:
    """Render a single UserTextOption as an HTML card for the dialog."""
    tags = ", ".join(opt.intent_patch.get("style_tags", []))
    must = ", ".join(opt.intent_patch.get("must_have", []))
    avoid_str = ", ".join(opt.intent_patch.get("avoid", []))
    parts = [f"<strong>{opt.id}. {opt.interpretation}</strong>"]
    if tags:
        parts.append(f"<br><span style='font-size:12px;'>style: {tags}</span>")
    if must:
        parts.append(f"<br><span style='font-size:12px;'>must have: {must}</span>")
    if avoid_str:
        parts.append(f"<br><span style='font-size:12px;color:#b00;'>avoid: {avoid_str}</span>")
    return f"<div class='ab-card'>{''.join(parts)}</div>"


def on_dialog_open(dialog_locked):
    """Open the dialog. Blocked when FAB is locked (direction chosen, Recommend pending).
    Always resets the input area to a clean state so the user sees an empty text box.
    """
    _no_change = (
        gr.update(),  # style_dialog — stay hidden
        gr.update(), gr.update(), gr.update(),  # input_section, section_b, section_d
        gr.update(value=""),  # dialog_chat_input
        gr.update(value=""),  # dialog_processing_html
        gr.update(value=""),  # direction_badge
        gr.update(interactive=True), gr.update(interactive=True),  # choose_a, choose_b
        gr.update(value="Recommend Now"),  # dialog_reco_btn
        gr.update(interactive=True), gr.update(interactive=True),  # dialog_send_btn, dialog_new_task_btn
        gr.update(interactive=True), gr.update(interactive=True),
        gr.update(interactive=True), gr.update(interactive=True),  # sc_btns 1-4
        gr.update(value=""),  # dialog_friend_input
    )
    if dialog_locked:
        return _no_change
    return (
        gr.update(visible=True),          # style_dialog
        gr.update(visible=True),          # dialog_input_section (show input area)
        gr.update(visible=False),         # dialog_section_b (hide stale summary)
        gr.update(visible=False),         # dialog_section_d (hide stale A/B)
        gr.update(value=""),              # dialog_chat_input (clear text)
        gr.update(value=""),              # dialog_processing_html (clear spinner)
        gr.update(value=""),              # direction_badge (clear confirm)
        gr.update(interactive=True),      # dialog_choose_a_btn
        gr.update(interactive=True),      # dialog_choose_b_btn
        gr.update(value="Recommend Now"), # dialog_reco_btn
        gr.update(interactive=True),      # dialog_send_btn
        gr.update(interactive=True),      # dialog_new_task_btn
        gr.update(interactive=True),      # dialog_sc_btn_1
        gr.update(interactive=True),      # dialog_sc_btn_2
        gr.update(interactive=True),      # dialog_sc_btn_3
        gr.update(interactive=True),      # dialog_sc_btn_4
        gr.update(value=""),              # dialog_friend_input (clear stale suggestion)
    )


def on_dialog_close():
    """Close the dialog (✕ button) and restore the input section."""
    return (
        gr.update(visible=False),    # style_dialog
        gr.update(visible=True),     # dialog_input_section
        gr.update(interactive=True), # dialog_choose_a_btn
        gr.update(interactive=True), # dialog_choose_b_btn
        gr.update(value="Recommend Now"),  # dialog_reco_btn
        gr.update(value=""),         # direction_badge (clear confirm)
        gr.update(value=""),         # dialog_processing_html (clear spinner)
    )


def _on_new_task_reset(session_id):
    """Phase 1: start a new outfit task.

    - Assigns a new task_id (changes task boundary in backend)
    - Calls reco.start_new_task() → archives old pref_vec at 0.3 decay, clears neg_vecs/avoid_terms
    - Resets Gradio task-level state: trace_entries, feedback_done, session_history_html
    """
    import uuid as _uuid
    new_task_id = f"task_{_uuid.uuid4().hex[:8]}"
    reco.start_new_task(session_id, new_task_id)
    empty_history_html = (
        '<div style="font-size:12px;color:var(--body-text-color-subdued,#aaa);'
        'text-align:center;padding:16px 0;">'
        'New outfit goal started. Give feedback (👍/👎/🛒) to build your preference profile for this task.<br>'
        '<small>Signals from previous tasks are preserved at reduced weight.</small>'
        '</div>'
    )
    return (
        new_task_id,         # st_task_id
        [],                  # st_trace_entries (reset for new task)
        {},                  # st_feedback_done (unlock all items)
        empty_history_html,  # session_history_html
    )


def on_dialog_send_show(user_msg, chat_history, friend_suggestion_raw):
    """Phase 1 of dialog send: instantly show user turn + spinner.
    Captures friend_suggestion_raw into State before clearing the UI field.
    """
    if not user_msg or not user_msg.strip():
        # Show inline warning — keep input section visible, don't popup
        warn = (
            "<div style='text-align:center;padding:12px 0;font-size:13px;color:#e57373;'>"
            "⚠️ Please describe your outfit idea before continuing.</div>"
        )
        return (
            chat_history,                    # chatbot (unchanged)
            gr.update(),                     # dialog_send_btn
            gr.update(), gr.update(), gr.update(), gr.update(),  # sc_btns
            gr.update(),                     # dialog_check_status
            (friend_suggestion_raw or "").strip(),  # st_friend_suggestion
            gr.update(),                     # dialog_section_b
            gr.update(),                     # dialog_section_d
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
            gr.update(value=""), gr.update(value=""),
            gr.update(),                     # dialog_input_section (keep visible)
            gr.update(value=warn),           # dialog_processing_html
        )
    chat_history = list(chat_history or [])
    if chat_history:
        chat_history.append({"role": "assistant", "content": "---"})
    chat_history.append({"role": "user", "content": user_msg.strip()})
    chat_history.append({"role": "assistant", "content": "⏳ *Thinking...*"})
    return (
        chat_history,                                    # chatbot
        gr.update(interactive=False),                    # dialog_send_btn
        gr.update(interactive=False),                    # dialog_sc_btn_1
        gr.update(interactive=False),                    # dialog_sc_btn_2
        gr.update(interactive=False),                    # dialog_sc_btn_3
        gr.update(interactive=False),                    # dialog_sc_btn_4
        gr.update(value=""),                             # dialog_check_status (clear)
        (friend_suggestion_raw or "").strip(),           # st_friend_suggestion (capture)
        gr.update(visible=False),                        # dialog_section_b (hide stale)
        gr.update(visible=False),                        # dialog_section_d
        gr.update(value="⏳ *Interpreting...*"),          # dialog_occasion_md
        gr.update(value=""),                             # dialog_style_goal_md
        gr.update(value=""),                             # dialog_constraints_md
        gr.update(value=""),                             # dialog_option_a_html
        gr.update(value=""),                             # dialog_option_b_html
        gr.update(visible=False),                        # dialog_input_section (hide)
        gr.update(value="<div style='text-align:center;padding:20px 0;font-size:14px;color:#00695c;'>" 
                        "<b>⏳ Interpreting your style goal…</b></div>"),  # dialog_processing_html
    )


def on_dialog_send_llm(
    chat_history, palette, selected_ids, gender, style_hint, friend_suggestion, invite_code,
):
    """Phase 2 of dialog send: run LLM, populate Section B (summary) and C (friend).
    Returns enough to show the summary + Continue button; Section D shown after Continue.
    """
    if not palette:
        gr.Warning("Palette not ready — please complete the color analysis first.")
        return (chat_history, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    if not selected_ids:
        gr.Warning("Please select at least 1 palette color before submitting your style goal.")
        return (chat_history, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    user_msg = ""
    for msg in reversed(chat_history or []):
        if isinstance(msg, dict):
            role = msg.get("role", "")
        else:
            role = getattr(msg, "role", "")
        if role == "user":
            user_msg = _msg_text(msg)
            break
    if not user_msg:
        # Empty input — restore dialog state, clear inline warning
        return (
            chat_history,
            gr.update(), gr.update(), gr.update(),   # payload, chosen, summary
            gr.update(), gr.update(),                # log, memory_html
            gr.update(),                             # dialog_section_b
            gr.update(value=""), gr.update(value=""), gr.update(value=""),
            gr.update(value=""), gr.update(value=""),
            gr.update(interactive=True),             # dialog_send_btn
            gr.update(interactive=True),             # sc_btn_1
            gr.update(interactive=True),             # sc_btn_2
            gr.update(interactive=True),             # sc_btn_3
            gr.update(interactive=True),             # sc_btn_4
            gr.update(interactive=True),             # dialog_invite_btn
            gr.update(interactive=True),             # dialog_check_btn
            gr.update(value=""),                     # dialog_processing_html (clear)
        )

    pal_sel = palette_selected_dicts(palette, selected_ids)

    if invite_code:
        store.update_friend_context_intent(invite_code, user_msg)

    if llm_client is not None:
        summary, options = llm_interpret_options(
            llm=llm_client, user_text_raw=user_msg,
            gender=(gender or ""), style_hint=(style_hint or ""),
            palette_selected=pal_sel,
            friend_suggestion=(friend_suggestion or None),
        )
    else:
        summary = {"occasion": "—", "style_goal": user_msg, "constraints": ""}
        options = None

    # Build chatbot assistant message (mirrors old flow — appended to Session History)
    chat_history = list(chat_history or [])
    if options:
        lines = ["Two interpretations ready — choose one below:\n"]
        for opt in sorted(options, key=lambda x: x.id):
            tags = ", ".join(opt.intent_patch.get("style_tags", []))
            lines.append(f"**{opt.id}. {opt.interpretation}**")
            if tags:
                lines.append(f"  _{tags}_")
            lines.append("")
        chat_msg = "\n".join(lines)
    else:
        chat_msg = f"✅ Got it: *\"{user_msg}\"*"

    if chat_history and "Thinking" in _msg_text(chat_history[-1]):
        # Replace the "thinking" placeholder — but first insert friend msg if any
        if friend_suggestion and friend_suggestion.strip():
            chat_history[-1] = {"role": "user", "content": f"[Friend] {friend_suggestion.strip()}"}
            chat_history.append({"role": "assistant", "content": chat_msg})
        else:
            chat_history[-1] = {"role": "assistant", "content": chat_msg}
    else:
        if friend_suggestion and friend_suggestion.strip():
            chat_history.append({"role": "user", "content": f"[Friend] {friend_suggestion.strip()}"})
        chat_history.append({"role": "assistant", "content": chat_msg})

    payload = UserTextPayload(raw=user_msg, choice=None, options=options)

    # Build Section B HTML cards
    occasion_html = (
        f"<div class='summary-field'><b>Occasion:</b> {summary.get('occasion', '—')}</div>"
    )
    goal_html = (
        f"<div class='summary-field'><b>Style Goal:</b> {summary.get('style_goal', '—')}</div>"
    )
    constraints_val = summary.get("constraints", "")
    constraints_html = (
        f"<div class='summary-field'><b>Constraints:</b> {constraints_val}</div>"
        if constraints_val else ""
    )

    # Section D: A/B option cards
    if options:
        opts_sorted = sorted(options, key=lambda x: x.id)
        html_a = _opt_to_html_card(opts_sorted[0]) if len(opts_sorted) > 0 else ""
        html_b = _opt_to_html_card(opts_sorted[1]) if len(opts_sorted) > 1 else ""
    else:
        html_a = html_b = ""

    # Build partial dialog log (user input + summary + friend)
    log: dict = {"user_input": user_msg, "summary": summary}
    if friend_suggestion and friend_suggestion.strip():
        log["friend_suggestion"] = friend_suggestion.strip()
    if options:
        opts_for_log = sorted(options, key=lambda x: x.id)
        if len(opts_for_log) > 0:
            o = opts_for_log[0]
            log["option_a"] = {"interpretation": o.interpretation,
                               "style_tags": o.intent_patch.get("style_tags", [])}
        if len(opts_for_log) > 1:
            o = opts_for_log[1]
            log["option_b"] = {"interpretation": o.interpretation,
                               "style_tags": o.intent_patch.get("style_tags", [])}

    return (
        chat_history,                                # chatbot
        payload,                                     # st_user_text_payload
        None,                                        # st_chosen_option
        summary,                                     # st_dialog_summary
        log,                                         # st_dialog_log
        _build_simple_log_html(log),                 # session_memory_html
        gr.update(visible=True),                     # dialog_section_b
        gr.update(value=occasion_html),              # dialog_occasion_md
        gr.update(value=goal_html),                  # dialog_style_goal_md
        gr.update(value=constraints_html),           # dialog_constraints_md
        gr.update(value=html_a),                     # dialog_option_a_html
        gr.update(value=html_b),                     # dialog_option_b_html
        gr.update(interactive=True),                 # dialog_send_btn (re-enable)
        gr.update(interactive=True),                 # dialog_sc_btn_1
        gr.update(interactive=True),                 # dialog_sc_btn_2
        gr.update(interactive=True),                 # dialog_sc_btn_3
        gr.update(interactive=True),                 # dialog_sc_btn_4
        gr.update(interactive=True),                 # dialog_invite_btn
        gr.update(interactive=True),                 # dialog_check_btn
        gr.update(value=""),                         # dialog_processing_html (hide spinner)
    )


def on_dialog_continue():
    """Show Section D (A/B choice) after user reviews the summary."""
    return gr.update(visible=True)


def _dialog_choose(choice_value, chat_history, payload, dialog_log, session_id, palette, selected_ids, reco_mode, trace_entries):
    """Shared logic for dialog choose A / choose B.
    Keeps dialog open — disables A/B buttons and updates Recommend Now button.
    """
    results = on_choose_option(choice_value, chat_history, payload)
    new_chat, new_payload, chosen, *_ = results

    trace_entries = _add_trace(trace_entries, "", f"Chose interpretation {choice_value} (dialog)")

    # Build updated dialog log with chosen
    new_log = dict(dialog_log or {})
    new_log["chosen"] = choice_value
    interp = ""
    if payload and payload.options:
        for opt in payload.options:
            if opt.id == choice_value:
                interp = opt.interpretation
                break

    confirm_label = f"✦ Recommend Now — Direction {choice_value}"
    confirm_html = (
        f"<div style='margin:8px 0 4px;font-size:12px;color:#00695c;font-weight:600;"
        f"background:#e0f2f1;border-radius:8px;padding:6px 12px;'>"
        f"✅ Direction {choice_value} chosen — {interp[:60]}</div>"
        if interp else ""
    )

    return (
        new_chat,                         # chatbot
        new_payload,                      # st_user_text_payload
        chosen,                           # st_chosen_option
        new_log,                          # st_dialog_log
        trace_entries,                    # st_trace_entries
        _build_simple_log_html(new_log),  # session_memory_html
        gr.update(value=confirm_html),    # direction_badge (inline in dialog)
        gr.update(interactive=False),     # dialog_choose_a_btn (disable)
        gr.update(interactive=False),     # dialog_choose_b_btn (disable)
        gr.update(value=confirm_label),   # dialog_reco_btn label update
    )


def _dialog_choose_a(chat_history, payload, dialog_log, session_id, palette, selected_ids, reco_mode, trace_entries):
    return _dialog_choose("A", chat_history, payload, dialog_log, session_id, palette, selected_ids, reco_mode, trace_entries)


def _dialog_choose_b(chat_history, payload, dialog_log, session_id, palette, selected_ids, reco_mode, trace_entries):
    return _dialog_choose("B", chat_history, payload, dialog_log, session_id, palette, selected_ids, reco_mode, trace_entries)


# ── Recommend / Regenerate ──
def generate_reco(
    session_id, palette, selected_ids,
    gender, style_hint, user_text_payload, chosen_option,
    reco_mode, trace_entries,
):
    if not session_id:
        raise gr.Error("session_id missing.")
    if not palette:
        gr.Warning("Palette not ready — please go back and complete the color analysis first.")
        return (gr.update(), gr.update(), "⚠️ Palette not ready. Please complete the color analysis first.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    if not selected_ids:
        gr.Warning("Please select at least 1 color from your palette in the sidebar.")
        return (gr.update(), gr.update(), "⚠️ No colors selected. Toggle at least 1 color in the palette sidebar.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    ut = None
    if user_text_payload and user_text_payload.options:
        ut = UserTextPayload(
            raw=user_text_payload.raw,
            choice=chosen_option or user_text_payload.choice,
            options=user_text_payload.options,
        )
    elif user_text_payload and user_text_payload.raw:
        ut = UserTextPayload(raw=user_text_payload.raw, choice=None, options=None)

    req = GenerateRequest(
        session_id=session_id,
        request_id=None,
        palette18=palette,
        selected_palette_ids=selected_ids,
        filters=Filters(
            categories=[],  # no upfront category filter; user filters after
            styles=([style_hint] if style_hint else []),
            gender=(gender or None),
        ),
        k=DEFAULT_K,
        user_text=ut,
        mode=(reco_mode or "full"),
    )

    resp = reco.generate(req)
    cur_mode = reco_mode or "full"
    gallery = []
    for it in resp.items:
        img_path = os.path.join(IMAGES_DIR, it.image_uri)
        if not os.path.exists(img_path):
            img_path = _PLACEHOLDER_IMG
        expl = (it.explanation_text or "")[:60]
        caption = f"{_pretty_id(it.item_id)} · {it.category}" + (f" — {expl}" if expl else "")
        gallery.append((img_path, caption))

    if not gallery:
        status = (
            "No items matched your criteria.\n\n"
            "**Try:** select more palette colors · broaden your style "
            "request · or remove specific constraints"
        )
    else:
        status = ""

    # Log generate event
    log_event("generate", session_id=session_id, payload={
        "request_id": resp.request_id,
        "n_results": len(resp.items),
        "selected_palette_ids": selected_ids,
        "user_text": ut.raw if ut else None,
        "chosen_option": (ut.choice if ut else None),
        "mode": cur_mode,
    })

    # Update trace
    trace_entries = _add_trace(
        trace_entries, "",
        f"Generated {len(resp.items)} items (mode: {cur_mode})"
    )

    # Reset feedback tracking for new batch
    # Compute available categories for filter (with counts)
    cat_counts: dict[str, int] = {}
    for it in resp.items:
        c = it.category or ""
        if c:
            cat_counts[c] = cat_counts.get(c, 0) + 1
    available_cats = sorted(cat_counts.keys())
    cat_choices = [(f"{cat} ({cat_counts[cat]})", cat) for cat in available_cats]

    return (
        resp.request_id, gallery, status, resp.items, {},
        gr.update(choices=cat_choices, value=[]),  # filter_categories
        trace_entries,      # st_trace_entries
        gr.Column(visible=False),  # feedback_panel
        None,               # st_selected_feedback_item
        "",                 # feedback_status
    )


# ── helper: access attr from dict or dataclass (Gradio State may serialise) ──
def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ── Category filter (post-recommendation) ──
def on_category_filter(selected_cats, last_items):
    """Filter already-generated items by category on the gallery.

    NOTE: This handler is also auto-triggered when generate_reco resets
    filter_categories (value=[]).  In that case last_items may still be
    stale or empty — return gr.skip() so the gallery set by generate_reco
    is not overwritten.
    """
    if not last_items:
        # No items yet (first load) or stale state during reco reset
        return gr.skip(), gr.skip()

    # If filter was just reset to [] by generate_reco, don't touch gallery
    if not selected_cats:
        return gr.skip(), f"Showing all {len(last_items)} items"

    def _to_gallery(items_subset):
        g = []
        for it in items_subset:
            img_path = os.path.join(IMAGES_DIR, _get(it, "image_uri", ""))
            if not os.path.exists(img_path):
                img_path = _PLACEHOLDER_IMG
            caption = f"{_pretty_id(_get(it, 'item_id', ''))} · {_get(it, 'category', '')}"
            g.append((img_path, caption))
        return g

    filtered = [it for it in last_items if (_get(it, "category") or "") in selected_cats]
    gallery = _to_gallery(filtered)
    if not gallery:
        # Return ALL items instead of empty so user doesn't get stuck
        all_gallery = _to_gallery(last_items)
        return all_gallery, f"No items match: {', '.join(selected_cats)}. Showing all {len(all_gallery)} items instead."
    return gallery, f"Showing {len(gallery)} / {len(last_items)} items (filter: {', '.join(selected_cats)})"


# ── Gallery select → show feedback panel ──
def on_gallery_select(evt: gr.SelectData, last_items, feedback_done, reco_mode, session_id):
    _empty = (
        None, "", gr.Column(visible=False), gr.Column(visible=False),
        gr.skip(), gr.skip(), gr.skip(), "",
    )
    if not last_items:
        return _empty

    # Extract display name from gallery caption, then map to real item_id
    item_id = None
    cap = getattr(evt, "value", None)
    if isinstance(cap, dict):
        cap = cap.get("caption", "") or cap.get("label", "")
    if isinstance(cap, (list, tuple)):
        cap = cap[1] if len(cap) > 1 else str(cap[0]) if cap else ""
    cap = str(cap) if cap else ""
    if "·" in cap or "|" in cap:
        sep = "·" if "·" in cap else "|"
        display_name = cap.split(sep)[0].strip()
        # Map pretty display name back to real item_id
        for it in (last_items or []):
            if _pretty_id(_get(it, "item_id", "")) == display_name:
                item_id = _get(it, "item_id", "")
                break
        # Fallback: maybe caption still has the raw item_id
        if not item_id:
            item_id = display_name

    # Fallback: use index into last_items
    if not item_id:
        idx = evt.index
        if isinstance(idx, (list, tuple)):
            idx = idx[0] if idx else 0
        try:
            idx = int(idx)
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(last_items) - 1))
        item_id = _get(last_items[idx], "item_id", "")

    if not item_id:
        return _empty

    # Find item object by item_id
    item = None
    for it in last_items:
        if _get(it, "item_id") == item_id:
            item = it
            break
    if item is None:
        return _empty

    category = _get(item, "category") or ""
    score = _get(item, "score", 0)
    explanation = _get(item, "explanation_text", "")
    debug = _get(item, "debug", {}) or {}
    pretty = _pretty_id(item_id)
    mode = reco_mode or "full"
    dominant = compute_dominant_signal(debug, mode)

    # Get avoid terms from session state
    state = store.get_or_create(session_id) if session_id else None
    avoid_terms = sorted(state.avoid_terms) if state else []

    info = render_item_detail_html(
        item_id=item_id,
        category=category,
        score=score,
        explanation=explanation,
        debug=debug,
        dominant=dominant,
        avoid_terms=avoid_terms,
        mode=mode,
        pretty_name=pretty,
    )

    # Check what actions were already taken
    done = feedback_done or {}
    liked = f"{item_id}:like" in done
    disliked = f"{item_id}:dislike" in done
    carted = f"{item_id}:cart" in done
    any_done = liked or disliked or carted
    done_actions = []
    if liked:
        done_actions.append("👍 Liked")
    if disliked:
        done_actions.append("👎 Disliked")
    if carted:
        done_actions.append("🛒 In cart")
    if done_actions:
        info += f'\n\n*Already: {", ".join(done_actions)}*'

    return (
        item_id, info,
        gr.Column(visible=True),   # feedback_panel
        gr.Column(visible=False),  # dislike_modal
        gr.Button(interactive=not any_done),  # like_btn
        gr.Button(interactive=not any_done),  # dislike_btn
        gr.Button(interactive=not any_done),  # cart_btn
        "",  # clear feedback_status
    )


# ── Like ──
def on_like(session_id, request_id, item_id, feedback_done,
            trace_entries, palette, selected_ids, user_text_payload, chosen_option, reco_mode,
            last_items=None):
    _no_change = gr.skip()
    if not item_id or not request_id:
        return ("⚠️ Select an item first.", gr.Column(visible=False), feedback_done,
                _no_change, _no_change, _no_change, trace_entries, _no_change)
    done = feedback_done or {}
    key = f"{item_id}:like"
    pretty = _pretty_id(item_id)
    if key in done:
        return (f"ℹ️ Already liked {pretty}.", gr.Column(visible=False), done,
                gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False),
                trace_entries, _no_change)
    ev = FeedbackEvent(
        session_id=session_id, request_id=request_id,
        item_id=item_id, action="like",
    )
    reco.feedback(ev)
    done[key] = True
    log_event("feedback", session_id=session_id, payload={
        "action": "like", "item_id": item_id, "request_id": request_id,
    })
    trace_entries = _add_trace(trace_entries, "👍", f"Liked {pretty}")
    return (f"👍 Liked **{pretty}** — we'll find more like this.",
            gr.Column(visible=False), done,
            gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False),
            trace_entries,
            _build_session_history_html(session_id, trace_entries, done, last_items))


# ── Dislike click → show modal (instantly, no LLM call here) ──
def on_dislike_click(session_id, request_id, item_id, feedback_done):
    """Show the critique modal but do NOT fire backend feedback yet."""
    _no_change = gr.skip()
    if not item_id or not request_id:
        return "⚠️ Select an item first.", gr.Column(visible=False), feedback_done, _no_change, _no_change, _no_change
    done = feedback_done or {}
    key = f"{item_id}:dislike"
    pretty = _pretty_id(item_id)
    if key in done:
        return (f"ℹ️ Already disliked {pretty}.", gr.Column(visible=False), done,
                gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False))
    # Show critique modal — do NOT fire reco.feedback yet (moved to submit)
    return (
        f"👎 Tell us what didn't work about **{pretty}**:",
        gr.Column(visible=True),
        feedback_done,
        gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False),
    )


# ── Dislike submit feedback (now also fires reco.feedback) ──
def on_dislike_submit(session_id, request_id, item_id, chips, text_fb,
                      feedback_done, trace_entries, palette, selected_ids,
                      user_text_payload, chosen_option, reco_mode,
                      last_items=None):
    """Fire backend dislike feedback with unified critique chips + optional free text."""
    done = dict(feedback_done or {})
    key = f"{item_id}:dislike"
    done[key] = True

    critique_tags = list(chips or [])

    ev = FeedbackEvent(
        session_id=session_id, request_id=request_id,
        item_id=item_id, action="dislike",
    )
    result = reco.feedback(
        ev, critique_tags=critique_tags,
        free_text=(text_fb or "").strip(),
    )
    new_avoid = result.get("new_avoid_terms", []) if isinstance(result, dict) else []

    log_event("feedback", session_id=session_id, payload={
        "action": "dislike", "item_id": item_id, "request_id": request_id,
        "critique_tags": critique_tags,
        "free_text": (text_fb or "").strip(),
        "new_avoid_terms": new_avoid,
    })

    pretty = _pretty_id(item_id)
    parts = []
    if critique_tags:
        parts.append(f"Tags: {', '.join(critique_tags)}")
    if text_fb and text_fb.strip():
        parts.append(f"Comment: {text_fb.strip()}")
    fb_str = " | ".join(parts) if parts else "No additional feedback"
    print(f"[DISLIKE_DETAIL] item={item_id} | {fb_str}")

    # Build trace entries
    trace_entries = _add_trace(trace_entries, "👎", f"Disliked {pretty}")
    if new_avoid:
        trace_entries = _add_trace(trace_entries, "",
                                   f"Avoid terms added: {', '.join(new_avoid)}")
        trace_entries = _add_trace(trace_entries, "",
                                   "Reranker will penalize items matching those traits")
    if critique_tags:
        trace_entries = _add_trace(trace_entries, "",
                                   f"Critiques applied: {', '.join(critique_tags)}")

    # Build critique impact message
    impact_msg = ""
    if new_avoid:
        impact_msg = f"\n\n **Critique impact:** Added avoid terms: *{', '.join(new_avoid)}*. "
        if critique_tags:
            example = critique_tags[0]
            impact_msg += f'Because you said "{example}", we\'ll lower items matching that trait.'
    elif critique_tags:
        impact_msg = f"\n\n **Critique applied:** {', '.join(critique_tags)} will penalize matching items."

    return (
        f"👎 Noted for **{pretty}**.  \n*{fb_str}*{impact_msg}",
        gr.Column(visible=False),          # dislike_modal
        gr.Column(visible=False),          # feedback_panel
        gr.CheckboxGroup(value=[]),        # clear dislike_chips
        "",                                # clear text
        done,                              # st_feedback_done
        trace_entries,                     # st_trace_entries
        _build_session_history_html(session_id, trace_entries, done, last_items),
    )


# ── Dislike close ──
def on_dislike_close():
    return (
        gr.Column(visible=False),          # dislike_modal
        gr.Column(visible=False),          # feedback_panel
        gr.CheckboxGroup(value=[]),        # clear dislike_chips
        "",                                # clear text
    )


def _cart_to_gallery(cart_items):
    """Convert cart items to gallery format [(path, caption), ...]."""
    return [
        (c.get("image_path", ""), f'{_pretty_id(c.get("item_id", ""))} · {c.get("category", "")}')
        for c in (cart_items or [])
    ]


# ── Add to cart ──
def on_cart(session_id, request_id, item_id, last_items, cart_items, feedback_done,
            trace_entries, palette, selected_ids, user_text_payload, chosen_option, reco_mode):
    _no_change = gr.skip()
    if not item_id or not request_id:
        return (
            "⚠️ Select an item first.",
            gr.Column(visible=False),
            cart_items,
            _cart_to_gallery(cart_items),
            feedback_done,
            _no_change, _no_change, _no_change,
            gr.skip(),  # cart_heading
            trace_entries, _no_change,  # trace, memory
        )
    done = feedback_done or {}
    key = f"{item_id}:cart"
    pretty = _pretty_id(item_id)
    if key in done:
        return (
            f"ℹ️ {pretty} is already in your cart.",
            gr.Column(visible=False),
            cart_items,
            _cart_to_gallery(cart_items),
            done,
            gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False),
            gr.skip(),  # cart_heading
            trace_entries, _no_change,  # trace, memory
        )
    ev = FeedbackEvent(
        session_id=session_id, request_id=request_id,
        item_id=item_id, action="cart",
    )
    reco.feedback(ev)
    done[key] = True
    log_event("feedback", session_id=session_id, payload={
        "action": "cart", "item_id": item_id, "request_id": request_id,
    })

    cart_items = list(cart_items or [])
    for it in last_items:
        if _get(it, "item_id") == item_id:
            _cart_img = os.path.join(IMAGES_DIR, _get(it, "image_uri", ""))
            if not os.path.exists(_cart_img):
                _cart_img = _PLACEHOLDER_IMG
            cart_items.append({
                "item_id": item_id,
                "image_path": _cart_img,
                "category": _get(it, "category") or "",
            })
            break

    n = len(cart_items)
    trace_entries = _add_trace(trace_entries, "🛒", f"Added {pretty} to cart")
    return (
        f"🛒 Added **{pretty}** to cart — nice pick!",
        gr.Column(visible=False),
        cart_items,
        _cart_to_gallery(cart_items),
        done,
        gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False),
        f"---\n### Your Cart ({n})",  # cart_heading
        trace_entries,               # st_trace_entries
        _build_session_history_html(session_id, trace_entries, done, last_items),
    )


# =============================================================================
# UI — 5-Step Wizard
# =============================================================================
_APP_JS = """
() => {
    document.documentElement.lang = 'en';
}
"""

_APP_CSS = """
/* Compact CheckboxGroup: horizontal wrap, smaller text */
.compact-cb .wrap { display:flex; flex-wrap:wrap; gap:4px 8px; }
.compact-cb label { font-size:12px !important; padding:2px 6px !important;
    white-space:nowrap; min-height:auto !important; }
.compact-cb input[type="checkbox"] { width:14px; height:14px; }
.compact-cb .container > label.block { font-size:11px !important; margin-bottom:2px; }

/* Style Intent chatbot: unified 12px font */
.style-advisor-chat .message,
.style-advisor-chat .message p,
.style-advisor-chat .message li,
.style-advisor-chat .message strong,
.style-advisor-chat .message em,
.style-advisor-chat .message span { font-size:12px !important; line-height:1.5 !important; }
.style-advisor-chat .message { padding:5px 10px !important; }
.style-advisor-chat .message p { margin:2px 0 !important; }
.style-advisor-chat .message ul,
.style-advisor-chat .message ol { margin:2px 0 !important; padding-left:16px !important; }
.style-advisor-chat .message li { margin:1px 0 !important; }
.style-advisor-chat .message hr { margin:4px 0 !important; border-color:rgba(128,128,128,.2); }
.style-advisor-chat .message h1,
.style-advisor-chat .message h2,
.style-advisor-chat .message h3 { font-size:12px !important; margin:2px 0 !important; }

/* Feedback panel: compact & prominent */
.feedback-card { background:var(--background-fill-secondary, #f7f7f8);
    border-radius:10px; padding:10px 14px !important; }
.feedback-card .prose { font-size:13px !important; }

/* Gallery caption: smaller text */
.gallery-item .caption { font-size:10px !important; }

/* Scenario chip buttons inside dialog */
.scenario-chip-row { gap: 6px !important; margin: 2px 0 !important; }
.scenario-chip { font-size: 12px !important; border-radius: 16px !important;
    padding: 3px 10px !important; min-height: 30px !important;
    white-space: nowrap !important; }

/* A/B choice row: compact pill-style buttons */
.option-choice-row { margin: 4px 0 !important; gap: 6px !important; }
.option-choice-row button {
    border-radius: 20px !important; font-size: 13px !important;
    font-weight: 700 !important; min-height: 32px !important;
    padding: 4px 20px !important;
}

/* Recommend button pulse hint */
@keyframes pulse-border { 0%,100%{box-shadow:0 0 0 0 rgba(0,150,136,.4)} 50%{box-shadow:0 0 0 6px rgba(0,150,136,0)} }
.reco-btn { animation: pulse-border 2.5s ease-in-out infinite; }

/* Style profile panel */
.session-memory-panel { max-height:400px; overflow-y:auto; }

/* ── FAB button (Style Goal floating button) ── */
#style-goal-fab {
    position: fixed !important;
    bottom: 28px !important;
    right: 28px !important;
    z-index: 900 !important;
    border-radius: 50% !important;
    width: 52px !important;
    height: 52px !important;
    min-width: 52px !important;
    min-height: 52px !important;
    max-width: 52px !important;
    padding: 0 !important;
    box-shadow: 0 4px 18px rgba(0,150,136,.35) !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    line-height: 52px !important;
    text-align: center !important;
}

/* ── Dialog close button (circular) ── */
#dialog-close-btn {
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    min-height: 28px !important;
    max-width: 28px !important;
    border-radius: 50% !important;
    padding: 0 !important;
    font-size: 13px !important;
    line-height: 28px !important;
    text-align: center !important;
    flex-shrink: 0 !important;
}
/* ── Style Goal Dialog overlay ── */
#style-dialog-overlay {
    position: fixed !important;
    top: 0 !important; left: 0 !important;
    width: 100vw !important; height: 100vh !important;
    background: rgba(0,0,0,.52) !important;
    z-index: 1000 !important;
    display: flex;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important; padding: 0 !important;
    gap: 0 !important;
}
/* When Gradio hides the overlay (visible=False), the inline style="display:none" wins
   over the CSS display:flex above — no extra rule needed. */
/* Inner dialog card */
#style-dialog-card {
    background: var(--background-fill-primary, #fff) !important;
    border-radius: 18px !important;
    max-width: 860px !important;
    width: 92vw !important;
    max-height: 90vh !important;
    overflow-y: auto !important;
    padding: 28px 32px !important;
    box-shadow: 0 8px 40px rgba(0,0,0,.22) !important;
    margin: 0 !important;
    flex-shrink: 0 !important;
}
/* Section headers inside dialog */
.dialog-section-header {
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: .04em !important;
    color: var(--body-text-color-subdued, #888) !important;
    text-transform: uppercase !important;
    margin: 16px 0 6px !important;
}
/* Summary field cards in Section B */
.summary-field {
    background: var(--background-fill-secondary, #f7f7f8) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
    font-size: 13px !important;
}
/* A/B option cards in Section D */
.ab-card-row { gap: 12px !important; }
.ab-card {
    background: var(--background-fill-secondary, #f7f7f8) !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    font-size: 13px !important;
    line-height: 1.55 !important;
    flex: 1 !important;
}
"""

with gr.Blocks(title="AuraWear — Personal Color-Aware Fashion Recommender") as demo:

    # ── States ──
    st_session_id = gr.State(on_init_session())
    st_selfie_path = gr.State(None)
    st_gender = gr.State("")
    st_style_hint = gr.State("")
    st_palette = gr.State([])
    st_analysis = gr.State({})
    st_selected_ids = gr.State([])
    st_selected_ids_step4 = gr.State([])
    st_last_items = gr.State([])
    st_request_id = gr.State("")
    st_user_text_payload = gr.State(None)
    st_chosen_option = gr.State(None)
    st_cart = gr.State([])
    st_selected_feedback_item = gr.State(None)
    st_feedback_done = gr.State({})
    st_reco_mode = gr.State("full")
    st_trace_entries = gr.State([])
    st_task_id = gr.State(None)           # current task UUID; changes on "Start New Outfit Goal"
    st_friend_suggestion = gr.State("")   # captures friend_input before it's cleared
    st_dialog_locked = gr.State(False)    # FAB locked after A/B selection until Recommend
    st_dialog_summary = gr.State({})      # structured summary from LLM (occasion/style_goal/constraints)
    st_dialog_log = gr.State({})          # simplified conversation log for Style Goal Log panel
    st_invite_code = gr.State("")         # active invite code for friend mode

    # ================================================================
    # PAGE 1: Upload Selfie
    # ================================================================
    with gr.Column(visible=True) as page1:
        gr.HTML(_step_indicator_html(1))
        gr.Markdown(
            "# AuraWear — Personal Color-Aware Fashion Recommender\n"
            "### Step 1 — Upload Your Selfie\n"
            "Take or upload a clear, well-lit selfie. We'll analyze your skin tone, "
            "hair color, and eye color to build your personal color profile."
        )
        selfie_input = gr.Image(label="Upload Selfie", type="filepath", height=400)
        with gr.Row():
            step1_next = gr.Button("Next →", variant="primary", size="lg")
            demo_btn = gr.Button(
                "Try Demo", variant="secondary", size="lg",
                visible=os.path.isfile(DEMO_SELFIE_PATH),
            )
        gr.Markdown(
            "<div style='text-align:center;margin-top:8px;"
            "color:var(--body-text-color-subdued,#aaa);font-size:12px;'>"
            "A friend shared a code with you? </div>"
        )
        friend_mode_entry_btn = gr.Button(
            "Open Friend Mode", variant="secondary", size="sm",
        )

    # ================================================================
    # PAGE 2: Select Styles + Gender
    # ================================================================
    with gr.Column(visible=False) as page2:
        gr.HTML(_step_indicator_html(2))
        gr.Markdown(
            "### Step 2 — Choose Your Style Preferences\n"
            "Select one or more styles that match your taste."
        )
        style_select = gr.CheckboxGroup(
            choices=STYLE_OPTIONS,
            label="Style Preferences",
            value=[],
        )
        gender_select = gr.Radio(
            choices=["Male", "Female", "Non-binary", "Prefer not to say"],
            label="Gender",
            value="Prefer not to say",
        )
        with gr.Row():
            step2_back = gr.Button("← Back")
            step2_next = gr.Button(
                "Analyze My Colors →", variant="primary", size="lg",
            )

    # ================================================================
    # PAGE 3: Analyzing (Loading)
    # ================================================================
    with gr.Column(visible=False) as page3:
        gr.HTML(_step_indicator_html(3))
        gr.Markdown(
            "### Step 3 — Analyzing Your Colors\n\n"
            "Please wait while we analyze your skin tone, hair color, "
            "and eye color to determine your seasonal color type...\n\n"
            "This may take a few seconds."
        )

    # ================================================================
    # PAGE 4: Palette Selection
    # ================================================================
    with gr.Column(visible=False) as page4:
        gr.HTML(_step_indicator_html(4))
        gr.Markdown("### Step 4 — Select Your Colors")
        step4_season_text = gr.Markdown("")
        gr.Markdown(
            "Choose the colors you'd like to see in your outfit recommendations "
            "(select at least 1):"
        )
        step4_palette_html = gr.HTML("")
        step4_palette_cb = gr.CheckboxGroup(
            choices=[], label="Toggle Colors", value=[],
            elem_classes=["compact-cb"],
        )
        with gr.Row():
            step4_back = gr.Button("← Back to Styles")
            step4_next = gr.Button(
                "Go to Recommendations →", variant="primary", size="lg",
            )

    # ================================================================
    # PAGE 5: HITL Recommendation Page
    # ================================================================
    with gr.Column(visible=False) as page5:
        gr.HTML(_step_indicator_html(5))
        gr.Markdown(
            "### Step 5 — Interactive Recommendation & Feedback\n"
            "<small style='color:var(--body-text-color-subdued,#888);'>Tell us the vibe you're going for, "
            "browse recommendations, and give feedback to refine your results.</small>"
        )

        with gr.Row():
            # ── LEFT SIDEBAR (1/3) ──
            with gr.Column(scale=1, min_width=300):
                sidebar_selfie = gr.Image(
                    label="Your Selfie", height=180, interactive=False,
                )
                sidebar_analysis = gr.HTML("")

                gr.Markdown("**Palette**")
                sidebar_palette_html = gr.HTML("")
                sidebar_palette_cb = gr.CheckboxGroup(
                    choices=[], label="Toggle Colors", value=[],
                    elem_classes=["compact-cb"],
                )

                # chatbot: kept hidden — still used internally for dialog wiring
                chatbot = gr.Chatbot(
                    visible=False,
                    value=_initial_chat_history(),
                )

                gr.Markdown("---")
                recommend_btn = gr.Button(
                    "Recommend", variant="primary",
                    elem_classes=["reco-btn"],
                )
                gr.Markdown(
                    "<small style='color:var(--body-text-color-subdued,#888);'>\n"
                    "Tip: Give feedback (👍/👎/🛒) then click again for better results.</small>",
                )

                with gr.Accordion("Advanced: Recommendation Mode", open=False):
                    gr.Markdown(
                        "<small style='color:var(--body-text-color-subdued,#888);'>"
                        "Compare how different ranking strategies affect your results</small>"
                    )
                    mode_select = gr.Radio(
                        choices=RECO_MODE_CHOICES,
                        value="full",
                        label="Active Mode",
                        info="Switch modes to compare recommender behavior",
                    )
                    mode_desc_display = gr.Markdown(
                        f"<small>{_MODE_DESCRIPTIONS['full']}</small>"
                    )

                with gr.Accordion("How recommendations work", open=False):
                    gr.Markdown(
                        "We rank items by combining several factors:\n\n"
                        f"| What we consider | Importance |\n"
                        f"|---|---|\n"
                        f"| How well colors match your palette | {reco.cfg.w_color:.0%} |\n"
                        f"| How well it fits what you described | {reco.cfg.w_intent:.0%} |\n"
                        f"| How similar to items you've liked | {reco.cfg.w_pref:.0%} |\n"
                        f"| Avoiding things you disliked | {reco.cfg.neg_penalty_weight:.0%} |\n"
                        f"| Keeping results diverse | {reco.cfg.w_dup:.0%} |\n\n"
                        f"*We pick from the top {reco.cfg.candidate_pool_size} color-matched items, "
                        f"then refine to show you the best {reco.cfg.top_k_default}.*"
                    )

            # ── RIGHT MAIN AREA (2/3) ──
            with gr.Column(scale=2):
                # Session History — system learning panel (above Style Goal Log)
                with gr.Accordion("Session History", open=False, elem_id="session-history-accordion"):
                    session_history_html = gr.HTML(
                        '<div style="font-size:12px;color:var(--body-text-color-subdued,#aaa);'
                        'text-align:center;padding:16px 0;">'
                        'Generate recommendations and give feedback (👍/👎/🛒) to see how the system learns.<br>'
                        '<small>Liked/disliked patterns, avoid signals, and your activity log will appear here.</small>'
                        '</div>'
                    )

                # Style Goal Log panel
                with gr.Accordion("Style Goal Log", open=False):
                    session_memory_html = gr.HTML(
                        '<div style="font-size:12px;color:var(--body-text-color-subdued,#aaa);'
                        'text-align:center;padding:12px 0;">'
                        'Use the ✦ Style Goal button to describe your outfit idea. '
                        'Your goal, summary, and choice will appear here.</div>'
                    )

                reco_status = gr.Markdown(
                    "<div style='text-align:center; color:var(--body-text-color-subdued,#999); "
                    "padding:8px 0; font-size:13px;'>Describe the look you want, or click "
                    "<b>Recommend</b> to generate results</div>"
                )
                filter_categories = gr.CheckboxGroup(
                    choices=CATEGORY_OPTIONS,
                    label="Filter by Category",
                    value=[],
                    info="Filter results after generation (leave empty = show all)",
                )
                reco_gallery = gr.Gallery(
                    label="Recommendations", columns=4,
                    height=420, preview=False,
                )

                # Feedback panel (hidden until item selected)
                with gr.Column(visible=False, elem_classes=["feedback-card"]) as feedback_panel:
                    feedback_close_btn = gr.Button(
                        "← Back to Gallery", variant="secondary", size="sm",
                    )
                    feedback_info = gr.Markdown("")
                    with gr.Row():
                        like_btn = gr.Button("👍 Like", variant="secondary", size="sm")
                        dislike_btn = gr.Button("👎 Dislike", variant="stop", size="sm")
                        cart_btn = gr.Button("🛒 Add to Cart", variant="primary", size="sm")

                # Dislike detail modal (hidden until dislike clicked)
                with gr.Column(visible=False) as dislike_modal:
                    gr.Markdown(
                        "**What didn't you like?** The more you tell us, the better we can filter."
                    )
                    dislike_chips = gr.CheckboxGroup(
                        choices=DISLIKE_CHIPS,
                        label="What felt off?",
                        elem_classes=["compact-cb"],
                    )
                    dislike_text = gr.Textbox(
                        label="Anything else? (optional)",
                        placeholder="e.g. The silhouette is too boxy, I'd prefer something more fitted...",
                    )
                    with gr.Row():
                        dislike_submit = gr.Button(
                            "Submit Critique", variant="primary",
                        )
                        dislike_close = gr.Button("Skip / Close")

                feedback_status = gr.Markdown("")

                # Cart area
                cart_heading = gr.Markdown("---\n### Your Cart")
                cart_gallery = gr.Gallery(
                    label="Cart", columns=8, height=110,
                    preview=False, interactive=False,
                    object_fit="cover",
                )

                gr.Markdown("---")
                reset_btn = gr.Button("New Session", variant="secondary", size="sm")

    # ================================================================
    # PAGE FRIEND: Friend Mode (code-based async suggestion)
    # ================================================================
    with gr.Column(visible=False) as page_friend:
        gr.Markdown(
            "## AuraWear — Friend Mode\n"
            "A friend shared their style session with you. "
            "Enter their invite code below to see their context and send a suggestion."
        )
        with gr.Row():
            friend_code_input = gr.Textbox(
                label="Invite Code",
                placeholder="Enter 6-character code (e.g. A3BZ7K)",
                max_lines=1,
                scale=3,
            )
            friend_lookup_btn = gr.Button("Look Up", variant="primary", scale=1)
        friend_ctx_display = gr.Markdown(
            "<div style='color:var(--body-text-color-subdued,#aaa);'>"
            "Enter the invite code above to see your friend's context.</div>"
        )
        with gr.Column(visible=False) as friend_reply_area:
            friend_suggestion_box = gr.Textbox(
                label="Your suggestion",
                placeholder="e.g. 'Something bolder — try a statement piece or brighter color'",
                lines=3,
            )
            friend_submit_btn = gr.Button("Send Suggestion", variant="primary")
            friend_submit_status = gr.Markdown("")
        with gr.Row():
            friend_back_btn = gr.Button("← Back to AuraWear", variant="secondary")

    # ================================================================
    # STYLE GOAL DIALOG — FAB + overlay (outside all pages)
    # ================================================================

    # FAB & direction badge (fixed position via CSS; hidden until Step 5)
    style_goal_fab_btn = gr.Button(
        "✦",
        elem_id="style-goal-fab",
        variant="primary",
        interactive=True,
        visible=False,
    )
    # Dialog overlay column (position:fixed via CSS, initially hidden)
    with gr.Column(visible=False, elem_id="style-dialog-overlay") as style_dialog:
        with gr.Column(elem_id="style-dialog-card"):
            # ── Header row ──
            with gr.Row(elem_classes=["dialog-header-row"]):
                gr.Markdown("### Describe Your Style Goal")
                dialog_close_btn = gr.Button("✕", elem_id="dialog-close-btn", scale=0)

            # ── Section A + C — input, chips, friend invite (hidden after Interpret) ──
            with gr.Column() as dialog_input_section:
                gr.HTML("<div class='dialog-section-header'>What are you dressing for?</div>")
                dialog_chat_input = gr.Textbox(
                    placeholder="e.g. Relaxed weekend brunch — flowy and effortless",
                    label="Your outfit idea",
                    lines=2,
                    show_label=False,
                    elem_id="dialog_chat_input",
                )
                with gr.Row(elem_classes=["scenario-chip-row"]):
                    dialog_sc_btn_1 = gr.Button("Polished office meeting", size="sm", elem_classes=["scenario-chip"])
                    dialog_sc_btn_2 = gr.Button("Casual weekend brunch",   size="sm", elem_classes=["scenario-chip"])
                with gr.Row(elem_classes=["scenario-chip-row"]):
                    dialog_sc_btn_3 = gr.Button("Elegant date night",      size="sm", elem_classes=["scenario-chip"])
                    dialog_sc_btn_4 = gr.Button("Cozy and relaxed at home", size="sm", elem_classes=["scenario-chip"])

                # ── Section C — friend invite (always visible inside input section) ──
                gr.HTML("<hr style='margin:10px 0;border-color:rgba(128,128,128,.15);'>")
                gr.HTML("<div class='dialog-section-header'>Optional: Include a Friend's Opinion</div>")
                dialog_friend_input = gr.Textbox(
                    placeholder="Friend's suggestion will appear here after you Check",
                    label="Friend's suggestion (shapes option B)",
                    lines=1,
                    show_label=False,
                    elem_id="dialog_friend_input",
                    interactive=False,
                )
                with gr.Row():
                    dialog_invite_btn = gr.Button(
                        "Invite Friend", variant="secondary", size="sm", scale=1,
                    )
                    dialog_check_btn = gr.Button(
                        "Check Friend Input", variant="secondary", size="sm", scale=1,
                        visible=False,
                    )
                dialog_invite_status = gr.Markdown("")
                dialog_check_status  = gr.Markdown("")

                gr.HTML("<hr style='margin:10px 0;border-color:rgba(128,128,128,.15);'>")
                gr.HTML(
                    '<div style="font-size:11px;color:var(--body-text-color-subdued,#999);'
                    'line-height:1.6;margin-bottom:8px;">'
                    '<b>Refine</b> keeps your current task context, A/B history, and feedback signals.<br>'
                    '<b>New Outfit Goal</b> resets task-specific signals (dislikes, avoid terms) '
                    'and starts fresh — your broader style taste is preserved at reduced weight.'
                    '</div>'
                )
                with gr.Row():
                    dialog_send_btn = gr.Button("Refine Current Goal →", variant="secondary", scale=1)
                    dialog_new_task_btn = gr.Button("↺ Start New Outfit Goal", variant="primary", scale=1)

            # ── Processing spinner (shown while LLM runs) ──
            dialog_processing_html = gr.HTML("", elem_id="dialog-processing")

            # ── Section B — summary (hidden until LLM returns) ──
            with gr.Column(visible=False) as dialog_section_b:
                gr.HTML("<hr style='margin:12px 0;border-color:rgba(128,128,128,.15);'>")
                gr.HTML("<div class='dialog-section-header'>Your Style Summary</div>")
                dialog_occasion_md    = gr.HTML("")
                dialog_style_goal_md  = gr.HTML("")
                dialog_constraints_md = gr.HTML("")
                dialog_continue_btn   = gr.Button("Continue →", variant="secondary")

            # ── Section D — A/B options (hidden until Continue clicked) ──
            with gr.Column(visible=False) as dialog_section_d:
                gr.HTML("<hr style='margin:12px 0;border-color:rgba(128,128,128,.15);'>")
                gr.HTML("<div class='dialog-section-header'>Choose Your Direction</div>")
                with gr.Row(elem_classes=["ab-card-row"]):
                    dialog_option_a_html = gr.HTML("")
                    dialog_option_b_html = gr.HTML("")
                with gr.Row(elem_classes=["option-choice-row"]):
                    dialog_choose_a_btn = gr.Button("Choose A →", variant="secondary", scale=1)
                    dialog_choose_b_btn = gr.Button("Choose B →", variant="secondary", scale=1)
                direction_badge = gr.HTML("", elem_id="dialog-direction-confirm")
                dialog_reco_btn = gr.Button("Recommend Now", variant="primary")

    # Legacy A/B row (hidden; kept for wiring compatibility — replaced by dialog)
    with gr.Row(visible=False, elem_classes=["option-choice-row"]) as option_btn_row:
        choose_a_btn = gr.Button("Choose A", variant="secondary", scale=1)
        choose_b_btn = gr.Button("Choose B", variant="secondary", scale=1)

    # Legacy chat_input + chips + send (hidden; kept for wiring compatibility)
    friend_input   = gr.Textbox(visible=False, elem_id="friend_input_legacy")
    chat_input     = gr.Textbox(visible=False, elem_id="style_chat_input_legacy")
    with gr.Row(visible=False):
        sc_btn_1 = gr.Button("Polished office meeting", size="sm")
        sc_btn_2 = gr.Button("Casual weekend brunch",   size="sm")
        sc_btn_3 = gr.Button("Elegant date night",      size="sm")
        sc_btn_4 = gr.Button("Cozy and relaxed at home", size="sm")
    chat_send_btn = gr.Button("Send (legacy)", visible=False)
    # Legacy invite controls (hidden; replaced by dialog Section C)
    invite_btn       = gr.Button("Invite Friend (legacy)", visible=False)
    check_friend_btn = gr.Button("Check Friend Input (legacy)", visible=False)
    invite_status_md  = gr.Markdown("", visible=False)
    check_friend_status = gr.Markdown("", visible=False)

    # ================================================================
    # WIRING
    # ================================================================

    # -- Step 1 → 2 --
    step1_next.click(
        fn=go_step1_to_step2,
        inputs=[selfie_input],
        outputs=[st_selfie_path, page1, page2],
    )

    # -- Demo mode: load sample selfie → step 2 --
    def _load_demo():
        if not os.path.isfile(DEMO_SELFIE_PATH):
            raise gr.Error("Demo selfie not found. Place an image at assets/demo_selfie.jpg")
        return (
            DEMO_SELFIE_PATH,          # selfie_input (show preview)
            DEMO_SELFIE_PATH,          # st_selfie_path
            gr.update(visible=False),  # page1
            gr.update(visible=True),   # page2
        )
    demo_btn.click(
        fn=_load_demo,
        outputs=[selfie_input, st_selfie_path, page1, page2],
    )

    # -- Step 2 back → 1 --
    step2_back.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[page2, page1],
    )

    # -- Step 2 → 3 (loading) → analysis → 4 --
    step2_next.click(
        fn=go_step2_to_step3,
        inputs=[style_select, gender_select],
        outputs=[st_gender, st_style_hint, page2, page3],
    ).then(
        fn=run_analysis,
        inputs=[st_selfie_path],
        outputs=[
            page3, page4,
            step4_palette_html, step4_palette_cb, st_selected_ids_step4,
            st_palette, st_analysis, step4_season_text,
        ],
    )

    # -- Step 4: palette selection changed --
    step4_palette_cb.change(
        fn=lambda sel, pal: on_palette_change(sel, pal, prefix="step4"),
        inputs=[step4_palette_cb, st_palette],
        outputs=[step4_palette_html, st_selected_ids_step4],
    )

    # -- Step 4 back → 2 --
    step4_back.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[page4, page2],
    )

    # -- Step 4 → 5 --
    step4_next.click(
        fn=go_step4_to_step5,
        inputs=[
            st_selected_ids_step4, st_palette, st_analysis,
            st_selfie_path, st_style_hint,
        ],
        outputs=[
            page4, page5,
            sidebar_selfie, sidebar_analysis,
            sidebar_palette_html, sidebar_palette_cb, st_selected_ids,
            chatbot, style_goal_fab_btn,
        ],
    )

    # -- Mode switch → update description + session memory --
    def _on_mode_change(mode, trace_entries):
        desc = f"<small>{_MODE_DESCRIPTIONS.get(mode, '')}</small>"
        trace_entries = _add_trace(trace_entries, "", f"Switched to mode: {mode}")
        return mode, desc, trace_entries

    mode_select.change(
        fn=_on_mode_change,
        inputs=[mode_select, st_trace_entries],
        outputs=[st_reco_mode, mode_desc_display, st_trace_entries],
    )

    # -- Step 5: sidebar palette changed --
    sidebar_palette_cb.change(
        fn=lambda sel, pal: on_palette_change(sel, pal, prefix="sidebar"),
        inputs=[sidebar_palette_cb, st_palette],
        outputs=[sidebar_palette_html, st_selected_ids],
    )

    # -- Scenario chip buttons → fill dialog_chat_input --
    _DIALOG_SCENARIO_FILLS = [
        (dialog_sc_btn_1, "Polished office meeting — sharp and professional"),
        (dialog_sc_btn_2, "Casual weekend brunch — relaxed and effortless"),
        (dialog_sc_btn_3, "Elegant date night — refined and romantic"),
        (dialog_sc_btn_4, "Cozy and relaxed at home — soft and comfortable"),
    ]
    for _sc_btn, _sc_text in _DIALOG_SCENARIO_FILLS:
        _sc_btn.click(fn=lambda t=_sc_text: gr.update(value=t), outputs=[dialog_chat_input])

    # -- Feedback panel close button --
    feedback_close_btn.click(
        fn=lambda: (None, "", gr.Column(visible=False), gr.Column(visible=False), ""),
        outputs=[st_selected_feedback_item, feedback_info, feedback_panel, dislike_modal, feedback_status],
    )

    # ── Style Goal FAB: open dialog ──
    _js_show = (
        "() => { const el = document.getElementById('style-dialog-overlay');"
        " if (el) { el.style.removeProperty('display');"
        " el.style.removeProperty('pointer-events'); } }"
    )
    style_goal_fab_btn.click(
        fn=on_dialog_open,
        inputs=[st_dialog_locked],
        outputs=[
            style_dialog,
            dialog_input_section, dialog_section_b, dialog_section_d,
            dialog_chat_input,
            dialog_processing_html, direction_badge,
            dialog_choose_a_btn, dialog_choose_b_btn, dialog_reco_btn,
            dialog_send_btn, dialog_new_task_btn,
            dialog_sc_btn_1, dialog_sc_btn_2, dialog_sc_btn_3, dialog_sc_btn_4,
            dialog_friend_input,
        ],
        js=_js_show,
    )

    # ── Dialog close (✕) ──
    dialog_close_btn.click(
        fn=on_dialog_close,
        outputs=[style_dialog, dialog_input_section, dialog_choose_a_btn, dialog_choose_b_btn, dialog_reco_btn, direction_badge, dialog_processing_html],
        js="() => { const el=document.getElementById('style-dialog-overlay'); if(el){el.style.setProperty('display','none','important');el.style.setProperty('pointer-events','none','important');} }",
    )

    # ── Dialog send — two-phase ──
    _dialog_show_inputs  = [dialog_chat_input, chatbot, dialog_friend_input]
    _dialog_show_outputs = [
        chatbot,
        dialog_send_btn,
        dialog_sc_btn_1, dialog_sc_btn_2, dialog_sc_btn_3, dialog_sc_btn_4,
        dialog_check_status,
        st_friend_suggestion,
        dialog_section_b, dialog_section_d,
        dialog_occasion_md, dialog_style_goal_md, dialog_constraints_md,
        dialog_option_a_html, dialog_option_b_html,
        dialog_input_section,
        dialog_processing_html,
    ]
    _dialog_llm_inputs = [
        chatbot,
        st_palette, st_selected_ids,
        st_gender, st_style_hint,
        st_friend_suggestion,
        st_invite_code,
    ]
    _dialog_llm_outputs = [
        chatbot,
        st_user_text_payload, st_chosen_option, st_dialog_summary,
        st_dialog_log, session_memory_html,
        dialog_section_b,
        dialog_occasion_md, dialog_style_goal_md, dialog_constraints_md,
        dialog_option_a_html, dialog_option_b_html,
        dialog_send_btn,
        dialog_sc_btn_1, dialog_sc_btn_2, dialog_sc_btn_3, dialog_sc_btn_4,
        dialog_invite_btn, dialog_check_btn,
        dialog_processing_html,
    ]

    dialog_send_btn.click(
        fn=on_dialog_send_show, inputs=_dialog_show_inputs, outputs=_dialog_show_outputs,
    ).then(
        fn=on_dialog_send_llm, inputs=_dialog_llm_inputs, outputs=_dialog_llm_outputs,
    )
    dialog_chat_input.submit(
        fn=on_dialog_send_show, inputs=_dialog_show_inputs, outputs=_dialog_show_outputs,
    ).then(
        fn=on_dialog_send_llm, inputs=_dialog_llm_inputs, outputs=_dialog_llm_outputs,
    )

    # ── Dialog: Start New Outfit Goal ──
    # Step 1: reset task-level Gradio state + backend pref decay
    # Step 2-3: same show + LLM chain as Refine
    dialog_new_task_btn.click(
        fn=_on_new_task_reset,
        inputs=[st_session_id],
        outputs=[st_task_id, st_trace_entries, st_feedback_done, session_history_html],
    ).then(
        fn=on_dialog_send_show, inputs=_dialog_show_inputs, outputs=_dialog_show_outputs,
    ).then(
        fn=on_dialog_send_llm, inputs=_dialog_llm_inputs, outputs=_dialog_llm_outputs,
    )

    # ── Dialog Continue → show Section D + auto-scroll ──
    dialog_continue_btn.click(
        fn=on_dialog_continue,
        outputs=[dialog_section_d],
    ).then(
        fn=None,
        js="() => { const c = document.getElementById('style-dialog-card'); if(c) setTimeout(()=>c.scrollTo({top:c.scrollHeight,behavior:'smooth'}),150); }",
    )

    # ── Dialog A / B choose ──
    _dialog_choose_inputs = [
        chatbot, st_user_text_payload, st_dialog_log,
        st_session_id, st_palette, st_selected_ids, st_reco_mode, st_trace_entries,
    ]
    _dialog_choose_outputs = [
        chatbot, st_user_text_payload, st_chosen_option,
        st_dialog_log, st_trace_entries, session_memory_html,
        direction_badge,
        dialog_choose_a_btn, dialog_choose_b_btn, dialog_reco_btn,
    ]

    dialog_choose_a_btn.click(
        fn=_dialog_choose_a,
        inputs=_dialog_choose_inputs,
        outputs=_dialog_choose_outputs,
    ).then(
        fn=_refresh_session_history,
        inputs=[st_session_id, st_trace_entries, st_feedback_done, st_last_items],
        outputs=[session_history_html],
    )
    dialog_choose_b_btn.click(
        fn=_dialog_choose_b,
        inputs=_dialog_choose_inputs,
        outputs=_dialog_choose_outputs,
    ).then(
        fn=_refresh_session_history,
        inputs=[st_session_id, st_trace_entries, st_feedback_done, st_last_items],
        outputs=[session_history_html],
    )

    # ── Chat (legacy two-phase wiring: hidden inputs, kept for compatibility) ──
    _show_inputs = [chat_input, chatbot, friend_input]
    _show_outputs = [chatbot, chat_input, chat_send_btn,
                     sc_btn_1, sc_btn_2, sc_btn_3, sc_btn_4,
                     invite_btn, check_friend_btn,
                     friend_input, check_friend_status,
                     st_friend_suggestion]

    _llm_inputs = [
        chatbot,
        st_palette, st_selected_ids,
        st_gender, st_style_hint,
        st_friend_suggestion,
        st_invite_code,
    ]
    _llm_outputs = [
        chatbot, option_btn_row,
        st_user_text_payload, st_chosen_option,
        choose_a_btn, choose_b_btn,
        chat_input, chat_send_btn,
        sc_btn_1, sc_btn_2, sc_btn_3, sc_btn_4,
        invite_btn, check_friend_btn,
    ]

    chat_send_btn.click(
        fn=on_chat_submit_show, inputs=_show_inputs, outputs=_show_outputs,
    ).then(
        fn=on_chat_submit_llm, inputs=_llm_inputs, outputs=_llm_outputs,
    )
    chat_input.submit(
        fn=on_chat_submit_show, inputs=_show_inputs, outputs=_show_outputs,
    ).then(
        fn=on_chat_submit_llm, inputs=_llm_inputs, outputs=_llm_outputs,
    )

    # Re-enable Send when user focuses on chat_input
    chat_input.focus(
        fn=lambda: gr.update(interactive=True),
        outputs=[chat_send_btn],
    )

    _choose_out = [chatbot, st_user_text_payload, st_chosen_option,
                    choose_a_btn, choose_b_btn, chat_send_btn,
                    st_trace_entries, session_memory_html,
                    option_btn_row]  # hide row after selection

    def _choose_a(h, p, session_id, palette, selected_ids, reco_mode, trace_entries):
        results = on_choose_option("A", h, p)
        trace_entries = _add_trace(trace_entries, "", "Chose interpretation A")
        memory = _build_memory_html(
            session_id, palette, selected_ids, results[1], "A", reco_mode, trace_entries,
        )
        return (*results, trace_entries, memory, gr.update(visible=False))

    def _choose_b(h, p, session_id, palette, selected_ids, reco_mode, trace_entries):
        results = on_choose_option("B", h, p)
        trace_entries = _add_trace(trace_entries, "", "Chose interpretation B")
        memory = _build_memory_html(
            session_id, palette, selected_ids, results[1], "B", reco_mode, trace_entries,
        )
        return (*results, trace_entries, memory, gr.update(visible=False))

    _choose_extra_inputs = [st_session_id, st_palette, st_selected_ids, st_reco_mode, st_trace_entries]

    choose_a_btn.click(
        fn=_choose_a,
        inputs=[chatbot, st_user_text_payload] + _choose_extra_inputs,
        outputs=_choose_out,
    )
    choose_b_btn.click(
        fn=_choose_b,
        inputs=[chatbot, st_user_text_payload] + _choose_extra_inputs,
        outputs=_choose_out,
    )

    # -- Recommend / Regenerate --
    _reco_inputs = [
        st_session_id, st_palette, st_selected_ids,
        st_gender, st_style_hint,
        st_user_text_payload, st_chosen_option,
        st_reco_mode, st_trace_entries,
    ]
    _reco_outputs = [
        st_request_id, reco_gallery, reco_status,
        st_last_items, st_feedback_done,
        filter_categories,
        st_trace_entries,
        feedback_panel, st_selected_feedback_item, feedback_status,
    ]

    recommend_btn.click(
        fn=lambda: gr.Button(value="\u23f3 Generating\u2026", interactive=False),
        outputs=[recommend_btn],
    ).then(
        fn=generate_reco, inputs=_reco_inputs, outputs=_reco_outputs,
    ).then(
        fn=lambda: (
            gr.Button(value="Recommend", interactive=True),
            gr.update(interactive=True),   # re-enable FAB
            False,                         # st_dialog_locked
        ),
        outputs=[recommend_btn, style_goal_fab_btn, st_dialog_locked],
    ).then(
        fn=_refresh_session_history,
        inputs=[st_session_id, st_trace_entries, st_feedback_done, st_last_items],
        outputs=[session_history_html],
    )

    # ── Dialog Recommend Now (alias for recommend_btn inside dialog) ──
    _js_hide = (
        "() => { const el = document.getElementById('style-dialog-overlay');"
        " if (el) { el.style.setProperty('display','none','important');"
        " el.style.setProperty('pointer-events','none','important'); } }"
    )
    dialog_reco_btn.click(
        fn=lambda: (
            gr.Button(value="\u23f3 Generating\u2026", interactive=False),
            gr.update(visible=False),     # close dialog
            gr.update(visible=True),      # restore dialog_input_section
            gr.update(interactive=True),  # reset dialog_choose_a_btn
            gr.update(interactive=True),  # reset dialog_choose_b_btn
            gr.update(value=""),          # clear direction confirm
        ),
        outputs=[dialog_reco_btn, style_dialog, dialog_input_section,
                 dialog_choose_a_btn, dialog_choose_b_btn, direction_badge],
        js=_js_hide,                      # instant client-side overlay hide on click
    ).then(
        fn=generate_reco, inputs=_reco_inputs, outputs=_reco_outputs,
    ).then(
        fn=lambda: (
            gr.Button(value="Recommend Now", interactive=True),
            gr.update(interactive=True),  # unlock FAB
            False,                        # st_dialog_locked
        ),
        outputs=[dialog_reco_btn, style_goal_fab_btn, st_dialog_locked],
    ).then(
        fn=_refresh_session_history,
        inputs=[st_session_id, st_trace_entries, st_feedback_done, st_last_items],
        outputs=[session_history_html],
    )

    # -- Category filter (post-generation) --
    filter_categories.change(
        fn=on_category_filter,
        inputs=[filter_categories, st_last_items],
        outputs=[reco_gallery, reco_status],
    )

    # -- Gallery select → show feedback panel --
    reco_gallery.select(
        fn=on_gallery_select,
        inputs=[st_last_items, st_feedback_done, st_reco_mode, st_session_id],
        outputs=[
            st_selected_feedback_item, feedback_info,
            feedback_panel, dislike_modal,
            like_btn, dislike_btn, cart_btn,
            feedback_status,
        ],
    )

    # -- Like --
    like_btn.click(
        fn=on_like,
        inputs=[
            st_session_id, st_request_id,
            st_selected_feedback_item,
            st_feedback_done,
            st_trace_entries, st_palette, st_selected_ids,
            st_user_text_payload, st_chosen_option, st_reco_mode,
            st_last_items,
        ],
        outputs=[feedback_status, feedback_panel, st_feedback_done,
                 like_btn, dislike_btn, cart_btn,
                 st_trace_entries, session_history_html],
    )

    # -- Dislike → show modal --
    dislike_btn.click(
        fn=on_dislike_click,
        inputs=[
            st_session_id, st_request_id,
            st_selected_feedback_item,
            st_feedback_done,
        ],
        outputs=[feedback_status, dislike_modal, st_feedback_done,
                 like_btn, dislike_btn, cart_btn],
    )

    # -- Dislike submit --
    dislike_submit.click(
        fn=on_dislike_submit,
        inputs=[
            st_session_id, st_request_id,
            st_selected_feedback_item, dislike_chips,
            dislike_text,
            st_feedback_done, st_trace_entries,
            st_palette, st_selected_ids,
            st_user_text_payload, st_chosen_option, st_reco_mode,
            st_last_items,
        ],
        outputs=[
            feedback_status, dislike_modal, feedback_panel,
            dislike_chips, dislike_text,
            st_feedback_done, st_trace_entries,
            session_history_html,
        ],
    )

    # -- Dislike close --
    dislike_close.click(
        fn=on_dislike_close,
        inputs=[],
        outputs=[dislike_modal, feedback_panel, dislike_chips, dislike_text],
    )

    # -- Cart --
    cart_btn.click(
        fn=on_cart,
        inputs=[
            st_session_id, st_request_id,
            st_selected_feedback_item, st_last_items, st_cart,
            st_feedback_done,
            st_trace_entries, st_palette, st_selected_ids,
            st_user_text_payload, st_chosen_option, st_reco_mode,
        ],
        outputs=[
            feedback_status, feedback_panel, st_cart,
            cart_gallery, st_feedback_done,
            like_btn, dislike_btn, cart_btn,
            cart_heading,
            st_trace_entries,
            session_history_html,
        ],
    )

    # -- Friend Mode entry (from page1) --
    def _open_friend_mode():
        return gr.update(visible=False), gr.update(visible=True)

    friend_mode_entry_btn.click(
        fn=_open_friend_mode,
        outputs=[page1, page_friend],
    )

    # -- Friend Mode: look up context by code --
    def _friend_lookup(code):
        code = (code or "").strip().upper()
        if not code:
            return (
                "<div style='color:#e57373;'>Please enter a code.</div>",
                gr.update(visible=False),
                gr.update(),  # suggestion_box: no change
                gr.update(),  # submit_status: no change
            )
        ctx = store.get_friend_context(code)
        if ctx is None:
            return (
                "<div style='color:#e57373;'>Code not found. The session may have expired — "
                "ask your friend to invite you again.</div>",
                gr.update(visible=False),
                gr.update(),
                gr.update(),
            )
        hexes = ctx.get("palette_hexes", [])
        swatches = "".join(
            f'<span style="display:inline-block;width:16px;height:16px;'
            f'border-radius:4px;background:{h};margin:0 2px;'
            f'border:1px solid #ccc;vertical-align:middle;"></span>'
            for h in hexes
        )
        intent_text = ctx.get("user_intent", "") or "*(no description yet)*"
        md = (
            f"**Your friend's context**\n\n"
            f"- Season type: **{ctx.get('season', 'Unknown')}**\n"
            f"- Palette: {swatches}\n"
            f"- What they're looking for: *{intent_text}*\n\n"
            f"Write your suggestion below and click **Send Suggestion**."
        )
        # Clear previous suggestion + status so friend writes fresh for this round
        return md, gr.update(visible=True), gr.update(value=""), gr.update(value="")

    friend_lookup_btn.click(
        fn=_friend_lookup,
        inputs=[friend_code_input],
        outputs=[friend_ctx_display, friend_reply_area, friend_suggestion_box, friend_submit_status],
    )

    # -- Friend Mode: submit suggestion --
    def _friend_submit(code, suggestion):
        code = (code or "").strip().upper()
        suggestion = (suggestion or "").strip()
        if not suggestion:
            return "<div style='color:#e57373;'>Please write a suggestion before submitting.</div>"
        ok = store.submit_friend_input(code, suggestion)
        if not ok:
            return "<div style='color:#e57373;'>Code expired or not found. Ask your friend for a new code.</div>"
        return "✅ Suggestion sent! Your friend will see it when they click **Check Friend Input**."

    friend_submit_btn.click(
        fn=_friend_submit,
        inputs=[friend_code_input, friend_suggestion_box],
        outputs=[friend_submit_status],
    )

    # -- Friend Mode: back button --
    friend_back_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[page1, page_friend],
    )

    # -- Invite Friend (dialog, on page5) --
    dialog_invite_btn.click(
        fn=on_invite_friend,
        inputs=[st_session_id, st_palette, st_selected_ids, st_analysis, st_user_text_payload, dialog_chat_input, st_invite_code],
        outputs=[dialog_invite_status, st_invite_code, dialog_invite_btn, dialog_check_btn],
    )

    # -- Check Friend Input (dialog) --
    def _check_friend_dialog(invite_code):
        status_md, suggestion = on_check_friend_input(invite_code)
        if not suggestion:
            return status_md, gr.update(), gr.update(interactive=True)
        hint = "✅ Friend's suggestion filled in above. Now press Interpret to generate A / B options."
        return hint, gr.update(value=suggestion), gr.update(interactive=False)

    dialog_check_btn.click(
        fn=_check_friend_dialog,
        inputs=[st_invite_code],
        outputs=[dialog_check_status, dialog_friend_input, dialog_check_btn],
    )

    # -- Legacy invite/check (hidden, kept for compatibility) --
    invite_btn.click(
        fn=on_invite_friend,
        inputs=[st_session_id, st_palette, st_selected_ids, st_analysis, st_user_text_payload, chat_input, st_invite_code],
        outputs=[invite_status_md, st_invite_code, invite_btn, check_friend_btn],
    )

    # -- Check Friend Input (user side, legacy) --
    def _check_friend(invite_code):
        status_md, suggestion = on_check_friend_input(invite_code)
        if not suggestion:
            return status_md, gr.update(), gr.update(interactive=True)
        hint = "✅ Friend's suggestion filled in above. Now press **Send** to generate A / B options."
        return hint, gr.update(value=suggestion), gr.update(interactive=False)

    check_friend_btn.click(
        fn=_check_friend,
        inputs=[st_invite_code],
        outputs=[check_friend_status, friend_input, check_friend_btn],
    )

    # -- Reset session --
    def _reset_session():
        new_sid = f"gr_{uuid.uuid4().hex[:10]}"
        log_event("reset_session", session_id=new_sid)
        return (
            new_sid,                       # st_session_id
            None,                          # st_selfie_path
            "", "",                        # st_gender, st_style_hint
            [], {},                        # st_palette, st_analysis
            [], [],                        # st_selected_ids, st_selected_ids_step4
            [], "",                        # st_last_items, st_request_id
            None, None,                    # st_user_text_payload, st_chosen_option
            [], None, {},                  # st_cart, st_selected_feedback_item, st_feedback_done
            "full", [],                    # st_reco_mode, st_trace_entries
            None,                          # st_task_id (cleared on full session reset)
            gr.update(visible=True),       # page1
            gr.update(visible=False),      # page5
            gr.update(visible=False),      # page_friend (hide friend mode on reset)
            '<div style="font-size:12px;color:var(--body-text-color-subdued,#aaa);'
            'text-align:center;padding:12px 0;">'
            'Your style profile will build up as you interact — '
            'describe what you want, like items, and give feedback.</div>',
            '<div style="font-size:12px;color:var(--body-text-color-subdued,#aaa);'
            'text-align:center;padding:16px 0;">'
            'Generate recommendations and give feedback (👍/👎/🛒) to see how the system learns.<br>'
            '<small>Liked/disliked patterns, avoid signals, and your activity log will appear here.</small>'
            '</div>',
            _initial_chat_history(),       # chatbot
            gr.update(value=""),           # friend_input
            # --- invite / friend UI reset ---
            "",                            # st_invite_code
            gr.update(visible=True, interactive=True),   # invite_btn
            gr.update(visible=False),      # check_friend_btn
            gr.update(value=""),           # invite_status_md
            gr.update(value=""),           # check_friend_status
            gr.update(visible=False),      # option_btn_row
            # --- chat input / chips reset ---
            gr.update(value="", interactive=True),       # chat_input
            gr.update(interactive=True),   # chat_send_btn
            gr.update(interactive=True),   # sc_btn_1
            gr.update(interactive=True),   # sc_btn_2
            gr.update(interactive=True),   # sc_btn_3
            gr.update(interactive=True),   # sc_btn_4
            "",                            # st_friend_suggestion
            # --- dialog state reset ---
            False,                         # st_dialog_locked
            {},                            # st_dialog_summary
            {},                            # st_dialog_log
            gr.update(visible=False),      # style_dialog (close)
            gr.update(visible=False, interactive=True),  # style_goal_fab_btn (hide on reset)
            gr.update(visible=False),      # dialog_section_b
            gr.update(visible=False),      # dialog_section_d
            gr.update(value=""),           # dialog_chat_input
            gr.update(interactive=True),   # dialog_send_btn
            gr.update(interactive=True),   # dialog_new_task_btn
            gr.update(visible=True),       # dialog_input_section (restore)
            gr.update(interactive=True),   # dialog_choose_a_btn
            gr.update(interactive=True),   # dialog_choose_b_btn
            gr.update(value="Recommend Now"),  # dialog_reco_btn
            gr.update(value=""),           # direction_badge (clear inline confirm)
            gr.update(value=""),           # dialog_processing_html
            # --- dialog friend controls reset ---
            gr.update(value=""),           # dialog_friend_input
            gr.update(visible=True, interactive=True),   # dialog_invite_btn
            gr.update(visible=False),      # dialog_check_btn
            gr.update(value=""),           # dialog_invite_status
            gr.update(value=""),           # dialog_check_status
        )
    reset_btn.click(
        fn=_reset_session,
        outputs=[
            st_session_id,
            st_selfie_path,
            st_gender, st_style_hint,
            st_palette, st_analysis,
            st_selected_ids, st_selected_ids_step4,
            st_last_items, st_request_id,
            st_user_text_payload, st_chosen_option,
            st_cart, st_selected_feedback_item, st_feedback_done,
            st_reco_mode, st_trace_entries,
            st_task_id,
            page1, page5, page_friend,
            session_memory_html,
            session_history_html,
            chatbot,
            friend_input,
            # invite / friend UI
            st_invite_code,
            invite_btn, check_friend_btn,
            invite_status_md, check_friend_status,
            option_btn_row,
            # chat input / chips
            chat_input, chat_send_btn,
            sc_btn_1, sc_btn_2, sc_btn_3, sc_btn_4,
            st_friend_suggestion,
            # dialog state
            st_dialog_locked, st_dialog_summary, st_dialog_log,
            style_dialog, style_goal_fab_btn,
            dialog_section_b, dialog_section_d,
            dialog_chat_input, dialog_send_btn,
            dialog_new_task_btn,
            dialog_input_section, dialog_choose_a_btn, dialog_choose_b_btn, dialog_reco_btn,
            direction_badge, dialog_processing_html,
            # dialog friend controls
            dialog_friend_input, dialog_invite_btn, dialog_check_btn,
            dialog_invite_status, dialog_check_status,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", server_port=7860, share=True,
        allowed_paths=[IMAGES_DIR],
        theme=gr.themes.Soft(primary_hue="teal"),
        css=_APP_CSS,
        js=_APP_JS,
    )
