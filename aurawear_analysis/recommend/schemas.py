from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import (
    Any, Dict, List, Literal, Optional, Protocol, Set,
    runtime_checkable,
)

import numpy as np


ActionType = Literal["like", "dislike", "cart"]

@dataclass
class UserTextOption:
    id: str                          # "A" / "B"
    interpretation: str              # 顯示給使用者看的說法
    intent_patch: Dict[str, Any]     # 給模型用的結構化約束


@dataclass
class UserTextPayload:
    raw: str                         # 使用者原始輸入
    choice: Optional[str] = None     # "A" / "B"
    options: Optional[List[UserTextOption]] = None


@dataclass
class Filters:
    # minimal filters; extend freely
    categories: List[str] = field(default_factory=list)  # e.g. ["top", "outerwear"]
    styles: List[str] = field(default_factory=list)      # optional (locked style can be passed)
    gender: Optional[str] = None                         # optional (locked gender can be passed)


@dataclass
class GenerateRequest:
    session_id: str
    request_id: Optional[str]
    palette18: List[Dict[str, Any]]
    selected_palette_ids: List[str]
    filters: "Filters"
    k: int = 12
    user_text: Optional[UserTextPayload] = None
    mode: str = "full"  # "full", "color_only", "text_only", "color_text"


@dataclass
class FeedbackEvent:
    session_id: str
    request_id: str
    item_id: str
    action: ActionType


@dataclass
class RecommendedItem:
    item_id: str
    image_uri: str
    category: Optional[str]
    score: float
    debug: Dict[str, float] = field(default_factory=dict)
    explanation_text: str = ""


@dataclass
class GenerateResponse:
    ok: bool
    session_id: str
    request_id: str
    items: List[RecommendedItem] = field(default_factory=list)
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# Data Models (HITL interface definitions)
# Backend engineer should provide production implementations of
# ItemIndex (DB/vector store) and SessionStore (Redis/DB).
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Item:
    """Single product item in the recommendation index."""
    item_id: str
    image_uri: str
    category: str
    dominant_hex: List[str]
    emb: np.ndarray

    # Optional metadata for better avoid_terms matching
    title: str = ""
    style: str = ""
    brand: str = ""
    tags: List[str] = field(default_factory=list)
    meta_text: str = ""

    # ── serialization ────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict (np.ndarray → list)."""
        return {
            "item_id": self.item_id,
            "image_uri": self.image_uri,
            "category": self.category,
            "dominant_hex": list(self.dominant_hex),
            "emb": self.emb.tolist(),
            "title": self.title,
            "style": self.style,
            "brand": self.brand,
            "tags": list(self.tags),
            "meta_text": self.meta_text,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Item":
        """Deserialize from dict (list → np.ndarray)."""
        return cls(
            item_id=d["item_id"],
            image_uri=d["image_uri"],
            category=d["category"],
            dominant_hex=d.get("dominant_hex", []),
            emb=np.asarray(d["emb"], dtype=np.float32),
            title=d.get("title", ""),
            style=d.get("style", ""),
            brand=d.get("brand", ""),
            tags=d.get("tags", []),
            meta_text=d.get("meta_text", ""),
        )


@dataclass
class SessionState:
    """HITL session state — tracks user preferences, feedback, and suppression signals.

    Design contract — two memory scopes:

    TASK-SCOPED  (cleared by start_new_task() on each "Start New Outfit Goal"):
        disliked      — item IDs that are hard-banned in candidate generation.
                        Task-specific; a formal coat blocked for commute may suit a wedding.
        neg_vecs      — per-item embedding suppressors built from disliked items.
        avoid_terms   — text-level avoid signals extracted from critique chips + LLM.
        critique_tags — raw chip labels; subset of avoid_terms.
        intent_vec    — embedding of the current LLM-generated intent query.

    STABLE / CROSS-TASK  (retained across tasks; past tasks down-weighted at 0.3):
        liked         — item ID set; kept as a display trace. Preference influence
                        is captured in pref_vec which is decayed on task switch.
        cart          — item ID set; same as liked — display / export only.
        pref_vec      — blended preference direction. On task switch: archived into
                        task_pref_vecs at 1.0, then rebuilt from all past tasks at 0.3.
        task_pref_vecs — per-task archived pref_vecs used for cross-task blending.
        seen          — seen item IDs for novelty penalty; cross-task intentionally.

    UI interaction locks are driven by Gradio's st_feedback_done (cleared on new task),
    NOT by the liked/cart sets above — so UI unlocks cleanly without data loss.
    """
    session_id: str

    # Vectors (all normalized)
    pref_vec: Optional[np.ndarray] = None
    intent_vec: Optional[np.ndarray] = None

    # Dislike semantics (from GPT)
    avoid_terms: Set[str] = field(default_factory=set)

    # Quick critique tags (from UI chips)
    critique_tags: Set[str] = field(default_factory=set)

    # Feedback sets
    liked: Set[str] = field(default_factory=set)
    disliked: Set[str] = field(default_factory=set)
    cart: Set[str] = field(default_factory=set)

    # Negative suppression (from disliked items)
    neg_vecs: List[np.ndarray] = field(default_factory=list)

    # Novelty
    seen: Set[str] = field(default_factory=set)

    last_request_id: Optional[str] = None
    updated_at: float = field(default_factory=lambda: time.time())

    # ── Task-level tracking (Phase 1/2) ────────────────────────────
    # Current active task ID — changes on each "Start New Outfit Goal"
    current_task_id: Optional[str] = None
    # Archived pref_vec per completed task, for cross-task blending
    task_pref_vecs: Dict[str, Any] = field(default_factory=dict)

    # ── serialization helpers ────────────────────────────────
    @staticmethod
    def _vec_to_list(v: Optional[np.ndarray]) -> Optional[List[float]]:
        return v.tolist() if v is not None else None

    @staticmethod
    def _list_to_vec(v: Optional[List[float]]) -> Optional[np.ndarray]:
        return np.asarray(v, dtype=np.float32) if v is not None else None

    @staticmethod
    def _vecs_to_lists(vecs: List[np.ndarray]) -> List[List[float]]:
        return [v.tolist() for v in vecs]

    @staticmethod
    def _lists_to_vecs(lists: List[List[float]]) -> List[np.ndarray]:
        return [np.asarray(v, dtype=np.float32) for v in lists]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-safe dict for Redis / DB storage.

        All ``np.ndarray`` fields are converted to plain Python lists.
        All ``set`` fields are converted to sorted lists for deterministic output.
        """
        return {
            "session_id": self.session_id,
            "pref_vec": self._vec_to_list(self.pref_vec),
            "intent_vec": self._vec_to_list(self.intent_vec),
            "avoid_terms": sorted(self.avoid_terms),
            "critique_tags": sorted(self.critique_tags),
            "liked": sorted(self.liked),
            "disliked": sorted(self.disliked),
            "cart": sorted(self.cart),
            "neg_vecs": self._vecs_to_lists(self.neg_vecs),
            "seen": sorted(self.seen),
            "last_request_id": self.last_request_id,
            "updated_at": self.updated_at,
            "current_task_id": self.current_task_id,
            "task_pref_vecs": {
                k: self._vec_to_list(v)
                for k, v in self.task_pref_vecs.items()
                if v is not None
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionState":
        """Deserialize from a dict (e.g. loaded from Redis / DB)."""
        return cls(
            session_id=d["session_id"],
            pref_vec=cls._list_to_vec(d.get("pref_vec")),
            intent_vec=cls._list_to_vec(d.get("intent_vec")),
            avoid_terms=set(d.get("avoid_terms", [])),
            critique_tags=set(d.get("critique_tags", [])),
            liked=set(d.get("liked", [])),
            disliked=set(d.get("disliked", [])),
            cart=set(d.get("cart", [])),
            neg_vecs=cls._lists_to_vecs(d.get("neg_vecs", [])),
            seen=set(d.get("seen", [])),
            last_request_id=d.get("last_request_id"),
            updated_at=d.get("updated_at", time.time()),
            current_task_id=d.get("current_task_id"),
            task_pref_vecs={
                k: cls._list_to_vec(v)
                for k, v in d.get("task_pref_vecs", {}).items()
                if v is not None
            },
        )

# ══════════════════════════════════════════════════════════════
# Interface Protocols
# Backend must provide concrete implementations of these.
# ══════════════════════════════════════════════════════════════

@runtime_checkable
class ItemIndexProtocol(Protocol):
    """Read-only product index used by AuraWearRecommender.

    Backend should implement this with a real DB / vector store.
    See ``SimpleItemIndex`` in app_gradio.py for a reference implementation.
    """

    @property
    def items(self) -> List[Item]:
        """Return all items (used for candidate generation scan)."""
        ...

    @property
    def emb_dim(self) -> int:
        """Embedding dimensionality (e.g. 512 for CLIP ViT-B-32)."""
        ...

    def get(self, item_id: str) -> Optional[Item]:
        """Look up a single item by ID. Return None if not found."""
        ...


@runtime_checkable
class SessionStoreProtocol(Protocol):
    """Session persistence used by AuraWearRecommender.

    Backend should implement this with Redis / DB.
    See ``SimpleSessionStore`` in app_gradio.py for a reference implementation.
    """

    def get_or_create(self, session_id: str) -> SessionState:
        """Return existing session or create a new empty one."""
        ...

    def save(self, state: SessionState) -> None:
        """Persist the session state."""
        ...
