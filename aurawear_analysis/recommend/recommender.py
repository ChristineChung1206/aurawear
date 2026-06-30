from __future__ import annotations


from typing import Any, Dict, List, Optional, Tuple, Set
import json
import random
import uuid
from datetime import datetime
from pathlib import Path
import numpy as np

from aurawear_analysis.recommend.schemas import (
    GenerateRequest, GenerateResponse, FeedbackEvent, RecommendedItem, Item,
    ItemIndexProtocol, SessionStoreProtocol,
)
from aurawear_analysis.recommend.color_match import color_score_min_deltaE
from aurawear_analysis.recommend.reranker import (
    cosine_sim, update_preference, neg_suppression_penalty, dup_penalty, normalize
)
from aurawear_analysis.config import RecoConfig
from aurawear_analysis.recommend.text_embedder import TextEmbedder, TextEmbedderConfig


def _item_tagset(it: Item) -> Set[str]:
    """Build a normalized tag set from Item fields."""
    tags = set()
    for t in (it.tags or []):
        if isinstance(t, str) and t.strip():
            tags.add(t.strip().lower())
    if it.category:
        tags.add(it.category.strip().lower())
    if it.style:
        tags.add(it.style.strip().lower())
    if it.brand:
        tags.add(it.brand.strip().lower())
    return tags


def _item_haystack(it: Item) -> str:
    """Build free-text for substring match (fallback for avoid_terms)."""
    parts = [
        it.category or "",
        it.title or "",
        it.style or "",
        it.brand or "",
        " ".join(it.tags or []),
    ]

    dh = it.dominant_hex
    if isinstance(dh, list):
        parts.append(" ".join(str(x) for x in dh if x))
    elif isinstance(dh, str) and dh:
        parts.append(dh)

    parts.append(it.meta_text or "")

    return " ".join(parts).lower()



def avoid_match_score(it: Item, avoid_terms: Set[str]) -> float:
    """
    Returns a soft match score in [0,1].
    - Prefer exact tag match (more reliable)
    - Fall back to substring match (less reliable)
    """
    if not avoid_terms:
        return 0.0

    tagset = _item_tagset(it)
    hay = None

    hit = 0
    for term in avoid_terms:
        t = (term or "").strip().lower()
        if not t:
            continue
        # 1) exact match against tags (best)
        if t in tagset:
            hit += 2
            continue
        # 2) substring fallback (weaker)
        if hay is None:
            hay = _item_haystack(it)
        if t in hay:
            hit += 1

    # squash to [0,1]
    if hit <= 0:
        return 0.0
    return min(1.0, hit / 4.0)


def _palette_subset_hex(palette18: List[Dict[str, Any]], selected_ids: List[str]) -> List[str]:
    m = {p.get("id"): p.get("hex") for p in palette18 if p.get("id") and p.get("hex")}
    return [m[i] for i in selected_ids if i in m]


def _match_filters(item: Item, req: GenerateRequest) -> bool:
    if req.filters.categories:
        if item.category is None or item.category not in req.filters.categories:
            return False
    # Gender filtering based on item_id prefix (df_MEN- / df_WOMEN-)
    if req.filters.gender:
        g = req.filters.gender.strip().lower()
        iid = (item.item_id or "").upper()
        if g in ("male", "men"):
            if "WOMEN-" in iid:
                return False
        elif g in ("female", "women"):
            if "MEN-" in iid and "WOMEN-" not in iid:
                return False
        # non-binary / prefer not to say → no filter
    return True


class AuraWearRecommender:
    def __init__(
        self,
        index: ItemIndexProtocol,
        store: SessionStoreProtocol,
        llm: Optional[object] = None,
        cfg: Optional[RecoConfig] = None,
    ):
        self.index = index
        self.store = store
        self.llm = llm
        self.cfg = cfg or RecoConfig()

        self.text_embedder = TextEmbedder(
            TextEmbedderConfig(
                backend=self.cfg.text_embedder_backend,
                expected_dim=self.index.emb_dim,
                clip_model=self.cfg.clip_model,
                clip_pretrained=self.cfg.clip_pretrained,
                device=self.cfg.clip_device,
            )
        )

    # ---------------- GENERATE / REGENERATE ----------------
    def generate(self, req: GenerateRequest) -> GenerateResponse:
        state = self.store.get_or_create(req.session_id)
        request_id = req.request_id or f"req_{uuid.uuid4().hex[:10]}"

        # Auto-assign task_id on first generation so the badge shows as #1
        # (current_task_id is only set explicitly when "Start New Outfit Goal" is clicked;
        # for the initial task we use a stable sentinel key).
        if state.current_task_id is None:
            state.current_task_id = "task_initial"

        k = req.k or self.cfg.top_k_default
        selected_hex = _palette_subset_hex(req.palette18, req.selected_palette_ids)

        # --- intent lifecycle control (VERY IMPORTANT) ---
        if self.llm is None:
            state.intent_vec = None
            print("[STATE][intent_vec_cleared]")
        else:
            from aurawear_analysis.recommend.llm.intent_generator import generate_intent

            palette_selected = [p for p in req.palette18 if p.get("id") in req.selected_palette_ids]

            intent = generate_intent(
                llm=self.llm,
                gender=(req.filters.gender if req.filters else ""),
                style=(req.filters.styles[0] if (req.filters and req.filters.styles) else ""),
                palette_selected=palette_selected,
                user_text=req.user_text,
                existing_avoid_terms=(
                    sorted(list(state.avoid_terms)) if state.avoid_terms else []
                ),
            )


            print("[LLM][intent]", intent)

            intent_text = (intent.get("query_text") or "").strip()
            if intent_text:
                state.intent_vec = self.text_embedder.embed_one(intent_text)
                print("[STATE][intent_vec_set]", intent_text, state.intent_vec.shape)
            else:
                state.intent_vec = None
                print("[STATE][intent_vec_empty_cleared]")


        if not selected_hex:
            return GenerateResponse(
                ok=False,
                session_id=req.session_id,
                request_id=request_id,
                error="no_selected_colors"
            )

        # 1) candidate generation: filter + color score
        # text_only mode: relax color gate to let more items through
        mode = getattr(req, 'mode', 'full') or 'full'
        color_gate = self.cfg.color_min_score if mode != 'text_only' else 0.05
        scored: List[Tuple[Item, float]] = []
        for it in self.index.items:
            if it.item_id in state.disliked:  # hard ban
                continue
            if not _match_filters(it, req):
                continue

            cs = color_score_min_deltaE(it.dominant_hex, selected_hex, tau=self.cfg.color_tau)
            if cs < color_gate:
                continue
            scored.append((it, cs))

        scored.sort(key=lambda x: x[1], reverse=True)
        pool = scored[: self.cfg.candidate_pool_size]

        # 2) rerank — mode controls active signals
        picked_embs: List[np.ndarray] = []
        items_out: List[RecommendedItem] = []
        ranked_dbg: List[Dict[str, Any]] = []

        # Mode-based weight overrides
        use_pref = mode == 'full'
        use_intent = mode in ('full', 'text_only', 'color_text')
        use_neg = mode == 'full'
        use_avoid = mode == 'full'
        use_novelty = mode == 'full'
        use_diversity = mode in ('full', 'color_text')
        use_color = mode in ('full', 'color_only', 'color_text')

        remaining = list(pool)

        for _ in range(min(k, len(remaining))):
            best = None
            best_score = -1e18
            best_dbg = {}

            for it, cs in remaining:

                if it.item_id in state.disliked:
                    continue

                neg_pen = neg_suppression_penalty(it.emb, state.neg_vecs) if use_neg else 0.0
                # diversity penalty: hard suppress + soft penalty
                dpen = dup_penalty(it.emb, picked_embs, threshold=self.cfg.dup_sim_threshold) if use_diversity else 0.0

                # hard suppress (dup_penalty returns a huge number for hard case)
                if dpen >= 1e5:
                    continue

                pref_sim = cosine_sim(state.pref_vec, it.emb) if (state.pref_vec is not None and use_pref) else 0.0
                intent_sim = cosine_sim(state.intent_vec, it.emb) if (state.intent_vec is not None and use_intent) else 0.0
                novelty_pen = (self.cfg.seen_penalty if it.item_id in state.seen else 0.0) if use_novelty else 0.0
                rule_pen = avoid_match_score(it, state.avoid_terms) if use_avoid else 0.0

                color_w = self.cfg.w_color if use_color else 0.0


                final = (
                    # ── Ranking priority (AuraWear design contract) ──────────────────────
                    # 1. Current task intent  (w_intent, ~50%)  — primary driver
                    # 2. Palette match        (w_color,  ~30%)  — hard constraint
                    # 3. Stable preferences   (w_pref,   ~15%)  — blended at 0.3 after task switch
                    # 4. Diversity / novelty  (w_dup, w_novelty) — exploration bonus
                    # 5. Neg suppression      (neg_penalty_weight) — cleared on new task
                    # 6. Avoid-term rule      (w_dislike_rule)  — cleared on new task
                    # ─────────────────────────────────────────────────────────────────────
                    color_w * cs
                    + self.cfg.w_pref * pref_sim
                    + self.cfg.w_intent * intent_sim
                    - self.cfg.neg_penalty_weight * neg_pen
                    - self.cfg.w_novelty * novelty_pen
                    - self.cfg.w_dup * dpen
                    - self.cfg.w_dislike_rule * rule_pen
                )

                if final > best_score:
                    best_score = final
                    best = (it, cs)
                    best_dbg = {
                        "color_score": float(cs),
                        "pref_sim": float(pref_sim),
                        "neg_pen": float(neg_pen),
                        "dup_pen": float(dpen),
                        "novelty_pen": float(novelty_pen),
                        "intent_sim": float(intent_sim),
                        "dislike_rule_pen": float(rule_pen),
                        "mode": mode,
                    }

            if best is None:
                break

            it, cs = best
            items_out.append(
                RecommendedItem(
                    item_id=it.item_id,
                    image_uri=it.image_uri,
                    category=it.category,
                    score=float(best_score),
                    debug=best_dbg,
                )
            )

            ranked_dbg.append({
                "rank": len(items_out),
                "item_id": it.item_id,
                "final": float(best_score),
                **best_dbg,
            })


            picked_embs.append(it.emb)
            remaining = [(x, c) for (x, c) in remaining if x.item_id != it.item_id]


        items_out.sort(key=lambda x: x.score, reverse=True)

        # Generate user-friendly explanations (LLM batch → rule-based fallback)
        explanations_map = self._batch_explain_llm(items_out)
        for ri in items_out:
            ri.explanation_text = explanations_map.get(ri.item_id) or self._generate_item_explanation(
                item=ri,
                debug=ri.debug,
            )
            state.seen.add(ri.item_id)
        state.last_request_id = request_id
        self.store.save(state)

        # --- DEBUG: dump ranking breakdown (optional) ---
        self._maybe_dump_rank_debug(req, ranked_dbg, request_id=request_id)

        return GenerateResponse(
            ok=True,
            session_id=req.session_id,
            request_id=request_id,
            items=items_out,
        )

    def _maybe_dump_rank_debug(self, req: GenerateRequest, ranked_dbg: List[Dict[str, Any]], request_id: str) -> None:
        dump_path = self.cfg.debug_dump_path
        if not dump_path:
            return

        p = Path(dump_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session_id": req.session_id,
            "request_id": request_id,
            "selected_palette_ids": req.selected_palette_ids,
            "filters": req.filters.__dict__ if req.filters else None,
            "top_debug": ranked_dbg[: min(len(ranked_dbg), self.cfg.debug_dump_topk)],
        }
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ---- LLM BATCH EXPLANATION ----
    def _batch_explain_llm(self, items: List[RecommendedItem]) -> Dict[str, str]:
        """
        Call the LLM once to generate varied explanations for all recommended items.
        Returns {item_id: explanation_text}.  Returns {} on failure (caller falls back).
        """
        if self.llm is None or not items:
            return {}

        from aurawear_analysis.recommend.llm.prompts import EXPLANATION_BATCH_INSTRUCTIONS
        from aurawear_analysis.recommend.llm.schemas import EXPLANATION_BATCH_SCHEMA

        # Build compact item list for the prompt
        rows = []
        for rank_idx, ri in enumerate(items, 1):
            dbg = ri.debug or {}
            rows.append({
                "id": ri.item_id,
                "category": ri.category or "",
                "rank": rank_idx,
                "color_score": round(dbg.get("color_score", 0), 3),
                "pref_sim": round(dbg.get("pref_sim", 0), 3),
                "intent_sim": round(dbg.get("intent_sim", 0), 3),
                "novelty_pen": round(dbg.get("novelty_pen", 0), 3),
                "dup_pen": round(dbg.get("dup_pen", 0), 3),
            })

        input_text = json.dumps(rows, ensure_ascii=False)

        try:
            result = self.llm.json_call(
                instructions=EXPLANATION_BATCH_INSTRUCTIONS,
                input_text=input_text,
                schema=EXPLANATION_BATCH_SCHEMA,
            )
            mapping: Dict[str, str] = {}
            for entry in result.get("explanations", []):
                iid = entry.get("item_id", "")
                expl = (entry.get("explanation") or "").strip()
                if iid and expl:
                    mapping[iid] = expl
            print(f"[LLM] batch explanations OK: {len(mapping)}/{len(items)}")
            return mapping
        except Exception as e:
            print(f"[LLM] batch explanations FAILED, falling back: {e!r}")
            return {}

    # ---- RULE-BASED EXPLAINABILITY (fallback) ----
    def _generate_item_explanation(self, item: RecommendedItem, debug: Dict[str, float]) -> str:
        """
        Convert debug scores into a user-friendly recommendation reason.

        Uses varied vocabulary and item-specific context so explanations
        don't all look the same.
        """
        if not debug:
            return "Personalized pick"

        reasons: list[str] = []

        # ── 1. Color match ──
        cs = debug.get("color_score", 0)
        cat = (item.category or "").lower()
        _color_excellent = [
            "Excellent palette match",
            "Colors perfectly complement your palette",
            "Right in your best color zone",
            "Spot-on color harmony",
        ]
        _color_good = [
            "Great color harmony with your palette",
            "Palette-friendly tones",
            "Harmonious hues for you",
            f"Flattering color for a {cat}" if cat else "Flattering color choice",
        ]
        _color_ok = [
            "Palette-compatible tones",
            "Works within your color range",
            "Blends with your palette",
        ]
        if cs >= 0.9:
            reasons.append(random.choice(_color_excellent))
        elif cs >= 0.7:
            reasons.append(random.choice(_color_good))
        elif cs > 0:
            reasons.append(random.choice(_color_ok))

        # ── 2. Preference match ──
        pref_sim = debug.get("pref_sim", 0)
        _pref_strong = [
            "Closely matches your style taste",
            "Very aligned with your preferences",
            "Right in line with what you've liked",
        ]
        _pref_moderate = [
            "Similar to styles you've favored",
            "Echoes your personal style",
            "Resonates with your taste",
        ]
        _pref_mild = [
            "Fits your general style",
            "Solid style match",
        ]
        if pref_sim >= 0.8:
            reasons.append(random.choice(_pref_strong))
        elif pref_sim >= 0.6:
            reasons.append(random.choice(_pref_moderate))
        elif pref_sim > 0.2:
            reasons.append(random.choice(_pref_mild))

        # ── 3. Intent match ──
        intent_sim = debug.get("intent_sim", 0)
        _intent_strong = [
            "Directly matches your search request",
            "Nails the vibe you described",
            "Exactly what you asked for",
        ]
        _intent_mild = [
            "Relevant to your request",
            "Fits the direction you described",
        ]
        if intent_sim >= 0.7:
            reasons.append(random.choice(_intent_strong))
        elif intent_sim > 0.3:
            reasons.append(random.choice(_intent_mild))

        # ── 4. Novelty ──
        novelty_pen = debug.get("novelty_pen", 0)
        if novelty_pen < 0.1:
            _fresh = ["Fresh discovery", "New find for you", "Something you haven't seen"]
            reasons.append(random.choice(_fresh))

        # ── 5. Diversity ──
        dup_pen = debug.get("dup_pen", 0)
        if dup_pen < 0.15:
            _variety = ["Adds variety to your mix", "Brings a different angle", "Diversifies your options"]
            reasons.append(random.choice(_variety))

        if not reasons:
            _fallback = ["Personalized pick", "Curated for you", "Selected for your profile"]
            return random.choice(_fallback)

        selected = reasons[:3]
        if len(selected) == 1:
            return selected[0]
        elif len(selected) == 2:
            return f"{selected[0]} · {selected[1]}"
        else:
            return f"{selected[0]} · {selected[1]} · {selected[2]}"

    # -------------------- TASK MANAGEMENT --------------------
    def start_new_task(self, session_id: str, new_task_id: str) -> None:
        """Phase 2: archive current task pref_vec, reblend old tasks at 0.3 decay,
        clear neg_vecs and avoid_terms for a fresh task context.

        Priority after reset:
          current task intent (1.0) > palette/constraints > blended past prefs (0.3) > neg suppression cleared
        """
        state = self.store.get_or_create(session_id)

        # Archive current task's pref_vec before switching.
        # If current_task_id was never set (first-time use before any explicit task
        # boundary), archive under a synthetic key so cross-task blending still works.
        archive_key = state.current_task_id or "task_initial"
        if state.pref_vec is not None:
            state.task_pref_vecs[archive_key] = state.pref_vec

        # Rebuild effective pref_vec: blend all past tasks at 0.3 weight
        old_vecs = [v for v in state.task_pref_vecs.values() if v is not None]
        if old_vecs:
            blended = np.zeros(old_vecs[0].shape, dtype=np.float32)
            for v in old_vecs:
                blended = blended + 0.3 * v
            state.pref_vec = normalize(blended)
        else:
            state.pref_vec = None

        # Clear task-specific negative signals — old-task dislikes shouldn't hard-block
        # items that fit a different occasion.
        #
        # disliked (item ID set): used as hard ban in candidate generation.
        #   Cleared so e.g. a formal coat blocked for commute can reappear for a wedding.
        # neg_vecs: per-item embedding suppressors. Cleared (hard ban removed).
        # avoid_terms / critique_tags: occasion-specific text signals. Cleared.
        # liked / cart item ID sets: kept — harmless (display only; preference
        #   influence already captured in pref_vec, which is decayed above).
        _cleared_disliked = len(state.disliked)
        _cleared_neg      = len(state.neg_vecs)
        state.disliked = set()
        state.neg_vecs = []
        state.avoid_terms = set()
        state.critique_tags = set()

        state.current_task_id = new_task_id
        self.store.save(state)
        print(
            f"[TASK] New task started"
            f" | new_task_id={new_task_id}"
            f" | archived_task_count={len(state.task_pref_vecs)}"
            f" | blended_pref_task_count={len(old_vecs)}"
            f" | decay_weight=0.30"
            f" | cleared_neg_vec_count={_cleared_neg}"
            f" | cleared_disliked_count={_cleared_disliked}"
            f" | retained_liked_count={len(state.liked)}"
            f" | retained_cart_count={len(state.cart)}"
            f" | blended_pref_vec={'yes' if state.pref_vec is not None else 'none'}"
        )

    # -------------------- FEEDBACK --------------------
    def feedback(self, ev: FeedbackEvent, critique_tags: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        state = self.store.get_or_create(ev.session_id)
        it = self.index.get(ev.item_id)
        if it is None:
            return {"ok": False, "error": "unknown_item"}

        if ev.action == "like":
            state.liked.add(ev.item_id)
            state.pref_vec = update_preference(state.pref_vec, it.emb, step=self.cfg.alpha_like)

        elif ev.action == "cart":
            state.cart.add(ev.item_id)
            state.pref_vec = update_preference(state.pref_vec, it.emb, step=self.cfg.gamma_cart)

        elif ev.action == "dislike":
            state.disliked.add(ev.item_id)
            state.neg_vecs.append(it.emb)

            # Quick critique tags → avoid_terms + critique_tags
            new_avoid: List[str] = []
            if critique_tags:
                for t in critique_tags:
                    t_clean = (t or "").strip().lower()
                    if t_clean:
                        state.avoid_terms.add(t_clean)
                        state.critique_tags.add(t_clean)
                        new_avoid.append(t_clean)

            # GPT dislike tagger: extract avoid_terms ONLY when user gave feedback
            if self.llm is not None and critique_tags:
                from aurawear_analysis.recommend.llm.dislike_tagger import tag_dislike

                item_meta = {
                    "category": it.category,
                }
                _reasons = kwargs.get("reasons", []) or []
                _free_text = kwargs.get("free_text", "") or ""

                rule = tag_dislike(
                    self.llm, item_id=ev.item_id, item_meta=item_meta,
                    critique_tags=critique_tags,
                    reasons=_reasons,
                    free_text=_free_text,
                )

                for t in (rule.get("avoid") or []):
                    if isinstance(t, str) and t.strip():
                        state.avoid_terms.add(t.strip().lower())
                        new_avoid.append(t.strip().lower())

            self.store.save(state)
            return {"ok": True, "session_id": ev.session_id, "action": ev.action,
                    "item_id": ev.item_id, "new_avoid_terms": list(set(new_avoid))}

        else:
            return {"ok": False, "error": "unknown_action"}

        self.store.save(state)
        return {"ok": True, "session_id": ev.session_id, "action": ev.action, "item_id": ev.item_id}
