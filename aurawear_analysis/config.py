from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RecoConfig:
    """
    Single source of truth config for palette → recommendation.

    Notes:
    - Hard constraint: color_min_score gates candidates (must pass palette compatibility).
    - Candidate pool: shortlist by color_score then rerank by final objective.
    - Diversity: dup_penalty gives hard suppress (very large) and soft penalty (0~soft_weight).
    """

    # --- candidate generation ---
    top_k_default: int = 50
    candidate_pool_size: int = 500
    color_tau: float = 18.0                 # exp(-ΔE/tau)
    color_min_score: float = 0.35           # HARD GATE

    # --- scoring weights ---
    w_color: float = 1.00
    w_pref: float = 0.60
    w_intent: float = 0.30                  # set 0 if llm disabled
    w_novelty: float = 1.00
    w_dup: float = 1.00

    # --- novelty / repeated exposure ---
    seen_penalty: float = 0.30

    # --- negative suppression (HITL: dislike) ---
    neg_penalty_weight: float = 1.00
    w_dislike_rule: float = 0.35         # avoid_terms soft penalty weight

    # --- diversity (dup suppression) ---
    dup_sim_threshold: float = 0.95      # hard suppress when sim > this

    text_embedder_backend: str = "clip"
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    clip_device: str = "cpu"

    debug_dump_path: str = "logs/rank_debug.json"
    debug_dump_topk: int = 200

    # HITL preference update
    alpha_like: float = 0.35     # like 對偏好向量的步長
    gamma_cart: float = 0.55     # cart 對偏好向量的步長（通常比 like 更強）
