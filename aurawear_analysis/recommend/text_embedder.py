from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union
import numpy as np

from aurawear_analysis.recommend.reranker import normalize as _l2_normalize


@dataclass
class TextEmbedderConfig:
    backend: str = "hash"  # "clip" | "hash"
    expected_dim: int = 512

    # CLIP backend options (open_clip)
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    device: str = "cpu"  # "cpu" or "cuda"


class TextEmbedder:
    """
    Clean interface for text embeddings with lazy loading.
    - backend="clip": real CLIP text encoder (open_clip_torch)
    - backend="hash": deterministic dev embedding with requested dim
    """
    def __init__(self, cfg: TextEmbedderConfig):
        self.cfg = cfg
        self._clip_model = None
        self._clip_tokenizer = None
        self._torch = None

    # ---------- public ----------
    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        if self.cfg.backend == "clip":
            vecs = self._embed_clip(texts)
        elif self.cfg.backend == "hash":
            vecs = self._embed_hash(texts, self.cfg.expected_dim)
        else:
            raise ValueError(f"Unknown TextEmbedder backend: {self.cfg.backend}")

        # enforce dim
        for v in vecs:
            if v.shape[0] != self.cfg.expected_dim:
                raise ValueError(
                    f"TextEmbedder produced dim={v.shape[0]} but expected_dim={self.cfg.expected_dim}. "
                    "Your item embeddings and text embeddings must match."
                )
        return vecs

    # ---------- backends ----------
    def _lazy_load_clip(self) -> None:
        if self._clip_model is not None:
            return

        import importlib

        try:
            torch = importlib.import_module("torch")
            open_clip = importlib.import_module("open_clip")
        except Exception as e:
            raise RuntimeError(
                "CLIP backend requires `torch` and `open_clip_torch`.\n"
                "Install:\n"
                "  pip install torch open_clip_torch\n"
                f"Original error: {repr(e)}"
            )

        self._torch = torch

        import warnings, logging
        # Suppress QuickGELU mismatch warning from open_clip
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="QuickGELU mismatch")
            # Suppress HF Hub unauthenticated request warning
            _hf_logger = logging.getLogger("huggingface_hub.utils._http")
            _prev_level = _hf_logger.level
            _hf_logger.setLevel(logging.ERROR)
            try:
                model, _, _ = open_clip.create_model_and_transforms(
                    self.cfg.clip_model,
                    pretrained=self.cfg.clip_pretrained,
                )
            finally:
                _hf_logger.setLevel(_prev_level)

        tokenizer = open_clip.get_tokenizer(self.cfg.clip_model)

        model.eval()
        model = model.to(self.cfg.device)

        self._clip_model = model
        self._clip_tokenizer = tokenizer


    def _embed_clip(self, texts: List[str]) -> List[np.ndarray]:
        self._lazy_load_clip()
        torch = self._torch

        # tokenize
        toks = self._clip_tokenizer([t if t else "" for t in texts]).to(self.cfg.device)

        with torch.no_grad():
            feats = self._clip_model.encode_text(toks)  # (B, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        feats = feats.detach().cpu().numpy().astype(np.float32)
        return [feats[i] for i in range(feats.shape[0])]

    def _embed_hash(self, texts: List[str], dim: int) -> List[np.ndarray]:
        import hashlib

        out = []
        for text in texts:
            t = (text or "").strip().lower() or "__empty__"
            h = hashlib.sha256(t.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "little", signed=False)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(dim).astype(np.float32)
            out.append(_l2_normalize(v))
        return out
