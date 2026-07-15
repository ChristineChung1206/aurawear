# Contribution Statement

## Primary System Designer and Implementer

**Pei-Chen Chung** is the primary system designer and implementer of the AuraWear software represented in this repository. This document records Pei-Chen Chung's software design, technical development, and implementation contributions to the public AuraWear software release contained in this repository. It does not replace, modify, or determine the official authorship record of the associated published C&C 2026 paper.

### System Contributions

| Component | Description |
|---|---|
| **Gradio-based multi-step interface** | Full 5-step wizard UI (selfie upload → style selection → color analysis → palette editing → recommendation). Includes layout architecture, Gradio state management, all callback chains, style goal dialog, session history panel, and friend collaboration flow. |
| **Personal color analysis workflow** | 8-module computer vision pipeline: image loading (M1), face landmark detection via MediaPipe (M2), person segmentation (M3), optional ONNX semantic face parsing (M3.5), ROI mask construction (M4), pixel filtering and normalization (M5), representative color estimation via KMeans (M6), seasonal feature extraction (M7), and 12-season rule classification (M8). |
| **Editable seasonal palette interface** | Interactive palette with clickable color swatches, real-time selection state, custom HTML rendering, and sidebar integration. Supports per-session palette customization with direct influence on recommendation scoring. |
| **LLM-mediated A/B intent clarification** | Two-option intent interpretation using GPT-4o-mini. Each option contains structured intent patches (style tags, must-haves, avoids) used to compute CLIP-space intent vectors. Includes graceful rule-based fallback when LLM is unavailable. |
| **Feedback-based reranking pipeline** | Multi-signal scoring: CIE Lab ΔE color compatibility, CLIP ViT-B-32 text–item cosine similarity, learned preference vectors (updated via like/dislike/cart), novelty penalty, diversity penalty (cosine deduplication), and negative suppression from dislike-tag extraction. |
| **Session memory design** | In-session preference vector accumulation with exponential moving average updates. Multi-task blending: archived task preference vectors are retained at decayed weight across outfit goal resets. Avoid-terms extracted from dislike tags via LLM and rule-based fallback. |
| **Recommendation explanation module** | Per-item rationale generation using LLM, aligned with the user's chosen style direction and palette. Batched explanation calls with graceful fallback to rule-based descriptions. |
| **Demo screenshots and system figures** | All interface screenshots, system architecture diagrams, and evaluation figures used in project documentation and reports were produced by Pei-Chen Chung. |

### Development Period

2025–2026

### Attribution

When citing or referencing the AuraWear software, please attribute Pei-Chen
Chung as the primary system designer and implementer of the software source code
in this repository. See [CITATION.cff](CITATION.cff) for the recommended
software citation format.

For citation of the published C&C 2026 paper, please use the official
publication record. See the `## Citation` section in [README.md](README.md).

---

## Documented Non-Software Research Contributions

The following records documented non-software contributions to the associated
C&C 2026 research and publication.

### Pei-Hua Chen

- Proposed the initial high-level research direction of combining machine
  learning and color theory for fashion recommendation
- Writing — review and editing
- Supervision

These documented research roles are distinct from authorship and implementation
of the software source code contained in this repository.
