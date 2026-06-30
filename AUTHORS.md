# Authors

## Primary Designer and Implementer

**Pei-Chen Chung**

Pei-Chen Chung is the primary system designer and implementer of AuraWear Analysis. Contributions include:

- **Gradio-based multi-step interface** — full 5-step wizard UI (selfie upload → style selection → color analysis → palette editing → recommendation), including layout, state management, and all callback logic
- **Personal color analysis workflow** — 8-module computer vision pipeline integrating face detection (MediaPipe), person segmentation, optional ONNX face parsing, ROI mask construction, color estimation (KMeans), and 12-season rule classification
- **Editable seasonal palette interface** — interactive clickable palette swatches with real-time selection state, custom HTML rendering, and sidebar integration
- **LLM-mediated A/B intent clarification** — two-option intent interpretation dialog with LLM-generated structured patches (style tags, must-haves, avoids), graceful fallback when LLM is unavailable
- **Feedback-based reranking pipeline** — multi-signal scoring combining CIE Lab color compatibility, CLIP-based text–item similarity, learned preference vectors (like/dislike/cart), diversity penalties, and negative suppression
- **Session memory design** — in-session preference vector accumulation, multi-task blending (archived task weights), avoid-terms extraction from dislike tags, and session history rendering
- **Recommendation explanation module** — per-item rationale generation aligned with the user's chosen style intent
- **Demo screenshots and system figures** — all interface screenshots, architecture diagrams, and evaluation figures included in project documentation
