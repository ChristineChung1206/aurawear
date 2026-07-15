# Authorship and Contributions

This file distinguishes authorship of the software source code contained in this
repository from authorship and contributions associated with the official C&C
2026 publication.

---

## Software Author

### Pei-Chen Chung

Primary system designer and implementer of the AuraWear software source code
contained in this repository.

Technical contributions include:

- system architecture;
- model selection;
- machine-learning and color-analysis pipeline (8-module computer vision
  pipeline: face detection via MediaPipe, person segmentation, optional ONNX
  face parsing, ROI mask construction, color estimation via KMeans, and
  12-season rule classification);
- recommendation and reranking logic (multi-signal scoring: CIE Lab color
  compatibility, CLIP-based text–item similarity, preference vector learning,
  diversity penalties, and negative suppression);
- user interface and Gradio integration (full 5-step wizard UI, state
  management, all callback logic, session history panel);
- LLM integration (intent clarification dialog, explanation generation,
  graceful fallback);
- API integration;
- session memory design;
- testing and debugging;
- software documentation; and
- preparation and maintenance of the public software release.

---

## Associated Research Publication

The official C&C 2026 publication is:

> Pei-Chen Chung and Pei-Hua Chen. 2026. AuraWear: Designing for Human-AI
> Co-Decision in Creative Fashion Recommendation. In *Proceedings of the 2026
> Conference on Creativity and Cognition* (C&C '26). Association for Computing
> Machinery, New York, NY, USA, 1527–1532.
> https://doi.org/10.1145/3803784.3816845

The official author order, affiliations, and publication record are as stated
above. Publication authorship is distinct from authorship of the software source
code contained in this repository.

---

## Documented Research Contributions

The following records contributions to the associated C&C 2026 research and
publication.

### Pei-Chen Chung

- Software implementation
- Technical methodology and system design
- Testing and validation
- Visualization and technical documentation
- Writing — original draft of technical and system-method content, where
  supported by version history

### Pei-Hua Chen

- Proposed the initial high-level research direction of combining machine
  learning and color theory for fashion recommendation
- Writing — review and editing
- Supervision

These documented research roles are distinct from authorship and implementation
of the software source code contained in this repository.
