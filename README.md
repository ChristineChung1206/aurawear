# AuraWear Analysis

AuraWear is a human-in-the-loop fashion recommendation system that integrates personal color analysis with LLM-driven intent understanding. Given a selfie, the system diagnoses the user's 12-season personal color type, constructs a curated seasonal palette, and recommends clothing items that harmonize with the user's natural coloring. Users can refine results through natural-language style goals, explicit item feedback, and iterative preference learning.

## Features

- **Personal color diagnosis** — 8-module computer vision pipeline (face detection, semantic segmentation, color estimation, 12-season rule classification)
- **Editable seasonal palette** — interactive palette editor with clickable color swatches
- **LLM-mediated A/B intent clarification** — two interpretations of the user's style goal are generated and presented for selection
- **Feedback-based reranking** — like, dislike, and cart actions update a preference vector in real time
- **Session memory** — the system tracks preference signals within and across outfit goals
- **Recommendation explanation** — each item shows a brief rationale aligned with the user's intent
- **Graceful degradation** — runs fully offline without OpenAI key or face parsing model (LLM and semantic segmentation features are disabled; all other functionality is intact)

## Demo Mode

The public repository ships with a synthetic sample catalog (`data/sample_catalog.csv`) and a placeholder image. No DeepFashion-MultiModal files, CelebAMask-HQ weights, or personal data are included. See [DATA_POLICY.md](DATA_POLICY.md) for the full policy.

## Quick Start

```bash
# 1. Create environment
conda create -n aurawear python=3.10
conda activate aurawear

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (copy and edit .env)
cp .env.example .env
# Optionally add OPENAI_API_KEY for LLM features

# 4. Run
python app_gradio.py
# Open http://127.0.0.1:7860
```

The default `.env.example` points to the synthetic demo catalog. To use the full DeepFashion-backed pipeline, obtain the dataset yourself (see below) and update `CATALOG_PATH` accordingly.

## Full Dataset Setup (Optional)

The full pipeline uses [DeepFashion-MultiModal](https://github.com/yumingj/DeepFashion-MultiModal) as the product catalog. This dataset is **not included** in this repository and has a non-commercial research-only license. To reproduce the full dataset-backed pipeline:

1. Obtain the dataset directly from the [DeepFashion-MultiModal release page](https://github.com/yumingj/DeepFashion-MultiModal) and accept their terms.
2. Place the downloaded files under `aurawear_analysis/data/products/`.
3. Run the preprocessing script:
   ```bash
   python tools/prepare_deepfashion.py \
       --data-dir aurawear_analysis/data/products \
       --output   aurawear_analysis/data/items_deepfashion.json \
       --device   cpu
   ```
4. In your `.env`, set:
   ```
   CATALOG_PATH=aurawear_analysis/data/items_deepfashion.json
   IMAGES_DIR=aurawear_analysis/data/products/images
   ```

## Face Parsing Model (Optional)

The face parsing module uses a BiSeNet model trained on CelebAMask-HQ. This model is **not included** due to license restrictions. If you have access to a compatible model, set `FACE_MODEL_PATH` in your `.env`. Without the model, the pipeline falls back to heuristic ROI masks automatically.

## Project Structure

```
aurawear_analysis/
├── app_gradio.py                  ← main application
├── requirements.txt
├── .env.example                   ← environment template
├── data/
│   └── sample_catalog.csv         ← synthetic demo catalog (bundled)
├── assets/
│   ├── palette18.json
│   └── synthetic_demo_images/
│       └── placeholder.png
├── aurawear_analysis/
│   ├── color_analysis/            ← personal color analysis pipeline
│   ├── recommend/                 ← recommendation and reranking engine
│   └── config.py
└── tools/
    └── prepare_deepfashion.py     ← dataset preprocessing (requires DeepFashion)
```

## License

Source code: [GNU Affero General Public License v3.0](LICENSE) (AGPL-3.0)  
Documentation and diagrams: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0), unless otherwise noted.

See [NOTICE.md](NOTICE.md) for third-party attributions.

## Citation

If you use this work, please cite it as described in [CITATION.cff](CITATION.cff).

## Authors

See [AUTHORS.md](AUTHORS.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Data Policy

See [DATA_POLICY.md](DATA_POLICY.md).
