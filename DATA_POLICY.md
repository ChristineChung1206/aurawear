# Data Policy

## What This Repository Does Not Include

This repository **does not redistribute** any of the following:

- **DeepFashion-MultiModal images** — product images from the DeepFashion-MultiModal dataset
- **DeepFashion-MultiModal annotations** — shape, texture, fabric, pattern label files
- **DeepFashion-derived embeddings** — pre-computed CLIP embeddings derived from DeepFashion images
- **DeepFashion-derived catalog files** — `items_deepfashion.json` or any processed catalog built from DeepFashion data
- **CelebAMask-HQ data** — face images or segmentation masks from the CelebAMask-HQ dataset
- **Model weights trained on CelebAMask-HQ** — face parsing ONNX model or PyTorch checkpoint
- **Participant data** — any real user sessions, interaction logs, or collected feedback
- **Real selfies** — no personal photographs of any individual
- **API keys** — no OpenAI or other service credentials

## What the Public Demo Uses

The public demo uses **synthetic, fully artificial sample data** only:

- `data/sample_catalog.csv` — 20 hand-crafted mock fashion items with no connection to any real dataset
- `assets/synthetic_demo_images/placeholder.png` — a programmatically generated grey placeholder image

## How to Access the Full Dataset

The full pipeline uses DeepFashion-MultiModal as the product catalog. To reproduce it:

1. Obtain the dataset from its official source and accept the dataset's terms:
   https://github.com/yumingj/DeepFashion-MultiModal
2. Place the downloaded files under `aurawear_analysis/data/products/`
3. Run `tools/prepare_deepfashion.py` to build the processed catalog locally
4. Set `CATALOG_PATH` and `IMAGES_DIR` in your `.env`

The dataset is **not available** through this repository or its maintainers.

## Contact

For questions about data usage or policy, open an issue on this repository.
