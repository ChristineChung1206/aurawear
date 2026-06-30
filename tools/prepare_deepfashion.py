#!/usr/bin/env python3
"""
prepare_deepfashion.py — DeepFashion-MultiModal → items.json 前處理 Pipeline

資料來源：https://github.com/yumingj/DeepFashion-MultiModal
    只需下載 image/, parsing/, labels/, textual_descriptions.json

Step 0 : 解壓 zip（如果尚未解壓）
Step 1 : 解析 labels (shape / fabric / color) + textual descriptions → metadata dict
Step 2 : 用 parsing mask 裁切衣物區域 → 提取 dominant_hex（KMeans）
Step 3 : 對原圖提取 CLIP embedding（ViT-B-32, 512-dim）
Step 4 : 組裝成 ItemIndex 格式的 items.json

使用方式：
    python tools/prepare_deepfashion.py \
        --data-dir aurawear_analysis/data/products \
        --output   aurawear_analysis/data/items_deepfashion.json \
        --device   mps          # cpu / cuda / mps
        --max-items 0           # 0=全部, >0=限制處理數量（測試用）
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# ─────────────── 常數 ──────────────────────────────────────────────────────

# DeepFashion-MultiModal 24-class parsing labels
PARSING_CLASSES = {
    0: "background", 1: "top", 2: "outer", 3: "skirt",
    4: "dress", 5: "pants", 6: "leggings", 7: "headwear",
    8: "eyeglass", 9: "neckwear", 10: "belt", 11: "footwear",
    12: "bag", 13: "hair", 14: "face", 15: "skin",
    16: "ring", 17: "wrist wearing", 18: "socks", 19: "gloves",
    20: "necklace", 21: "rompers", 22: "earrings", 23: "tie",
}

# 衣物類別 → 推薦系統 category 映射
# 只取主要衣物類別（排除配件、人體部位）
GARMENT_CLASSES = {
    1: "top",
    2: "outer",
    3: "skirt",
    4: "dress",
    5: "pants",
    6: "leggings",
    21: "rompers",
}

# shape 屬性名稱與值
SHAPE_ATTRS = {
    0: ("sleeve_length",       {0: "sleeveless", 1: "short-sleeve", 2: "medium-sleeve", 3: "long-sleeve", 4: "not-long-sleeve"}),
    1: ("lower_length",        {0: "three-point", 1: "medium-short", 2: "three-quarter", 3: "long"}),
    2: ("socks",               {0: "no-socks", 1: "socks", 2: "leggings"}),
    3: ("hat",                 {0: "no-hat", 1: "hat"}),
    4: ("glasses",             {0: "no-glasses", 1: "eyeglasses", 2: "sunglasses"}),
    5: ("neckwear",            {0: "no-neckwear", 1: "neckwear"}),
    6: ("wrist_wearing",       {0: "no-wrist-accessory", 1: "wrist-accessory"}),
    7: ("ring",                {0: "no-ring", 1: "ring"}),
    8: ("waist_accessories",   {0: "no-belt", 1: "belt", 2: "waist-clothing", 3: "hidden"}),
    9: ("neckline",            {0: "v-shape", 1: "square", 2: "round", 3: "standing", 4: "lapel", 5: "suspenders"}),
    10: ("is_cardigan",        {0: "cardigan", 1: "not-cardigan"}),
    11: ("covers_navel",       {0: "crop", 1: "covers-navel"}),
}

FABRIC_MAP = {0: "denim", 1: "cotton", 2: "leather", 3: "furry", 4: "knitted", 5: "chiffon", 6: "other"}
COLOR_MAP  = {0: "floral", 1: "graphic", 2: "striped", 3: "pure-color", 4: "lattice", 5: "other", 6: "color-block"}

FABRIC_REGIONS = ["upper", "lower", "outer"]
COLOR_REGIONS  = ["upper", "lower", "outer"]


# ─────────────── Step 0 : 解壓 ─────────────────────────────────────────────

def unzip_if_needed(data_dir: Path) -> None:
    """自動解壓 zip 檔案到 data_dir（保留 zip 內部結構）。

    images.zip  → data_dir/images/  (圖片)
    parsing.zip → data_dir/segm/    (parsing masks)
    labels.zip  → data_dir/labels/  (shape + texture)
    """
    # (zip 檔名, 解壓後應出現的資料夾名)  — 只要該資料夾存在就跳過
    zip_expect = [
        (["images.zip", "image.zip"], ["images", "image"]),
        (["parsing.zip"],             ["segm", "parsing"]),
        (["labels.zip"],              ["labels"]),
    ]
    for zip_names, expected_folders in zip_expect:
        # 檢查是否已有解壓結果
        already_exists = False
        for ef in expected_folders:
            target = data_dir / ef
            if target.exists() and any(target.iterdir()):
                print(f"  ✓ {ef}/ 已存在，跳過解壓")
                already_exists = True
                break
        if already_exists:
            continue

        # 嘗試解壓
        for zn in zip_names:
            zp = data_dir / zn
            if zp.exists():
                print(f"  ⏳ 解壓 {zn} → {data_dir}/ ...")
                with zipfile.ZipFile(zp, "r") as zf:
                    zf.extractall(data_dir)   # 直接解壓到 data_dir，保留 zip 內部結構
                print(f"  ✓ {zn} 解壓完成")
                break
        else:
            print(f"  ⚠  找不到 {zip_names}，也沒有 {expected_folders} 資料夾")


# ─────────────── Step 1 : 解析 Labels & Descriptions ───────────────────────

def _find_label_file(labels_dir: Path, candidates: List[str]) -> Optional[Path]:
    """在 labels_dir 下搜尋多個可能的標籤檔路徑。"""
    for cand in candidates:
        p = labels_dir / cand
        if p.exists():
            return p
    # 也在 labels_dir 的子資料夾中遞迴搜尋
    for cand in candidates:
        fname = Path(cand).name
        found = list(labels_dir.rglob(fname))
        if found:
            return found[0]
    return None


def _parse_label_file(path: Path, n_fields: int) -> Dict[str, List[int]]:
    """讀取 shape_anno_all.txt / fabric_ann.txt / pattern_ann.txt。"""
    result: Dict[str, List[int]] = {}
    if path is None or not path.exists():
        return result
    print(f"  ✓ 載入標籤: {path.name} ({path.parent.name}/)")
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 1 + n_fields:
                continue
            fname = parts[0]
            vals = [int(x) for x in parts[1 : 1 + n_fields]]
            result[fname] = vals
    return result


def load_metadata(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    讀取 labels + textual descriptions，回傳:
      { "000001.jpg": { "shape": [...], "fabric": [...], "color_pattern": [...],
                        "description": "...", "tags": [...], "style": "...",
                        "category": "..." }, ... }
    """
    labels_dir = data_dir / "labels"

    # ── 搜尋 shape 檔案（可能是 shape.txt 或 shape/shape_anno_all.txt） ──
    shape_path = _find_label_file(labels_dir, ["shape.txt", "shape/shape_anno_all.txt", "shape_anno_all.txt"])
    fabric_path = _find_label_file(labels_dir, ["fabric.txt", "texture/fabric_ann.txt", "fabric_ann.txt"])
    pattern_path = _find_label_file(labels_dir, ["color.txt", "texture/pattern_ann.txt", "pattern_ann.txt"])

    shape_data  = _parse_label_file(shape_path, 12) if shape_path else {}
    fabric_data = _parse_label_file(fabric_path, 3) if fabric_path else {}
    color_data  = _parse_label_file(pattern_path, 3) if pattern_path else {}
    if not shape_path:
        print("  ⚠  找不到 shape 標籤檔")
    if not fabric_path:
        print("  ⚠  找不到 fabric 標籤檔")
    if not pattern_path:
        print("  ⚠  找不到 pattern/color 標籤檔")

    # textual descriptions
    desc_data: Dict[str, str] = {}
    for desc_candidate in [
        data_dir / "textual_descriptions.json",
        data_dir / "textual descriptions.json",       # 有空格的版本
        data_dir / "textual_descriptions" / "textual_descriptions.json",
    ]:
        if desc_candidate.exists():
            print(f"  ✓ 載入描述: {desc_candidate.name}")
            with open(desc_candidate, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # 支援 dict (key=fname) 或 list (entry with 'name'+'description')
            if isinstance(raw, dict):
                desc_data = {k: v for k, v in raw.items()}
            elif isinstance(raw, list):
                for entry in raw:
                    if isinstance(entry, dict):
                        name = entry.get("name") or entry.get("image") or entry.get("file_name", "")
                        desc = entry.get("description") or entry.get("text") or entry.get("caption", "")
                        if name and desc:
                            desc_data[name] = desc
            break

    # 組合
    all_fnames = sorted(set(shape_data) | set(fabric_data) | set(color_data) | set(desc_data))
    print(f"  ✓ 標籤涵蓋 {len(all_fnames)} 張圖片 (shape={len(shape_data)}, fabric={len(fabric_data)}, color={len(color_data)}, desc={len(desc_data)})")

    meta: Dict[str, Dict[str, Any]] = {}
    for fname in all_fnames:
        tags: List[str] = []
        style_parts: List[str] = []

        # shape attributes → tags
        if fname in shape_data:
            for idx, val in enumerate(shape_data[fname]):
                attr_name, val_map = SHAPE_ATTRS.get(idx, (None, None))
                if attr_name and val_map and val in val_map:
                    tag = val_map[val]
                    if not tag.startswith("no-") and tag != "hidden":
                        tags.append(tag)
                        if attr_name in ("sleeve_length", "neckline"):
                            style_parts.append(tag)

        # fabric → tags
        if fname in fabric_data:
            for i, val in enumerate(fabric_data[fname]):
                if val in FABRIC_MAP:
                    tags.append(f"{FABRIC_REGIONS[i]}:{FABRIC_MAP[val]}")
                    if i == 0:  # upper fabric → style
                        style_parts.append(FABRIC_MAP[val])

        # color/pattern → tags
        if fname in color_data:
            for i, val in enumerate(color_data[fname]):
                if val in COLOR_MAP:
                    tags.append(f"{COLOR_REGIONS[i]}:{COLOR_MAP[val]}")

        meta[fname] = {
            "shape_raw":        shape_data.get(fname, []),
            "fabric_raw":       fabric_data.get(fname, []),
            "color_pattern_raw": color_data.get(fname, []),
            "description":      desc_data.get(fname, ""),
            "tags":             tags,
            "style":            " ".join(style_parts) if style_parts else "",
        }

    return meta


# ─────────────── Step 2 : Parsing Mask → Dominant Color ────────────────────

def _find_dir_flexible(data_dir: Path, names: List[str], description: str) -> Path:
    """在 data_dir 下尋找多個可能的子資料夾名稱（也處理多包一層的情況）。"""
    for name in names:
        candidate = data_dir / name
        if candidate.exists() and candidate.is_dir():
            # 有時 zip 解壓會多包一層同名資料夾
            for inner_name in names:
                inner = candidate / inner_name
                if inner.exists() and inner.is_dir():
                    return inner
            return candidate
    raise FileNotFoundError(
        f"找不到 {description} 資料夾 ({names}): {data_dir}\n"
        f"  目前有: {[p.name for p in data_dir.iterdir() if p.is_dir()]}"
    )


def _find_images_dir(data_dir: Path) -> Path:
    return _find_dir_flexible(data_dir, ["images", "image"], "image")


def _find_parsing_dir(data_dir: Path) -> Path:
    return _find_dir_flexible(data_dir, ["segm", "parsing", "segmentation"], "parsing")


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return f"#{r:02X}{g:02X}{b:02X}"


def extract_garment_dominant_color(
    img: np.ndarray,
    parsing: np.ndarray,
    garment_ids: List[int],
    n_colors: int = 1,
    max_pixels: int = 5000,
) -> Tuple[List[str], str]:
    """
    用 parsing mask 裁切衣物像素，跑 KMeans 取主色。

    Args:
        img:         (H, W, 3) RGB
        parsing:     (H, W) int, class indices 0-23
        garment_ids: 哪些 class id 視為「這件衣物」
        n_colors:    主色數量
        max_pixels:  KMeans 最大取樣量

    Returns:
        (dominant_hex_list, primary_category)
    """
    # 合併所有指定衣物類別的 mask
    mask = np.isin(parsing, garment_ids)
    pixel_count = int(mask.sum())

    if pixel_count < 50:
        return ["#808080"], "unknown"

    # 取出衣物像素
    garment_pixels = img[mask].reshape(-1, 3).astype(np.float32)

    # 隨機取樣
    if garment_pixels.shape[0] > max_pixels:
        idx = np.random.choice(garment_pixels.shape[0], size=max_pixels, replace=False)
        garment_pixels = garment_pixels[idx]

    # KMeans
    k = min(n_colors, garment_pixels.shape[0])
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(garment_pixels)

    centers = kmeans.cluster_centers_.astype(np.uint8)
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)

    hex_list = [rgb_to_hex(tuple(centers[i])) for i in sorted_idx[:n_colors]]

    # 判斷主要 category（佔最多像素的衣物類別）
    cat_counts = {}
    for gid in garment_ids:
        c = int((parsing == gid).sum())
        if c > 0:
            cat_counts[GARMENT_CLASSES.get(gid, "other")] = c
    primary_cat = max(cat_counts, key=cat_counts.get) if cat_counts else "other"

    return hex_list, primary_cat


def determine_garment_classes(parsing: np.ndarray) -> List[int]:
    """
    根據 parsing mask 決定這張圖的主要衣物類別。
    策略：找出佔像素最多的「上半身衣物」，建構合理的衣物類別組合。
    """
    present = {}
    for gid in GARMENT_CLASSES:
        c = int((parsing == gid).sum())
        if c > 50:
            present[gid] = c

    if not present:
        return list(GARMENT_CLASSES.keys())  # fallback: 全部

    # 如果有 dress 或 rompers（連身），直接用它們
    if 4 in present:   return [4]        # dress
    if 21 in present:  return [21]       # rompers

    # 否則分別建立上/下半身
    garment_ids = []
    # 上半身: top > outer
    for gid in [1, 2]:
        if gid in present:
            garment_ids.append(gid)
    # 下半身: pants > skirt > leggings
    for gid in [5, 3, 6]:
        if gid in present:
            garment_ids.append(gid)

    return garment_ids if garment_ids else list(present.keys())


# ─────────────── Step 3 : CLIP Embedding ───────────────────────────────────

def load_clip_model(device: str = "cpu"):
    """Lazy load CLIP model. Returns (model, preprocess, device_str)."""
    import torch
    import open_clip

    if device == "mps" and not torch.backends.mps.is_available():
        print("  ⚠  MPS 不可用，退回 CPU")
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("  ⚠  CUDA 不可用，退回 CPU")
        device = "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval().to(device)
    print(f"  ✓ CLIP ViT-B-32 loaded on {device}")
    return model, preprocess, device


def extract_clip_embedding(
    img_pil: Image.Image,
    model,
    preprocess,
    device: str,
) -> np.ndarray:
    """提取 512-dim L2-normalized CLIP image embedding。"""
    import torch

    img_t = preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img_t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    v = feat.squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(v)) + 1e-12
    return (v / norm).astype(np.float32)


# ─────────────── Step 4 : 主 Pipeline ──────────────────────────────────────

def build_items(
    data_dir: Path,
    output_path: Path,
    device: str = "cpu",
    n_dominant_colors: int = 1,
    max_items: int = 0,
    batch_save_every: int = 500,
) -> None:
    """
    完整 pipeline：解壓 → metadata → dominant color → CLIP → items.json
    """
    print("\n" + "=" * 70)
    print("🎨 DeepFashion-MultiModal → items.json Pipeline")
    print("=" * 70)

    # ── Step 0 : 解壓 ──
    print("\n📦 Step 0: 檢查並解壓資料...")
    unzip_if_needed(data_dir)

    # ── 找到實際路徑 ──
    images_dir  = _find_images_dir(data_dir)
    parsing_dir = _find_parsing_dir(data_dir)
    print(f"  ✓ images  → {images_dir}")
    print(f"  ✓ parsing → {parsing_dir}")

    # ── Step 1 : 載入 metadata ──
    print("\n📋 Step 1: 載入 labels & descriptions...")
    metadata = load_metadata(data_dir)

    # ── 列出有 parsing mask 的圖片（只處理全身照） ──
    parsing_files = sorted(parsing_dir.glob("*.png"))
    print(f"  ✓ 找到 {len(parsing_files)} 張 parsing mask")

    if max_items > 0:
        parsing_files = parsing_files[:max_items]
        print(f"  ⚠  限制處理前 {max_items} 張（測試模式）")

    # ── 載入 CLIP 模型 ──
    print(f"\n🧠 Step 3 (前置): 載入 CLIP 模型 (device={device})...")
    clip_model, clip_preprocess, clip_device = load_clip_model(device)

    # ── 逐張處理 ──
    print(f"\n🔄 Processing {len(parsing_files)} images...\n")
    items: List[Dict[str, Any]] = []
    errors = 0

    for i, parse_path in enumerate(parsing_files, 1):
        # 從 parsing 檔名推斷原圖檔名
        # parsing 可能是 000001.png 或 000001_segm.png
        stem = parse_path.stem.replace("_segm", "")
        img_fname = f"{stem}.jpg"
        img_path = images_dir / img_fname

        if not img_path.exists():
            # 嘗試 .png
            img_path = images_dir / f"{stem}.png"
            img_fname = f"{stem}.png"
            if not img_path.exists():
                if errors < 5:
                    print(f"  ⚠  [{i}] 找不到圖片: {stem}.*")
                errors += 1
                continue

        try:
            # 讀取圖片 & parsing mask
            img_pil = Image.open(img_path).convert("RGB")
            img_np  = np.array(img_pil)

            parse_np = np.array(Image.open(parse_path))
            # 確保 parsing 是單通道
            if parse_np.ndim == 3:
                parse_np = parse_np[:, :, 0]

            # 確保尺寸一致
            if img_np.shape[:2] != parse_np.shape[:2]:
                # resize parsing to match image
                parse_pil = Image.fromarray(parse_np).resize(
                    (img_np.shape[1], img_np.shape[0]), Image.NEAREST
                )
                parse_np = np.array(parse_pil)

            # ── Step 2 : 裁切衣物 → dominant color ──
            garment_ids = determine_garment_classes(parse_np)
            dominant_hex, category = extract_garment_dominant_color(
                img_np, parse_np, garment_ids, n_colors=n_dominant_colors
            )

            # ── Step 3 : CLIP embedding ──
            emb = extract_clip_embedding(img_pil, clip_model, clip_preprocess, clip_device)

            # ── Step 1 (cont.) : metadata ──
            meta = metadata.get(img_fname, {})
            # 也用不帶副檔名的 key 嘗試
            if not meta:
                meta = metadata.get(stem, {})

            tags = meta.get("tags", [])
            style = meta.get("style", "")
            description = meta.get("description", "")

            # 構建 item_id
            item_id = f"df_{stem}"

            item = {
                "item_id":       item_id,
                "image_uri":     img_fname,
                "category":      category,
                "dominant_hex":  dominant_hex,
                "emb":           emb.tolist(),
                "product_link":  None,
                "title":         description[:80] if description else "",
                "style":         style,
                "brand":         "",
                "tags":          tags,
                "meta_text":     description,
            }
            items.append(item)

            # 進度
            if i <= 5 or i % 200 == 0 or i == len(parsing_files):
                print(
                    f"  ✓ [{i:>5}/{len(parsing_files)}]  {item_id}  "
                    f"cat={category:<8s}  hex={dominant_hex[0]}  "
                    f"tags={len(tags)}  emb_dim={len(item['emb'])}"
                )

            # 中途存檔（防止中斷遺失）
            if batch_save_every > 0 and i % batch_save_every == 0:
                _save_items(items, output_path, partial=True)

        except Exception as e:
            if errors < 10:
                print(f"  ✗ [{i}] {img_fname}: {e}")
            errors += 1

    # ── Step 4 : 寫入 items.json ──
    print(f"\n💾 Step 4: 寫入 {len(items)} 筆商品...")
    _save_items(items, output_path, partial=False)

    # ── 統計 ──
    print("\n" + "=" * 70)
    print(f"✅ 完成！")
    print(f"   - 成功: {len(items)}")
    print(f"   - 失敗: {errors}")
    print(f"   - 輸出: {output_path}")

    # 類別統計
    cat_counts: Dict[str, int] = defaultdict(int)
    for it in items:
        cat_counts[it["category"]] += 1
    print(f"   - 類別分佈:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"       {cat:<12s} {cnt:>6,}")
    print("=" * 70)


def _save_items(items: List[Dict[str, Any]], path: Path, partial: bool = False) -> None:
    """存檔。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False)
    label = "(中途存檔)" if partial else ""
    print(f"  💾 已寫入 {len(items)} 筆 → {path} {label}")


# ─────────────── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DeepFashion-MultiModal → items.json 前處理 Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        default="aurawear_analysis/data/products",
        help="放置 zip / 解壓資料的根目錄 (default: aurawear_analysis/data/products)",
    )
    parser.add_argument(
        "--output",
        default="aurawear_analysis/data/items_deepfashion.json",
        help="輸出 items.json 路徑",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="CLIP 推理裝置",
    )
    parser.add_argument(
        "--n-colors",
        type=int,
        default=1,
        help="每張圖提取的主色數量",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="最多處理幾張（0=全部，>0 用於測試）",
    )
    parser.add_argument(
        "--batch-save",
        type=int,
        default=500,
        help="每處理 N 張中途存檔一次（0=關閉）",
    )
    args = parser.parse_args()

    build_items(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        device=args.device,
        n_dominant_colors=args.n_colors,
        max_items=args.max_items,
        batch_save_every=args.batch_save,
    )


if __name__ == "__main__":
    main()
