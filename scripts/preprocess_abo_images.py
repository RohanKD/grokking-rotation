#!/usr/bin/env python3
"""
Preprocess ABO spin images into 64x64 PNGs.

Cleaning:
  - verify image can be opened
  - convert to RGB
  - optional foreground crop using background heuristic
  - square pad
  - resize to 64x64
  - write normalized image

This script does not overwrite unless --overwrite is passed.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


def find_foreground_bbox(img_rgb: np.ndarray, threshold: int = 245):
    """
    Simple background heuristic:
    Many ABO product spin images are on bright backgrounds.
    We estimate foreground as pixels sufficiently different from near-white.

    Returns x0, y0, x1, y1 or None.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Pixel is foreground if not close to white background.
    mask = gray < threshold

    # Also catch colorful/light objects by using saturation.
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    mask = np.logical_or(mask, sat > 20)

    # Clean small noise.
    mask = mask.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(mask > 0)
    if len(xs) < 20 or len(ys) < 20:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    h, w = gray.shape
    area = (x1 - x0 + 1) * (y1 - y0 + 1)
    if area < 0.01 * h * w:
        return None

    return int(x0), int(y0), int(x1), int(y1)


def crop_pad_resize(
    in_path: str,
    out_path: str,
    size: int = 64,
    crop: bool = True,
    pad_value: int = 255,
):
    with Image.open(in_path) as im:
        im = im.convert("RGB")
        arr = np.array(im)

    if crop:
        bbox = find_foreground_bbox(arr)
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            pad = int(0.08 * max(x1 - x0 + 1, y1 - y0 + 1))
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(arr.shape[1] - 1, x1 + pad)
            y1 = min(arr.shape[0] - 1, y1 + pad)
            arr = arr[y0 : y1 + 1, x0 : x1 + 1]

    im = Image.fromarray(arr)

    # Square pad.
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (pad_value, pad_value, pad_value))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))

    canvas = canvas.resize((size, size), Image.BICUBIC)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/abo_processed/manifests/spins_clean.csv")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    n_ok = 0
    n_fail = 0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        raw_path = r["raw_image_path"]
        out_path = r["processed_path"]

        if Path(out_path).exists() and not args.overwrite:
            n_ok += 1
            continue

        try:
            crop_pad_resize(
                raw_path,
                out_path,
                size=args.size,
                crop=not args.no_crop,
            )
            n_ok += 1
        except Exception as e:
            n_fail += 1

    print(f"Done. ok={n_ok:,}, fail={n_fail:,}")


if __name__ == "__main__":
    main()