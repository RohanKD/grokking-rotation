#!/usr/bin/env python3
"""
Build cleaned ABO manifests for novel view synthesis.

Outputs:
  data/abo_processed/manifests/spins_clean.csv
  data/abo_processed/manifests/objects_split.csv
  data/abo_processed/manifests/pairs_train.csv
  data/abo_processed/manifests/pairs_val.csv
  data/abo_processed/manifests/pairs_test.csv

Expected task format:
  source_image, target_image, delta_angle, source_angle, target_angle,
  spin_id, object_id, complexity_score, complexity_quartile

This script is intentionally robust to small ABO directory layout changes.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


REQUIRED_SPIN_COLS = {"spin_id", "azimuth", "image_id", "path"}


def find_csv_with_columns(root: Path, required_cols: set) -> Optional[Path]:
    candidates = list(root.rglob("*.csv"))
    for p in candidates:
        try:
            df = pd.read_csv(p, nrows=5)
            if required_cols.issubset(set(df.columns)):
                return p
        except Exception:
            continue
    return None


def find_json_listing_files(root: Path) -> List[Path]:
    # ABO listings are often JSON lines or compressed/extracted JSON.
    patterns = ["*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz"]
    files = []
    for pat in patterns:
        files.extend(root.rglob(pat))
    return files


def load_listing_metadata(raw_root: Path) -> pd.DataFrame:
    """
    Attempts to extract spin_id -> item_id/product_type/category metadata.
    If unavailable, returns empty dataframe.
    """
    rows = []
    files = find_json_listing_files(raw_root)

    for file in files:
        try:
            compression = "gzip" if file.suffix == ".gz" else None
            with pd.read_json(file, lines=True, compression=compression, chunksize=5000) as reader:
                for chunk in reader:
                    if "spin_id" not in chunk.columns:
                        continue

                    for _, r in chunk.iterrows():
                        spin_id = r.get("spin_id", None)
                        if pd.isna(spin_id) or spin_id is None:
                            continue

                        item_id = r.get("item_id", spin_id)

                        product_type = None
                        pt = r.get("product_type", None)
                        if isinstance(pt, list) and len(pt) > 0 and isinstance(pt[0], dict):
                            product_type = pt[0].get("value")
                        elif isinstance(pt, str):
                            product_type = pt

                        rows.append(
                            {
                                "spin_id": str(spin_id),
                                "item_id": str(item_id),
                                "product_type": product_type,
                            }
                        )
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["spin_id", "item_id", "product_type"])

    out = pd.DataFrame(rows).drop_duplicates("spin_id")
    return out


def resolve_image_path(raw_root: Path, relative_path: str) -> Optional[str]:
    """
    ABO metadata has a relative image path. Try several likely locations.
    """
    if not isinstance(relative_path, str):
        return None

    candidates = [
        raw_root / relative_path,
        raw_root / "spins" / relative_path,
        raw_root / "images" / relative_path,
    ]

    # Sometimes metadata path already begins with spins/
    for c in candidates:
        if c.exists():
            return str(c)

    # Fallback: search by filename, but avoid doing this too much in large dirs.
    fname = Path(relative_path).name
    hits = list(raw_root.rglob(fname))
    if hits:
        return str(hits[0])

    return None


def valid_image(path: str, min_side: int = 32) -> bool:
    try:
        with Image.open(path) as im:
            w, h = im.size
            if w < min_side or h < min_side:
                return False
            return True
    except Exception:
        return False


def image_complexity(path: str) -> Dict[str, float]:
    """
    Cheap object/image complexity features.
    These are not semantic; they are intended for quartile stratification.

    Features:
      - edge_density: fraction of Canny edge pixels
      - entropy: grayscale entropy
      - asymmetry: left-right pixel difference after resize
      - foreground_area_proxy: non-white/non-black visual mass proxy
    """
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2 failed")
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 80, 160)
        edge_density = float((edges > 0).mean())

        hist = cv2.calcHist([gray], [0], None, [64], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        entropy = float(-(hist * np.log(hist + 1e-8)).sum())

        left = gray[:, :64].astype(np.float32)
        right = np.fliplr(gray[:, 64:]).astype(np.float32)
        asymmetry = float(np.mean(np.abs(left - right)) / 255.0)

        # Foreground proxy: assumes product photos often have bright/neutral backgrounds.
        # This is intentionally weak and only used for stratification.
        std = np.std(img, axis=2)
        foreground_area_proxy = float((std > 8).mean())

        return {
            "edge_density": edge_density,
            "entropy": entropy,
            "asymmetry": asymmetry,
            "foreground_area_proxy": foreground_area_proxy,
        }
    except Exception:
        return {
            "edge_density": np.nan,
            "entropy": np.nan,
            "asymmetry": np.nan,
            "foreground_area_proxy": np.nan,
        }


def circular_delta(source_angle: float, target_angle: float) -> float:
    """
    Returns delta in degrees in [0, 360).
    """
    return float((target_angle - source_angle) % 360)


def make_pairs_for_split(
    df_views: pd.DataFrame,
    split_ids: set,
    pairs_per_object: int,
    seed: int,
    heldout_angle_mode: str = "none",
) -> pd.DataFrame:
    """
    Build capped source-target pairs.

    heldout_angle_mode:
      none: use all angles.
      train_sparse_30: for train, sources/targets from multiples of 30 only.
      eval_intermediate: for eval, target angles are non-multiples of 30.
    """
    rng = random.Random(seed)
    rows = []

    subset = df_views[df_views["object_id"].isin(split_ids)].copy()

    for object_id, g in tqdm(subset.groupby("object_id"), desc="Making pairs"):
        g = g.sort_values("azimuth").reset_index(drop=True)

        if len(g) < 2:
            continue

        views = g.to_dict("records")

        if heldout_angle_mode == "train_sparse_30":
            views = [v for v in views if int(v["azimuth"]) % 30 == 0]
        elif heldout_angle_mode == "eval_intermediate":
            views = [v for v in views if int(v["azimuth"]) % 30 != 0]

        if len(views) < 2:
            continue

        max_pairs = min(pairs_per_object, len(views) * (len(views) - 1))

        seen = set()
        tries = 0
        while len(seen) < max_pairs and tries < max_pairs * 20:
            tries += 1
            s, t = rng.sample(views, 2)
            key = (s["image_id"], t["image_id"])
            if key in seen:
                continue
            seen.add(key)

            rows.append(
                {
                    "object_id": object_id,
                    "spin_id": s["spin_id"],
                    "source_image": s["processed_path"],
                    "target_image": t["processed_path"],
                    "source_angle": float(s["azimuth"]),
                    "target_angle": float(t["azimuth"]),
                    "delta_angle": circular_delta(float(s["azimuth"]), float(t["azimuth"])),
                    "complexity_score": float(s["complexity_score"]),
                    "complexity_quartile": int(s["complexity_quartile"]),
                    "product_type": s.get("product_type", None),
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=str, default="data/abo_raw")
    parser.add_argument("--processed-root", type=str, default="data/abo_processed")
    parser.add_argument("--min-views", type=int, default=72)
    parser.add_argument("--require-72", action="store_true")
    parser.add_argument("--pairs-per-object", type=int, default=512)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--heldout-angle-mode",
        type=str,
        default="none",
        choices=["none", "train_sparse_30"],
        help="Use train_sparse_30 to train only on 30-degree multiples.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    manifest_dir = processed_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print("Finding spin metadata CSV...")
    spin_csv = find_csv_with_columns(raw_root, REQUIRED_SPIN_COLS)
    if spin_csv is None:
        raise FileNotFoundError(
            "Could not find spin metadata CSV with columns: "
            f"{sorted(REQUIRED_SPIN_COLS)} under {raw_root}"
        )

    print(f"Using spin metadata: {spin_csv}")
    spins = pd.read_csv(spin_csv)
    spins["spin_id"] = spins["spin_id"].astype(str)
    spins["image_id"] = spins["image_id"].astype(str)
    spins["azimuth"] = spins["azimuth"].astype(float)

    print("Loading listing metadata if available...")
    listings = load_listing_metadata(raw_root)
    if len(listings) > 0:
        spins = spins.merge(listings, on="spin_id", how="left")
    else:
        spins["item_id"] = spins["spin_id"]
        spins["product_type"] = None

    spins["object_id"] = spins["item_id"].fillna(spins["spin_id"]).astype(str)

    print("Resolving image paths...")
    resolved_paths = []
    is_valid = []
    for p in tqdm(spins["path"].tolist(), desc="Resolving paths"):
        local = resolve_image_path(raw_root, p)
        resolved_paths.append(local)
        is_valid.append(local is not None and valid_image(local))

    spins["raw_image_path"] = resolved_paths
    spins["valid_image"] = is_valid
    spins = spins[spins["valid_image"]].copy()

    print("Filtering valid sequences...")
    seq_counts = spins.groupby("spin_id")["azimuth"].nunique().reset_index(name="n_views")

    if args.require_72:
        keep_spin_ids = set(seq_counts[seq_counts["n_views"] == 72]["spin_id"])
    else:
        keep_spin_ids = set(seq_counts[seq_counts["n_views"] >= args.min_views]["spin_id"])

    spins = spins[spins["spin_id"].isin(keep_spin_ids)].copy()

    # Remove duplicate azimuths within spin_id.
    spins = spins.sort_values(["spin_id", "azimuth"]).drop_duplicates(["spin_id", "azimuth"])

    print(f"Remaining images: {len(spins):,}")
    print(f"Remaining spin sequences: {spins['spin_id'].nunique():,}")
    print(f"Remaining objects: {spins['object_id'].nunique():,}")

    print("Computing complexity from a representative image per object...")
    rep = spins.sort_values(["object_id", "azimuth"]).groupby("object_id").head(1).copy()

    comp_rows = []
    for _, r in tqdm(rep.iterrows(), total=len(rep), desc="Complexity"):
        feats = image_complexity(r["raw_image_path"])
        feats["object_id"] = r["object_id"]
        comp_rows.append(feats)

    comp = pd.DataFrame(comp_rows)
    for col in ["edge_density", "entropy", "asymmetry", "foreground_area_proxy"]:
        comp[col] = comp[col].fillna(comp[col].median())

    # Normalize and combine.
    for col in ["edge_density", "entropy", "asymmetry", "foreground_area_proxy"]:
        lo, hi = comp[col].quantile(0.01), comp[col].quantile(0.99)
        comp[col + "_norm"] = ((comp[col] - lo) / (hi - lo + 1e-8)).clip(0, 1)

    comp["complexity_score"] = (
        0.35 * comp["edge_density_norm"]
        + 0.30 * comp["entropy_norm"]
        + 0.25 * comp["asymmetry_norm"]
        + 0.10 * comp["foreground_area_proxy_norm"]
    )

    comp["complexity_quartile"] = pd.qcut(
        comp["complexity_score"],
        q=4,
        labels=[0, 1, 2, 3],
        duplicates="drop",
    ).astype(int)

    spins = spins.merge(
        comp[["object_id", "complexity_score", "complexity_quartile"]],
        on="object_id",
        how="left",
    )

    # processed_path is where the next preprocessing script will put resized images.
    spins["processed_path"] = spins.apply(
        lambda r: str(
            processed_root
            / "images_64"
            / str(r["spin_id"])
            / f"{int(round(float(r['azimuth']))):03d}.png"
        ),
        axis=1,
    )

    spins_clean_path = manifest_dir / "spins_clean.csv"
    spins.to_csv(spins_clean_path, index=False)
    print(f"Wrote {spins_clean_path}")

    print("Creating object-level split...")
    object_df = (
        spins[["object_id", "spin_id", "product_type", "complexity_score", "complexity_quartile"]]
        .drop_duplicates("object_id")
        .reset_index(drop=True)
    )

    objects = object_df["object_id"].tolist()

    train_ids, test_ids = train_test_split(
        objects,
        test_size=args.test_frac,
        random_state=args.seed,
        stratify=object_df["complexity_quartile"] if object_df["complexity_quartile"].nunique() == 4 else None,
    )

    train_df_tmp = object_df[object_df["object_id"].isin(train_ids)]
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=args.val_frac / (1.0 - args.test_frac),
        random_state=args.seed,
        stratify=train_df_tmp["complexity_quartile"] if train_df_tmp["complexity_quartile"].nunique() == 4 else None,
    )

    split_rows = []
    for oid in train_ids:
        split_rows.append({"object_id": oid, "split": "train"})
    for oid in val_ids:
        split_rows.append({"object_id": oid, "split": "val"})
    for oid in test_ids:
        split_rows.append({"object_id": oid, "split": "test"})

    split_df = pd.DataFrame(split_rows)
    object_df = object_df.merge(split_df, on="object_id", how="left")
    split_path = manifest_dir / "objects_split.csv"
    object_df.to_csv(split_path, index=False)
    print(f"Wrote {split_path}")

    train_set = set(object_df[object_df["split"] == "train"]["object_id"])
    val_set = set(object_df[object_df["split"] == "val"]["object_id"])
    test_set = set(object_df[object_df["split"] == "test"]["object_id"])

    print("Building pair manifests...")
    train_mode = args.heldout_angle_mode
    eval_mode = "eval_intermediate" if args.heldout_angle_mode == "train_sparse_30" else "none"

    pairs_train = make_pairs_for_split(
        spins,
        train_set,
        pairs_per_object=args.pairs_per_object,
        seed=args.seed,
        heldout_angle_mode=train_mode,
    )
    pairs_val = make_pairs_for_split(
        spins,
        val_set,
        pairs_per_object=max(128, args.pairs_per_object // 4),
        seed=args.seed + 1,
        heldout_angle_mode=eval_mode,
    )
    pairs_test = make_pairs_for_split(
        spins,
        test_set,
        pairs_per_object=max(128, args.pairs_per_object // 4),
        seed=args.seed + 2,
        heldout_angle_mode=eval_mode,
    )

    for name, df in [
        ("pairs_train.csv", pairs_train),
        ("pairs_val.csv", pairs_val),
        ("pairs_test.csv", pairs_test),
    ]:
        out = manifest_dir / name
        df.to_csv(out, index=False)
        print(f"Wrote {out}: {len(df):,} pairs")

    stats = {
        "n_images": int(len(spins)),
        "n_spins": int(spins["spin_id"].nunique()),
        "n_objects": int(spins["object_id"].nunique()),
        "n_train_objects": int(len(train_set)),
        "n_val_objects": int(len(val_set)),
        "n_test_objects": int(len(test_set)),
        "pairs_train": int(len(pairs_train)),
        "pairs_val": int(len(pairs_val)),
        "pairs_test": int(len(pairs_test)),
        "heldout_angle_mode": args.heldout_angle_mode,
    }

    stats_path = manifest_dir / "build_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()