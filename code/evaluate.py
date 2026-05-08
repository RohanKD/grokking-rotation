"""Evaluation script: load a checkpoint and report per-object / quartile metrics.

Usage
-----
    python evaluate.py \
        --checkpoint runs/exp_default/final.pt \
        --data_root /path/to/coil-100 \
        --output_dir runs/exp_default/eval

Outputs
-------
  eval_per_object.csv  – per-object LPIPS and SSIM
  eval_summary.csv     – mean±std per complexity quartile
  Prints a summary table stratified by complexity quartile.
  Also computes a copy-source baseline (predict target = source).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from data.coil100 import COIL100Dataset, get_train_test_split, build_dataloader
from models.unet import UNet
from models.ddpm import ConditionalDDPM, EMAHelper
from metrics.perceptual import PerceptualMetrics
from complexity import compute_all_complexities, get_quartile_splits
from train import load_config, build_model, load_checkpoint


# ---------------------------------------------------------------------------
# Copy-source baseline
# ---------------------------------------------------------------------------


def copy_source_baseline(
    dataloader: torch.utils.data.DataLoader,
    metrics: PerceptualMetrics,
    device: torch.device,
) -> Dict[int, Dict[str, float]]:
    """Evaluate the trivial baseline: predict target = source.

    Returns dict mapping object_id → {'lpips': float, 'ssim': float}.
    """
    per_object: Dict[int, List] = {}
    for batch in tqdm(dataloader, desc="Copy-source baseline"):
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        obj_ids = batch["object_id"].tolist()

        for i, oid in enumerate(obj_ids):
            lpips_val = metrics.compute_lpips(source[i:i+1], target[i:i+1])
            ssim_val = metrics.compute_ssim(source[i:i+1], target[i:i+1])
            if oid not in per_object:
                per_object[oid] = {"lpips": [], "ssim": []}
            per_object[oid]["lpips"].append(lpips_val)
            per_object[oid]["ssim"].append(ssim_val)

    return {oid: {k: float(np.mean(v)) for k, v in vals.items()} for oid, vals in per_object.items()}


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def evaluate_model(
    model: ConditionalDDPM,
    dataloader: torch.utils.data.DataLoader,
    metrics: PerceptualMetrics,
    device: torch.device,
    n_ddim_steps: int = 50,
) -> Dict[int, Dict[str, float]]:
    """Run DDIM sampling and compute per-object LPIPS + SSIM.

    Returns dict mapping object_id → {'lpips': float, 'ssim': float}.
    """
    model.eval()
    per_object: Dict[int, List] = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating model"):
            source = batch["source"].to(device)
            target = batch["target"].to(device)
            delta = batch["delta_theta"].float().to(device)
            obj_ids = batch["object_id"].tolist()

            source_n = source * 2.0 - 1.0
            pred_n = model.ddim_sample(source_n, delta, n_steps=n_ddim_steps)
            pred_01 = ((pred_n + 1.0) / 2.0).clamp(0.0, 1.0)

            for i, oid in enumerate(obj_ids):
                lpips_val = metrics.compute_lpips(pred_01[i:i+1], target[i:i+1])
                ssim_val = metrics.compute_ssim(pred_01[i:i+1], target[i:i+1])
                if oid not in per_object:
                    per_object[oid] = {"lpips": [], "ssim": []}
                per_object[oid]["lpips"].append(lpips_val)
                per_object[oid]["ssim"].append(ssim_val)

    return {oid: {k: float(np.mean(v)) for k, v in vals.items()} for oid, vals in per_object.items()}


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary_table(
    per_object: Dict[int, Dict[str, float]],
    complexity_scores: Dict[int, float],
    label: str = "Model",
) -> None:
    """Print a Markdown-style table stratified by complexity quartile."""
    qs = get_quartile_splits(complexity_scores)
    print(f"\n{'='*60}")
    print(f"  {label}  —  Results by complexity quartile")
    print(f"{'='*60}")
    print(f"{'Quartile':<10} {'N':>4} {'LPIPS (↓)':>14} {'SSIM (↑)':>14}")
    print("-" * 44)
    all_lpips, all_ssim = [], []
    for qname in ["Q1", "Q2", "Q3", "Q4"]:
        ids = qs[qname]
        valid = [per_object[o] for o in ids if o in per_object]
        if not valid:
            continue
        lpips_vals = [v["lpips"] for v in valid]
        ssim_vals = [v["ssim"] for v in valid]
        all_lpips.extend(lpips_vals)
        all_ssim.extend(ssim_vals)
        lbl = f"{qname} (simple)" if qname == "Q1" else (f"{qname} (complex)" if qname == "Q4" else qname)
        print(
            f"{lbl:<10} {len(valid):>4} "
            f"{np.mean(lpips_vals):>6.4f}±{np.std(lpips_vals):.4f} "
            f"{np.mean(ssim_vals):>6.4f}±{np.std(ssim_vals):.4f}"
        )
    print("-" * 44)
    print(
        f"{'Overall':<10} {len(all_lpips):>4} "
        f"{np.mean(all_lpips):>6.4f}±{np.std(all_lpips):.4f} "
        f"{np.mean(all_ssim):>6.4f}±{np.std(all_ssim):.4f}"
    )
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Save CSV helpers
# ---------------------------------------------------------------------------


def save_per_object_csv(
    path: Path,
    per_object_model: Dict[int, Dict[str, float]],
    per_object_baseline: Dict[int, Dict[str, float]],
    complexity_scores: Dict[int, float],
) -> None:
    """Write per-object CSV with model and baseline metrics plus complexity."""
    qs = get_quartile_splits(complexity_scores)
    obj_to_quartile = {}
    for qname, ids in qs.items():
        for oid in ids:
            obj_to_quartile[oid] = qname

    fields = ["object_id", "complexity", "quartile", "model_lpips", "model_ssim", "baseline_lpips", "baseline_ssim"]
    rows = []
    for oid in sorted(per_object_model.keys()):
        rows.append({
            "object_id": oid,
            "complexity": complexity_scores.get(oid, float("nan")),
            "quartile": obj_to_quartile.get(oid, "?"),
            "model_lpips": per_object_model[oid]["lpips"],
            "model_ssim": per_object_model[oid]["ssim"],
            "baseline_lpips": per_object_baseline.get(oid, {}).get("lpips", float("nan")),
            "baseline_ssim": per_object_baseline.get(oid, {}).get("ssim", float("nan")),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved per-object CSV: {path}")


def save_summary_csv(
    path: Path,
    per_object: Dict[int, Dict[str, float]],
    per_object_baseline: Dict[int, Dict[str, float]],
    complexity_scores: Dict[int, float],
) -> None:
    """Write summary CSV with mean±std per quartile for model and baseline."""
    qs = get_quartile_splits(complexity_scores)
    fields = [
        "quartile", "n",
        "model_lpips_mean", "model_lpips_std",
        "model_ssim_mean", "model_ssim_std",
        "baseline_lpips_mean", "baseline_lpips_std",
        "baseline_ssim_mean", "baseline_ssim_std",
    ]
    rows = []
    for qname in ["Q1", "Q2", "Q3", "Q4"]:
        ids = qs[qname]
        m_valid = [per_object[o] for o in ids if o in per_object]
        b_valid = [per_object_baseline[o] for o in ids if o in per_object_baseline]
        if not m_valid:
            continue
        m_lpips = [v["lpips"] for v in m_valid]
        m_ssim = [v["ssim"] for v in m_valid]
        b_lpips = [v["lpips"] for v in b_valid] if b_valid else [float("nan")]
        b_ssim = [v["ssim"] for v in b_valid] if b_valid else [float("nan")]
        rows.append({
            "quartile": qname,
            "n": len(m_valid),
            "model_lpips_mean": np.mean(m_lpips),
            "model_lpips_std": np.std(m_lpips),
            "model_ssim_mean": np.mean(m_ssim),
            "model_ssim_std": np.std(m_ssim),
            "baseline_lpips_mean": np.mean(b_lpips),
            "baseline_lpips_std": np.std(b_lpips),
            "baseline_ssim_mean": np.mean(b_ssim),
            "baseline_ssim_std": np.std(b_ssim),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved summary CSV: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation on held-out test objects."""
    # Load config from the checkpoint.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", load_config(args.config))

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Data splits -------------------------------------------------------
    n_train = cfg["data"]["train_objects"]
    seed = cfg["data"]["seed"]
    _, test_ids = get_train_test_split(n_train=n_train, seed=seed)
    print(f"Test objects: {len(test_ids)}")

    # --- Complexity scores -------------------------------------------------
    complexity_json = Path(args.checkpoint).parent / "test_complexity.json"
    if complexity_json.exists():
        with open(complexity_json) as f:
            raw = json.load(f)
        complexity_scores = {int(k): v for k, v in raw.items()}
        # Keep only test_ids.
        complexity_scores = {o: complexity_scores[o] for o in test_ids if o in complexity_scores}
        print(f"Loaded complexity scores from {complexity_json}")
    else:
        print("Computing complexity scores (this may take a while) ...")
        complexity_scores = compute_all_complexities(
            args.data_root, test_ids, alpha_deg=cfg["complexity"]["alpha_deg"], device=str(device)
        )

    # --- Build model & load weights ----------------------------------------
    model = build_model(cfg, device)
    ema = EMAHelper(model)
    step = load_checkpoint(args.checkpoint, model, ema, device=str(device))
    # Use EMA weights for evaluation.
    ema.copy_to(model)
    model.eval()

    # --- DataLoader --------------------------------------------------------
    loader = build_dataloader(
        root=args.data_root,
        object_ids=test_ids,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        length=len(test_ids) * 72,  # all pairs
    )

    # --- Metrics -----------------------------------------------------------
    metrics = PerceptualMetrics(device=str(device))
    n_ddim = cfg["diffusion"]["ddim_steps"]

    print("\nEvaluating model ...")
    per_object_model = evaluate_model(model, loader, metrics, device, n_ddim_steps=n_ddim)

    print("\nEvaluating copy-source baseline ...")
    per_object_baseline = copy_source_baseline(loader, metrics, device)

    # --- Print tables ------------------------------------------------------
    print_summary_table(per_object_model, complexity_scores, label=f"Model (step {step})")
    print_summary_table(per_object_baseline, complexity_scores, label="Copy-source baseline")

    # --- Save CSVs ---------------------------------------------------------
    save_per_object_csv(
        output_dir / "eval_per_object.csv",
        per_object_model,
        per_object_baseline,
        complexity_scores,
    )
    save_summary_csv(
        output_dir / "eval_summary.csv",
        per_object_model,
        per_object_baseline,
        complexity_scores,
    )
    print(f"\nEvaluation complete. Results saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained DDPM checkpoint on held-out objects.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    parser.add_argument("--data_root", required=True, help="Path to COIL-100 directory.")
    parser.add_argument("--output_dir", default=None, help="Directory for output CSVs.")
    parser.add_argument("--config", default="configs/base.yaml", help="Fallback config if not in checkpoint.")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = str(Path(args.checkpoint).parent / "eval")
    evaluate(args)
