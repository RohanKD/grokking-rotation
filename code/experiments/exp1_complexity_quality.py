"""Experiment 1: Complexity vs. Generalization Quality.

Research question
-----------------
Does the model's generalization quality on held-out objects depend on
object *complexity* (view-variance)?  We expect performance to degrade
for complex objects (high C(o; alpha)) relative to simple ones.

Protocol
--------
1. Train one model on all 80 training objects.
2. Evaluate on 20 test objects, stratified by complexity quartile.
3. Compute copy-source baseline.
4. Report Table 1: mean ± std LPIPS and SSIM per quartile.
5. Save box-plot data as CSV.

Usage
-----
    python experiments/exp1_complexity_quality.py \
        --config configs/base.yaml \
        --data_root /path/to/coil-100 \
        --output_dir results/exp1
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.coil100 import get_train_test_split, build_dataloader
from models.unet import UNet
from models.ddpm import ConditionalDDPM, EMAHelper
from metrics.perceptual import PerceptualMetrics
from complexity import compute_all_complexities, get_quartile_splits, summarize_complexity
from train import (
    load_config, build_model, load_checkpoint, save_checkpoint,
    set_seed, get_lr_scheduler, evaluate_test_set
)
from evaluate import evaluate_model, copy_source_baseline, print_summary_table


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_training(
    cfg: Dict,
    data_root: str,
    output_dir: Path,
    device: str,
    total_steps: int,
    seed: int = 42,
) -> Path:
    """Train a model and return the path to the final checkpoint.

    Delegates to train.py via subprocess so we get the exact same training
    logic without code duplication.
    """
    import subprocess, shlex
    cmd = [
        sys.executable, str(Path(__file__).parent.parent / "train.py"),
        "--config", str(Path(__file__).parent.parent / "configs/base.yaml"),
        "--data_root", data_root,
        "--output_dir", str(output_dir),
        "--steps", str(total_steps),
        "--seed", str(seed),
        "--n_train_objects", str(cfg["data"]["train_objects"]),
        "--device", device,
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    final_ckpt = output_dir / "final.pt"
    if not final_ckpt.exists():
        raise FileNotFoundError(f"Training did not produce {final_ckpt}")
    return final_ckpt


def save_boxplot_csv(
    path: Path,
    per_object: Dict[int, Dict[str, float]],
    complexity_scores: Dict[int, float],
    label: str,
) -> None:
    """Save per-object results with complexity for box-plot generation."""
    rows = []
    qs = get_quartile_splits(complexity_scores)
    obj_to_q = {oid: q for q, ids in qs.items() for oid in ids}
    for oid, vals in sorted(per_object.items()):
        rows.append({
            "object_id": oid,
            "complexity": complexity_scores.get(oid, float("nan")),
            "quartile": obj_to_q.get(oid, "?"),
            "lpips": vals["lpips"],
            "ssim": vals["ssim"],
            "source": label,
        })
    fields = ["object_id", "complexity", "quartile", "lpips", "ssim", "source"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {path}")


def plot_boxplots(
    boxplot_data_path: Path,
    output_dir: Path,
) -> None:
    """Generate box plots of LPIPS per complexity quartile."""
    import pandas as pd
    df = pd.read_csv(boxplot_data_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    labels = ["Q1\n(simple)", "Q2", "Q3", "Q4\n(complex)"]

    for ax, metric in zip(axes, ["lpips", "ssim"]):
        model_data = [
            df[(df["quartile"] == q) & (df["source"] == "model")][metric].dropna().values
            for q in quartiles
        ]
        baseline_data = [
            df[(df["quartile"] == q) & (df["source"] == "baseline")][metric].dropna().values
            for q in quartiles
        ]

        positions_m = [1 + 3 * i for i in range(4)]
        positions_b = [2 + 3 * i for i in range(4)]

        bp_m = ax.boxplot(
            model_data, positions=positions_m, widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.7),
            medianprops=dict(color="navy", linewidth=2),
        )
        bp_b = ax.boxplot(
            baseline_data, positions=positions_b, widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor="tomato", alpha=0.7),
            medianprops=dict(color="darkred", linewidth=2),
        )

        ax.set_xticks([1.5 + 3 * i for i in range(4)])
        ax.set_xticklabels(labels)
        arrow_label = "(↓ better)" if metric == "lpips" else "(↑ better)"
        ax.set_ylabel(f"{metric.upper()} {arrow_label}")
        ax.set_title(f"{metric.upper()} by complexity quartile")
        ax.legend(
            [bp_m["boxes"][0], bp_b["boxes"][0]],
            ["Model", "Copy-source"],
            loc="upper left" if metric == "lpips" else "lower left",
        )
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Exp 1: Generalization Quality vs. Object Complexity", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "exp1_boxplots.png", dpi=150)
    plt.close()
    print(f"  Saved {output_dir / 'exp1_boxplots.png'}")


def print_table1(
    per_object_model: Dict[int, Dict[str, float]],
    per_object_baseline: Dict[int, Dict[str, float]],
    complexity_scores: Dict[int, float],
) -> None:
    """Print Table 1 in the paper style."""
    qs = get_quartile_splits(complexity_scores)
    header = f"{'Quartile':<14}{'LPIPS model':>14}{'LPIPS base':>14}{'SSIM model':>12}{'SSIM base':>12}{'N':>4}"
    print("\n" + "=" * len(header))
    print("Table 1: Generalization quality vs. object complexity")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for qname in ["Q1", "Q2", "Q3", "Q4"]:
        ids = qs[qname]
        m = [per_object_model[o] for o in ids if o in per_object_model]
        b = [per_object_baseline[o] for o in ids if o in per_object_baseline]
        if not m:
            continue
        m_l = [v["lpips"] for v in m]
        m_s = [v["ssim"] for v in m]
        b_l = [v["lpips"] for v in b] if b else [float("nan")]
        b_s = [v["ssim"] for v in b] if b else [float("nan")]
        tag = " (simple)" if qname == "Q1" else (" (complex)" if qname == "Q4" else "")
        print(
            f"{qname+tag:<14}"
            f"{np.mean(m_l):>7.4f}±{np.std(m_l):.4f}"
            f"{np.mean(b_l):>7.4f}±{np.std(b_l):.4f}"
            f"{np.mean(m_s):>7.4f}±{np.std(m_s):.4f}"
            f"{np.mean(b_s):>7.4f}±{np.std(b_s):.4f}"
            f"{len(m):>4}"
        )
    print("=" * len(header))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    """Execute Experiment 1."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    device = args.device if args.device else ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    seed = cfg["data"]["seed"]
    total_steps = args.steps if args.steps else cfg["training"]["total_steps"]

    print("=" * 60)
    print("Experiment 1: Complexity vs. Generalization Quality")
    print("=" * 60)

    # --- Step 1: train (or load pre-trained) --------------------------------
    ckpt_path = output_dir / "model" / "final.pt"
    if ckpt_path.exists() and not args.retrain:
        print(f"\nFound existing checkpoint at {ckpt_path}. Skipping training.")
        print("  (Use --retrain to force retraining.)")
    else:
        print(f"\nStep 1: Training on {cfg['data']['train_objects']} objects for {total_steps} steps ...")
        run_training(cfg, args.data_root, output_dir / "model", device, total_steps, seed)

    # --- Step 2: load checkpoint -------------------------------------------
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg_saved = ckpt.get("config", cfg)
    dev = torch.device(device)

    model = build_model(cfg_saved, dev)
    ema = EMAHelper(model)
    load_checkpoint(str(ckpt_path), model, ema, device=str(dev))
    ema.copy_to(model)
    model.eval()

    # --- Step 3: get test objects + complexity -----------------------------
    n_train = cfg_saved["data"]["train_objects"]
    _, test_ids = get_train_test_split(n_train=n_train, seed=cfg_saved["data"]["seed"])

    complexity_json = output_dir / "model" / "test_complexity.json"
    if complexity_json.exists():
        with open(complexity_json) as f:
            raw = json.load(f)
        complexity_scores = {int(k): v for k, v in raw.items() if int(k) in test_ids}
    else:
        print("\nStep 2: Computing complexity scores ...")
        complexity_scores = compute_all_complexities(
            args.data_root, test_ids,
            alpha_deg=cfg_saved["complexity"]["alpha_deg"],
            device=device, verbose=True
        )

    print("\nComplexity distribution:")
    summarize_complexity(complexity_scores)

    # --- Step 4: evaluate --------------------------------------------------
    print("\nStep 3: Evaluating model and baseline ...")
    metrics = PerceptualMetrics(device=device)

    loader = build_dataloader(
        root=args.data_root,
        object_ids=test_ids,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        length=len(test_ids) * 72,
    )

    per_object_model = evaluate_model(
        model, loader, metrics, dev,
        n_ddim_steps=cfg_saved["diffusion"]["ddim_steps"]
    )
    per_object_baseline = copy_source_baseline(loader, metrics, dev)

    # --- Step 5: print and save --------------------------------------------
    print_table1(per_object_model, per_object_baseline, complexity_scores)

    # Box-plot data CSV (combined model + baseline rows).
    boxplot_path = output_dir / "exp1_boxplot_data.csv"
    # Write model rows.
    save_boxplot_csv(boxplot_path, per_object_model, complexity_scores, "model")
    # Append baseline rows.
    import pandas as pd
    df_m = pd.read_csv(boxplot_path)
    rows_b = []
    qs_b = get_quartile_splits(complexity_scores)
    obj_to_q = {oid: q for q, ids in qs_b.items() for oid in ids}
    for oid, vals in sorted(per_object_baseline.items()):
        rows_b.append({
            "object_id": oid,
            "complexity": complexity_scores.get(oid, float("nan")),
            "quartile": obj_to_q.get(oid, "?"),
            "lpips": vals["lpips"],
            "ssim": vals["ssim"],
            "source": "baseline",
        })
    df_b = pd.DataFrame(rows_b)
    df_all = pd.concat([df_m, df_b], ignore_index=True)
    df_all.to_csv(boxplot_path, index=False)
    print(f"  Saved combined box-plot data: {boxplot_path}")

    plot_boxplots(boxplot_path, output_dir)

    # Per-object full CSV.
    full_csv = output_dir / "exp1_per_object.csv"
    rows = []
    for oid in sorted(per_object_model.keys()):
        rows.append({
            "object_id": oid,
            "complexity": complexity_scores.get(oid, float("nan")),
            "quartile": obj_to_q.get(oid, "?"),
            "model_lpips": per_object_model[oid]["lpips"],
            "model_ssim": per_object_model[oid]["ssim"],
            "baseline_lpips": per_object_baseline.get(oid, {}).get("lpips", float("nan")),
            "baseline_ssim": per_object_baseline.get(oid, {}).get("ssim", float("nan")),
        })
    fields = ["object_id", "complexity", "quartile", "model_lpips", "model_ssim", "baseline_lpips", "baseline_ssim"]
    with open(full_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved per-object CSV: {full_csv}")

    print(f"\nExperiment 1 complete. Results saved to {output_dir}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Exp 1: Complexity vs. generalization quality.")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", default="results/exp1")
    p.add_argument("--steps", type=int, default=None, help="Override training steps.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--retrain", action="store_true", help="Re-train even if checkpoint exists.")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
