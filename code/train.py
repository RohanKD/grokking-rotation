"""Training script for the conditional DDPM rotation generalization study.

Usage
-----
    python train.py \
        --config configs/base.yaml \
        --data_root /path/to/coil-100 \
        --output_dir runs/exp_default

The script:
  1. Loads config and CLI overrides.
  2. Splits COIL-100 into train/test objects.
  3. Builds model, optimizer, and LR scheduler.
  4. Runs the training loop with periodic logging, evaluation, and checkpointing.
  5. Saves a CSV log with columns: step, train_loss, test_lpips_simple, test_lpips_complex.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports — add parent dir so the script can be run from anywhere.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from data.coil100 import COIL100Dataset, get_train_test_split, build_dataloader
from models.unet import UNet
from models.ddpm import ConditionalDDPM, EMAHelper
from metrics.perceptual import PerceptualMetrics
from complexity import compute_all_complexities, get_quartile_splits


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    """Load a YAML config file and return as a nested dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_config_args(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Override config values with any explicitly set CLI arguments."""
    if args.n_train_objects is not None:
        cfg["data"]["train_objects"] = args.n_train_objects
    if args.seed is not None:
        cfg["data"]["seed"] = args.seed
    if args.steps is not None:
        cfg["training"]["total_steps"] = args.steps
    return cfg


def build_model(cfg: Dict, device: torch.device) -> ConditionalDDPM:
    """Instantiate UNet + ConditionalDDPM from config."""
    mcfg = cfg["model"]
    dcfg = cfg["diffusion"]
    unet = UNet(
        base_ch=mcfg["base_channels"],
        n_blocks=mcfg["n_blocks"],
        time_emb_dim=mcfg["time_emb_dim"],
        cond_emb_dim=mcfg["cond_emb_dim"],
    ).to(device)
    model = ConditionalDDPM(
        unet=unet,
        T=dcfg["T"],
        beta_schedule=dcfg["beta_schedule"],
    ).to(device)
    return model


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    use_cosine: bool,
    warmup_steps: int = 2000,
) -> optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warm-up."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        if not use_cosine:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    output_dir: Path,
    step: int,
    model: ConditionalDDPM,
    ema: EMAHelper,
    optimizer: optim.Optimizer,
    scheduler,
    cfg: Dict,
    filename: Optional[str] = None,
) -> None:
    """Save a training checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fname = filename or f"checkpoint_{step:07d}.pt"
    path = output_dir / fname
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": cfg,
        },
        path,
    )
    print(f"  [ckpt] Saved {path}")


def load_checkpoint(
    path: str,
    model: ConditionalDDPM,
    ema: Optional[EMAHelper] = None,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
) -> int:
    """Load a checkpoint into an existing model (and optionally optimizer/scheduler).

    Returns the training step at which the checkpoint was saved.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if ema and "ema_state_dict" in ckpt:
        ema.load_state_dict(ckpt["ema_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    step = ckpt.get("step", 0)
    print(f"  [ckpt] Loaded {path} at step {step}")
    return step


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_test_set(
    model: ConditionalDDPM,
    ema: EMAHelper,
    test_objects_simple: List[int],
    test_objects_complex: List[int],
    data_root: str,
    device: torch.device,
    cfg: Dict,
    n_samples_per_group: int = 8,
) -> Tuple[float, float]:
    """Run DDIM on a small eval batch and return (lpips_simple, lpips_complex).

    We swap in EMA weights for evaluation, then restore the original weights.
    """
    # Temporarily copy EMA weights into the model for evaluation.
    original_params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
    ema.copy_to(model)
    model.eval()

    dcfg = cfg["diffusion"]
    metrics = PerceptualMetrics(device=str(device))

    results = {}
    for tag, obj_ids in [("simple", test_objects_simple), ("complex", test_objects_complex)]:
        if not obj_ids:
            results[tag] = float("nan")
            continue
        # Use a fixed small batch for speed.
        eval_ids = obj_ids[:min(len(obj_ids), n_samples_per_group)]
        loader = build_dataloader(
            root=data_root,
            object_ids=eval_ids,
            batch_size=n_samples_per_group,
            shuffle=False,
            num_workers=0,
            length=n_samples_per_group,
        )
        batch = next(iter(loader))
        source = (batch["source"].to(device) * 2.0 - 1.0)
        target_01 = batch["target"].to(device)
        delta = batch["delta_theta"].float().to(device)

        with torch.no_grad():
            pred_n = model.ddim_sample(source, delta, n_steps=dcfg["ddim_steps"])
        pred_01 = ((pred_n + 1.0) / 2.0).clamp(0.0, 1.0)

        results[tag] = metrics.compute_lpips(pred_01, target_01)

    # Restore original weights.
    for name, param in model.named_parameters():
        if name in original_params:
            param.data.copy_(original_params[name])
    model.train()

    return results.get("simple", float("nan")), results.get("complex", float("nan"))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Run the full training procedure."""
    # --- Config & device --------------------------------------------------
    cfg = load_config(args.config)
    cfg = merge_config_args(cfg, args)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    seed = cfg["data"]["seed"]
    set_seed(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config alongside the run.
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # --- Data splits -------------------------------------------------------
    n_train = cfg["data"]["train_objects"]
    train_ids, test_ids = get_train_test_split(n_train=n_train, seed=seed)
    print(f"Train objects: {len(train_ids)}, Test objects: {len(test_ids)}")

    # Compute complexity scores for the test set to enable stratified logging.
    print("Computing complexity scores for test objects ...")
    try:
        test_complexity = compute_all_complexities(
            args.data_root, test_ids, alpha_deg=cfg["complexity"]["alpha_deg"], device=str(device)
        )
        qs = get_quartile_splits(test_complexity)
        test_simple = qs["Q1"]
        test_complex = qs["Q4"]
        print(f"  Test Q1 (simple): {test_simple}")
        print(f"  Test Q4 (complex): {test_complex}")
        # Save for reproducibility.
        with open(output_dir / "test_complexity.json", "w") as f:
            json.dump({str(k): v for k, v in test_complexity.items()}, f, indent=2)
    except Exception as e:
        print(f"  WARNING: Could not compute complexity ({e}). Disabling stratified eval.")
        test_simple = test_ids[:5]
        test_complex = test_ids[-5:]

    # --- DataLoader --------------------------------------------------------
    tcfg = cfg["training"]
    dcfg = cfg["data"]

    train_loader = build_dataloader(
        root=args.data_root,
        object_ids=train_ids,
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=dcfg["num_workers"],
        length=tcfg["total_steps"] * tcfg["batch_size"],
    )
    train_iter = iter(train_loader)

    # --- Model, optimizer, scheduler --------------------------------------
    model = build_model(cfg, device)
    ema = EMAHelper(model, decay=tcfg.get("ema_decay", 0.9999))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=tcfg["lr"])
    scheduler = get_lr_scheduler(
        optimizer,
        total_steps=tcfg["total_steps"],
        use_cosine=tcfg["cosine_schedule"],
    )

    # --- Resume from checkpoint if requested ------------------------------
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, ema, optimizer, scheduler, device=str(device))

    # --- CSV logger -------------------------------------------------------
    log_path = output_dir / "training_log.csv"
    log_fields = ["step", "train_loss", "lr", "test_lpips_simple", "test_lpips_complex"]
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writeheader()

    # --- Training loop ----------------------------------------------------
    log_every = cfg["logging"]["log_every"]
    eval_every = cfg["logging"]["eval_every"]
    save_every = cfg["logging"]["save_every"]
    total_steps = tcfg["total_steps"]
    grad_clip = tcfg.get("grad_clip", 1.0)

    model.train()
    running_loss = 0.0
    t0 = time.time()

    pbar = tqdm(
        range(start_step, total_steps),
        initial=start_step,
        total=total_steps,
        desc="Training",
        dynamic_ncols=True,
    )

    test_lpips_simple = float("nan")
    test_lpips_complex = float("nan")

    for step in pbar:
        # --- Fetch a batch ------------------------------------------------
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        source = batch["source"].to(device)
        target = batch["target"].to(device)
        delta = batch["delta_theta"].float().to(device)

        # Normalize to [-1, 1].
        source_n = source * 2.0 - 1.0
        target_n = target * 2.0 - 1.0

        # --- Forward + loss -----------------------------------------------
        optimizer.zero_grad()
        loss = model.p_losses(target_n, source_n, delta)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        running_loss += loss.item()

        # --- Logging ------------------------------------------------------
        if (step + 1) % log_every == 0:
            avg_loss = running_loss / log_every
            running_loss = 0.0
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            steps_per_sec = log_every / elapsed
            t0 = time.time()
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                lr=f"{lr_now:.2e}",
                sps=f"{steps_per_sec:.1f}",
                lpips_s=f"{test_lpips_simple:.3f}",
                lpips_c=f"{test_lpips_complex:.3f}",
            )
            row = {
                "step": step + 1,
                "train_loss": avg_loss,
                "lr": lr_now,
                "test_lpips_simple": test_lpips_simple,
                "test_lpips_complex": test_lpips_complex,
            }
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        # --- Evaluation ---------------------------------------------------
        if (step + 1) % eval_every == 0:
            print(f"\n[Step {step+1}] Running evaluation ...")
            test_lpips_simple, test_lpips_complex = evaluate_test_set(
                model, ema, test_simple, test_complex, args.data_root, device, cfg
            )
            print(
                f"  LPIPS simple={test_lpips_simple:.4f}  complex={test_lpips_complex:.4f}"
            )

        # --- Checkpoint ---------------------------------------------------
        if (step + 1) % save_every == 0:
            save_checkpoint(output_dir, step + 1, model, ema, optimizer, scheduler, cfg)

    # --- Final save -------------------------------------------------------
    save_checkpoint(output_dir, total_steps, model, ema, optimizer, scheduler, cfg, filename="final.pt")
    print(f"\nTraining complete. Outputs saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train conditional DDPM for COIL-100 rotation generalization study."
    )
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--data_root", required=True, help="Path to COIL-100 directory.")
    parser.add_argument("--output_dir", default="runs/default", help="Where to save outputs.")
    parser.add_argument("--n_train_objects", type=int, default=None, help="Override number of training objects.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps.")
    parser.add_argument("--device", type=str, default=None, help="Torch device (e.g. 'cuda', 'cpu').")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
