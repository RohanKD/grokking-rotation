"""View-variance complexity metric C(o; alpha) for COIL-100 objects.

Definition
----------
For object *o* and angular gap *alpha*:
  C(o; alpha) = mean_{theta} LPIPS(view(o, theta), view(o, theta + alpha))

A high value means the object's appearance changes a lot with rotation
(e.g. elongated or asymmetric objects), which we call "complex".
A low value means the object looks similar from all angles (e.g. round mugs),
which we call "simple".

We compute this over all 72 angles of COIL-100, wrapping modulo 72.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import lpips as lpips_lib


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

_DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)


def _load_all_views(root: str, obj_id: int) -> torch.Tensor:
    """Load all 72 views of *obj_id* and return a (72, 3, 128, 128) tensor.

    Parameters
    ----------
    root:
        COIL-100 root directory.
    obj_id:
        1-based object index.

    Returns
    -------
    Float tensor of shape (72, 3, 128, 128) with values in [0, 1].
    """
    root_path = Path(root)
    imgs = []
    for angle_idx in range(72):
        path = root_path / f"obj{obj_id}__image{angle_idx}.png"
        img = Image.open(path).convert("RGB")
        imgs.append(_DEFAULT_TRANSFORM(img))
    return torch.stack(imgs, dim=0)  # (72, 3, 128, 128)


# ---------------------------------------------------------------------------
# Core complexity computation
# ---------------------------------------------------------------------------


def compute_view_variance(
    root: str,
    object_id: int,
    alpha_deg: float = 90.0,
    device: str = "cpu",
) -> float:
    """Compute C(o; alpha) = mean LPIPS over all (theta, theta+alpha) pairs.

    Parameters
    ----------
    root:
        COIL-100 root directory.
    object_id:
        1-based object index.
    alpha_deg:
        Angular gap in degrees.  Must be a multiple of 5 (COIL-100 step size).
    device:
        Torch device for LPIPS computation (``'cpu'`` or ``'cuda'``).

    Returns
    -------
    Scalar complexity score (float).
    """
    alpha_idx = round(alpha_deg / 5.0) % 72  # number of angle steps

    dev = torch.device(device)
    lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(dev)
    lpips_fn.eval()

    views = _load_all_views(root, object_id).to(dev)  # (72, 3, 128, 128)
    # Normalize to [-1, 1] for LPIPS.
    views_n = views * 2.0 - 1.0

    scores = []
    with torch.no_grad():
        for theta_idx in range(72):
            img_a = views_n[theta_idx : theta_idx + 1]          # (1, 3, H, W)
            img_b = views_n[(theta_idx + alpha_idx) % 72 : (theta_idx + alpha_idx) % 72 + 1]
            dist = lpips_fn(img_a, img_b).item()
            scores.append(dist)

    return float(np.mean(scores))


def compute_all_complexities(
    root: str,
    object_ids: List[int],
    alpha_deg: float = 90.0,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[int, float]:
    """Compute complexity scores for a list of objects.

    Parameters
    ----------
    root:
        COIL-100 root directory.
    object_ids:
        Objects to score.
    alpha_deg:
        Angular gap in degrees.
    device:
        Torch device.
    verbose:
        Print progress if True.

    Returns
    -------
    dict mapping object_id → C(o; alpha).
    """
    scores: Dict[int, float] = {}
    n = len(object_ids)
    for i, oid in enumerate(object_ids):
        c = compute_view_variance(root, oid, alpha_deg=alpha_deg, device=device)
        scores[oid] = c
        if verbose:
            print(f"  complexity [{i+1}/{n}] obj{oid}: {c:.4f}")
    return scores


# ---------------------------------------------------------------------------
# Quartile splits
# ---------------------------------------------------------------------------


def get_quartile_splits(
    complexity_scores: Dict[int, float],
) -> Dict[str, List[int]]:
    """Partition objects into four complexity quartiles.

    Parameters
    ----------
    complexity_scores:
        Mapping from object_id to scalar complexity score.

    Returns
    -------
    dict with keys ``'Q1'``, ``'Q2'``, ``'Q3'``, ``'Q4'``
    (Q1 = simplest, Q4 = most complex).  Each value is a sorted list of
    object IDs.
    """
    oids = sorted(complexity_scores.keys())
    values = np.array([complexity_scores[o] for o in oids])

    q25 = np.percentile(values, 25)
    q50 = np.percentile(values, 50)
    q75 = np.percentile(values, 75)

    Q1, Q2, Q3, Q4 = [], [], [], []
    for oid, v in zip(oids, values):
        if v <= q25:
            Q1.append(oid)
        elif v <= q50:
            Q2.append(oid)
        elif v <= q75:
            Q3.append(oid)
        else:
            Q4.append(oid)

    return {"Q1": sorted(Q1), "Q2": sorted(Q2), "Q3": sorted(Q3), "Q4": sorted(Q4)}


def summarize_complexity(
    complexity_scores: Dict[int, float],
) -> None:
    """Print a summary table of complexity distribution."""
    values = list(complexity_scores.values())
    qs = get_quartile_splits(complexity_scores)
    print(f"{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    print(f"{'N objects':<20} {len(values):>10d}")
    print(f"{'Mean':<20} {np.mean(values):>10.4f}")
    print(f"{'Std':<20} {np.std(values):>10.4f}")
    print(f"{'Min':<20} {np.min(values):>10.4f}")
    print(f"{'Q25':<20} {np.percentile(values, 25):>10.4f}")
    print(f"{'Median':<20} {np.median(values):>10.4f}")
    print(f"{'Q75':<20} {np.percentile(values, 75):>10.4f}")
    print(f"{'Max':<20} {np.max(values):>10.4f}")
    print()
    for qname in ["Q1", "Q2", "Q3", "Q4"]:
        ids = qs[qname]
        qvals = [complexity_scores[o] for o in ids]
        print(f"{qname}: n={len(ids)}, mean={np.mean(qvals):.4f} ({ids[:5]}...)")
