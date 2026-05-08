"""COIL-100 dataset utilities for the rotation generalization study.

COIL-100 structure: obj{N}__{angle}.png
  N in 1..100 (object index, 1-based)
  angle in {0, 5, 10, ..., 355} degrees
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class COIL100Dataset(Dataset):
    """Paired-view dataset built from COIL-100.

    Each sample consists of a source image at angle theta_s and a target
    image at angle theta_t = theta_s + delta_theta.  The model is trained
    to predict the target from (source, delta_theta).

    Parameters
    ----------
    root:
        Path to the COIL-100 directory that contains the .png files.
    object_ids:
        1-based object indices to include (subset of 1..100).
    angle_src:
        If given, fix the source angle index (0..71) for all samples.
        If None, sample uniformly at random each time.
    angle_delta:
        If given, fix the delta angle index (0..71) for all samples.
        If None, sample uniformly at random each time.
    transform:
        Optional torchvision transform applied to both source and target
        images *before* the default [0,1] float conversion.
    length:
        Virtual dataset length.  Defaults to len(object_ids) * 72.
    """

    N_ANGLES: int = 72          # 72 views per object
    ANGLE_STEP: float = 5.0     # degrees between consecutive views

    def __init__(
        self,
        root: str,
        object_ids: List[int],
        angle_src: Optional[int] = None,
        angle_delta: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        length: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.object_ids = list(object_ids)
        self.angle_src = angle_src
        self.angle_delta = angle_delta
        self.transform = transform
        self._length = length if length is not None else len(self.object_ids) * self.N_ANGLES

        # Verify at least one file is readable (fail fast on bad root).
        sample_path = self._img_path(self.object_ids[0], 0)
        if not sample_path.exists():
            raise FileNotFoundError(
                f"COIL-100 file not found: {sample_path}\n"
                f"Expected layout: {{root}}/obj{{N}}__{{angle}}.png  (angle in degrees)"
            )

        # Default transform: resize to 128x128, convert to tensor in [0, 1].
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                ]
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _img_path(self, obj_id: int, angle_idx: int) -> Path:
        """Return the file path for object *obj_id* at angle index *angle_idx*."""
        angle_deg = angle_idx * 5
        return self.root / f"obj{obj_id}__{angle_deg}.png"

    def _load_image(self, obj_id: int, angle_idx: int) -> torch.Tensor:
        """Load a single COIL-100 image and apply the transform."""
        path = self._img_path(obj_id, angle_idx)
        img = Image.open(path).convert("RGB")
        return self.transform(img)  # shape: (3, H, W)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict:
        """Return a paired (source, target) sample.

        Returns
        -------
        dict with keys:
            source      – (3, H, W) float tensor in [0, 1]
            target      – (3, H, W) float tensor in [0, 1]
            delta_theta – rotation delta in **radians** (float)
            object_id   – 1-based object index (int)
            src_angle   – source angle index 0..71 (int)
            tgt_angle   – target angle index 0..71 (int)
        """
        # Choose object randomly from the pool.
        obj_id = self.object_ids[idx % len(self.object_ids)]

        # Source angle.
        if self.angle_src is not None:
            src_idx = int(self.angle_src) % self.N_ANGLES
        else:
            src_idx = random.randint(0, self.N_ANGLES - 1)

        # Delta angle.
        if self.angle_delta is not None:
            delta_idx = int(self.angle_delta) % self.N_ANGLES
        else:
            delta_idx = random.randint(0, self.N_ANGLES - 1)

        tgt_idx = (src_idx + delta_idx) % self.N_ANGLES

        source = self._load_image(obj_id, src_idx)
        target = self._load_image(obj_id, tgt_idx)

        delta_rad = math.radians(delta_idx * self.ANGLE_STEP)

        return {
            "source": source,
            "target": target,
            "delta_theta": delta_rad,
            "object_id": obj_id,
            "src_angle": src_idx,
            "tgt_angle": tgt_idx,
        }


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------


def get_train_test_split(
    n_train: int = 80,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Randomly split 100 COIL-100 objects into train and test sets.

    Parameters
    ----------
    n_train:
        Number of objects in the training split.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (train_ids, test_ids)
        Two lists of 1-based object indices.
    """
    rng = random.Random(seed)
    all_ids = list(range(1, 101))
    rng.shuffle(all_ids)
    train_ids = sorted(all_ids[:n_train])
    test_ids = sorted(all_ids[n_train:])
    return train_ids, test_ids


def get_complexity_split(
    root: str,
    object_ids: List[int],
    alpha_deg: float = 90.0,
) -> Tuple[Dict[int, float], List[float]]:
    """Compute per-object complexity scores and return quartile boundaries.

    This is a *lightweight* version that computes L1 pixel distance rather
    than LPIPS (no GPU required, no model loading).  The full LPIPS-based
    complexity is provided in ``complexity.py``.

    Parameters
    ----------
    root:
        COIL-100 root directory.
    object_ids:
        Objects to score.
    alpha_deg:
        Angular gap used for the view-variance metric.

    Returns
    -------
    (scores_dict, quartile_boundaries)
        scores_dict maps object_id → scalar score.
        quartile_boundaries is [q25, q50, q75].
    """
    from complexity import compute_all_complexities  # local import to avoid circular deps

    scores = compute_all_complexities(root, object_ids, alpha_deg=alpha_deg)
    values = np.array([scores[oid] for oid in sorted(scores)])
    q25, q50, q75 = float(np.percentile(values, 25)), float(np.percentile(values, 50)), float(np.percentile(values, 75))
    return scores, [q25, q50, q75]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def build_dataloader(
    root: str,
    object_ids: List[int],
    batch_size: int = 64,
    angle_src: Optional[int] = None,
    angle_delta: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 4,
    length: Optional[int] = None,
) -> DataLoader:
    """Convenience factory that wraps COIL100Dataset in a DataLoader.

    Parameters
    ----------
    root:
        COIL-100 root directory.
    object_ids:
        Objects to include.
    batch_size:
        Samples per batch.
    angle_src:
        Fixed source angle index, or None for random.
    angle_delta:
        Fixed delta angle index, or None for random.
    shuffle:
        Whether to shuffle the dataset.
    num_workers:
        DataLoader worker processes.
    length:
        Virtual epoch length.  Defaults to len(object_ids) * 72.

    Returns
    -------
    DataLoader that yields batches of paired views.
    """
    dataset = COIL100Dataset(
        root=root,
        object_ids=object_ids,
        angle_src=angle_src,
        angle_delta=angle_delta,
        length=length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
