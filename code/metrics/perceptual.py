"""Perceptual quality metrics: LPIPS and SSIM.

All methods accept images as torch.Tensor in [0, 1] or [-1, 1] (auto-detected
by the ``normalize`` flag — LPIPS internally wants [-1, 1], SSIM wants [0, 1]).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as skimage_ssim

import lpips as lpips_lib


class PerceptualMetrics:
    """Wrapper that provides LPIPS and SSIM computation on a given device.

    Parameters
    ----------
    device:
        Torch device string or object (e.g. ``'cpu'``, ``'cuda:0'``).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        # LPIPS with AlexNet backbone (fast and correlates well with human judgement).
        self.lpips_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(self.device)
        self.lpips_fn.eval()

    # ------------------------------------------------------------------
    # Per-batch / per-image metrics
    # ------------------------------------------------------------------

    def compute_lpips(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        already_normalized: bool = False,
    ) -> float:
        """Compute mean LPIPS distance between predicted and target images.

        Parameters
        ----------
        pred:
            Predicted images (B, 3, H, W) in **[0, 1]**.
        target:
            Ground-truth images (B, 3, H, W) in **[0, 1]**.
        already_normalized:
            If True, inputs are already in [-1, 1] and no rescaling is done.

        Returns
        -------
        Mean LPIPS scalar (float, lower is better).
        """
        pred = pred.to(self.device)
        target = target.to(self.device)
        if not already_normalized:
            pred = pred * 2.0 - 1.0
            target = target * 2.0 - 1.0
        with torch.no_grad():
            dist = self.lpips_fn(pred, target)  # (B, 1, 1, 1)
        return dist.mean().item()

    def compute_ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        """Compute mean SSIM between predicted and target images.

        Uses scikit-image's implementation with the multichannel flag.

        Parameters
        ----------
        pred:
            Predicted images (B, 3, H, W) in **[0, 1]**.
        target:
            Ground-truth images (B, 3, H, W) in **[0, 1]**.

        Returns
        -------
        Mean SSIM scalar (float, higher is better, max 1.0).
        """
        pred_np = pred.detach().cpu().permute(0, 2, 3, 1).numpy()    # (B, H, W, 3)
        target_np = target.detach().cpu().permute(0, 2, 3, 1).numpy()
        pred_np = np.clip(pred_np, 0.0, 1.0)
        target_np = np.clip(target_np, 0.0, 1.0)

        scores = []
        for p, t in zip(pred_np, target_np):
            s = skimage_ssim(p, t, data_range=1.0, channel_axis=-1)
            scores.append(float(s))
        return float(np.mean(scores))

    # ------------------------------------------------------------------
    # Batch evaluation over a full DataLoader
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: Optional[str] = None,
        n_ddim_steps: int = 50,
    ) -> Dict:
        """Run the model on every batch and aggregate LPIPS / SSIM scores.

        The model is expected to be a ``ConditionalDDPM`` instance.

        Parameters
        ----------
        model:
            ``ConditionalDDPM`` model in eval mode.
        dataloader:
            DataLoader yielding dicts with keys ``source``, ``target``,
            ``delta_theta``, ``object_id``.
        device:
            Override device; defaults to ``self.device``.
        n_ddim_steps:
            Number of DDIM steps for inference.

        Returns
        -------
        dict with keys:
            ``mean_lpips``   – float
            ``mean_ssim``    – float
            ``per_object``   – dict mapping object_id → {'lpips': [...], 'ssim': [...]}
        """
        dev = torch.device(device) if device else self.device
        model = model.to(dev)
        model.eval()

        all_lpips: List[float] = []
        all_ssim: List[float] = []
        per_object: Dict[int, Dict[str, List[float]]] = {}

        from models.ddpm import ConditionalDDPM  # avoid circular at module level

        with torch.no_grad():
            for batch in dataloader:
                source = batch["source"].to(dev)
                target = batch["target"].to(dev)
                delta_theta = batch["delta_theta"].float().to(dev)
                obj_ids: List[int] = batch["object_id"].tolist()

                # Normalize to [-1, 1] for the model.
                source_n = source * 2.0 - 1.0
                target_n = target * 2.0 - 1.0  # ground truth in [-1,1] for reference

                if isinstance(model, ConditionalDDPM):
                    pred_n = model.ddim_sample(source_n, delta_theta, n_steps=n_ddim_steps)
                else:
                    raise TypeError("model must be a ConditionalDDPM instance.")

                # Back to [0, 1].
                pred = (pred_n + 1.0) / 2.0
                pred = pred.clamp(0.0, 1.0)

                batch_lpips = self.compute_lpips(pred, target)
                batch_ssim = self.compute_ssim(pred, target)
                all_lpips.append(batch_lpips)
                all_ssim.append(batch_ssim)

                # Per-image breakdown for per-object tracking.
                for i, oid in enumerate(obj_ids):
                    img_lpips = self.compute_lpips(pred[i:i+1], target[i:i+1])
                    img_ssim = self.compute_ssim(pred[i:i+1], target[i:i+1])
                    if oid not in per_object:
                        per_object[oid] = {"lpips": [], "ssim": []}
                    per_object[oid]["lpips"].append(img_lpips)
                    per_object[oid]["ssim"].append(img_ssim)

        return {
            "mean_lpips": float(np.mean(all_lpips)),
            "mean_ssim": float(np.mean(all_ssim)),
            "per_object": per_object,
        }
