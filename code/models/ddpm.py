"""Conditional DDPM wrapper around the U-Net backbone.

Implements:
  - Cosine noise schedule (Nichol & Dhariwal 2021).
  - Forward diffusion q(x_t | x_0).
  - Training loss (simple MSE on predicted noise, epsilon-parameterization).
  - DDIM deterministic sampling for fast inference.
  - EMA parameter tracking for stable evaluation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import UNet


# ---------------------------------------------------------------------------
# Schedule utilities
# ---------------------------------------------------------------------------


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Compute cosine noise schedule betas.

    Following the formula in Nichol & Dhariwal (2021):
        alpha_bar(t) = cos((t/T + s) / (1 + s) * pi/2)^2 / cos(s/(1+s)*pi/2)^2

    Parameters
    ----------
    T:
        Total number of diffusion steps.
    s:
        Small offset to prevent beta from being too small near t=0.

    Returns
    -------
    Tensor of shape (T,) with beta values in (0, 1).
    """
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-5, max=0.9999)


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear noise schedule as a fallback option."""
    return torch.linspace(beta_start, beta_end, T)


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------


class EMAHelper:
    """Exponential moving average of model parameters.

    Parameters
    ----------
    model:
        The model whose parameters to track.
    decay:
        EMA decay rate (e.g. 0.9999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA shadow parameters from the current model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA parameters into ``model`` (for inference)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> Dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# ---------------------------------------------------------------------------
# Conditional DDPM
# ---------------------------------------------------------------------------


class ConditionalDDPM(nn.Module):
    """Conditional DDPM that generates x_t given (x_source, Δθ).

    Parameters
    ----------
    unet:
        U-Net noise predictor.
    T:
        Number of diffusion timesteps.
    beta_schedule:
        Either ``'cosine'`` or ``'linear'``.
    """

    def __init__(
        self,
        unet: UNet,
        T: int = 1000,
        beta_schedule: str = "cosine",
    ) -> None:
        super().__init__()
        self.unet = unet
        self.T = T
        self.beta_schedule = beta_schedule

        betas, alphas, alpha_bars = self.get_schedule()
        # Register as buffers so they move with .to(device).
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def get_schedule(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the noise schedule.

        Returns
        -------
        (betas, alphas, alpha_bars)
            Each of shape (T,).
        """
        if self.beta_schedule == "cosine":
            betas = cosine_beta_schedule(self.T)
        elif self.beta_schedule == "linear":
            betas = linear_beta_schedule(self.T)
        else:
            raise ValueError(f"Unknown beta_schedule: {self.beta_schedule}")
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return betas, alphas, alpha_bars

    # ------------------------------------------------------------------
    # Forward diffusion
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0) = N(sqrt(ᾱ_t) x_0, (1-ᾱ_t) I).

        Parameters
        ----------
        x0:
            Clean target image (B, 3, H, W) in [-1, 1].
        t:
            Timestep indices (B,), each in [0, T-1].
        noise:
            Optional pre-sampled noise.  If None, sampled from N(0, I).

        Returns
        -------
        Noisy image x_t of the same shape as x0.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        return sqrt_ab * x0 + sqrt_1mab * noise

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def p_losses(
        self,
        x0_target: torch.Tensor,
        x_source: torch.Tensor,
        delta_theta: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the epsilon-prediction MSE training loss.

        Parameters
        ----------
        x0_target:
            Clean target image (B, 3, H, W) in [-1, 1].
        x_source:
            Source conditioning image (B, 3, H, W) in [-1, 1].
        delta_theta:
            Rotation delta in radians (B,).
        t:
            Optional integer timesteps (B,).  Sampled uniformly if None.

        Returns
        -------
        Scalar MSE loss.
        """
        B = x0_target.shape[0]
        device = x0_target.device

        if t is None:
            t = torch.randint(0, self.T, (B,), device=device)

        noise = torch.randn_like(x0_target)
        x_noisy = self.q_sample(x0_target, t, noise)

        noise_pred = self.unet(x_noisy, x_source, t, delta_theta)
        return F.mse_loss(noise_pred, noise)

    # ------------------------------------------------------------------
    # Rotation encoding
    # ------------------------------------------------------------------

    def encode_rotation(self, delta_theta: torch.Tensor) -> torch.Tensor:
        """Encode rotation delta as a 2-D unit vector.

        Parameters
        ----------
        delta_theta:
            Rotation delta in radians — shape (B,) or scalar tensor.

        Returns
        -------
        Tensor of shape (B, 2) containing [sin(Δθ), cos(Δθ)].
        """
        delta_theta = delta_theta.view(-1)
        return torch.stack([torch.sin(delta_theta), torch.cos(delta_theta)], dim=-1)

    # ------------------------------------------------------------------
    # DDIM sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        x_source: torch.Tensor,
        delta_theta: torch.Tensor,
        n_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate x_0 via DDIM deterministic reverse diffusion.

        Parameters
        ----------
        x_source:
            Source conditioning image (B, 3, H, W) in [-1, 1].
        delta_theta:
            Rotation delta in radians (B,).
        n_steps:
            Number of DDIM denoising steps.
        eta:
            Stochasticity parameter (0 = deterministic DDIM, 1 = DDPM).

        Returns
        -------
        Generated image (B, 3, H, W) in [-1, 1].
        """
        device = x_source.device
        B = x_source.shape[0]

        # Build evenly-spaced timestep subsequence descending from T-1 to 0.
        # ddim_timesteps[i] is the *current* t; the *previous* t is ddim_timesteps[i+1].
        # We include an extra 0 at the end for the "previous" of the last step.
        ddim_timesteps = torch.linspace(self.T - 1, 0, n_steps + 1, device=device).long()
        # ddim_timesteps[0]=T-1, ddim_timesteps[-1]=0.

        x = torch.randn_like(x_source)  # start from pure noise x_T

        for i in range(n_steps):
            t_cur = ddim_timesteps[i]        # current noisy step
            t_prev = ddim_timesteps[i + 1]   # previous (less noisy) step

            t_batch = t_cur.expand(B)
            ab_cur = self.alpha_bars[t_cur]
            # alpha_bar at t=0 is the very first entry (already close to 1).
            ab_prev = self.alpha_bars[t_prev]

            eps_pred = self.unet(x, x_source, t_batch, delta_theta)

            # Predicted x_0 from current x_t and predicted noise.
            x0_pred = (x - (1.0 - ab_cur).sqrt() * eps_pred) / ab_cur.sqrt()
            x0_pred = x0_pred.clamp(-1.0, 1.0)

            # DDIM / DDPM interpolation via eta.
            sigma = (
                eta
                * ((1.0 - ab_prev) / (1.0 - ab_cur)).sqrt()
                * (1.0 - ab_cur / ab_prev).sqrt()
            )
            dir_xt = (1.0 - ab_prev - sigma ** 2).clamp(min=0.0).sqrt() * eps_pred
            noise = sigma * torch.randn_like(x) if eta > 0.0 else 0.0

            x = ab_prev.sqrt() * x0_pred + dir_xt + noise

        return x

    # ------------------------------------------------------------------
    # Convenience: tensor rescaling
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        """Rescale images from [0, 1] to [-1, 1]."""
        return x * 2.0 - 1.0

    @staticmethod
    def denormalize(x: torch.Tensor) -> torch.Tensor:
        """Rescale images from [-1, 1] to [0, 1]."""
        return (x + 1.0) / 2.0
