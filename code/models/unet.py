"""U-Net backbone for conditional image-to-image diffusion.

Architecture
------------
* Input: concatenation of [noisy_target (3 ch), source_image (3 ch)] → 6 channels.
* Time embedding: sinusoidal positional encoding passed through a 2-layer MLP.
* Rotation conditioning: r(Δθ) = [sin(Δθ), cos(Δθ)] projected to 256-dim and
  **added** to the time embedding so a single conditioning vector drives AdaGN.
* Encoder: 4 resolution scales with channel widths [64, 128, 256, 512].
  Each scale has 2 ResBlocks followed by a 2× average-pool downsampler.
* Bottleneck: ResBlock → SelfAttention → ResBlock.
* Decoder: 4 upsampling scales mirroring the encoder, with skip connections.
  Attention is applied at scale indices 1, 2, 3 (16×16, 8×8, 4×4 if input is 128×128).
* Output: 3-channel noise prediction at the original spatial resolution.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Compute sinusoidal positional embeddings for diffusion timesteps.

    Parameters
    ----------
    timesteps:
        1-D integer tensor of shape (B,).
    dim:
        Output embedding dimensionality (must be even).

    Returns
    -------
    Tensor of shape (B, dim).
    """
    assert dim % 2 == 0, "Embedding dim must be even."
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / (half - 1)
    )  # (half,)
    args = timesteps.float()[:, None] * freqs[None, :]  # (B, half)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    return embedding


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    """Residual block with AdaGN (Adaptive Group Normalization).

    Conditioning signal (time + rotation) is injected via two learned
    scale/shift parameters applied after the first group-norm, following
    the design in Dhariwal & Nichol (2021).

    Parameters
    ----------
    in_ch:
        Input channel count.
    out_ch:
        Output channel count.
    time_emb_dim:
        Dimensionality of the combined time+rotation embedding.
    num_groups:
        Number of groups for GroupNorm.  Must divide both in_ch and out_ch.
    dropout:
        Dropout probability between the two convolutions.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_emb_dim: int,
        num_groups: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Clamp num_groups so it always divides the channel counts.
        g_in = min(num_groups, in_ch)
        while g_in > 1 and in_ch % g_in != 0:
            g_in -= 1
        g_out = min(num_groups, out_ch)
        while g_out > 1 and out_ch % g_out != 0:
            g_out -= 1

        self.norm1 = nn.GroupNorm(g_in, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # AdaGN projection: maps conditioning → 2 * out_ch (scale + shift).
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * out_ch),
        )

        self.norm2 = nn.GroupNorm(g_out, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Skip connection when channel counts differ.
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Feature map (B, in_ch, H, W).
        cond:
            Conditioning vector (B, time_emb_dim).

        Returns
        -------
        Feature map (B, out_ch, H, W).
        """
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # AdaGN: scale and shift after first norm.
        scale_shift = self.cond_proj(cond)[:, :, None, None]  # (B, 2*out_ch, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale) + shift

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention over spatial features.

    The spatial dimensions are flattened to a sequence, attention is applied,
    and the result is reshaped back.  Layer norm is applied before attention
    (pre-norm formulation).

    Parameters
    ----------
    channels:
        Number of feature channels (= sequence embedding dimension).
    num_heads:
        Number of attention heads.  Must divide ``channels``.
    """

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        # Ensure num_heads divides channels.
        while num_heads > 1 and channels % num_heads != 0:
            num_heads //= 2
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Parameters
        ----------
        x:
            Feature map (B, C, H, W).

        Returns
        -------
        Feature map (B, C, H, W) — same shape as input.
        """
        B, C, H, W = x.shape
        # Flatten spatial dims → sequence length N = H*W.
        seq = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
        seq = self.norm(seq)
        attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
        # Residual connection and reshape.
        out = (seq + attn_out).permute(0, 2, 1).reshape(B, C, H, W)
        return out


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------


class UNet(nn.Module):
    """Conditional U-Net for diffusion-based image-to-image prediction.

    The network takes a 6-channel input (noisy target concatenated with the
    source conditioning image) and predicts the noise residual (3 channels).

    Rotation conditioning is encoded as [sin(Δθ), cos(Δθ)], projected to
    ``cond_emb_dim`` dimensions, and added to the time embedding.

    Parameters
    ----------
    base_ch:
        Base channel width.  Subsequent scales multiply by [1, 2, 4, 8].
    n_blocks:
        Number of encoder/decoder resolution scales.
    time_emb_dim:
        Dimension of sinusoidal time embedding and MLP output.
    cond_emb_dim:
        Output dimension of the rotation projection MLP.
        Should equal ``time_emb_dim`` so the two can be simply summed.
    """

    CHANNEL_MULTS: List[int] = [1, 2, 4, 8]

    def __init__(
        self,
        base_ch: int = 64,
        n_blocks: int = 4,
        time_emb_dim: int = 256,
        cond_emb_dim: int = 256,
    ) -> None:
        super().__init__()
        assert n_blocks <= len(self.CHANNEL_MULTS), (
            f"n_blocks={n_blocks} > len(CHANNEL_MULTS)={len(self.CHANNEL_MULTS)}"
        )
        assert time_emb_dim == cond_emb_dim, (
            "time_emb_dim must equal cond_emb_dim so they can be summed."
        )

        self.base_ch = base_ch
        self.n_blocks = n_blocks
        self.time_emb_dim = time_emb_dim
        ch_widths = [base_ch * m for m in self.CHANNEL_MULTS[:n_blocks]]  # e.g. [64,128,256,512]

        # ---- Time embedding MLP ----------------------------------------
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # ---- Rotation conditioning MLP ---------------------------------
        # Input: 2 (sin/cos), output: cond_emb_dim.
        self.rot_mlp = nn.Sequential(
            nn.Linear(2, cond_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_emb_dim * 4, cond_emb_dim),
        )

        # Combined conditioning dimensionality fed into ResBlocks.
        combined_dim = time_emb_dim  # time + rot are summed → same dim

        # ---- Initial projection ----------------------------------------
        # 6-channel input (3 noisy target + 3 source).
        self.input_conv = nn.Conv2d(6, base_ch, 3, padding=1)

        # ---- Encoder ---------------------------------------------------
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.downsamplers: nn.ModuleList = nn.ModuleList()
        in_ch = base_ch
        for i, out_ch in enumerate(ch_widths):
            self.enc_blocks.append(
                nn.ModuleList(
                    [
                        ResBlock(in_ch, out_ch, combined_dim),
                        ResBlock(out_ch, out_ch, combined_dim),
                        # Attention at all but the coarsest scale.
                        SelfAttention(out_ch) if i > 0 else nn.Identity(),
                    ]
                )
            )
            self.downsamplers.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            in_ch = out_ch

        # ---- Bottleneck ------------------------------------------------
        btl_ch = ch_widths[-1]
        self.mid_block1 = ResBlock(btl_ch, btl_ch, combined_dim)
        self.mid_attn = SelfAttention(btl_ch)
        self.mid_block2 = ResBlock(btl_ch, btl_ch, combined_dim)

        # ---- Decoder ---------------------------------------------------
        # At decoder level i (0 = coarsest, n_blocks-1 = finest):
        #   - upsample from enc_out[n_blocks-1-i] channels to the same ch count
        #   - concatenate with skip from encoder at that level
        #   - two ResBlocks reducing to enc_out[n_blocks-2-i] channels (or base_ch at finest)
        self.dec_blocks: nn.ModuleList = nn.ModuleList()
        self.upsamplers: nn.ModuleList = nn.ModuleList()
        # Keep track of the current channel width coming out of the previous stage.
        # After bottleneck, we have ch_widths[-1] channels.
        cur_ch = ch_widths[-1]
        for i in range(n_blocks):
            # The skip comes from encoder stage (n_blocks-1-i).
            skip_ch = ch_widths[n_blocks - 1 - i]
            # Upsample cur_ch → skip_ch channels.
            self.upsamplers.append(
                nn.ConvTranspose2d(cur_ch, skip_ch, 4, stride=2, padding=1)
            )
            # After concat with skip: 2 * skip_ch channels.
            dec_in = skip_ch * 2
            # Output channel width.
            if i < n_blocks - 1:
                dec_out = ch_widths[n_blocks - 2 - i]
            else:
                dec_out = base_ch
            self.dec_blocks.append(
                nn.ModuleList(
                    [
                        ResBlock(dec_in, dec_out, combined_dim),
                        ResBlock(dec_out, dec_out, combined_dim),
                        # Attention at all but the finest (highest-res) decoder stage.
                        SelfAttention(dec_out) if i < n_blocks - 1 else nn.Identity(),
                    ]
                )
            )
            cur_ch = dec_out

        # ---- Output projection -----------------------------------------
        g = min(32, base_ch)
        while base_ch % g != 0:
            g //= 2
        self.out_norm = nn.GroupNorm(g, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 3, 3, padding=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_noisy: torch.Tensor,
        x_source: torch.Tensor,
        t: torch.Tensor,
        delta_theta: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise residual.

        Parameters
        ----------
        x_noisy:
            Noisy target image at diffusion step *t* — (B, 3, H, W).
        x_source:
            Source conditioning image — (B, 3, H, W).
        t:
            Integer diffusion timesteps — (B,).
        delta_theta:
            Rotation delta in radians — (B,) or (B, 1).

        Returns
        -------
        Predicted noise tensor (B, 3, H, W).
        """
        B = x_noisy.shape[0]
        delta_theta = delta_theta.view(B)

        # --- Build conditioning vector ----------------------------------
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)  # (B, D)
        t_emb = self.time_mlp(t_emb)

        rot_enc = torch.stack([torch.sin(delta_theta), torch.cos(delta_theta)], dim=-1)  # (B, 2)
        rot_emb = self.rot_mlp(rot_enc)  # (B, D)

        cond = t_emb + rot_emb  # (B, D)

        # --- U-Net forward ----------------------------------------------
        h = self.input_conv(torch.cat([x_noisy, x_source], dim=1))  # (B, base_ch, H, W)

        # Encoder.
        skips: List[torch.Tensor] = []
        for res1, res2, attn in self.enc_blocks:
            h = res1(h, cond)
            h = res2(h, cond)
            if not isinstance(attn, nn.Identity):
                h = attn(h)
            skips.append(h)
            h = self.downsamplers[len(skips) - 1](h)

        # Bottleneck.
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)

        # Decoder.
        for i, (res1, res2, attn) in enumerate(self.dec_blocks):
            h = self.upsamplers[i](h)
            skip = skips[self.n_blocks - 1 - i]
            h = torch.cat([h, skip], dim=1)
            h = res1(h, cond)
            h = res2(h, cond)
            if not isinstance(attn, nn.Identity):
                h = attn(h)

        # Output.
        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
