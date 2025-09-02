import math
from typing import Tuple

import torch

# the implement of yarn is from https://github.com/jquesnelle/yarn


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class FlashYaRNRotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    This implements the YaRN extension method.
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scaling_factor=1.0,
        pos_idx_in_fp32=True,
        max_position_embeddings=2048,
        original_max_position_embeddings=2048,
        extrapolation_factor=1,
        attn_factor=1,
        beta_fast=32,
        beta_slow=1,
        dynamic=False,
        finetuned=False,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        scaling_factor: RotaryEmbedding extended with YaRN scaling.
        """
        super().__init__()

        self.dim = dim
        self.base = float(base)
        self.interleaved = interleaved
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = (
            original_max_position_embeddings
            if original_max_position_embeddings
            else max_position_embeddings
        )
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        self.mscale = float(
            _yarn_get_mscale(self.scaling_factor) * attn_factor
        )  # Get n-d magnitude scaling corrected for interpolation
        self.dynamic = dynamic
        self.finetuned = finetuned

        # Generate and save the inverse frequency buffer (non trainable)
        if not dynamic:
            self._compute_inv_freq(scaling_factor, device)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _compute_inv_freq(self, scaling_factor, device=None):
        pos_freqs = self.base ** (
            torch.arange(0, self.dim, 2).float().to(device) / self.dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        # import pdb; pdb.set_trace()

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = (
            1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float().to(device)
        ) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq_original(self, device=None):
        inv_freq = 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen

            if self.dynamic:
                scaling_factor = None
                if seqlen <= self.max_position_embeddings:
                    if self.finetuned:
                        scaling_factor = self.scaling_factor
                else:
                    scaling_factor = seqlen / self.original_max_position_embeddings
                if scaling_factor:
                    self._compute_inv_freq(scaling_factor, device)
                    self.mscale = float(
                        _yarn_get_mscale(scaling_factor) * self.attn_factor
                    )
                else:
                    self._compute_inv_freq_original(device)

            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            freqs = torch.outer(t, inv_freq.to(device)).to(device)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = (torch.cos(emb) * self.mscale).to(dtype)
            self._sin_cached = (torch.sin(emb) * self.mscale).to(dtype)

    def forward(
        self, q: torch.Tensor, seqlen_offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(
            q.shape[2] + seqlen_offset, device=q.device, dtype=q.dtype
        )
        return self._cos_cached, self._sin_cached
