"""Startup-only LoRA on the canonical local projection, before executor packing."""

import torch

from rtp_llm.models_py.runtime_weight_update import checked_lora_sum


def _merged(base, update):
    delta = update.delta().to(base)
    if delta.shape != base.shape:
        raise ValueError(f"Invalid startup LoRA delta for {update.target!r}")
    return checked_lora_sum(update.target, base, delta)


class FloatingStartupLoRA:
    """Keep factors, not a dense delta, while loading floating source shards."""

    def can_stage_startup_lora(self, layer):
        return layer.weight.dtype in (torch.float16, torch.bfloat16, torch.float32)

    def stage_startup_lora(self, layer, update):
        if not self.can_stage_startup_lora(layer):
            raise ValueError("Startup LoRA requires unpacked floating weights")
        layer._startup_lora_update = update

    def weight_before_quantization(self, layer):
        update = getattr(layer, "_startup_lora_update", None)
        if update is None:
            return layer.weight.detach()
        weight = _merged(layer.weight.detach(), update)
        del layer._startup_lora_update
        return weight


@torch.no_grad()
def merge_calibrated_projection(layer, update, *, per_row, dtype):
    """FT importer has already decoded and validated [N,K] codes and steps.

    Preserve the original step unless the adapter exceeds its range. Activation
    calibration is independent of these weight coding steps and stays unchanged.
    """
    weight, scale = layer.weight, layer.weight_scale
    expected = (weight.shape[0], 1) if per_row else (1,)
    if (weight.ndim != 2 or weight.dtype != dtype or scale.shape != expected
            or scale.dtype != torch.float32 or not torch.isfinite(scale).all()
            or not (scale > 0).all()):
        raise ValueError(f"Invalid canonical FT weights or calibration for {update.target!r}")
    merged = _merged(weight.float() * scale, update)
    lo, hi = (-128, 127) if dtype == torch.int8 else (-448, 448)
    dims = (1,) if per_row else (0, 1)
    needed = torch.maximum(merged.amax(dims) / hi, merged.amin(dims) / lo).reshape_as(scale)
    step = torch.maximum(scale, needed)
    codes = (merged / step).clamp(lo, hi)
    if dtype == torch.int8:
        codes = codes.round()
    weight.copy_(codes.to(dtype))
    scale.copy_(step)


@torch.no_grad()
def merge_block_fp8_projection(layer, update, block):
    """Merge a local FT delta before DeepGEMM/ROCm pack the canonical grid."""
    weight, scale = layer.weight, layer.weight_scale_inv
    n, k = weight.shape
    nb, kb = (n + block - 1) // block, (k + block - 1) // block
    if (
        weight.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.float32
        or scale.shape != (nb, kb)
        or not torch.isfinite(scale).all()
        or not (scale > 0).all()
    ):
        raise ValueError(
            f"Invalid calibrated FT block FP8 projection for {update.target!r}"
        )

    def tiles(matrix):
        pad_k, pad_n = kb * block - k, nb * block - n
        padded = torch.nn.functional.pad(matrix, (0, pad_k, 0, pad_n)) if pad_k or pad_n else matrix
        return padded.view(nb, block, kb, block)

    def unpad(grid):
        return grid.reshape(nb * block, kb * block)[:n, :k]

    base = unpad(tiles(weight.float()) * scale[:, None, :, None])
    merged = tiles(_merged(base, update))
    limit = torch.finfo(weight.dtype).max
    needed = merged.abs().amax(dim=(1, 3)) / limit
    step = torch.maximum(scale, needed)
    if "weight_scale_inv" in getattr(layer, "_packed_block_fp8_ft", ()):
        # The source was UE8M0; the runtime will repack scales losslessly.
        # Expanded scales must remain exactly representable as exponent bytes.
        mantissa, exponent = torch.frexp(step)
        step = torch.where(
            mantissa == 0.5, step, torch.ldexp(torch.ones_like(step), exponent)
        )
    codes = (merged / step[:, None, :, None]).clamp_(-limit, limit)
    weight.copy_(unpad(codes).to(weight.dtype))
    scale.copy_(step)
