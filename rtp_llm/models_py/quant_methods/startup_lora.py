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
def merge_calibrated_projection(layer, update, *, per_row):
    """FT importer has already decoded and validated [N,K] codes and steps.

    Preserve the original step unless the adapter exceeds its range. Activation
    calibration is independent of these weight coding steps and stays unchanged.
    """
    weight, scale = layer.weight, layer.weight_scale
    dtype = torch.int8 if per_row else torch.float8_e4m3fn
    expected = (weight.shape[0], 1) if per_row else (1,)
    if (weight.ndim != 2 or weight.dtype != dtype or scale.shape != expected
            or scale.dtype != torch.float32 or not torch.isfinite(scale).all()
            or not (scale > 0).all()):
        raise ValueError(f"Invalid canonical FT weights or calibration for {update.target!r}")
    merged = _merged(weight.float() * scale, update)
    lo, hi = (-128, 127) if per_row else (-448, 448)
    dims = (1,) if per_row else (0, 1)
    needed = torch.maximum(merged.amax(dims) / hi, merged.amin(dims) / lo).reshape_as(scale)
    step = torch.maximum(scale, needed)
    codes = (merged / step).clamp(lo, hi)
    if per_row:
        codes = codes.round()
    weight.copy_(codes.to(dtype))
    scale.copy_(step)
