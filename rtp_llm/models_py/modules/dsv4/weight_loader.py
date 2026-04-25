"""Weight loader for DeepSeek-V4 standalone Transformer.

Maps HF safetensors keys (which mirror official `inference/model.py` naming)
to our `V4Transformer.state_dict()` keys, dequantizing FP8 (e4m3fn + UE8M0
block scale) to BF16 on the fly.

Block size is fixed at 128 per `config.json:quantization_config.weight_block_size`.

Usage:
    from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer, V4Args
    from rtp_llm.models_py.modules.dsv4.weight_loader import load_v4_safetensors

    args = V4Args(...)
    model = V4Transformer(args)
    load_v4_safetensors(model, ckpt_dir, dtype=torch.bfloat16, device='cpu')
"""

import json
import os
from typing import Dict, Iterable, Optional, Set

import torch
from safetensors import safe_open


FP8_BLOCK = 128
FP4_BLOCK = 32


def _repack_v4_fp8_scale_to_int32(scale: torch.Tensor) -> torch.Tensor:
    """Convert V4's FP8 weight-scale layout to DeepGEMM's MN-major
    TMA-aligned UE8M0-packed int32 tensor.

    V4 ckpt ships FP8 attention/indexer-linear scales as
    ``torch.float8_e8m0fnu`` with shape ``[N/128, K/128]`` — one byte per
    128×128 weight block. DeepGEMM's ``fp8_gemm_nt`` (SM100+ E8M0 path)
    consumes scales via its ``get_mn_major_tma_aligned_packed_ue8m0_tensor``
    utility which takes an FP32 scale of shape ``[N, K/128]``, one row per
    weight row.  We:

    1. Cast UE8M0 → FP32 via ``.float()`` (semantically ``2^(byte-127)``)
    2. Row-repeat along N by 128 so each weight row gets its own scale row
    3. Hand off to DeepGEMM's helper which returns a column-major
       int32-packed UE8M0 tensor in TMA-aligned layout.

    Must be called on-device (the DeepGEMM helper is a CUDA op).
    """
    assert scale.dtype == torch.float8_e8m0fnu, f"unexpected scale dtype {scale.dtype}"
    assert scale.dim() == 2, f"unexpected scale dim {scale.dim()}"
    from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

    N_blk, K_blk = scale.shape
    N = N_blk * 128
    scale_fp32 = scale.float()
    idx = torch.arange(N, device=scale.device) // 128
    scale_rep = scale_fp32.index_select(-2, idx)
    return get_mn_major_tma_aligned_packed_ue8m0_tensor(scale_rep)


def _should_repack_scale(weight_dtype: torch.dtype, scale_dtype: torch.dtype) -> bool:
    """Only FP8 e4m3fn weights with UE8M0 scale need repacking. FP4 stays
    as-is (DeepGEMM FP4 kernels consume UE8M0 directly) and BF16/FP32
    params have no scale."""
    return (
        weight_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        and scale_dtype == torch.float8_e8m0fnu
    )


def load_v4_weights_dict(
    ckpt_dir: str,
    device: str = "cpu",
    keys_filter: Optional[Iterable[str]] = None,
    repack_fp8_scale: bool = True,
    cast_bf16_fp32_params: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load V4-Flash safetensors into a flat dict ready to feed into
    ``LinearFactory.create_linear_from_weights``.

    - FP8 weights keep their native ``float8_e4m3fn`` dtype.
    - FP8 scales are repacked into int32 (``_repack_v4_fp8_scale_to_int32``)
      so ``CudaFp8DeepGEMMLinear`` on SM100+ consumes them directly.
    - FP4 weights (packed int8) and FP4 scales (UE8M0) are returned
      unchanged — DeepGEMM's ``fp8_fp4_gemm_nt`` consumes them natively.
    - BF16/FP32 params (norms, gate, embed, lm_head, hc_*) are returned
      as-is so callers can wrap them in ``nn.Parameter`` directly.

    Memory-efficient via ``safe_open(..., device=device)`` — tensors land
    straight in target-device memory with no CPU staging.
    """
    idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map: Dict[str, str] = json.load(f)["weight_map"]

    filters: Optional[Set[str]] = None
    if keys_filter is not None:
        filters = set(keys_filter)

    def _matches(key: str) -> bool:
        if filters is None:
            return True
        return any(key.startswith(p) for p in filters)

    # Group keys by shard for sequential reads per file.
    by_shard: Dict[str, list] = {}
    for key, shard in weight_map.items():
        if not _matches(key):
            continue
        by_shard.setdefault(shard, []).append(key)

    out: Dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        path = os.path.join(ckpt_dir, shard)
        with safe_open(path, framework="pt", device=device) as f:
            for key in keys:
                out[key] = f.get_tensor(key)

    if repack_fp8_scale:
        # NOTE: we do NOT repack here — some call sites (e.g. the grouped
        # output projection wo_a, routed FP4 experts) want the original
        # ckpt layout. Call sites that route through LinearFactory are
        # responsible for invoking `_repack_v4_fp8_scale_to_int32` on the
        # specific scale tensor they consume. This parameter is kept for
        # API stability and future opt-in batch repacking.
        pass

    # Some params (norms, attn_sink) ship as BF16/F32 in ckpt but the
    # standalone reference code built them as FP32 Parameters. Callers
    # that assign directly as nn.Parameter tolerate dtype mismatch; no
    # eager cast here (saves RAM) unless explicitly asked.
    return out


def _dequant_fp8_block128(weight_fp8: torch.Tensor, scale: torch.Tensor,
                          out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize an FP8 e4m3fn weight matrix using a UE8M0 block-wise scale.

    Args:
      weight_fp8: [out, in] in float8_e4m3fn
      scale:      [ceil(out/128), ceil(in/128)] in float8_e8m0fnu
                  scale[i, j] applies to weight_fp8[i*128:(i+1)*128, j*128:(j+1)*128]

    Returns: weight in `out_dtype` (BF16 by default).
    """
    out_dim, in_dim = weight_fp8.shape
    scale_f = scale.to(torch.float32)
    w_f = weight_fp8.to(torch.float32)
    scale_full = scale_f.repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    scale_full = scale_full[:out_dim, :in_dim]
    return (w_f * scale_full).to(out_dtype)


# FP4 e2m1 lookup table (16 values, indexed by 4-bit raw)
# Layout: sign bit + 2 exponent bits + 1 mantissa bit, biased
_FP4_LUT = torch.tensor([
     0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def _dequant_fp4_block32(weight_int8: torch.Tensor, scale: torch.Tensor,
                         out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize FP4 e2m1 (packed 2-per-byte in int8) with UE8M0 32-block scale.

    Args:
      weight_int8: [out, in/2] int8 — 2 fp4 values per byte (low nibble first)
      scale:       [out, in/32] UE8M0  — scale per 32 fp4 elements along K

    Returns: dequantized weight [out, in] in out_dtype.
    """
    out_dim, packed_in = weight_int8.shape
    in_dim = packed_in * 2

    # Unpack: low nibble + high nibble.
    w_uint = weight_int8.to(torch.int32) & 0xFF
    low = w_uint & 0x0F
    high = (w_uint >> 4) & 0x0F
    # interleave: result[..., 2i] = low[..., i], result[..., 2i+1] = high[..., i]
    interleaved = torch.empty(out_dim, in_dim, dtype=torch.int64, device=weight_int8.device)
    interleaved[:, 0::2] = low.long()
    interleaved[:, 1::2] = high.long()

    lut = _FP4_LUT.to(weight_int8.device)
    w_f = lut[interleaved]                              # [out, in] fp32

    scale_f = scale.to(torch.float32)
    scale_full = scale_f.repeat_interleave(FP4_BLOCK, 1)[:, :in_dim]
    # scale is per-row × per-32-K-block; only column repeat needed
    return (w_f * scale_full).to(out_dtype)


def _load_index(ckpt_dir: str) -> Dict[str, str]:
    """Returns {key -> shard_filename} from model.safetensors.index.json."""
    idx_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        return json.load(f)["weight_map"]


def load_v4_safetensors(
    model: torch.nn.Module,
    ckpt_dir: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    keys_filter: Optional[Iterable[str]] = None,
    strict: bool = False,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """Populate `model.state_dict()` from V4-Flash safetensors checkpoint.

    Args:
      model:        V4Transformer instance
      ckpt_dir:     directory holding model-NNNNN-of-NNNNN.safetensors and index json
      dtype:        target dtype for FP8-dequantized linear weights
      device:       where to place loaded tensors
      keys_filter:  if given, only load ckpt keys whose prefix matches one of these
                    (e.g. ['embed', 'layers.0.', 'norm', 'head', 'hc_head'] for layer-0 only)
      strict:       if True, raise on any missing/unexpected keys
      verbose:      print per-key dtype info

    Returns: dict of {key: tensor} actually loaded.
    """
    weight_map = _load_index(ckpt_dir)
    state = dict(model.state_dict())  # the destination keyset

    # Group keys by shard for efficient loading.
    by_shard: Dict[str, list] = {}
    for k, shard in weight_map.items():
        if keys_filter is not None and not any(k.startswith(p) for p in keys_filter):
            continue
        by_shard.setdefault(shard, []).append(k)

    loaded: Dict[str, torch.Tensor] = {}
    seen_state_keys = set()

    # Open all shards lazily and keep handles cached so we can read scales
    # that may live in a different shard than their weight.
    open_shards: Dict[str, "safe_open"] = {}

    def _get_tensor(key: str) -> Optional[torch.Tensor]:
        shard = weight_map.get(key)
        if shard is None:
            return None
        if shard not in open_shards:
            ctx = safe_open(os.path.join(ckpt_dir, shard), framework="pt", device=device)
            open_shards[shard] = ctx.__enter__()
        return open_shards[shard].get_tensor(key)

    # For the "preserve native dtype" path (QuantizedLinear), we copy weight and
    # scale directly without dequantizing. Model Parameter dtype for QuantizedLinear
    # is int8 (fp4) or float8_e4m3fn (fp8), and scale Parameter is float8_e8m0fnu.
    # Only when the model Parameter is BF16/FP32 and ckpt is FP4/FP8 do we dequant.

    def _maybe_dequant(w: torch.Tensor, s: Optional[torch.Tensor], target_param: torch.Tensor):
        """Return a tensor matching target_param.dtype.

        - If ckpt already matches (e.g. ckpt int8 + model int8): pass through.
        - If ckpt is FP4/FP8 but model is BF16/FP32: dequantize.
        - Otherwise: cast.
        """
        if s is None:
            # No scale: may need cast only.
            return w
        # Scale present — ckpt is quantized.
        if w.dtype == target_param.dtype:
            return w   # native preservation path (QuantizedLinear)
        # Fallback: dequant (used for the rare case where caller wants BF16 eagerly).
        if w.dtype == torch.int8:
            return _dequant_fp4_block32(w, s, out_dtype=dtype)
        return _dequant_fp8_block128(w, s, out_dtype=dtype)

    try:
        for shard, keys in by_shard.items():
            base_keys = sorted({k[:-6] if k.endswith(".scale") else k for k in keys})
            for base in base_keys:
                w = _get_tensor(base)
                if w is None:
                    continue
                base_root = base[:-7] if base.endswith(".weight") else base
                s = _get_tensor(base_root + ".scale")

                target = base
                if target not in state:
                    if strict:
                        raise KeyError(f"checkpoint key {target!r} has no model destination")
                    continue
                target_param = state[target]
                w_final = _maybe_dequant(w, s, target_param)
                if w_final.dtype != target_param.dtype:
                    w_final = w_final.to(target_param.dtype)
                if w_final.shape != target_param.shape:
                    raise RuntimeError(
                        f"shape mismatch loading {target}: ckpt {tuple(w_final.shape)} vs model {tuple(target_param.shape)}"
                    )
                target_param.copy_(w_final)
                loaded[target] = target_param
                seen_state_keys.add(target)

                # If this is a QuantizedLinear weight with matching native dtype,
                # also copy the scale into state[target_root + ".scale"].
                if s is not None:
                    scale_target = base_root + ".scale"
                    if scale_target in state:
                        scale_param = state[scale_target]
                        s_cast = s if s.dtype == scale_param.dtype else s.to(scale_param.dtype)
                        if s_cast.shape != scale_param.shape:
                            raise RuntimeError(
                                f"scale shape mismatch {scale_target}: ckpt {tuple(s_cast.shape)} vs model {tuple(scale_param.shape)}"
                            )
                        scale_param.copy_(s_cast)
                        loaded[scale_target] = scale_param
                        seen_state_keys.add(scale_target)
                if verbose:
                    tag = "quant-native" if (s is not None and w.dtype == target_param.dtype) else (
                          "deq-to-bf16" if s is not None else "direct")
                    print(f"  [{tag:13s}] {base} -> {tuple(target_param.shape)} {target_param.dtype}")
    finally:
        for h in open_shards.values():
            h.__exit__(None, None, None)

    missing = set(state.keys()) - seen_state_keys
    # Filter out persistent=False buffers (kv_cache, kv_state, score_state, freqs_cis)
    # and the unloaded experts in MoE TP setups (we load all experts here, world_size=1)
    benign_missing = {k for k in missing if any(s in k for s in [
        "kv_cache", "kv_state", "score_state", "freqs_cis",
    ])}
    real_missing = missing - benign_missing
    if real_missing and strict:
        raise KeyError(f"missing keys after load: {sorted(real_missing)[:20]} ...")
    if verbose and real_missing:
        print(f"WARNING: {len(real_missing)} model keys not found in ckpt (e.g. {sorted(real_missing)[:5]})")
    return loaded
