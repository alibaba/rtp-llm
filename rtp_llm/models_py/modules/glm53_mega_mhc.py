"""SM100 MegaMHC; Fn uses TF32. GLM53_MEGA_MHC=0 restores FP32 HC."""

import functools
import importlib.util
import logging
import os
import pathlib

import torch

_ENABLED = os.environ.get("GLM53_MEGA_MHC", "1") == "1"


@functools.lru_cache(maxsize=1)
def _get_library():
    # These optional JIT assets are distributed with the runtime overlay, not
    # every RTP installation. Missing/incompatible assets keep the HC fallback.
    root = pathlib.Path(
        os.environ.get(
            "GLM53_MEGA_MHC_LIBRARY_DIR",
            str(pathlib.Path(__file__).parents[2] / "libs/mega_mhc_isolated"),
        )
    )
    required = (
        "csrc/jit_kernels/impls/sm100_mega_mhc.hpp",
        "include/deep_jit/utils/parser.hpp",
    )
    binaries = list(root.glob("mhc_isolated*.so"))
    if len(binaries) != 1 or not all((root / name).is_file() for name in required):
        logging.info("GLM53 MegaMHC unavailable at %s; using ordinary HC", root)
        return None
    sms = int(os.environ.get("GLM53_MEGA_MHC_SMS", "96"))
    if (
        not 1
        <= sms
        <= torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    ):
        raise ValueError("GLM53_MEGA_MHC_SMS exceeds the device SM count")
    try:
        spec = importlib.util.spec_from_file_location("mhc_isolated", binaries[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.init(str(root))
        module.set_num_sms(sms)
        module.set_pdl(os.environ.get("GLM53_DG_PDL", "1") == "1")
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        logging.warning(
            "GLM53 MegaMHC extension unavailable; using ordinary HC: %s", error
        )
        return None
    logging.info(
        "GLM53 isolated MegaMHC initialized, SMs=%d, library=%s", sms, binaries[0]
    )
    return module


def can_use_mega_mhc(x, residual, post, comb, unit, norm):
    """Guard all fixed ABI assumptions before using the optional SM100 kernel."""
    if not (
        _ENABLED
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[0] in (48, 192, 240, 288)
        and x.shape[1] == 4096
        and torch.cuda.get_device_capability(x.device) == (10, 0)
        and unit.hc_mult == 4
        and unit.dim == 4096
    ):
        return False
    rows = x.shape[0]
    tensors = (
        (x, (rows, 4096), torch.bfloat16),
        (residual, (rows, 4, 4096), torch.bfloat16),
        (post, (rows, 4, 1), torch.float32),
        (comb, (rows, 4, 4), torch.float32),
        (unit.fn, (24, 16384), torch.float32),
        (norm.weight, (4096,), torch.bfloat16),
    )
    if any(
        t.shape != shape
        or t.dtype != dtype
        or t.device != x.device
        or not t.is_contiguous()
        for t, shape, dtype in tensors
    ):
        return False
    if any(
        t.numel() != size
        or t.dtype != torch.float32
        or t.device != x.device
        or not t.is_contiguous()
        for t, size in ((unit.scale, 3), (unit.base, 24))
    ):
        return False
    return _get_library() is not None


def fused_post_pre_rms(x, residual, post, comb, unit, norm, mega_inputs=None):
    if x.ndim != 2 or x.shape not in (
        (48, 4096),
        (192, 4096),
        (240, 4096),
        (288, 4096),
    ):
        raise ValueError(
            "MegaMHC candidate is validated only for M48/M192/M240/M288,H4096"
        )
    library = _get_library()
    if library is None:
        raise RuntimeError(
            "MegaMHC requires validated runtime assets; call can_use_mega_mhc first"
        )
    updated = torch.empty_like(residual)
    next_post = torch.empty_like(post)
    next_comb = torch.empty_like(comb)
    y = torch.empty_like(x)
    extra = {}
    if mega_inputs is not None:
        extra = {
            "y_fp8": mega_inputs["x_fp8"],
            "y_routed_sf": mega_inputs["routed_sf"],
            "y_shared_sf": mega_inputs["shared_sf"],
            "shared_sf_block_m": mega_inputs["block_m"],
        }
    library.mega_mhc(
        x=x,
        residual=residual,
        shifted_prev_mix=None,
        post_mix=post,
        comb_res_mix=comb,
        fn=unit.fn,
        mix_scales=unit.scale.reshape(3),
        mix_bases=unit.base.reshape(24),
        hc_mult=4,
        hc_norm_eps=float(unit.norm_eps),
        hc_pre_eps=float(unit.hc_eps),
        hc_post_scale=2.0,
        sinkhorn_eps=float(unit.hc_eps),
        num_sinkhorn_iters=int(unit.hc_sinkhorn_iters),
        rmsnorm_weight=norm.weight.data,
        rmsnorm_eps=float(norm.variance_epsilon),
        rmsnorm_scale=1.0,
        new_residual=updated,
        new_prev_mix=None,
        new_post_mix=next_post,
        new_comb_res_mix=next_comb,
        y_bf16=y,
        **extra,
    )
    return updated, y, next_post, next_comb
