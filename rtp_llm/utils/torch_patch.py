import logging
from typing import List

import torch

# 1. 保存原始 concat
original_concat = torch.concat


# 2. 定义自定义 concat
def custom_concat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    try:
        return original_concat(tensors, dim=dim).contiguous()
    except RuntimeError as e:
        if "cat_cuda" in str(e) and "Float8_e4m3fn" in str(e):
            logging.info("Caught Float8_e4m3fn concat error. Falling back to CPU.")
            original_device = tensors[0].device
            tensors_cpu = [t.cpu() for t in tensors]
            result_cpu = original_concat(tensors_cpu, dim=dim).contiguous()
            return result_cpu.to(original_device)
        else:
            raise


# 3. 劫持 torch.concat
torch.concat = custom_concat


# 4. UE8M0 dist.broadcast compat shim
#
# NCCL refuses ``Float8_e8m0fnu`` ("Input tensor data type is not supported
# for NCCL process group: Float8_e8m0fnu"), which blocks any multi-rank
# collective on UE8M0 tensors.  In particular fastsafetensors' torch
# backend (frameworks/_torch.py:broadcast) calls ``dist.broadcast`` to
# reshuffle weights across ranks during multi-rank checkpoint loading,
# and that shuffles UE8M0 weight scales (DSv4-Flash etc).  Scatter is
# not exercised by fastsafetensors -- both ``ParallelLoader`` and
# RTP-LLM's ``PerExpertParallelLoader`` route every UE8M0 transfer
# through ``pg.broadcast`` (dim=-1 in fastsafetensors / per-expert
# broadcast in PerExpertParallelLoader); only the broadcast wrapper is
# needed.
#
# Workaround: view the tensor as uint8 (zero-copy reinterpret -- same
# storage, same shape, 1 byte per element) before handing it to NCCL.
# The byte-level NCCL transfer lands in the original storage; the
# receiver's tensor dtype is still UE8M0.  Forward-path collectives are
# unaffected because UE8M0 scales are local and never broadcast outside
# the loader.
#
# Version gate: we patch on the torch.distributed call site, so the
# fragile dependency is the torch version (signature, name).  Validated
# against torch 2.11.x; warn loudly on anything else so an unexpected
# upgrade surfaces here instead of as a silent no-op.
import torch.distributed as _dist

_TORCH_TESTED_MAJOR_MINOR = (2, 11)
_torch_mm = tuple(int(x) for x in torch.__version__.split(".")[:2])
if _torch_mm != _TORCH_TESTED_MAJOR_MINOR:
    logging.warning(
        "torch version %s is outside the validated %d.%d for the UE8M0 "
        "dist.broadcast compat shim. The shim is still installed (only "
        "triggers on float8_e8m0fnu tensors) but verify multi-rank ckpt "
        "loading of UE8M0 weight scales and bump _TORCH_TESTED_MAJOR_MINOR "
        "in torch_patch.py once re-validated.",
        torch.__version__,
        *_TORCH_TESTED_MAJOR_MINOR,
    )

_UE8M0_DTYPE = torch.float8_e8m0fnu
_orig_broadcast = _dist.broadcast


def _ue8m0_broadcast(tensor, *args, **kwargs):
    if tensor.dtype is _UE8M0_DTYPE:
        return _orig_broadcast(tensor.view(torch.uint8), *args, **kwargs)
    return _orig_broadcast(tensor, *args, **kwargs)


_dist.broadcast = _ue8m0_broadcast
