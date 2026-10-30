"""GLM-5 routed FP4 GC MoE with external shared DenseMLP."""

from __future__ import annotations

import importlib.metadata
import logging
import os

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.glm5_mega_moe.input_packer import (
    get_mega_moe_input_packer,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe import (
    GLM5MegaMoE,
    _sync_cuda_graph_warmup_ranks,
)
from rtp_llm.models_py.modules.glm5_mega_moe.quant_layouts import FP4_BLOCK

from .gc_buf import (
    get_or_create_gc_buffer,
    get_or_create_gc_clone_buffer,
    get_or_create_gc_output,
)

logger = logging.getLogger(__name__)
_DEPENDENCIES_LOGGED = False


def transform_gc_routed_fp4(l1, l2, w1_layout: str):
    """Convert RTP's W1 order, then invoke GC's own routed transform."""
    import gc_moe

    if w1_layout == "up_gate":
        w1, s1 = l1
        half = w1.shape[1] // 2
        # DeepGEMM's packed scales are MN-major. A plain cat would turn them
        # into row-major storage, which GC's TMA validation rejects.
        reordered_s1 = torch.empty_strided(
            s1.shape, s1.stride(), dtype=s1.dtype, device=s1.device
        )
        reordered_s1[:, :half].copy_(s1[:, half:])
        reordered_s1[:, half:].copy_(s1[:, :half])
        l1 = (
            torch.cat((w1[:, half:], w1[:, :half]), dim=1),
            reordered_s1,
        )
    elif w1_layout != "gate_up":
        raise ValueError(f"Unsupported FP4 W1 layout: {w1_layout}")
    return gc_moe.transform_weights_for_gc_moe(l1, l2, activation="swiglu")


def gc_warmup_token_counts(capacity: int) -> list[int]:
    """Eagerly prepare RTP decode and MTP draft graph token geometries."""
    raw = os.environ.get("GLM5_GC_MOE_JIT_WARMUP_TOKENS")
    if raw:
        try:
            counts = {int(item) for item in raw.split(",") if item.strip()}
        except ValueError as exc:
            raise ValueError("Invalid GLM5_GC_MOE_JIT_WARMUP_TOKENS") from exc
        if not counts or min(counts) <= 0:
            raise ValueError("GC MoE JIT warmup tokens must be positive")
    else:
        # CudaGraphRunner captures decode buckets 1..8 and 16. With Eagle's
        # four tokens per batch, draft-prefill captures 4,8,..,32 and 64.
        counts = set(range(1, 9)) | {16} | {4 * n for n in (*range(1, 9), 16)}
        counts.add(capacity)
    return sorted(n for n in counts if n <= capacity)


def log_gc_dependencies_once():
    global _DEPENDENCIES_LOGGED
    if _DEPENDENCIES_LOGGED:
        return
    import cuda.bindings
    import cutlass
    import gc_moe

    logger.info(
        "[GLM5 GC MoE] runtime gc_moe=%s %s cutlass=%s %s "
        "cuda_bindings=%s %s torch=%s capability=%s",
        importlib.metadata.version("gc-moe"),
        gc_moe.__file__,
        importlib.metadata.version("nvidia-cutlass-dsl"),
        cutlass.__file__,
        importlib.metadata.version("cuda-bindings"),
        cuda.bindings.__file__,
        torch.__version__,
        torch.cuda.get_device_capability(),
    )
    _DEPENDENCIES_LOGGED = True


def setup_gc_buffer(module, num_shared_experts: int):
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("GLM5 GC MoE requires initialized torch.distributed")
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("GC MoE wheel requires B300 compute capability 10.3")
    log_gc_dependencies_once()
    cfg = module.cfg
    module._mega_group = dist.group.WORLD
    module._mega_buf = get_or_create_gc_buffer(
        module._mega_group, cfg, num_shared_experts
    )
    if module._mega_buf.num_shared_experts != num_shared_experts:
        raise RuntimeError("GC MoE buffer/shared weight configuration mismatch")
    module._mega_y = get_or_create_gc_output(
        module._mega_buf.num_max_tokens_per_rank,
        cfg.dim,
        module._mega_l1_w.device,
    )


def clone_gc_module(source, num_shared_experts: int, packer):
    clone = object.__new__(type(source))
    nn.Module.__init__(clone)
    clone.cfg = source.cfg
    for name in (
        "_mega_l1_w",
        "_mega_l1_sf",
        "_mega_l2_w",
        "_mega_l2_sf",
        "_shared_l1_w",
        "_shared_l1_sf",
        "_shared_l2_w",
        "_shared_l2_sf",
        "_num_shared_experts",
    ):
        if hasattr(source, name):
            setattr(clone, name, getattr(source, name))
    clone._mega_group = source._mega_group
    clone._input_packer = packer
    clone._mega_y = torch.empty_like(source._mega_y)
    return clone


class GLM5GCMoE(GLM5MegaMoE):
    """Reuse stable GLM5 weight/packer contracts with a GC-owned runtime."""

    def _transform_routed_fp4_weights(self, l1, l2, w1_layout: str):
        return transform_gc_routed_fp4(l1, l2, w1_layout)

    def _setup_buffer_and_warmup(self) -> None:
        setup_gc_buffer(self, 0)
        self._input_packer = get_mega_moe_input_packer()
        self._maybe_warmup_jit_once()

    def _maybe_warmup_jit_once(self) -> None:
        if getattr(self, "_gc_jit_warmed", False):
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "GC MoE JIT eager warmup cannot run during graph capture"
            )
        counts = gc_warmup_token_counts(self.cfg.max_tokens_per_rank)
        logger.info(
            "[GLM5 GC MoE] eager warmup: layer=%d shared=0 tokens=%s",
            self.cfg.layer_id,
            counts,
        )
        self._warmup_jit(counts)
        self._gc_jit_warmed = True

    def clone_for_cuda_graph(self, *, share_mega_buf: bool = False) -> "GLM5GCMoE":
        clone = clone_gc_module(self, 0, get_mega_moe_input_packer())
        if share_mega_buf:
            clone._mega_buf = self._mega_buf
        else:
            clone._mega_buf = get_or_create_gc_clone_buffer(
                self._mega_buf, self._mega_group, self.cfg, 0
            )
        clone._gc_jit_warmed = share_mega_buf and self._gc_jit_warmed
        if not clone._gc_jit_warmed:
            clone._maybe_warmup_jit_once()
        return clone

    def _forward_impl(self, x, weights, indices, *, inputs_prepacked: bool):
        import gc_moe

        tokens = int(x.size(0))
        buf = self._mega_buf
        if tokens > buf.num_max_tokens_per_rank or tokens > self._mega_y.size(0):
            raise RuntimeError("GLM5 GC MoE input exceeds actual buffer capacity")
        if not inputs_prepacked:
            if weights is None or indices is None:
                raise ValueError("GC MoE routing weights and ids are required")
            self._input_packer.pack(x, weights, indices, buf, tokens)
        _sync_cuda_graph_warmup_ranks(
            f"glm5.gc_moe.layer{self.cfg.layer_id}.before_gc", x.device
        )
        y = self._mega_y[:tokens]
        gc_moe.fp8_fp4_gc_moe(
            y,
            (self._mega_l1_w, self._mega_l1_sf),
            (self._mega_l2_w, self._mega_l2_sf),
            buf,
            recipe=(1, 1, FP4_BLOCK),
            activation="swiglu",
            activation_clamp=None,
            fast_math=False,
        )
        return y
