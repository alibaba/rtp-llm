"""PPU attention-post / FFN-pre / RMSNorm transition.

The normalized tensor has its own result type: PPUStepMix.hidden always
means an unnormalized readout. Mega-mHC is enabled by default in ordinary
Decode Graph.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from rtp_llm.config.v41_runtime import env_bool

from .mhc import MHCConfig
from .optimization_config import CROSS_LAYER_MHC, VERIFY_MEGA_MHC, enabled

MHC_BACKEND_ENV = "RTP_V41_PPU_MHC_BACKEND"
GRAPH_MEGA_MHC_ENV = "RTP_V41_PPU_GRAPH_MEGA_MHC"
QUALIFIED_EXTENSION_SHA256 = (
    "816ff22272d7636c8ee57b960189ada869910af74016f35d53638248fcdc3942"
)
BUILD_RECEIPT_ENV = "RTP_V41_PPU_DEEPGEMM_BUILD_RECEIPT"
REVIEWED_NATIVE_SOURCE = "4effb76f083b739f549b69d6b6210e6f4daf3ad7"


@dataclass(frozen=True)
class PPUMegaMHCTransition:
    residual: torch.Tensor
    normalized: torch.Tensor
    pre_mix: torch.Tensor
    post_mix: torch.Tensor
    residual_mix: torch.Tensor


@lru_cache(maxsize=1)
def _extension_identity():
    # Identity is checked once before any native launch, not inferred from a
    # package version. A selected but incompatible installation fails loudly.
    module = importlib.import_module("deep_gemm")
    native = getattr(module, "_C", None)
    extension = native or importlib.import_module("deep_gemm.deep_gemm_cpp")
    with Path(extension.__file__).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if native is not None:
        # Explicit benchmark admission binds the reviewed ABI to the private
        # built package. This is provenance, not numerical qualification.
        receipt_path = os.environ.get(BUILD_RECEIPT_ENV)
        if not receipt_path:
            raise RuntimeError(f"native Mega-mHC requires {BUILD_RECEIPT_ENV}")
        receipt = json.loads(Path(receipt_path).read_text())
        root = Path(module.__file__).resolve().parent.parent
        reviewed_copy_block_trial = (
            os.environ.get("RTP_V41_PPU_MEGAMOE_E448_COPY_BLOCKS") == "4"
            and receipt.get("source_pin") == "3a635699e5d5cd53b62d21f6e4d39e3bf01a61be"
            and digest == "4bc8f9b35419b5315b00cc80ba91c16573ff33470a811cb6da115c9f93e3599e"
        )
        if (
            (receipt.get("source_pin") != REVIEWED_NATIVE_SOURCE and not reviewed_copy_block_trial)
            or Path(receipt["installed"]).resolve() != root
            or receipt.get("evidence") != "BUILT_WHEEL_FILE_INSPECTION_ONLY_NO_KERNEL_EXECUTION"
        ):
            raise RuntimeError("native Mega-mHC build receipt identity differs")
        for name, expected in receipt["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid DeepGEMM receipt path")
            with (root / relative).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError("native Mega-mHC package hash differs")
        key = "deep_gemm/" + Path(extension.__file__).name
        if receipt["files"].get(key) != digest:
            raise RuntimeError("native Mega-mHC extension is absent from receipt")
    elif digest != QUALIFIED_EXTENSION_SHA256:
        raise RuntimeError("Mega-mHC requires the qualified PPU extension")
    return digest


@lru_cache(maxsize=1)
def _qualified_operation():
    _extension_identity()
    module = importlib.import_module("deep_gemm")
    operation = getattr(module, "mega_mhc", None)
    if not callable(operation):
        raise RuntimeError("qualified deep_gemm.mega_mhc is unavailable")
    return operation


def _dense(value, shape, dtype, device):
    return (
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == shape
        and value.dtype == dtype
        and value.device == device
        and value.is_contiguous()
        and value.data_ptr() % 16 == 0
        and not value.requires_grad
    )


class PPUMegaMHC:
    def __init__(self):
        self.backend = os.environ.get(MHC_BACKEND_ENV, "mega").strip().lower()
        if self.backend not in ("legacy", "mega"):
            raise ValueError(f"{MHC_BACKEND_ENV} must be legacy or mega")
        self.graph_enabled = env_bool(os.environ, GRAPH_MEGA_MHC_ENV, default=True)
        self.graph_capture_calls = 0
        self.calls = Counter()
        self.fallbacks = Counter()
        self._new_graph_warmed = set()

    def execution_stats(self):
        return {
            "backend": self.backend,
            "calls": dict(self.calls),
            "fallbacks": dict(self.fallbacks),
            "graph_capture_calls": self.graph_capture_calls,
            "extension_sha256": _extension_identity() if self.calls else None,
        }

    def _reason(self, branch, residual, state, parameters, norm, eps, config, phase):
        if self.backend != "mega":
            return "legacy_selected"
        cross = phase.endswith("_cross")
        base_phase = phase.removesuffix("_cross")
        verify = base_phase in ("verify", "graph_verify")
        if cross and not enabled(CROSS_LAYER_MHC):
            return "cross_layer_disabled"
        if verify and not enabled(VERIFY_MEGA_MHC):
            return "verify_disabled"
        graph = base_phase in ("graph", "graph_verify") and self.graph_enabled
        if not graph and base_phase not in ("prefill", "decode", "verify"):
            return "unqualified_phase"
        if cross and base_phase not in ("decode", "graph", "verify", "graph_verify"):
            return "unqualified_cross_phase"
        if (
            config.hc_mult != 4
            or config.sinkhorn_iters != 20
            or config.hc_eps != 1e-6
            or config.post_mult != 2.0
            or config.rms_eps not in (1e-20, 1e-6)
            or eps not in (1e-20, 1e-6)
            or config.strict_bf16_fn
            or config.use_triton is False
        ):
            return "unqualified_config"
        from rtp_llm.device.device_type import is_ppu

        if residual.device.type != "cuda" or not is_ppu():
            return "not_ppu"
        if not graph:
            with torch.cuda.device(residual.device):
                if torch.cuda.is_current_stream_capturing():
                    return "graph_capture"
                if torch.cuda.current_stream() != torch.cuda.default_stream():
                    return "unqualified_stream"
        if residual.ndim != 3 or tuple(residual.shape[1:]) != (4, 5120):
            return "layout"
        rows = residual.shape[0]
        if not 0 < rows <= 16384:
            return "rows"
        device = residual.device
        contracts = (
            (branch, (rows, 5120), torch.bfloat16),
            (residual, (rows, 4, 5120), torch.bfloat16),
            (getattr(state, "pre_mix", None), (rows, 4), torch.float32),
            (getattr(state, "post_mix", None), (rows, 4, 1), torch.float32),
            (getattr(state, "residual_mix", None), (rows, 4, 4), torch.float32),
            (parameters.get("fn"), (24, 20480), torch.float32),
            (parameters.get("base"), (24,), torch.float32),
            (parameters.get("scale"), (3,), torch.float32),
            (norm, (5120,), torch.bfloat16),
        )
        if any(
            not _dense(value, shape, dtype, device) for value, shape, dtype in contracts
        ):
            return "tensor_contract"
        return None

    def try_transition(
        self,
        branch,
        residual,
        state,
        parameters,
        norm,
        eps,
        config: MHCConfig,
        *,
        phase,
    ):
        reason = self._reason(
            branch, residual, state, parameters, norm, eps, config, phase
        )
        if reason is not None:
            self.fallbacks[f"{phase}:{reason}"] += 1
            return None
        rows = residual.shape[0]
        graph = phase.removesuffix("_cross") in ("graph", "graph_verify")
        new_graph = graph and phase != "graph"
        with torch.cuda.device(residual.device):
            capturing = graph and torch.cuda.is_current_stream_capturing()
            warm_key = (
                residual.device, rows, phase, torch.cuda.current_stream().cuda_stream,
                config.rms_eps, eps,
            )
        if new_graph and capturing and warm_key not in self._new_graph_warmed:
            self.fallbacks[f"{phase}:cold_capture_stream"] += 1
            return None
        operation = _qualified_operation()
        updated = torch.empty_like(residual)
        pre = torch.empty((rows, 4, 1), device=residual.device, dtype=torch.float32)
        post = torch.empty_like(pre)
        comb = torch.empty((rows, 4, 4), device=residual.device, dtype=torch.float32)
        normalized = torch.empty_like(branch)
        # Each output has independent fresh storage. The carried attention pre
        # reads the new BF16 residual; new FFN pre is only for the next readout.
        # Once native execution begins, all errors propagate; never retry.
        with torch.cuda.device(residual.device):
            # The runner warms this bucket on the capture stream. The native
            # operator allocates capture-private workspace from the graph pool.
            operation(
                x=branch,
                residual=residual,
                shifted_prev_mix=state.pre_mix.view(rows, 4, 1),
                post_mix=state.post_mix,
                comb_res_mix=state.residual_mix,
                fn=parameters["fn"],
                mix_scales=parameters["scale"],
                mix_bases=parameters["base"],
                hc_mult=4,
                hc_norm_eps=config.rms_eps,
                hc_pre_eps=config.hc_eps,
                hc_post_scale=config.post_mult,
                sinkhorn_eps=config.hc_eps,
                num_sinkhorn_iters=config.sinkhorn_iters,
                rmsnorm_weight=norm,
                rmsnorm_eps=eps,
                rmsnorm_scale=1.0,
                new_residual=updated,
                new_prev_mix=pre,
                new_post_mix=post,
                new_comb_res_mix=comb,
                y_bf16=normalized,
            )
        self.calls[phase] += 1
        self.graph_capture_calls += int(capturing)
        if new_graph and not capturing:
            self._new_graph_warmed.add(warm_key)
        return PPUMegaMHCTransition(updated, normalized, pre.view(rows, 4), post, comb)
