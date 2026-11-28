"""Direct checkpoint owners for Qwen4 hyper connections.

Consumers are the existing Qwen4 runtime, not a second implementation of HC.
Raw zero-centred norm parameters deliberately do not fold in one at load time.
"""

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule


class Qwen4TensorWeight(RtpModule):
    def __init__(self, shape, *, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class Qwen4ProjectionWeight(Qwen4TensorWeight):
    """Model-owned per-tensor storage recipe, not a checkpoint-name adapter."""

    def __init__(self, shape, *, dtype=torch.bfloat16, fp8_storage=False):
        super().__init__(shape, dtype=dtype)
        self.fp8_storage = fp8_storage
        self.register_buffer("storage_scale", None)
        self._storage_ready = False

    def process_weights_after_loading(self):
        if self._storage_ready:
            return
        if self.fp8_storage and all(size % 16 == 0 for size in self.weight.shape):
            # Match current-main Fp8StorageWeight, after source projection layout
            # and TP selection, with one scalar amax over this owned tensor.
            values = self.weight.float()
            scale = values.abs().amax().clamp_min(1e-12).reshape(1) / 448.0
            quantized = (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            self.weight.data = quantized
            self.storage_scale = scale
            self.weight._rtp_requires_restart_for_update = True
            self.storage_scale._rtp_requires_restart_for_update = True
        self._storage_ready = True


class Qwen4HyperConnection(RtpModule):
    def __init__(
        self,
        hidden_size,
        hc_mult,
        lowrank,
        *,
        with_inject=True,
        norm_eps=1e-6,
        dtype=torch.bfloat16,
        fp8_storage=False,
        role="PDFUSION",
    ):
        super().__init__()
        if any(type(x) is not int or x <= 0 for x in (hidden_size, hc_mult, lowrank)):
            raise ValueError("Qwen4 HC dimensions must be positive integers")
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.role = role
        width = hidden_size * hc_mult
        self.hc_norm = Qwen4TensorWeight((width,), dtype=dtype)
        self.input_mix_weight_down = Qwen4ProjectionWeight(
            (lowrank, width),
            dtype=dtype,
            fp8_storage=fp8_storage,
        )
        self.input_mix_weight_up = Qwen4ProjectionWeight(
            (width, lowrank),
            dtype=dtype,
            fp8_storage=fp8_storage,
        )
        self.block_inject_weight = (
            Qwen4ProjectionWeight(
                (hc_mult, width), dtype=dtype, fp8_storage=fp8_storage
            )
            if with_inject
            else None
        )
        object.__setattr__(self, "_runtime", None)

    def process_weights_after_loading(self):
        if self._runtime is not None:
            return
        from rtp_llm.models_py.modules.qwen4_exp.gated_residual import (
            Qwen4GatedResidual,
        )

        projections = (
            self.input_mix_weight_down,
            self.input_mix_weight_up,
            self.block_inject_weight,
        )
        for projection in projections:
            if projection is not None:
                projection.process_weights_after_loading()
        down, up, inject = projections
        runtime = Qwen4GatedResidual(
            self.hc_norm.weight,
            down.weight,
            up.weight,
            inject.weight if inject is not None else None,
            hc_mult=self.hc_mult,
            norm_eps=self.norm_eps,
            role=self.role,
            weight_scales=tuple(
                p.storage_scale if p is not None else None for p in projections
            ),
        )
        if runtime.merged_down_inject is not None:
            # The optimized consumer combines these two projections before
            # capture. Rebind owner Parameters to that storage, retaining ids
            # and eliminating duplicate checkpoint allocation.
            merged = runtime.merged_down_inject
            down.weight.data = merged[: down.weight.shape[0]]
            inject.weight.data = merged[
                down.weight.shape[0] : down.weight.shape[0] + self.hc_mult
            ]
        for parameter in self.parameters():
            parameter._rtp_requires_restart_for_update = True
        # Runtime borrows the owned Parameters; it is not another checkpoint
        # subtree and must not introduce duplicate load destinations.
        object.__setattr__(self, "_runtime", runtime)

    @property
    def runtime(self):
        if self._runtime is None:
            raise RuntimeError("Qwen4 HC weights have not been finalized")
        return self._runtime

    def forward(self, hyper_input):
        if self._runtime is None:
            raise RuntimeError("Qwen4 HC weights have not been finalized")
        return self._runtime(hyper_input)

    def forward_with_injection(self, hyper_input, sublayer_out, inject_weights):
        if self._runtime is None:
            raise RuntimeError("Qwen4 HC weights have not been finalized")
        return self._runtime.forward_with_injection(
            hyper_input, sublayer_out, inject_weights
        )
