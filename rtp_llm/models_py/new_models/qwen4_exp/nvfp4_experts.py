"""Direct Qwen4 routed-expert parameters for the split ModelOpt NVFP4 source."""

import re

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.read_spec import ReadSpec, unwrap_checkpoint_tensor


class Qwen4Nvfp4Experts(RtpModule):
    """Contiguous EP expert ownership for nvfp4_marlin, not a token dispatcher.

    Mount under `mlp.experts` in the target or MTP layer. The surrounding MoE
    supplies global top-k IDs and weights and reduces this EP-local contribution.
    Shared experts/router and other dense parameters are independent BF16 leaves.
    """

    _source_key = re.compile(
        r"(0|[1-9]\d*)\.(up|gate|down)_proj\.(weight|weight_scale|weight_scale_2|input_scale)"
    )
    _projection = ("up", "gate", "down")
    _suffix = ("weight", "weight_scale", "weight_scale_2", "input_scale")

    def __init__(self, hidden_size, intermediate_size, num_experts, ep_size, ep_rank):
        super().__init__()
        k, n = hidden_size, intermediate_size
        if (
            min(k, n, num_experts, ep_size) <= 0
            or k % 128
            or n % 128
            or num_experts % ep_size
            or not 0 <= ep_rank < ep_size
        ):
            raise ValueError("Invalid Qwen4 NVFP4 Marlin dimensions or EP topology")
        e = num_experts // ep_size
        self.ep_size, self.ep_rank = ep_size, ep_rank
        self.num_experts, self.first, self.local = num_experts, ep_rank * e, e
        self.n = n
        fp8 = torch.float8_e4m3fn
        for name, shape, dtype in (
            ("w13", (e, 2 * n, k // 2), torch.uint8),
            ("w2", (e, k, n // 2), torch.uint8),
            ("s13", (e, 2 * n, k // 16), fp8),
            ("s2", (e, k, n // 16), fp8),
            ("outer_scales", (e, 3), torch.float32),
            ("input_scales", (e, 3), torch.float32),
        ):
            self.register_parameter(
                name, nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            )
        self._loaded = set()
        self._marlin = None

    def _parse(self, name):
        match = self._source_key.fullmatch(name)
        if match is None or int(match[1]) >= self.num_experts:
            raise ValueError(f"Unknown Qwen4 NVFP4 expert source: {name!r}")
        return int(match[1]), match[2], match[3]

    def checkpoint_read_spec(self, name, shape):
        expert, _, _ = self._parse(name)
        return ReadSpec(
            name, name, shape, load=self.first <= expert < self.first + self.local
        )

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, dict) else weights
        for name, value in items:
            expert, projection, suffix = self._parse(name)
            i, p = expert - self.first, self._projection.index(projection)
            if not 0 <= i < self.local:
                raise RuntimeError(f"Nonlocal expert was materialized: {name}")
            if self._marlin is not None or name in self._loaded:
                raise ValueError(f"Repeated NVFP4 expert source: {name}")
            value, _ = unwrap_checkpoint_tensor(value)
            if suffix in ("weight_scale_2", "input_scale"):
                storage = (
                    self.outer_scales
                    if suffix == "weight_scale_2"
                    else self.input_scales
                )
                dest = storage[i, p]
                if value.numel() != 1 or value.dtype not in (
                    torch.float32,
                    torch.float16,
                    torch.bfloat16,
                ):
                    raise ValueError(f"Expected floating scalar: {name}")
                value = value.reshape(())
            else:
                is_weight = suffix == "weight"
                if p == 2:
                    dest = (self.w2 if is_weight else self.s2)[i]
                else:
                    dest = (self.w13 if is_weight else self.s13)[
                        i, p * self.n : (p + 1) * self.n
                    ]
                if value.dtype != dest.dtype or value.shape != dest.shape:
                    raise ValueError(
                        f"Invalid packed tensor {name}: {value.shape} {value.dtype}; expected {dest.shape} {dest.dtype}"
                    )
            dest.copy_(value)
            self._loaded.add(name)

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        expected = {
            f"{expert}.{proj}_proj.{suffix}"
            for expert in range(self.first, self.first + self.local)
            for proj in self._projection
            for suffix in self._suffix
        }
        missing = expected - self._loaded
        if missing:
            raise ValueError(
                f"Missing Qwen4 NVFP4 expert tensors: {sorted(missing)[:8]}"
            )
        if not torch.equal(self.outer_scales[:, 0], self.outer_scales[:, 1]):
            raise ValueError("Qwen4 NVFP4 requires equal up/gate outer scales")
        if (
            not (torch.isfinite(self.outer_scales) & (self.outer_scales >= 0))
            .all()
            .item()
        ):
            raise ValueError("Qwen4 NVFP4 requires finite nonnegative outer scales")
        # W4A16 Marlin does not quantize activations. Preserve the checkpoint
        # input scales exactly, without imposing CUTLASS uniformity on them.
        # E4M3FN encodes NaN as 0x7f/0xff; allow both encodings of zero.
        for scales in (self.s13, self.s2):
            encoded = scales.view(torch.uint8)
            if ((encoded == 127) | (encoded > 128)).any().item():
                raise ValueError("Qwen4 NVFP4 requires finite nonnegative block scales")

    def validate_runtime_device(self, device):
        if device.type == "cpu":
            return  # Checkpoint staging only.
        if device.type != "cuda":
            raise RuntimeError(f"Qwen4 Marlin does not support {device}")
        from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4 import (
            Nvfp4MarlinMoE,
            load_library,
        )

        with torch.cuda.device(device):
            if not Nvfp4MarlinMoE.is_supported():
                raise RuntimeError(
                    "Qwen4 Marlin is unavailable on this CUDA/Torch/Triton runtime"
                )
            # Confirm the packaged CUDA library and both entry points before
            # streaming the model. load_library is cached across expert layers.
            load_library()
            for op in ("gptq_marlin_repack", "moe_wna16_marlin_gemm"):
                getattr(torch.ops.rtp_qwen4_marlin, op)

    def process_weights_after_loading(self):
        if not self.w13.is_cuda or self._marlin is not None:
            return
        from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4 import (
            Nvfp4MarlinMoE,
        )

        moe = Nvfp4MarlinMoE(
            self.w13,
            self.s13,
            self.outer_scales[:, 0].contiguous(),
            self.w2,
            self.s2,
            self.outer_scales[:, 2].contiguous(),
            self.ep_size,
            self.ep_rank,
        )
        # Retain only one set of large matrices. Marlin owns the repacked
        # tensors and these Parameters expose the same storage to the model.
        for name, tensor in (
            ("w13", moe.w1),
            ("w2", moe.w2),
            ("s13", moe.s1),
            ("s2", moe.s2),
        ):
            setattr(self, name, nn.Parameter(tensor, requires_grad=False))
        self._marlin = moe

    def forward(self, x, global_ids, weights):
        if self._marlin is None:
            raise RuntimeError("Qwen4 NVFP4 experts require initialized Marlin on CUDA")
        if x.shape[0] == 0:
            return torch.empty_like(x)
        return self._marlin(x, global_ids.to(torch.int32), weights.to(torch.float32))
