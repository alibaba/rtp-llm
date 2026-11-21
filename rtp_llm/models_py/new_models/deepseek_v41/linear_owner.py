"""Direct owner for V4.1 checkpoint-native MXFP8 projections."""

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.read_spec import (
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)


class Dsv41MXFP8Linear(RtpModule):
    """Own E4M3 and one E8M0 scale per 32x32 tile; reuse main's CUDA kernel."""

    def __init__(
        self, in_features, out_features, *, tp_size=1, tp_rank=0, partition="replicated"
    ):
        super().__init__()
        if any(
            type(n) is not int or n <= 0 for n in (in_features, out_features, tp_size)
        ):
            raise ValueError(
                "V4.1 MXFP8 dimensions and TP size must be positive integers"
            )
        if type(tp_rank) is not int or not 0 <= tp_rank < tp_size:
            raise ValueError("Invalid V4.1 MXFP8 TP rank")
        if partition not in ("replicated", "column", "row"):
            raise ValueError("Invalid V4.1 MXFP8 partition")
        if in_features % 32 or out_features % 32:
            raise ValueError("V4.1 MXFP8 dimensions must align to 32")
        self.in_features, self.out_features = in_features, out_features
        self.tp_size, self.tp_rank, self.partition = tp_size, tp_rank, partition
        self._shard_dim = {"replicated": None, "column": 0, "row": 1}[partition]
        shape = [out_features, in_features]
        if self._shard_dim is not None:
            if shape[self._shard_dim] % (32 * tp_size):
                raise ValueError("V4.1 MXFP8 TP shard splits a 32-element scale block")
            shape[self._shard_dim] //= tp_size
        self.weight = nn.Parameter(torch.empty(shape, dtype=torch.float8_e4m3fn), False)
        self.scale = nn.Parameter(
            torch.empty(shape[0] // 32, shape[1] // 32, dtype=torch.float8_e8m0fnu),
            False,
        )
        # Derived runtime cache, not a second registered owner of the weight.
        object.__setattr__(self, "_executor", None)

    def checkpoint_read_spec(self, name, shape):
        if name not in ("weight", "scale"):
            return None
        divisor = 32 if name == "scale" else 1
        expected = (self.out_features // divisor, self.in_features // divisor)
        if tuple(shape) != expected:
            raise ValueError(
                f"V4.1 MXFP8 {name} shape must be {expected}, got {tuple(shape)}"
            )
        if self._shard_dim is None:
            return ReadSpec.full(name, expected)
        return make_sharded_read_spec(
            name, expected, self._shard_dim, self.tp_size, self.tp_rank
        )

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, dict) else weights
        for name, value in items:
            if name not in ("weight", "scale"):
                raise ValueError(f"Unknown V4.1 MXFP8 checkpoint tensor {name!r}")
            if name in getattr(self, "_rtp_loaded_weight_names", ()):
                raise RuntimeError(f"Duplicate V4.1 MXFP8 tensor {name!r}")
            tensor, applied = unwrap_checkpoint_tensor(value)
            if applied is None:
                spec = self.checkpoint_read_spec(name, tensor.shape)
                tensor, _ = unwrap_checkpoint_tensor(spec.materialize(tensor))
            else:
                expected = self.checkpoint_read_spec(name, applied.source_shape)
                if (
                    applied.slices != expected.slices
                    or applied.shard_rank != expected.shard_rank
                    or applied.shard_count != expected.shard_count
                    or applied.shard_dim != expected.shard_dim
                    or applied.row_indices is not None
                    or applied.segments is not None
                ):
                    raise ValueError(f"Wrong V4.1 MXFP8 ReadSpec for {name}")
            copy_weight_(getattr(self, name), tensor, f"V4.1 MXFP8 {name}")
            self._mark_weight_loaded(name)
            object.__setattr__(self, "_executor", None)

    def validate_runtime_device(self, device):
        device = torch.device(device)
        if device.type != "cuda" or torch.version.hip:
            raise RuntimeError("V4.1 MXFP8 execution requires CUDA DeepGEMM")
        if torch.cuda.get_device_capability(device)[0] not in (10, 12):
            raise RuntimeError("V4.1 MXFP8 requires a packed UE8M0 CUDA backend")
        try:
            import deep_gemm
            from deep_gemm.utils.layout import (
                get_mn_major_tma_aligned_packed_ue8m0_tensor,
            )

            from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
                sgl_per_token_group_quant_fp8,
            )
        except (ImportError, OSError, AttributeError) as exc:
            raise RuntimeError("V4.1 MXFP8 backend cannot be imported") from exc
        if not all(
            callable(fn)
            for fn in (
                getattr(deep_gemm, "fp8_fp4_gemm_nt", None),
                get_mn_major_tma_aligned_packed_ue8m0_tensor,
                sgl_per_token_group_quant_fp8,
            )
        ):
            raise RuntimeError(
                "V4.1 MXFP8 requires native FP8 GEMM, input quantization and UE8M0 packing"
            )

    def process_weights_after_loading(self):
        if self._executor is not None:
            return
        self.validate_weights_loaded()
        self.validate_runtime_device(self.weight.device)
        # Preserve the original V4.1 group32 rounding, quantized-input ABI
        # and grouped projection dispatch. The owner still owns the two
        # checkpoint Parameters and invalidates the executor when they move.
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        object.__setattr__(self, "_executor", V41MXFP8Linear(self.weight, self.scale))

    @property
    def v41_runtime_linear(self):
        return self._executor

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        object.__setattr__(self, "_executor", None)
        return result

    @property
    def K(self):
        return self.weight.shape[1]

    def _runtime(self):
        if self._executor is None:
            raise RuntimeError(
                "V4.1 MXFP8 executor must be prepared after checkpoint loading"
            )
        return self._executor

    @property
    def weight_scales(self):
        return self._runtime().weight_scales

    def forward(self, *args, **kwargs):
        return self._runtime()(*args, **kwargs)

    def forward_quantized(self, *args, **kwargs):
        return self._runtime().forward_quantized(*args, **kwargs)
