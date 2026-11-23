"""Checkpoint-owned native UE8M0 FP8 projections for DeepSeek V4/V4.1.

The stored names are the checkpoint's ``weight``/``scale``. Runtime execution
reuses the existing V4 DeepGEMM linear; no legacy weight-name dictionary is
maintained by this module.
"""

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule, copy_weight_
from rtp_llm.models_py.read_spec import (
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)


class Dsv4FP8Parameters(RtpModule):
    """Checkpoint ownership shared by native block-128 V4 backends."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        partition: str = "replicated",
    ):
        super().__init__()
        for name, value in (
            ("in_features", in_features),
            ("out_features", out_features),
            ("tp_size", tp_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(tp_rank, bool)
            or not isinstance(tp_rank, int)
            or not 0 <= tp_rank < tp_size
        ):
            raise ValueError("Invalid V4 FP8 TP rank")
        if partition not in ("replicated", "column", "row"):
            raise ValueError(f"Unknown V4 FP8 partition {partition!r}")
        if in_features % 128 or out_features % 128:
            raise ValueError("Native V4 UE8M0 weight dimensions must align to 128")
        self.in_features, self.out_features = in_features, out_features
        self.tp_size, self.tp_rank, self.partition = tp_size, tp_rank, partition
        self._shard_dim = {"replicated": None, "column": 0, "row": 1}[partition]
        shape = [out_features, in_features]
        if self._shard_dim is not None:
            if shape[self._shard_dim] % (128 * tp_size):
                raise ValueError(
                    "V4 FP8 TP boundary must align to complete 128-element scale blocks"
                )
            shape[self._shard_dim] //= tp_size
        self.weight = nn.Parameter(
            torch.empty(shape, dtype=torch.float8_e4m3fn), requires_grad=False
        )
        self.scale = nn.Parameter(
            torch.empty((shape[0] // 128, shape[1] // 128), dtype=torch.float8_e8m0fnu),
            requires_grad=False,
        )
        # Executor refers to these owned parameters; it is not a second owner
        # in the checkpoint/module tree. Invalidated on migration/reloading.
        object.__setattr__(self, "_executor", None)

    def checkpoint_read_spec(self, name, shape):
        if name not in ("weight", "scale"):
            return None
        divisor = 128 if name == "scale" else 1
        expected = (self.out_features // divisor, self.in_features // divisor)
        if tuple(shape) != expected:
            raise ValueError(
                f"V4 FP8 {name} checkpoint shape must be {expected}, got {tuple(shape)}"
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
                raise ValueError(f"Unknown V4 FP8 checkpoint tensor {name!r}")
            if name in getattr(self, "_rtp_loaded_weight_names", set()):
                raise RuntimeError(f"Duplicate V4 FP8 checkpoint tensor {name!r}")
            tensor, applied = unwrap_checkpoint_tensor(value)
            if applied is None:
                spec = self.checkpoint_read_spec(name, tuple(tensor.shape))
                tensor, _ = unwrap_checkpoint_tensor(spec.materialize(tensor))
            else:
                expected = self.checkpoint_read_spec(name, applied.source_shape)
                if (
                    applied.slices != expected.slices
                    or applied.shard_rank != expected.shard_rank
                ):
                    raise ValueError(
                        f"V4 FP8 {name} ReadSpec belongs to a different shard"
                    )
            copy_weight_(getattr(self, name), tensor, f"Dsv4FP8Linear.{name}")
            self._mark_weight_loaded(name)
        object.__setattr__(self, "_executor", None)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        object.__setattr__(self, "_executor", None)
        return result


class Dsv4FP8Linear(Dsv4FP8Parameters):
    def validate_runtime_device(self, device):
        device = torch.device(device)
        if device.type != "cuda" or getattr(torch.version, "hip", None):
            raise RuntimeError("Native V4 UE8M0 FP8 execution requires CUDA DeepGEMM")
        # The reused executor interprets [N,K] weights with packed UE8M0
        # scales in native E8M0 mode; do not silently use its SM90 reshaping.
        if torch.cuda.get_device_capability(device)[0] not in (10, 12):
            raise RuntimeError(
                "V4 native UE8M0 executor requires DeepGEMM E8M0 device mode"
            )
        try:
            from deep_gemm import fp8_gemm_nt
            from deep_gemm.utils.layout import (
                get_mn_major_tma_aligned_packed_ue8m0_tensor,
            )
        except (ImportError, OSError, AttributeError) as exc:
            raise RuntimeError(
                "V4 FP8 executor or UE8M0 scale packing is unavailable"
            ) from exc
        if not callable(fp8_gemm_nt) or not callable(
            get_mn_major_tma_aligned_packed_ue8m0_tensor
        ):
            raise RuntimeError("V4 FP8 GEMM or UE8M0 layout symbol is not callable")

    def process_weights_after_loading(self):
        if self._executor is not None:
            return
        self.validate_weights_loaded()
        self.validate_runtime_device(self.weight.device)
        from rtp_llm.models_py.modules.dsv4.fp8.attention import _v4_fp8_linear

        executor = _v4_fp8_linear(self.weight, self.scale)
        backend = getattr(executor, "prequantized_input_backend", executor)
        if not all(
            callable(getattr(backend, name, None))
            for name in ("quantize_input", "forward_quantized")
        ):
            raise RuntimeError(
                f"V4 FP8 executor {type(executor).__name__} cannot consume shared input scales"
            )
        object.__setattr__(self, "_executor", executor)

    @property
    def K(self):
        return self.weight.shape[1]

    def dense_fp8_warmup_backend(self):
        # Do not register the executor as a child: its tensor views belong
        # to this owner and are invalidated on migration or reloading.
        return None if self._executor is None else self._shared_input_backend

    @property
    def scale_ue8m0(self):
        return True

    @property
    def supports_fused_prefill_norm_quant(self):
        # The owner can reuse a pair via a dispatcher's DeepGEMM backend.
        # That does not opt the dispatcher into replacing the model's BF16
        # RMSNorm: other attention consumers see that normalized tensor too.
        return callable(getattr(self._executor, "quantize_input", None)) and callable(
            getattr(self._executor, "forward_quantized", None)
        )

    @property
    def _shared_input_backend(self):
        executor = self._executor
        if executor is None:
            raise RuntimeError(
                "V4 FP8 executor must be prepared after checkpoint loading"
            )
        return getattr(executor, "prequantized_input_backend", executor)

    def quantize_input(self, *args, **kwargs):
        return self._shared_input_backend.quantize_input(*args, **kwargs)

    def forward_quantized(self, *args, **kwargs):
        return self._shared_input_backend.forward_quantized(*args, **kwargs)

    @property
    def weight_scales(self):
        if self._executor is None:
            raise RuntimeError(
                "V4 FP8 executor must be prepared after checkpoint loading"
            )
        return self._executor.weight_scales

    def maybe_cache_quant_scale(self, max_len):
        if self._executor is None:
            raise RuntimeError(
                "V4 FP8 executor must be prepared after checkpoint loading"
            )
        return self._executor.maybe_cache_quant_scale(max_len)

    def forward(self, inputs, *args, **kwargs):
        if self._executor is None:
            raise RuntimeError(
                "V4 FP8 executor must be prepared after checkpoint loading"
            )
        return self._executor(inputs, *args, **kwargs)
