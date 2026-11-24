"""Module-owned loading and runtime layout for ModelOpt NVFP4 dense weights.

The existing CUDA FP4 executor owns GEMM. This method owns packed checkpoint
parameters, TP/GQA slicing, projection fusion and scale layout preparation.
It deliberately does not use OldLoader descriptors or DeviceImpl.
"""

import importlib
import math
import os
from dataclasses import dataclass

import torch
from torch import nn

from rtp_llm.models_py.fp4_read_spec import modelopt_fp4_read_spec
from rtp_llm.models_py.module_base import copy_weight_
from rtp_llm.models_py.quant_methods.base import (
    QuantizeMethodBase,
    register_quant_method,
)
from rtp_llm.models_py.read_spec import (
    ReadSpec,
    make_sharded_read_spec,
    unwrap_checkpoint_tensor,
)
from rtp_llm.models_py.utils.quant_layout import swizzle_nvfp4_scales


@dataclass(frozen=True)
class _Projection:
    name: str
    global_rows: int
    local_rows: int
    shards: int
    rank: int
    offset: int


@register_quant_method("modelopt_fp4")
class ModelOptFp4LinearMethod(QuantizeMethodBase):
    owns_checkpoint_loading = True
    required_checkpoint_parameters = (
        "weight",
        "weight_scale",
        "weight_scale_2",
        "input_scale",
    )

    def __init__(self, quant_config=None):
        super().__init__(quant_config)
        self._pieces = {}
        self._scalar_pieces = {}
        self._executor = None
        self._backend = None

    def create_weights(self, layer, input_size, output_size, params_dtype):
        if input_size % 16:
            raise ValueError(
                "NVFP4 local input width must align to 16-value scale blocks"
            )
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("NVFP4 compute dtype must be float16 or bfloat16")
        if (
            os.getenv("RTP_LLM_FP4_GEMM_BACKEND", "cutlass") == "trtllm"
            and params_dtype != torch.bfloat16
        ):
            raise ValueError("NVFP4 TRTLLM backend requires bfloat16 compute dtype")
        for name, shape, dtype in (
            ("weight", (output_size, input_size // 2), torch.uint8),
            ("weight_scale", (output_size, input_size // 16), torch.float8_e4m3fn),
            ("weight_scale_2", (), torch.float32),
            ("input_scale", (), torch.float32),
        ):
            layer.register_parameter(
                name, nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            )

    @staticmethod
    def _layout(layer):
        if hasattr(layer, "num_kv_heads"):
            kv_shards = math.gcd(layer.num_kv_heads, layer.tp_size)
            projections = (
                _Projection(
                    "q_proj",
                    layer.num_heads * layer.head_dim,
                    layer.q_size,
                    layer.tp_size,
                    layer.tp_rank,
                    0,
                ),
                _Projection(
                    "k_proj",
                    layer.num_kv_heads * layer.head_dim,
                    layer.kv_size,
                    kv_shards,
                    layer.kv_head_rank,
                    layer.q_size,
                ),
                _Projection(
                    "v_proj",
                    layer.num_kv_heads * layer.head_dim,
                    layer.kv_size,
                    kv_shards,
                    layer.kv_head_rank,
                    layer.q_size + layer.kv_size,
                ),
            )
            return projections, layer.input_size, 0
        if layer.shard_names:
            local = layer.output_size_per_partition // len(layer.shard_names)
            return (
                tuple(
                    _Projection(
                        name,
                        local * layer.tp_size,
                        local,
                        layer.tp_size,
                        layer.tp_rank,
                        index * local,
                    )
                    for index, name in enumerate(layer.shard_names)
                ),
                layer.input_size,
                0,
            )
        if hasattr(layer, "input_size_per_partition"):
            return (
                (
                    _Projection(
                        "",
                        layer.output_size,
                        layer.output_size,
                        layer.tp_size,
                        layer.tp_rank,
                        0,
                    ),
                ),
                layer.input_size * layer.tp_size,
                1,
            )
        return (
            (
                _Projection(
                    "",
                    layer.output_size * layer.tp_size,
                    layer.output_size,
                    layer.tp_size,
                    layer.tp_rank,
                    0,
                ),
            ),
            layer.input_size,
            0,
        )

    def _source(self, layer, name, shape):
        projections, global_input, dim = self._layout(layer)
        parts = name.rsplit(".", 1)
        parameter = parts[-1]
        projection_name = parts[0] if len(parts) == 2 else ""
        if parameter not in self.required_checkpoint_parameters and parameter != "bias":
            raise ValueError(f"Unknown NVFP4 checkpoint parameter {name!r}")
        if parameter == "bias" and layer.bias is None:
            raise ValueError("NVFP4 checkpoint contains bias for a bias-free module")
        if projection_name:
            selected = tuple(p for p in projections if p.name == projection_name)
            if not selected:
                raise ValueError(f"Unknown NVFP4 projection {projection_name!r}")
        else:
            selected = projections
        rows = sum(p.global_rows for p in selected)
        shape = tuple(shape)
        # Scalar scales may be saved as [] or [1]. Never accept a projection
        # vector as one scale, or confuse it with per-block weight scales.
        if parameter in ("weight_scale_2", "input_scale"):
            if math.prod(shape) != 1:
                raise ValueError(f"NVFP4 {name} must contain one scalar")
        else:
            expected = {
                "weight": (rows, global_input // 2),
                "weight_scale": (rows, global_input // 16),
                "bias": (rows,),
            }[parameter]
            if shape != expected:
                raise ValueError(
                    f"NVFP4 {name} checkpoint shape must be {expected}, got {shape}"
                )
        return parameter, selected, dim

    @staticmethod
    def _is_kv_cache_scale(layer, name, shape):
        # ModelOpt can export FP8 KV-cache calibration alongside NVFP4 GEMM
        # tensors. The runtime manages KV-cache scales; these are not QKV
        # projection input_scale or weight_scale_2, and do not complete either.
        if not hasattr(layer, "num_kv_heads") or name not in (
            "k_proj.k_scale",
            "v_proj.v_scale",
        ):
            return False
        if tuple(shape) not in ((), (1,)):
            raise ValueError(f"ModelOpt KV-cache scale {name!r} must be scalar")
        return True

    def checkpoint_read_spec(self, layer, name, shape):
        if self._is_kv_cache_scale(layer, name, shape):
            return ReadSpec.skip(name, tuple(shape))
        parameter, projections, dim = self._source(layer, name, shape)
        if len(projections) == 1 and parameter != "bias":
            projection = projections[0]
            return modelopt_fp4_read_spec(
                name,
                tuple(shape),
                input_size=self._layout(layer)[1],
                output_size=projection.global_rows,
                tp_size=projection.shards,
                tp_rank=projection.rank,
                partition_dim=dim,
            )
        if parameter in ("weight_scale_2", "input_scale") or (
            parameter == "bias" and dim == 1
        ):
            return ReadSpec.full(name, tuple(shape))
        if len(projections) > 1:
            segments = []
            offset = 0
            for projection in projections:
                start = offset + projection.rank * projection.local_rows
                segments.append((start, start + projection.local_rows))
                offset += projection.global_rows
            return ReadSpec(
                name,
                name,
                tuple(shape),
                segments=tuple(segments),
                partition="nvfp4_fused",
            )
        projection = projections[0]
        if projection.shards == 1:
            return ReadSpec.full(name, tuple(shape))
        return make_sharded_read_spec(
            name, tuple(shape), dim, projection.shards, projection.rank
        )

    def load_weights(self, layer, weights):
        for name, value in weights.items():
            tensor, spec = unwrap_checkpoint_tensor(value)
            source_shape = (
                spec.source_shape if spec is not None else tuple(tensor.shape)
            )
            if self._is_kv_cache_scale(layer, name, source_shape):
                # Direct/legacy checkpoint iterators need not provide a ReadSpec.
                continue
            parameter, projections, dim = self._source(layer, name, source_shape)
            target = getattr(layer, parameter)
            keys = {projection.name for projection in projections}
            seen = self._pieces.get(parameter, set())
            if parameter in layer._loaded_parameter_names or keys & seen:
                raise RuntimeError(
                    f"Duplicate or mixed fused/separate NVFP4 tensor {name}"
                )
            if spec is not None:
                expected_spec = self.checkpoint_read_spec(layer, name, source_shape)
                if (
                    spec.slices != expected_spec.slices
                    or spec.segments != expected_spec.segments
                    or spec.row_indices != expected_spec.row_indices
                ):
                    raise ValueError(
                        f"NVFP4 {name} ReadSpec does not match this TP rank"
                    )
            if parameter in ("weight_scale_2", "input_scale"):
                if not tensor.is_floating_point() or tensor.numel() != 1:
                    raise ValueError(f"NVFP4 {name} must be a floating-point scalar")
                scalar = tensor.detach().float().reshape(())
                if not bool(torch.isfinite(scalar)) or not bool(scalar > 0):
                    raise ValueError(f"NVFP4 {name} must be finite and positive")
                values = self._scalar_pieces.setdefault(parameter, {})
                for projection in projections:
                    values[projection.name] = scalar.to(device="cpu", copy=True)
                # Match existing ModelOpt QKV and gate/up max_scalar fusion.
                # Projection values are retained until all pieces have arrived.
                target.data.copy_(torch.stack(tuple(values.values())).max())
            else:
                global_offset = 0
                staged = []
                for projection in projections:
                    if spec is not None:
                        piece = (
                            tensor.narrow(0, projection.offset, projection.local_rows)
                            if spec.segments is not None
                            else tensor
                        )
                    else:
                        piece = tensor.narrow(0, global_offset, projection.global_rows)
                        if not (parameter == "bias" and dim == 1):
                            width = piece.shape[dim] // projection.shards
                            piece = piece.narrow(dim, projection.rank * width, width)
                    destination = target.narrow(
                        0, projection.offset, projection.local_rows
                    )
                    if tuple(piece.shape) != tuple(destination.shape):
                        raise ValueError(
                            f"NVFP4 {name} local shape mismatch: {tuple(piece.shape)} vs {tuple(destination.shape)}"
                        )
                    if (
                        parameter in ("weight", "weight_scale")
                        and piece.dtype != destination.dtype
                    ):
                        raise TypeError(
                            f"NVFP4 {name} dtype must be {destination.dtype}, got {piece.dtype}"
                        )
                    staged.append((destination, piece))
                    global_offset += projection.global_rows
                for destination, piece in staged:
                    copy_weight_(destination, piece, name)
            self._pieces.setdefault(parameter, set()).update(keys)
            expected = {p.name for p in self._layout(layer)[0]}
            if self._pieces[parameter] == expected:
                layer._record_parameter_loaded(parameter)

    def validate_runtime_device(self, device):
        device = torch.device(device)
        if (
            device.type != "cuda"
            or torch.version.hip is not None
            or not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "ModelOpt NVFP4 requires a supported NVIDIA CUDA accelerator"
            )
        if torch.cuda.get_device_capability(device)[0] < 10:
            raise RuntimeError("ModelOpt NVFP4 execution requires SM100 or newer")
        backend = os.getenv("RTP_LLM_FP4_GEMM_BACKEND")
        if backend not in (None, "cute-dsl", "cutlass", "trtllm", "sgl_cutlass"):
            raise ValueError(f"Unknown NVFP4 GEMM backend {backend!r}")
        try:
            flashinfer = importlib.import_module("flashinfer")
            required = ["fp4_quantize", "mm_fp4"]
            if backend == "trtllm":
                required.extend(("shuffle_matrix_a", "shuffle_matrix_sf_a"))
            for name in required:
                if not callable(getattr(flashinfer, name, None)):
                    raise RuntimeError(f"flashinfer.{name} is unavailable")
            executor_module = importlib.import_module(
                "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear"
            )
            quantize_kwargs, backend = executor_module.fp4_backend_options()
            if backend == "cute-dsl" and not quantize_kwargs:
                raise RuntimeError(
                    "installed FlashInfer does not support the cute-dsl backend"
                )
            self._executor_type = executor_module.CudaFp4GEMMLinear
            if not callable(self._executor_type):
                raise RuntimeError("CudaFp4GEMMLinear is unavailable")
            if backend == "sgl_cutlass":
                if not callable(
                    getattr(executor_module, "cutlass_scaled_fp4_mm_wrapper", None)
                ):
                    raise RuntimeError("cutlass_scaled_fp4_mm_wrapper is unavailable")
                # Check the backend actually captured by the serving wrapper.
                # Importing the wrapper alone also succeeds when its wheel is missing.
                fp4_kernel = importlib.import_module(
                    "rtp_llm.models_py.kernels.cuda.fp4_kernel.fp4_kernel"
                )
                fp4_kernel._require_nvfp4()
                if not callable(getattr(fp4_kernel, "cutlass_scaled_fp4_mm", None)):
                    raise RuntimeError("rtp_kernel.nvfp4.cutlass_scaled_fp4_mm is unavailable")
        except (ImportError, AttributeError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"NVFP4 backend {backend!r} failed loading-time validation: {exc}"
            ) from exc
        self._backend = backend

    def process_weights_after_loading(self, layer):
        self.validate_runtime_device(layer.weight.device)
        layer.validate_weights_loaded()
        if self._backend == "trtllm":
            from flashinfer import shuffle_matrix_a, shuffle_matrix_sf_a

            weight = shuffle_matrix_a(layer.weight.view(torch.uint8), 128)
            scale = (
                shuffle_matrix_sf_a(layer.weight_scale.view(torch.uint8), 128)
                .reshape(layer.weight_scale.shape)
                .view(torch.float8_e4m3fn)
            )
        else:
            weight, scale = layer.weight, swizzle_nvfp4_scales(layer.weight_scale)
        layer.weight.data = weight.detach()
        layer.weight_scale.data = scale.detach()
        self._executor = self._executor_type(
            weight=layer.weight,
            weight_scales=layer.weight_scale,
            input_scales=layer.input_scale,
            bias=None,
            quant_config=self.quant_config.source_config,
            weight_scale_2=layer.weight_scale_2,
        )

    def apply(self, layer, x, bias=None):
        if self._executor is None:
            raise RuntimeError("NVFP4 executor has not been initialized")
        if self._backend == "trtllm" and x.dtype != torch.bfloat16:
            raise ValueError("NVFP4 TRTLLM backend requires bfloat16 input")
        output = self._executor.forward(x.reshape(-1, x.shape[-1]).contiguous())
        output = output.reshape(*x.shape[:-1], layer.output_size)
        return output if bias is None else output + bias
