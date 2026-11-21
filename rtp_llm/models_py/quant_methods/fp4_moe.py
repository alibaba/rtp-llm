"""ModelOpt NVFP4 expert parameters, checkpoint dispatch and runtime layouts."""

import importlib
import math

import torch

from rtp_llm.models_py.quant_methods.base import (
    FusedMoEMethodBase,
    register_moe_quant_method,
)
from rtp_llm.models_py.quant_methods.nvfp4_quantization import quantize_expert_to_nvfp4
from rtp_llm.models_py.utils.quant_layout import swizzle_nvfp4_scales
from rtp_llm.utils.model_weight import W


@register_moe_quant_method("modelopt_fp4", "modelopt_fp4_online")
class ModelOptFp4MoEMethod(FusedMoEMethodBase):
    requires_staged_device_postprocess = True
    owns_expert_dispatch = True

    def __init__(self, quant_config=None):
        super().__init__(quant_config)
        self.online = quant_config.quant_type == "modelopt_fp4_online"
        self.uses_packed_weights = not self.online
        self._backend = None
        source = quant_config.source_config
        self.source_fp8_block_size = getattr(source, "source_fp8_block_size", None)
        if self.source_fp8_block_size not in (None, 128):
            raise ValueError("Online NVFP4 supports only block128 FP8 source weights")
        if self.source_fp8_block_size is not None and not self.online:
            raise ValueError("FP8 source decoding requires online ModelOpt FP4")
        activation_amax = getattr(source, "activation_amax", None)
        self.input_scale = (
            1.0 if activation_amax is None else float(activation_amax) / (6 * 448)
        )
        if not math.isfinite(self.input_scale) or self.input_scale <= 0:
            raise ValueError("Online NVFP4 activation amax must be finite and positive")
        self._cache = None
        if self.online and getattr(source, "online_cache_dir", None):
            from rtp_llm.models_py.quant_methods.nvfp4_cache import Nvfp4ExpertCache

            self._cache = Nvfp4ExpertCache(source.online_cache_dir)

    def create_weights(
        self, layer, num_experts, hidden_size, intermediate_size, params_dtype, **kwargs
    ):
        if hidden_size % 16 or intermediate_size % 16:
            raise ValueError(
                "NVFP4 hidden and TP-local intermediate sizes must be divisible by 16"
            )
        source = self.quant_config.source_config
        group_size = getattr(source, "group_size", lambda: 16)()
        if group_size != 16 or getattr(source, "bits", 4) != 4:
            raise ValueError("ModelOpt NVFP4 requires bits=4 and group_size=16")
        self._backend = _resolve_backend(layer._moe_config)
        if self._backend not in ("cutedsl", "trtllm", "cutlass", "marlin"):
            raise ValueError(f"Unsupported NVFP4 MoE backend: {self._backend!r}")
        if self._backend in ("cutlass", "marlin"):
            if layer._dynamic_eplb_enabled or layer._expert_placement.has_redundancy:
                raise ValueError(
                    f"NVFP4 {self._backend} does not support dynamic EPLB or redundant experts in main"
                )
            if self._backend == "cutlass" and not self.online:
                raise ValueError("CUTLASS NVFP4 requires online source weights in main")
            if self._backend == "marlin" and getattr(
                layer._model_config, "model_type", None
            ) not in ("qwen4_exp", "qwen4_exp_mtp"):
                raise ValueError(
                    "Marlin NVFP4 is currently admitted only for Qwen4 models"
                )
        e, h, m = num_experts, hidden_size, intermediate_size
        if getattr(source, "require_contiguous_experts", False) or self._backend in (
            "cutlass",
            "marlin",
        ):
            count = layer.num_experts // layer.ep_size
            expected = tuple(range(count * layer.ep_rank, count * (layer.ep_rank + 1)))
            actual = layer._expert_placement.physical_to_logical[
                layer._ep_start_expert : layer._ep_start_expert + num_experts
            ]
            if tuple(actual) != expected:
                raise ValueError(
                    "This NVFP4 recipe requires contiguous logical expert ownership"
                )
        divisor = 2
        dtype = torch.uint8
        layer.register_parameter(
            "w13",
            torch.nn.Parameter(
                torch.empty(e, 2 * m, h // divisor, dtype=dtype), requires_grad=False
            ),
        )
        layer.register_parameter(
            "w2",
            torch.nn.Parameter(
                torch.empty(e, h, m // divisor, dtype=dtype), requires_grad=False
            ),
        )
        layer.register_buffer(
            "w13_scale", torch.empty(e, 2 * m, h // 16, dtype=torch.float8_e4m3fn)
        )
        layer.register_buffer(
            "w2_scale", torch.empty(e, h, m // 16, dtype=torch.float8_e4m3fn)
        )
        for name in ("w13_scale_2", "w2_scale_2", "w13_input_scale", "w2_input_scale"):
            layer.register_buffer(name, torch.zeros(e, dtype=torch.float32))
        layer._nvfp4_pending_float = {}
        layer._nvfp4_pending_source = {}
        layer._nvfp4_pending_source_scale = {}

    def required_aux_parameters(self):
        if self.online:
            return ("weight_scale_inv",) if self.source_fp8_block_size else ()
        return ("weight_scale", "weight_scale_2", "input_scale")

    def _slice(self, layer, projection, tensor, divisor):
        if tensor.ndim != 2:
            raise ValueError(
                f"NVFP4 {projection} requires a matrix, got {tuple(tensor.shape)}"
            )
        m, h = layer.moe_inter, layer.hidden_size
        expected = (h, m // divisor) if projection == "down_proj" else (m, h // divisor)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"NVFP4 {projection} expected {expected}, got {tuple(tensor.shape)}"
            )
        axis = 1 if projection == "down_proj" else 0
        size = layer.moe_inter_tp // divisor if axis == 1 else layer.moe_inter_tp
        return tensor.narrow(axis, layer.moe_expert_tp_rank * size, size)

    @staticmethod
    def _target(layer, local_id, projection, scale=False):
        name = "w2" if projection == "down_proj" else "w13"
        if scale:
            name += "_scale"
        tensor = getattr(layer, name).data[local_id]
        if projection != "down_proj":
            start = layer.moe_inter_tp if projection == "gate_proj" else 0
            tensor = tensor.narrow(0, start, layer.moe_inter_tp)
        return tensor

    def dispatch_weight(self, layer, local_id, projection, parameter_name, tensor):
        if parameter_name != "weight":
            return False
        if projection == "gate_up_proj":
            if tensor.ndim != 2 or tensor.shape[0] != 2 * layer.moe_inter:
                raise ValueError(
                    "NVFP4 fused gate_up weight requires [2*M, K/2] (or online [2*M,K])"
                )
            gate, up = tensor.chunk(2, dim=0)
            self.dispatch_weight(layer, local_id, "gate_proj", parameter_name, gate)
            self.dispatch_weight(layer, local_id, "up_proj", parameter_name, up)
            return True
        if self.source_fp8_block_size:
            if tensor.dtype != torch.float8_e4m3fn:
                raise TypeError("FP8 source weight must use float8_e4m3fn")
            layer._nvfp4_pending_source[(local_id, projection)] = tensor
            self._decode_source(layer, local_id, projection)
            return True
        expected = (
            (torch.float16, torch.bfloat16, torch.float32)
            if self.online
            else (torch.uint8,)
        )
        if tensor.dtype not in expected:
            raise TypeError(
                f"NVFP4 checkpoint weight dtype must be {expected}, got {tensor.dtype}"
            )
        if self.online:
            tensor = tensor.to(layer._params_dtype)
        sliced = self._slice(layer, projection, tensor, 1 if self.online else 2)
        if self.online:
            if not torch.isfinite(tensor).all():
                raise ValueError("Online NVFP4 checkpoint weights must be finite")
            self._quantize_projection(
                layer, local_id, projection, sliced, tensor.float().abs().amax()
            )
        else:
            self._target(layer, local_id, projection).copy_(sliced)
        return True

    def _decode_source(self, layer, local_id, projection):
        key = (local_id, projection)
        if (
            key not in layer._nvfp4_pending_source
            or key not in layer._nvfp4_pending_source_scale
        ):
            return
        weight = layer._nvfp4_pending_source.pop(key)
        scale = layer._nvfp4_pending_source_scale.pop(key)
        block = self.source_fp8_block_size
        expected = tuple((size + block - 1) // block for size in weight.shape)
        if weight.ndim != 2 or tuple(scale.shape) != expected:
            raise ValueError("FP8 source block scale geometry disagrees")
        if not torch.isfinite(scale).all() or (scale < 0).any():
            raise ValueError("FP8 source block scales must be finite and nonnegative")
        expanded = scale.float().repeat_interleave(block, 0).repeat_interleave(block, 1)
        decoded = (weight.float() * expanded[: weight.shape[0], : weight.shape[1]]).to(
            layer._params_dtype
        )
        if not torch.isfinite(decoded).all():
            raise ValueError("FP8 source decoded weight must be finite")
        sliced = self._slice(layer, projection, decoded, 1)
        self._quantize_projection(
            layer, local_id, projection, sliced, decoded.float().abs().amax()
        )

    def _quantize_projection(self, layer, local_id, projection, local_weight, maximum):
        stem = "w2" if projection == "down_proj" else "w13"
        if stem == "w13":
            pending = layer._nvfp4_pending_float
            pending[(local_id, projection)] = (local_weight.clone(), maximum)
            if any((local_id, p) not in pending for p in ("up_proj", "gate_proj")):
                return
            up, up_max = pending.pop((local_id, "up_proj"))
            gate, gate_max = pending.pop((local_id, "gate_proj"))
            local_weight = torch.cat((up, gate))
            maximum = torch.maximum(up_max, gate_max)
        if self._cache is None:
            packed, scale, outer = quantize_expert_to_nvfp4(local_weight, amax=maximum)
        else:
            logical_id = layer._expert_placement.physical_to_logical[
                layer._ep_start_expert + local_id
            ]
            packed, scale, outer = self._cache.get_or_quantize(
                local_weight,
                block_size=16,
                input_scale=self.input_scale,
                amax=maximum,
                identity=dict(
                    layer=layer.layer_idx,
                    prefix=layer.prefix,
                    expert=logical_id,
                    projection=stem,
                    tp_size=layer.moe_expert_tp_size,
                    tp_rank=layer.moe_expert_tp_rank,
                    source_format=(
                        "fp8-block128" if self.source_fp8_block_size else "floating"
                    ),
                ),
            )
        getattr(layer, stem).data[local_id].copy_(packed)
        getattr(layer, stem + "_scale")[local_id].copy_(scale)
        getattr(layer, stem + "_scale_2")[local_id].copy_(outer)
        getattr(layer, stem + "_input_scale")[local_id].fill_(self.input_scale)

    def dispatch_scale(self, layer, local_id, projection, parameter_name, tensor):
        if self.source_fp8_block_size and parameter_name == "weight_scale_inv":
            if tensor.ndim != 2 or not tensor.is_floating_point():
                raise ValueError("FP8 source block scale must be a floating matrix")
            layer._nvfp4_pending_source_scale[(local_id, projection)] = tensor
            self._decode_source(layer, local_id, projection)
            return True
        if self.online or parameter_name not in self.required_aux_parameters():
            return False
        if parameter_name == "weight_scale":
            if tensor.dtype != torch.float8_e4m3fn:
                raise TypeError("NVFP4 weight_scale must use float8_e4m3fn")
            values = tensor.float()
            if not torch.isfinite(values).all() or (values < 0).any():
                raise ValueError("NVFP4 block scales must be finite and nonnegative")
            self._target(layer, local_id, projection, scale=True).copy_(
                self._slice(layer, projection, tensor, 16)
            )
        else:
            if tensor.numel() != 1 or not tensor.is_floating_point():
                raise ValueError(f"NVFP4 {parameter_name} must be a floating scalar")
            value = tensor.float().reshape(())
            if not torch.isfinite(value) or value <= 0:
                raise ValueError(f"NVFP4 {parameter_name} must be finite and positive")
            stem = "w2" if projection == "down_proj" else "w13"
            suffix = (
                "_scale_2" if parameter_name == "weight_scale_2" else "_input_scale"
            )
            target = getattr(layer, stem + suffix)
            # OldLoader uses max_scalar(gate, up) for the fused expert GEMM.
            target[local_id].copy_(
                torch.maximum(target[local_id], value.to(target.device))
            )
        return True

    def validate_runtime_device(self, device):
        device = torch.device(device)
        if device.type != "cuda" or torch.version.hip is not None:
            raise ValueError("NVFP4 GPU execution requires a supported NVIDIA backend")
        capability = torch.cuda.get_device_capability(device)
        if self._backend == "marlin":
            if capability not in ((9, 0), (12, 0)):
                raise ValueError("NVFP4 Marlin requires SM90 or SM120")
            from rtp_llm.models_py.kernels.cuda.marlin_nvfp4.marlin_nvfp4 import (
                Nvfp4MarlinMoE,
                load_library,
            )

            # Main's admission/library helpers inspect the current CUDA device.
            # Validate the load target, which need not be the caller's current GPU.
            with torch.cuda.device(device):
                if not Nvfp4MarlinMoE.is_supported():
                    raise RuntimeError(
                        "NVFP4 Marlin requires Torch >= 2.10 and compatible Triton histogram"
                    )
                load_library()
            return
        if self._backend == "cutlass" and capability[0] != 12:
            raise ValueError("CUTLASS NVFP4 requires SM120")
        if self._backend in ("trtllm", "cutedsl") and capability[0] < 10:
            raise ValueError(
                "NVFP4 TRTLLM/CuteDSL execution requires NVIDIA SM100 or newer"
            )
        symbols = []
        if self._backend == "trtllm":
            symbols += [
                ("flashinfer", "nvfp4_block_scale_interleave"),
                ("flashinfer.fused_moe", "trtllm_fp4_block_scale_routed_moe"),
                ("flashinfer", "fp4_quantize"),
            ]
        elif self._backend == "cutlass":
            symbols += [("flashinfer.fused_moe", "cutlass_fused_moe")]
        else:
            symbols += [
                ("flashinfer", "scaled_fp4_grouped_quantize"),
                (
                    "rtp_llm.models_py.kernels.cuda.fp4_kernel",
                    "flashinfer_cutedsl_moe_masked",
                ),
            ]
        for module, symbol in symbols:
            try:
                implementation = getattr(importlib.import_module(module), symbol)
                if not callable(implementation):
                    raise TypeError(f"{symbol} is not callable")
            except Exception as exc:
                raise RuntimeError(f"NVFP4 backend requires {module}.{symbol}") from exc

    @staticmethod
    def _replace_parameter(layer, name, value):
        # Preserve the module parameter object while publishing converted storage.
        getattr(layer, name).data = value.contiguous()

    def process_weights_after_loading(self, layer):
        if (
            layer._nvfp4_pending_float
            or layer._nvfp4_pending_source
            or layer._nvfp4_pending_source_scale
        ):
            raise RuntimeError("NVFP4 expert source tensors are incomplete")
        if self._backend == "cutedsl":
            # Loader buffers are [up|gate], CuteDSL consumes [gate|up].
            m = layer.moe_inter_tp
            self._replace_parameter(
                layer,
                "w13",
                torch.cat((layer.w13.data[:, m:], layer.w13.data[:, :m]), dim=1),
            )
            s = layer.w13_scale.view(torch.uint8)
            s = torch.cat((s[:, m:], s[:, :m]), dim=1).view(torch.float8_e4m3fn)
            layer.w13_scale = swizzle_nvfp4_scales(s)
            layer.w2_scale = swizzle_nvfp4_scales(layer.w2_scale)
        elif self._backend in ("cutlass", "marlin"):
            # Main's CUTLASS/Marlin views retain packed [up|gate] weights.
            layer.w13_scale = swizzle_nvfp4_scales(layer.w13_scale)
            layer.w2_scale = swizzle_nvfp4_scales(layer.w2_scale)
        else:
            for stem in ("w13", "w2"):
                weight, scale = _prepare_trtllm(
                    getattr(layer, stem).data,
                    getattr(layer, stem + "_scale"),
                    stem == "w13",
                )
                self._replace_parameter(layer, stem, weight)
                setattr(layer, stem + "_scale", scale)
        # TRTLLM quantizes activations once per rank; CuteDSL quantizes per expert.
        # Compute from the newly loaded local experts, including on EPLB reload.
        input_scale = layer.w13_input_scale
        if self._backend == "trtllm":
            input_scale = input_scale.max().reshape(1)
        g1_alpha = input_scale * layer.w13_scale_2
        derived = {
            "nvfp4_w13_alpha": g1_alpha,
            "nvfp4_w2_alpha": layer.w2_input_scale * layer.w2_scale_2,
            "nvfp4_w13_input_scale_inv": input_scale.reciprocal(),
            "nvfp4_w2_input_scale_inv": layer.w2_input_scale.reciprocal(),
            "nvfp4_g1_scale_c": g1_alpha / layer.w2_input_scale,
        }
        for name, value in derived.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite NVFP4 runtime scale {name}")
            layer.register_buffer(name, value)

    def add_weight_tensors(self, layer, weights):
        for key, name in (
            (W.moe_s1, "w13_scale"),
            (W.moe_s2, "w2_scale"),
            (W.moe_w1_s2, "w13_scale_2"),
            (W.moe_w2_s2, "w2_scale_2"),
            (W.moe_w1_i_s, "w13_input_scale"),
            (W.moe_w2_i_s, "w2_input_scale"),
        ):
            weights[key] = getattr(layer, name)
        for name in self._derived_names():
            weights[name] = getattr(layer, name)

    @staticmethod
    def _derived_names():
        return (
            "nvfp4_w13_alpha",
            "nvfp4_w2_alpha",
            "nvfp4_w13_input_scale_inv",
            "nvfp4_w2_input_scale_inv",
            "nvfp4_g1_scale_c",
        )

    def eplb_aux_weight_names(self):
        return (
            W.moe_w1_s2,
            W.moe_w2_s2,
            W.moe_w1_i_s,
            W.moe_w2_i_s,
            *self._derived_names(),
        )


def _resolve_backend(config):
    backend = getattr(config, "fp4_moe_op", "auto")
    strategy = getattr(config, "moe_strategy", "auto")
    for selected in ("marlin", "cutlass"):
        if strategy == f"nvfp4_{selected}":
            if backend not in ("auto", selected):
                raise ValueError(
                    f"nvfp4_{selected} requires FP4_MOE_OP=auto or {selected}"
                )
            return selected
        if backend == selected:
            raise ValueError(
                f"FP4_MOE_OP={selected} requires MOE_STRATEGY=nvfp4_{selected}"
            )
    if backend == "auto":
        if getattr(config, "use_deepep_moe", False) and getattr(
            config, "use_deepep_low_latency", False
        ):
            return "cutedsl"
        return "trtllm"
    return backend


def _prepare_trtllm(weight, scale, gate_up):
    """Use the same FlashInfer permutation/interleave contract as OldLoader."""
    from flashinfer import nvfp4_block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    findices = (
        _maybe_get_cached_w3_w1_permute_indices
        if gate_up
        else get_w2_permute_indices_with_cache
    )
    weights, scales, cache = [], [], {}
    for w, s in zip(weight, scale):
        wi = findices(cache, w.view(torch.uint8), 128)
        si = findices(cache, s.view(torch.uint8), 128, num_elts_per_sf=16)
        weights.append(w.view(torch.uint8)[wi.to(w.device)].contiguous())
        scales.append(
            nvfp4_block_scale_interleave(
                s.view(torch.uint8)[si.to(s.device)].contiguous()
            )
        )
    return torch.stack(weights), torch.stack(scales).view(
        torch.float8_e4m3fn
    ).reshape_as(scale)
