"""PPU W8A8 expert parameters; execution remains in main's DeepEP backends."""

import importlib
import weakref

import torch
from torch import nn

from rtp_llm.models_py.module_base import copy_weight_
from rtp_llm.models_py.quant_methods.base import (
    FusedMoEMethodBase,
    register_moe_quant_method,
)
from rtp_llm.models_py.quant_methods.w8a8_int8 import (
    W8A8_METHOD,
    checked_channel_scale,
    require_ppu_device,
    resolved_backend_symbol,
)


def _require_grouped_symbols(low_latency):
    module = importlib.import_module("deep_gemm")
    masked = (
        "m_grouped_gemm_int8_int8_bf16_nt_masked",
        "m_grouped_int8_gemm_nt_masked",
    )
    contiguous = (
        "m_grouped_gemm_int8_int8_bf16_nt_contiguous",
        "m_grouped_int8_gemm_nt_contiguous",
    )
    nopad = (
        "m_grouped_gemm_int8_int8_bf16_nt_nopad",
        "m_grouped_int8_gemm_nt_nopad",
    )
    # Hybrid selects nopad whenever present, otherwise contiguous. A bad
    # preferred symbol must not be concealed by a callable alternate alias.
    normal = nopad if resolved_backend_symbol(module, nopad) is not None else contiguous
    for aliases in (masked,) if low_latency else (masked, normal):
        if not callable(resolved_backend_symbol(module, aliases)):
            raise RuntimeError(f"PPU DeepGEMM grouped INT8 symbol missing: {aliases}")
    buffer_cls = getattr(importlib.import_module("deep_ep"), "Buffer")
    names = (
        ("low_latency_dispatch", "low_latency_combine")
        if low_latency
        else ("dispatch", "combine", "get_dispatch_layout")
    )
    if not callable(buffer_cls) or not all(
        callable(getattr(buffer_cls, name, None)) for name in names
    ):
        raise RuntimeError(f"PPU DeepEP requires Buffer APIs {names}")


def require_ppu_moe_backend(layer, device):
    """Validate the selected real router/executor before GPU weight allocation."""
    require_ppu_device(device)
    if layer._params_dtype != torch.bfloat16 or layer.ep_size <= 1:
        raise RuntimeError(
            "PPU W8A8 MoE runtime requires BF16 compute and expert parallelism"
        )
    from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory
    from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
        MoEConfigAdapter,
    )

    low_latency = layer._moe_config.use_deepep_low_latency
    try:
        _require_grouped_symbols(low_latency)
        module = importlib.import_module(
            "rtp_llm.models_py.modules.factory.fused_moe.ppu_impl.strategy.w8a8_int8"
        )
        cls = (
            module.PpuW8A8Int8EpLowLatencyDeepGemmStrategy
            if low_latency
            else module.PpuW8A8Int8DpNormalDeepGemmStrategy
        )
        strategy = cls()
        adapter = MoEConfigAdapter(
            model_config=layer._model_config,
            parallelism_config=layer._parallelism_config,
            moe_config=layer._moe_config,
            quant_config=layer._effective_model_quant_config,
            enable_cuda_graph=layer._enable_cuda_graph(),
        )
        if adapter.physical_expert_num != layer.physical_num_experts:
            raise ValueError("NewLoader placement and PPU physical counts differ")
        adapter.expert_num = layer.physical_num_experts
        if not strategy.can_handle(adapter):
            raise RuntimeError(
                "PPU W8A8 router/executor rejected the model topology or backend"
            )
        registry = FusedMoeFactory._registry
        if registry is None:
            raise RuntimeError("MoE backend registry is not initialized")
        if not any(type(existing) is cls for existing in registry._strategies):
            registry.register(strategy)
    except (ImportError, OSError, RuntimeError, AttributeError, ValueError) as error:
        raise RuntimeError(f"PPU W8A8 MoE backend is unavailable: {error}") from error


@register_moe_quant_method(W8A8_METHOD)
class W8A8Int8MoEMethod(FusedMoEMethodBase):
    def create_weights(
        self, layer, num_experts, hidden_size, intermediate_size, params_dtype, **kwargs
    ):
        if params_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("W8A8 checkpoint owners require FP16/BF16 compute dtype")
        self._layer = weakref.ref(layer)
        for name, shape, dtype in (
            ("w13", (num_experts, 2 * intermediate_size, hidden_size), torch.int8),
            ("w2", (num_experts, hidden_size, intermediate_size), torch.int8),
            ("w13_scale", (num_experts, 2 * intermediate_size, 1), torch.float32),
            ("w2_scale", (num_experts, hidden_size, 1), torch.float32),
        ):
            layer.register_parameter(
                name, nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            )

    def validate_runtime_device(self, device):
        if torch.device(device).type != "cpu":
            require_ppu_moe_backend(self._layer(), device)

    def required_aux_parameters(self):
        return ("weight_scale",)

    def dispatch_weight(self, layer, local_id, projection, parameter_name, tensor):
        if parameter_name != "weight":
            return False
        if tensor.dtype != torch.int8:
            raise TypeError(
                "Compressed W8A8 experts require signed INT8 checkpoint storage"
            )
        return False  # BaseMoE owns the existing expert placement and TP copies.

    def dispatch_scale(self, layer, local_id, projection, parameter_name, tensor):
        if parameter_name != "weight_scale":
            return False
        if projection == "down_proj":
            source = checked_channel_scale(tensor, layer.hidden_size, projection)
            target = layer.w2_scale.data[local_id]
        else:
            source = checked_channel_scale(tensor, layer.moe_inter, projection)
            source = source.narrow(
                0, layer.moe_expert_tp_rank * layer.moe_inter_tp, layer.moe_inter_tp
            )
            offset = layer.moe_inter_tp if projection == "gate_proj" else 0
            target = layer.w13_scale.data[local_id].narrow(
                0, offset, layer.moe_inter_tp
            )
        copy_weight_(target, source, f"{projection}.{parameter_name}")
        return True

    def process_weights_after_loading(self, layer):
        for parameter in (layer.w13_scale, layer.w2_scale):
            checked_channel_scale(
                parameter.reshape(-1, 1), parameter.numel(), layer.prefix
            )
        if layer.w13.device.type != "cpu":
            require_ppu_moe_backend(layer, layer.w13.device)

    def add_weight_tensors(self, layer, weights):
        # Existing executor boundary only; no checkpoint name translation.
        from rtp_llm.utils.model_weight import W

        weights[W.moe_s1] = layer.w13_scale
        weights[W.moe_s2] = layer.w2_scale
