"""Select direct routed expert parameters for an inspected Qwen4 checkpoint."""

from types import SimpleNamespace

import torch

from rtp_llm.config.quant_config import QWEN4_FP8_NVFP4_RECIPE, ModelOptFp4Config
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.new_models.qwen4_exp.nvfp4_experts import Qwen4Nvfp4Experts
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


class Qwen4RoutedExperts(BaseMoEExperts):
    """Reuse main FusedMoe (including its router, reduction and graph hooks)."""

    @property
    def local(self):
        return self.num_local_experts

    @property
    def runtime(self):
        return self.fused_moe

    def validate_runtime_device(self, device):
        # CPU loads are checkpoint staging. BaseMoEExperts never installs a
        # fused executor on CPU; actual accelerator admission is unchanged.
        if device.type != "cpu":
            super().validate_runtime_device(device)


def make_qwen4_experts(plan, layer_idx, load, *, prefix=None):
    kind = plan.source_kind
    h = plan.hidden_size
    m = plan.positive("moe_intermediate_size")
    e = plan.positive("num_experts")
    if kind == "nvfp4":
        if load.tp_size != load.ep_size:
            raise ValueError("Qwen4 prequantized Marlin requires TP == EP")
        return Qwen4Nvfp4Experts(h, m, e, load.ep_size, load.ep_rank)
    if kind not in ("bf16", "fp8"):
        raise ValueError(f"Unrecognized Qwen4 expert source: {kind!r}")

    config = plan.model_config
    if isinstance(config, dict):
        # Dict-based CPU checkpoint inspection is supported without inventing
        # scheduler/parallelism/engine configuration for a runnable model.
        if torch.device(load.device).type != "cpu":
            raise ValueError(
                "Qwen4 accelerator loading requires a typed runtime configuration"
            )
        source = config.get("quant_config")
        if source is None and kind == "fp8":
            source = ModelOptFp4Config(
                bits=4,
                group_size=16,
                is_quanted=False,
                model_recipe=QWEN4_FP8_NVFP4_RECIPE,
            )
        config = SimpleNamespace(
            model_type=config.get("model_type", "qwen4_exp"),
            activation_type="SiGLU",
            num_layers=plan.positive("num_hidden_layers"),
            moe_k=config.get("moe_k", plan.text.get("num_experts_per_tok")),
            quant_config=source,
        )
    source = config.quant_config
    if kind == "bf16":
        if source is not None:
            raise ValueError("Qwen4 BF16 expert source requires unquantized runtime")
        method = "none"
    else:
        if (
            not isinstance(source, ModelOptFp4Config)
            or source.model_recipe != QWEN4_FP8_NVFP4_RECIPE
            or source.is_quanted()
            or source.source_fp8_block_size != 128
        ):
            raise ValueError(
                "Qwen4 FP8 experts require the block128 source to online NVFP4 recipe"
            )
        method = "modelopt_fp4_online"
    # Keep the runtime quant config itself on the typed model, not an alias
    # derived from checkpoint quant_method='fp8'. It controls CUTLASS factory
    # selection, activation scaling, the cache, and graph capture.
    quant = QuantizationConfig(
        method, source_config=source, hw_kernel_config=load.py_hw_kernel_config
    )
    if prefix is None:
        # EPLB compares WEIGHTS_MAPPER output to the registered owner path.
        prefix = f"layers.{layer_idx}.mlp.experts"
    owner = Qwen4RoutedExperts(
        num_experts=e,
        top_k=config.moe_k,
        hidden_size=h,
        moe_intermediate_size=m,
        tp_size=load.ffn_tp_size,
        tp_rank=load.ffn_tp_rank,
        ep_size=load.ep_size,
        ep_rank=load.ep_rank,
        params_dtype=load.compute_dtype,
        model_config=config,
        parallelism_config=load.parallelism_config,
        moe_config=load.moe_config,
        quant_config=quant,
        layer_idx=layer_idx,
        prefix=prefix,
    )
    owner.has_shared_expert_gate = (
        f"{plan.prefix}layers.{layer_idx}.mlp.shared_expert_gate.weight" in plan.shapes
    )
    return owner
