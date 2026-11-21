"""Owned V4 parameters on the existing production prefill/decode runtime.

There is no ModelWeights checkpoint adapter here. The three runtime hooks keep
initialization, cache binding, warmup, prefill and decode in the existing model.
"""

from dataclasses import asdict

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v4.backbone import (
    OwnedV4Args,
    OwnedV4Transformer,
)
from rtp_llm.models_py.new_models.deepseek_v4.load_geometry import (
    validate_load_groups,
    validate_v4_static_placement,
)


class _BackboneMapper:
    def __init__(self, backbone):
        self.backbone = backbone

    def map_name(self, name):
        return "v4." + self.backbone.WEIGHTS_MAPPER.map_name(name)

    def apply(self, items):
        for name, value in items:
            yield self.map_name(name), value


class OwnedV4ModelMixin:
    """Shared loader hooks; each serving role keeps its own runtime class."""

    # These owners copy each selected checkpoint tensor into its final rank-local
    # destination; CPU source tensors do not become runtime parameters.
    arm_cuda_streamed_scratch = True

    runtime_parameter_paths = {
        "embedding": "v4.embed.weight",
        "lm_head": "v4.head.weight",
    }

    @classmethod
    def resolve_runtime_class(cls, model_config):
        from .ppu_platform import is_ppu_flash

        if not is_ppu_flash(model_config):
            return cls
        if model_config.model_type == "deepseek_v4_mtp":
            from .ppu_mtp import PpuDeepSeekV4MtpForCausalLM

            return PpuDeepSeekV4MtpForCausalLM
        from .ppu_language import PpuDeepSeekV4ForCausalLM

        return PpuDeepSeekV4ForCausalLM

    def validate_runtime_device(self, device):
        super().validate_runtime_device(device)
        if torch.device(device).type != "cpu":
            from rtp_llm.models_py.modules.dsv4.native_ops import (
                require_serving_bindings,
            )

            require_serving_bindings()

    def __init__(self, model_config, load_config):
        batch_size = load_config.max_generate_batch_size
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError(
                "V4 requires the service's positive max_generate_batch_size"
            )
        if load_config.compute_dtype != torch.bfloat16:
            raise ValueError("V4 native FP8/FP4 runtime requires BF16 compute dtype")
        if load_config.parallelism_config is None:
            raise ValueError("V4 requires the service parallelism configuration")
        super().__init__(
            model_config,
            load_config.parallelism_config,
            weights=None,
            moe_config=load_config.moe_config,
            max_generate_batch_size=batch_size,
            fmha_config=load_config.fmha_config,
            py_hw_kernel_config=load_config.py_hw_kernel_config,
        )
        args = OwnedV4Args(**asdict(self._v4_args))
        args.cuda_norm = (
            torch.device(load_config.device).type == "cuda"
            and torch.version.hip is None
        )
        if args.cuda_norm:
            from .cuda_rms import use_native_rms

            args.cuda_native_rms = use_native_rms(load_config.device)
        validate_load_groups(args, load_config)
        validate_v4_static_placement(args, model_config, load_config)
        args.lm_head_tp_size = load_config.lm_head_tp_size
        args.lm_head_tp_rank = load_config.lm_head_tp_rank
        # The main runtime has already validated checkpoint geometry and
        # resolved the role-independent MoE budget, including batch overrides.
        self._v4_args = args
        self.v4 = self.create_backbone(args, load_config)
        self.WEIGHTS_MAPPER = _BackboneMapper(self.v4)

    def create_backbone(self, args, load_config):
        return OwnedV4Transformer(
            args,
            params_dtype=load_config.compute_dtype,
            head_dtype=(
                torch.float32
                if getattr(self.config, "enable_fp32_lm_head", False)
                else load_config.compute_dtype
            ),
        )

    def checkpoint_weight_name_filter(self):
        return self.v4.checkpoint_weight_name_filter()

    def load_weights(self, weights):
        items = weights.items() if isinstance(weights, dict) else weights
        RtpModule.load_weights(self, self.WEIGHTS_MAPPER.apply(items))

    def runtime_weight_view(self):
        return {
            "embedding": self.v4.embed.weight,
            "lm_head": self.v4.head.weight,
            "final_layernorm.gamma": self.v4.norm.weight,
        }

    def _runtime_weight_device(self):
        return self.v4.layers[0].attn.wkv.weight.device

    def _build_runtime_backbone(self):
        # The root parameter tree was already loaded and postprocessed by
        # NewLoader. Only now is the engine role/token budget authoritative.
        from rtp_llm.models_py.model_loader import NewModelLoader
        from rtp_llm.models_py.modules.dsv4.chunk_env import chunked_moe_enabled
        from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_owned import (
            initialize_owned_mega_decode,
        )

        NewModelLoader._validate_loaded_weights(self)
        for block in self.v4.layers:
            if block.ffn is not None:
                block.ffn.configure_runtime(
                    is_decode_role=self._is_decode_role,
                    max_tokens_per_rank=self._v4_args.max_tokens_per_rank,
                    world_size=self._v4_args.world_size,
                    world_rank=self._v4_args.world_rank,
                    chunking_enabled=chunked_moe_enabled(),
                )
                block.ffn.initialize_runtime()
        initialize_owned_mega_decode(self.v4, self._v4_args)

    def _release_runtime_weight_source(self):
        from rtp_llm.models.direct_checkpoint import RuntimeWeightView

        # The direct serving boundary publishes references to our parameters,
        # not a checkpoint source. Keep those references available to the
        # engine and draft loans after model initialization.
        if self.weight is not None and not isinstance(self.weight, RuntimeWeightView):
            raise RuntimeError("Owned V4 must never carry a legacy ModelWeights source")


class DeepSeekV4ForCausalLM(OwnedV4ModelMixin, DeepSeekV4Model, RtpModule):
    """Direct Flash/Pro parameters with the original target execution path."""
