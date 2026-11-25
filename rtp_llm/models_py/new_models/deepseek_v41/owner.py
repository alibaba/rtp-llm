"""Runtime-neutral constructor/hooks shared by target and DSpARK owners."""

import json
import re
from pathlib import Path

import torch

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.deepseek_v4.load_geometry import (
    configure_lm_head,
    validate_load_groups,
)
from rtp_llm.models_py.new_models.deepseek_v41.backbone import runtime_backbone_class


def vision_router_layers(checkpoint_path):
    """Inspect checkpoint names only; never eagerly materialize model weights."""
    root = Path(checkpoint_path)
    names = set()
    for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        path = root / filename
        if path.is_file():
            with path.open() as stream:
                names.update(json.load(stream)["weight_map"])
            break
    else:
        from safetensors import safe_open

        from rtp_llm.models_py.weight_mapper import discover_ckpt_files

        for filename in discover_ckpt_files(str(root)):
            if filename.endswith(".safetensors"):
                with safe_open(filename, framework="pt", device="cpu") as source:
                    names.update(source.keys())
            else:
                metadata = torch.load(filename, map_location="meta", weights_only=True)
                names.update(metadata.get("state_dict", metadata))
    result = {"layers": [], "mtp": []}
    for name in names:
        match = re.fullmatch(r"(layers|mtp)\.(\d+)\.ffn\.gate\.bias_vl", name)
        if match:
            result[match[1]].append(int(match[2]))
        else:
            ft = re.fullmatch(
                r"rank_\d+_\d+_\d+\.layers\.(\d+)\.v4\.router\.bias_vl", name
            )
            if ft:
                result["layers"].append(int(ft[1]))
    return {prefix: tuple(sorted(set(ids))) for prefix, ids in result.items()}


class _BackboneMapper:
    def __init__(self, backbone):
        self.backbone = backbone

    def map_name(self, name):
        return "v4." + self.backbone.WEIGHTS_MAPPER.map_name(name)

    def apply(self, items):
        for name, value in items:
            yield self.map_name(name), value


class OwnedV41Mixin:
    """Place before either main target or main draft runtime in the MRO.

    This mixin deliberately inherits neither target runtime nor nn.Module;
    draft forward/commit/graph methods therefore remain the main draft's.
    """

    runtime_parameter_paths = {
        "embedding": "v4.embed.weight",
        "lm_head": "v4.head.weight",
    }

    def load_ft_checkpoint(self, checkpoint):
        from rtp_llm.models_py.new_models.deepseek_v4.ft_checkpoint import (
            load_v4_ft_checkpoint,
        )

        load_v4_ft_checkpoint(self, checkpoint)

    def validate_runtime_device(self, device):
        super().validate_runtime_device(device)
        if torch.device(device).type != "cpu":
            from rtp_llm.models_py.modules.dsv4.native_ops import (
                require_serving_bindings,
            )

            require_serving_bindings()

    def __init__(self, model_config, load_config):
        if (
            type(load_config.max_generate_batch_size) is not int
            or load_config.max_generate_batch_size < 1
        ):
            raise ValueError(
                "V4.1 requires the real positive service generation batch size"
            )
        if load_config.compute_dtype != torch.bfloat16:
            raise ValueError("V4.1 native runtime requires BF16 compute dtype")
        if load_config.parallelism_config is None:
            raise ValueError(
                "V4.1 requires the service's actual parallelism configuration"
            )
        super().__init__(
            model_config,
            load_config.parallelism_config,
            weights=None,
            moe_config=load_config.moe_config,
            max_generate_batch_size=load_config.max_generate_batch_size,
            fmha_config=load_config.fmha_config,
            py_hw_kernel_config=load_config.py_hw_kernel_config,
        )
        self._owned_load_config = load_config
        self._owned_checkpoint_path = model_config.ckpt_path
        self._owned_source_method = None
        requested = getattr(load_config.load_method, "value", load_config.load_method)
        if requested in ("scratch", "fastsafetensors"):
            self.set_checkpoint_source(requested)
        args = self._v4_args
        if not args.v41_config:
            raise ValueError("V4.1 cannot substitute V4 config or cache geometry")
        validate_load_groups(args, load_config)
        configure_lm_head(args, model_config, load_config)
        args.embedding_tp_size, args.embedding_tp_rank = args.tp_size, args.tp_rank
        args.n_shared_experts = int(model_config.n_shared_experts)
        args.moe_inter_dim = int(model_config.moe_inter_size)
        if getattr(load_config, "resident_construction", None) is None:
            args.vision_router_bias_layers = vision_router_layers(model_config.ckpt_path)
        else:
            args.vision_router_bias_layers = {
                key: tuple(load_config.resident_construction[key]) for key in ("layers", "mtp")
            }
        self.v4 = self.create_backbone(args, load_config)
        self.WEIGHTS_MAPPER = _BackboneMapper(self.v4)

    def resident_constructor_state(self):
        return self._v4_args.vision_router_bias_layers

    def create_backbone(self, args, load_config):
        return runtime_backbone_class(target_block_device=load_config.device)(
            args,
            checkpoint_path=self._owned_checkpoint_path,
            head_dtype=(
                torch.float32 if self.config.enable_fp32_lm_head else torch.bfloat16
            ),
        )

    def set_checkpoint_source(self, load_method):
        value = getattr(load_method, "value", load_method)
        if value not in ("scratch", "fastsafetensors"):
            raise ValueError(
                "V4.1 host sources require the resolved scratch/FST method"
            )
        if self._owned_source_method not in (None, value):
            raise RuntimeError(
                "V4.1 host source cannot silently change the selected loader"
            )
        self._owned_source_method = value

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

    def _runtime_weight_counts(self):
        return len(self.v4.layers), 0 if self.v4.commit_only else 3

    def _runtime_weight_device(self):
        return self.v4.layers[0].attn.wkv.weight.device

    def _create_runtime_transformer(self):
        from rtp_llm.models_py.model_loader import NewModelLoader
        from rtp_llm.models_py.modules.dsv41 import _profiler
        from rtp_llm.models_py.modules.dsv41.chunk_env import chunked_moe_enabled

        NewModelLoader._validate_loaded_weights(self)
        if self.v4.embed is not None:
            self.v4.embed.bind_embedding_runtime(self.config, self.parallelism_config)
        for block in self.v4.layers:
            if block.ffn is not None:
                block.ffn.configure_runtime(
                    is_decode_role=self._is_decode_role,
                    max_tokens_per_rank=self._v4_args.max_tokens_per_rank,
                    world_size=self._v4_args.world_size,
                    world_rank=self._v4_args.world_rank,
                    chunking_enabled=chunked_moe_enabled(),
                    record_function_scope=_profiler.moe_record_function_scope,
                )
                block.ffn.initialize_runtime()
                if not self._is_decode_role:
                    block.ffn.gate._prefill_gate_chunk_rows = 32768

    def _release_runtime_weight_source(self):
        if self.weight is not None:
            from rtp_llm.models.direct_checkpoint import RuntimeWeightView

            if not isinstance(self.weight, RuntimeWeightView):
                raise RuntimeError(
                    "V4.1 direct owner must not retain legacy ModelWeights"
                )
            # The serving boundary publishes a non-owning view before runtime
            # initialization. It must reference this same owner's parameters;
            # accepting a second loaded tensor set would violate ownership.
            expected = self.runtime_weight_view()
            if any(self.weight.weights) or set(self.weight.global_weights) != set(
                expected
            ):
                raise RuntimeError(
                    "V4.1 runtime view must contain only owned global aliases"
                )
            if any(
                self.weight.global_weights[name] is not value
                for name, value in expected.items()
            ):
                raise RuntimeError("V4.1 runtime view cannot replace owned parameters")
        self._load_owned_extra_weights()
        del self.weight

    def _load_owned_extra_weights(self):
        """Draft overrides this to bind its already-owned auxiliary modules."""
        return None
