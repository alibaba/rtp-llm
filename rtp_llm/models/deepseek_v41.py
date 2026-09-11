"""V4.1 configuration and checkpoint descriptors for the existing model factory.

The runtime guard is intentional until the typed cache, shared Engram mapping
and V4.1 attention driver are connected. V4's ratio dispatch is incompatible.
"""

import torch

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.config.dsv41_weights import (
    V41TensorSpec,
    build_v41_manifest,
    validate_inventory,
)
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.models_py.modules.dsv41.math import dequantize_block32
from rtp_llm.utils.model_weight import CkptWeightInfo, W, identity


def _wo_a_bf16(tensors):
    return dequantize_block32(tensors[0], tensors[1])


_CHECKPOINT_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "I8": torch.int8,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
}


class V41AtomicWeight(AtomicWeight):
    """Already EP-selected weights with the checkpoint's exact tensor layout."""

    def __init__(self, name, specs: list[V41TensorSpec], process_fun, data_type):
        if not name.startswith("v41.") or not specs:
            raise ValueError("V4.1 atomic weights need their checkpoint specifications")
        if any(spec.placement == "host_shared" for spec in specs):
            raise ValueError("Engram tables must use the host-shared loader")
        self.specs = tuple(specs)
        super().__init__(
            name,
            [CkptWeightInfo(spec.name, identity) for spec in specs],
            process_fun,
            data_type=data_type,
        )

    def _load_raw_tensor(self, tensor_source, layer_id, device, load_config):
        tensors = []
        for spec in self.specs:
            values = tensor_source.load_tensor(spec.name, None)
            if len(values) != 1:
                raise ValueError(f"{spec.name}: expected one checkpoint tensor")
            value = values[0]
            if (
                tuple(value.shape) != spec.shape
                or value.dtype != _CHECKPOINT_DTYPES[spec.dtype]
            ):
                raise ValueError(f"{spec.name}: checkpoint shape or dtype mismatch")
            tensors.append(value.to(device))
        # Unlike the generic loader, do not reshape 1D mHC scales or cast a
        # paired FP8 weight/UE8M0 scale to one output dtype before conversion.
        return {self.name: self.process_fun(tensors).to(self.data_type)}

    def _split(self, tensor, load_config):
        if load_config.tp_size != 1:
            raise ValueError("V4.1 tensor descriptors require attention TP1")
        return tensor if isinstance(tensor, dict) else {self.name: tensor}


class DeepSeekV41Weight(ModelDeployWeightInfo):
    supports_fastsafetensors = False

    def _process_meta(self, meta_dict, weight_keys):
        validate_inventory(
            build_v41_manifest(self.model_config.dsv41_config), weight_keys
        )

    @property
    def host_weight_specs(self):
        return {
            name: spec
            for name, spec in build_v41_manifest(self.model_config.dsv41_config).items()
            if spec.placement == "host_shared"
        }

    def get_weight_info(self) -> ModelWeightInfo:
        # V4.1 carries its own raw FP8/FP4 scales. Generic V4 block128
        # conversions and merged-FFN transforms must not rewrite this graph.
        return self._get_weight_info()

    def _get_weight_info(self) -> ModelWeightInfo:
        config = self.model_config.dsv41_config
        config.validate_parallelism(tp_size=self.tp_size, ep_size=self.ep_size)
        if self.tp_size != 1:
            raise NotImplementedError(
                "V4.1 descriptors currently require attention TP1"
            )
        specs = build_v41_manifest(config)
        layers = [[] for _ in range(config.text["num_hidden_layers"])]
        globals_ = []
        aliases = {
            "embed.weight": W.embedding,
            "head.weight": W.lm_head,
            "norm.weight": W.final_ln_gamma,
        }
        for name, spec in specs.items():
            if name.startswith("mtp.") or spec.placement == "host_shared":
                continue
            if spec.conversion == "dequantize_bf16" and name.endswith(".scale"):
                continue
            parts = name.split(".")
            if ".ffn.experts." in name:
                expert = int(parts[4])
                local = config.text["n_routed_experts"] // self.ep_size
                if not self.ep_rank * local <= expert < (self.ep_rank + 1) * local:
                    continue
            source_specs = [spec]
            process = identity
            dtype = _CHECKPOINT_DTYPES[spec.dtype]
            if spec.conversion == "dequantize_bf16":
                source_specs.append(specs[name.removesuffix(".weight") + ".scale"])
                process, dtype = _wo_a_bf16, torch.bfloat16
            elif spec.conversion in ("fp32_logits", "fp32_norm"):
                dtype = torch.float32
            is_layer = name.startswith("layers.")
            if (
                is_layer
                and config.text["compress_ratios"][int(parts[1])] == 2
                and ".".join(parts[2:])
                in ("attn.compressor.wkv.weight", "attn.compressor.wgate.weight")
            ):
                dtype = torch.float32
            key = "v41." + (".".join(parts[2:]) if is_layer else name)
            if name in aliases:
                descriptor = AtomicWeight(
                    aliases[name],
                    [CkptWeightInfo(name, identity)],
                    process,
                    data_type=dtype,
                )
            else:
                descriptor = V41AtomicWeight(key, source_specs, process, dtype)
            if is_layer:
                layers[int(parts[1])].append(descriptor)
            else:
                globals_.append(descriptor)
        return ModelWeightInfo(weights=globals_, layer_weights=layers)


class DeepSeekV41PrefillDraftWeight(DeepSeekV41Weight):
    """Selective draft weights for P commit-only execution.

    The shared embedding/head descriptors preserve the ordinary ModelLoader
    interface. Pass target-owned global_weight_aliases when loading this
    descriptor so they retain the target storage without checkpoint I/O.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num_layers = 3
        self.moe_layer_index_ = []
        self.enable_eplb_ = False

    @property
    def host_weight_specs(self):
        return {}

    def _get_weight_info(self) -> ModelWeightInfo:
        config = self.model_config.dsv41_config
        config.validate_parallelism(tp_size=self.tp_size, ep_size=self.ep_size)
        if self.tp_size != 1 or config.text["num_nextn_predict_layers"] != 3:
            raise ValueError("V4.1 P draft weights require TP1 and three stages")
        specs = build_v41_manifest(config)

        def descriptor(name, key):
            spec = specs[name]
            return V41AtomicWeight(
                "v41." + key,
                [spec],
                identity,
                _CHECKPOINT_DTYPES[spec.dtype],
            )

        globals_ = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo("embed.weight", identity)],
                identity,
                data_type=torch.bfloat16,
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo("head.weight", identity)],
                identity,
                data_type=torch.float32,
            ),
        ]
        globals_.extend(
            descriptor("mtp.0." + name, "mtp.0." + name)
            for name in ("main_proj.weight", "main_proj.scale", "main_norm.weight")
        )
        layers = [
            [
                descriptor(f"mtp.{stage}." + name, name)
                for name in (
                    "attn.wkv.weight",
                    "attn.wkv.scale",
                    "attn.kv_norm.weight",
                )
            ]
            for stage in range(3)
        ]
        return ModelWeightInfo(weights=globals_, layer_weights=layers)


class DeepSeekV41(DeepSeekV2):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        parsed = V41Config.from_path(ckpt_path)
        t = parsed.text
        config = ModelConfig()
        config.dsv41_config = parsed
        config.model_type = "deepseek_v41"
        config.num_layers = t["num_hidden_layers"]
        config.hidden_size = t["hidden_size"]
        config.vocab_size = t["vocab_size"]
        config.max_seq_len = t["max_position_embeddings"]
        config.layernorm_eps = t["rms_norm_eps"]
        config.norm_type = "rmsnorm"
        config.activation_type = "SiGLU"
        config.has_post_decoder_layernorm = True
        config.tie_word_embeddings = False
        config.enable_fp32_lm_head = True
        config.config_dtype = parsed.dtype
        config.special_tokens.bos_token_id = parsed.bos_token_id
        config.special_tokens.eos_token_id = parsed.eos_token_id
        config.special_tokens.pad_token_id = parsed.pad_token_id
        a = config.attn_config
        a.dsv41_cache_layout_version = 1
        a.head_num = t["num_attention_heads"]
        a.kv_head_num = t["num_key_value_heads"]
        a.size_per_head = t["head_dim"]
        a.rope_head_dim = t["qk_rope_head_dim"]
        a.nope_head_dim = t["head_dim"] - t["qk_rope_head_dim"]
        a.v_head_dim = t["head_dim"]
        a.q_lora_rank = t["q_lora_rank"]
        a.kv_lora_rank = 0
        a.use_mla = False
        a.o_groups = t["o_groups"]
        a.o_lora_rank = t["o_lora_rank"]
        a.sliding_window = t["sliding_window"]
        a.layer_compress_ratios = t["compress_ratios"][: config.num_layers]
        a.compress_rope_theta = t["compress_rope_theta"]
        a.rope_config.base = t["rope_theta"]
        a.rope_config.dim = t["qk_rope_head_dim"]
        a.rope_config.offset = t["head_dim"] - t["qk_rope_head_dim"]
        a.rope_config.style = 0
        scaling = t["rope_scaling"]
        a.rope_config.scale = scaling["factor"]
        a.rope_config.factor1 = float(scaling["beta_slow"])
        a.rope_config.factor2 = float(scaling["beta_fast"])
        a.rope_config.max_pos = scaling["original_max_position_embeddings"]
        a.is_sparse = True
        a.indexer_head_dim = t["index_head_dim"]
        a.indexer_head_num = t["index_n_heads"]
        a.indexer_topk = t["index_topk"]
        config.hc_mult = t["hc_mult"]
        config.hc_sinkhorn_iters = t["hc_sinkhorn_iters"]
        config.hc_eps = t["hc_eps"]
        config.swiglu_limit = t["swiglu_limit"]
        config.num_hash_layers = 0
        config.scoring_func = 2
        config.expert_num = t["n_routed_experts"]
        config.moe_k = t["num_experts_per_tok"]
        config.moe_style = 2
        config.moe_n_group = 0
        config.moe_topk_group = 0
        config.has_moe_norm = t["norm_topk_prob"]
        config.moe_layer_index = list(range(config.num_layers))
        config.routed_scaling_factor = t["routed_scaling_factor"]
        config.moe_inter_size = t["moe_intermediate_size"]
        config.inter_size = t["n_shared_experts"] * t["moe_intermediate_size"]
        config.dspark_noise_token_id = t["dspark_noise_token_id"]
        config.dspark_target_layer_ids = list(t["dspark_target_layer_ids"])
        config.dspark_markov_rank = t["dspark_markov_rank"]
        return config

    @classmethod
    def from_config(cls, *args, **kwargs):
        raise NotImplementedError(
            "V4.1 runtime requires its typed cache/attention and host-shared Engram integration; V4 execution is not compatible"
        )

    def _create_python_model(self):
        raise NotImplementedError("V4.1 runtime integration is incomplete")

    @staticmethod
    def get_weight_cls():
        return DeepSeekV41Weight


register_model("deepseek_v41", DeepSeekV41, ["DeepseekV41ForCausalLM"])
