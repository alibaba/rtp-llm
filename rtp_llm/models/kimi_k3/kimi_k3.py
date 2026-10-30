import json
import logging
import os
from dataclasses import dataclass, replace
from math import gcd
from typing import Any, Dict, Iterable, List, Optional, Set

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.hybrid_kv_cache import build_hybrid_kv_cache_spec_descs
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3MtpWeight, KimiK3Weight
from rtp_llm.ops import (
    DataType,
    HybridAttentionType,
    KvCacheDataType,
    KVCacheSpecType,
    QuantAlgo,
    RoleType,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.weight_type import WEIGHT_TYPE


@dataclass(frozen=True)
class KimiK3RuntimeConfig:
    """Checkpoint fields consumed only by the Python Kimi K3 implementation."""

    dense_intermediate_size: int
    routed_expert_hidden_size: int
    num_shared_experts: int
    activation_situ_beta: float
    activation_situ_linear_beta: Optional[float]
    attn_res_block_size: int
    latent_moe_use_norm: bool
    mla_use_nope: bool
    mla_use_output_gate: bool
    kda_gate_lower_bound: Optional[float]
    kda_use_full_rank_gate: bool
    mtp_source_layer: Optional[int] = None
    media_placeholder_token_id: Optional[int] = None


class KimiK3ModelConfig(ModelConfig):
    """RTP model config with a K3-scoped, strongly typed Python extension."""

    _python_fields = ModelConfig._python_fields | {
        "k3_runtime_config",
        "attention_projection_quant_config",
    }
    attention_projection_quant_config = None
    k3_runtime_config: KimiK3RuntimeConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Target and MTP use compact PageRR tables in both P and D roles.
        self.local_kv_cache_sharded = True

    def init_precision_config(self, kv_cache_config, act_type):
        # K3's nested quantization metadata describes only routed experts.
        # The generic loader promotes it to a model-wide scheme, which would
        # quantize ordinary projections as well. Keep this policy model-local.
        is_draft = self.model_type == "kimi_k3_mtp"
        selected = self.config_dtype if is_draft else (act_type or self.config_dtype)
        if not selected or WEIGHT_TYPE.from_str(selected) != WEIGHT_TYPE.BF16:
            raise ValueError("Kimi K3 requires BF16 compute")
        quantization = (self.quantization or "").upper()
        if is_draft and quantization:
            raise ValueError("K3 draft requires BF16 projections")
        if quantization not in ("", "FP8_PER_BLOCK"):
            raise ValueError("K3 supports only FP8_PER_BLOCK projection quantization")
        fp8_gemm = quantization == "FP8_PER_BLOCK"
        fp8_mla = os.environ.get("FP8_MLA", "0") == "1" and not is_draft
        self.quant_algo = QuantAlgo()
        self.quant_config = None
        self.attention_projection_quant_config = Fp8BlockWiseQuantConfig() if fp8_gemm else None
        self.data_type = WEIGHT_TYPE.BF16.to_str()
        self.attn_config.kv_cache_dtype = (
            KvCacheDataType.FP8 if fp8_mla else KvCacheDataType.BASE
        )
        self.attn_config.mla_fp8_compute = fp8_mla
        logging.info(
            "K3 precision: role=%s attention=%s linear=%s cache=%s experts=MXFP4xFP8",
            "draft" if is_draft else "target",
            "FP8" if fp8_mla else "BF16",
            "FP8" if fp8_gemm else "BF16",
            "FP8" if fp8_mla else "BF16",
        )


class KimiK3(BaseModel):
    """Kimi K3 text target using RTP's model and cache interfaces."""

    WEIGHT_PREFIX = "language_model."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(self) is KimiK3:
            parallelism = self.parallelism_config
            cp = parallelism.prefill_cp_config
            local_shards = parallelism.tp_size if cp.kv_cache_sharded else 1
            upstream_shards = (
                max(1, cp.prefill_cp_size)
                if parallelism.role_type == RoleType.DECODE
                else local_shards
            )
            # Keep basic-page indices; retain one checkpoint per CP stripe.
            # Target and MLA-only MTP share this cache config, so scale once.
            self.kv_cache_config.linear_step = max(
                1, self.kv_cache_config.linear_step
            ) * max(local_shards, upstream_shards)

    @classmethod
    def create_dash_sc_request_adapter(cls):
        from rtp_llm.models.kimi_k3.kimi_k3_dash_sc_adapter import (
            KimiK3DashScRequestAdapter,
        )

        return KimiK3DashScRequestAdapter()

    @classmethod
    def create_grammar_format(cls):
        from rtp_llm.models.kimi_k3.kimi_k3_grammar import KimiK3GrammarFormat

        return KimiK3GrammarFormat()

    @classmethod
    def _post_build_model_config(cls, model_config):
        if model_config.local_kv_cache_sharded is not True:
            raise ValueError("K3 MLA requires compact local PageRR cache tables")
        if not model_config.kv_cache_spec_descs:
            model_config.kv_cache_spec_descs = build_hybrid_kv_cache_spec_descs(
                model_config.hybrid_attention_config.hybrid_attention_types,
                KVCacheSpecType.MLA,
            )
        # ModelFactory runs this hook after init_precision_config(), so the
        # hybrid cache descriptors first exist here in the production path.
        cache_descs = model_config.kv_cache_spec_descs
        for layer_descs in cache_descs:
            for desc in layer_descs:
                if desc.cache_type == KVCacheSpecType.LINEAR:
                    # KDA states use head shards, independently of MLA page ownership.
                    desc.head_sharded_transfer = True
                    if model_config.attn_config.mla_fp8_compute:
                        # FP8 MLA does not change recurrent or conv storage.
                        desc.dtype = DataType.TYPE_BF16
                elif desc.cache_type == KVCacheSpecType.MLA and model_config.attn_config.mla_fp8_compute:
                    desc.dtype = DataType.TYPE_FP8_E4M3
        model_config.kv_cache_spec_descs = cache_descs

    @classmethod
    def _create_config(cls, ckpt_path: str) -> KimiK3ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")

        with open(config_path, encoding="utf-8") as reader:
            config_json = json.load(reader)
        config = cls._from_config_json(config_json, ckpt_path)
        if not config.is_mtp:
            from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_config import (
                configure_kimi_k3_multimodal,
            )

            configure_kimi_k3_multimodal(config, config_json)
        return config

    @classmethod
    def _from_config_json(
        cls, config_json: Dict[str, Any], ckpt_path: str = ""
    ) -> KimiK3ModelConfig:
        text_config = config_json.get("text_config")
        if not isinstance(text_config, dict):
            raise ValueError("Kimi K3 config must contain an object-valued text_config")

        model_type = config_json.get("model_type")
        if model_type != "kimi_k3":
            raise ValueError(f"expected outer model_type='kimi_k3', got {model_type!r}")
        if text_config.get("model_type") != "kimi_linear":
            raise ValueError(
                "expected text_config.model_type='kimi_linear', got "
                f"{text_config.get('model_type')!r}"
            )

        config = KimiK3ModelConfig()
        config.ckpt_path = ckpt_path
        config.tokenizer_path = ckpt_path
        config.model_type = "kimi_k3"

        cls._parse_basic_config(config_json, text_config, config)
        cls._parse_attention_config(text_config, config)
        cls._parse_moe_config(text_config, config)
        cls._parse_hybrid_attention_config(text_config, config)
        cls._parse_kimi_runtime_config(text_config, config)
        config.mm_model_config.is_multimodal = False

        logging.info(
            "Kimi K3 text config loaded: layers=%d hidden=%d heads=%d "
            "kda_layers=%d mla_layers=%d experts=%d topk=%d attn_res_block=%d",
            config.num_layers,
            config.hidden_size,
            config.attn_config.head_num,
            sum(
                layer_type == HybridAttentionType.LINEAR
                for layer_type in config.hybrid_attention_config.hybrid_attention_types
            ),
            sum(
                layer_type == HybridAttentionType.NONE
                for layer_type in config.hybrid_attention_config.hybrid_attention_types
            ),
            config.expert_num,
            config.moe_k,
            config.k3_runtime_config.attn_res_block_size,
        )
        return config

    @staticmethod
    def _required(config: Dict[str, Any], name: str) -> Any:
        if name not in config:
            raise ValueError(f"Kimi K3 text_config is missing required field {name!r}")
        return config[name]

    @classmethod
    def _parse_basic_config(
        cls,
        outer_config: Dict[str, Any],
        text_config: Dict[str, Any],
        config: KimiK3ModelConfig,
    ) -> None:
        config.num_layers = int(cls._required(text_config, "num_hidden_layers"))
        config.hidden_size = int(cls._required(text_config, "hidden_size"))
        config.vocab_size = int(cls._required(text_config, "vocab_size"))
        config.input_vocab_size = config.vocab_size
        config.max_seq_len = int(cls._required(text_config, "max_position_embeddings"))
        config.inter_size = int(cls._required(text_config, "intermediate_size"))
        config.layernorm_eps = float(text_config.get("rms_norm_eps", 1e-5))
        config.norm_type = "rmsnorm"
        config.has_pre_decoder_layernorm = False
        config.has_post_decoder_layernorm = True
        config.has_lm_head = True
        # C++ only needs to know that the FFN is gated.  The Python K3 modules
        # implement SiTU instead of dispatching the framework's SiGLU kernel.
        config.activation_type = "SiGLU"
        config.tie_word_embeddings = bool(
            text_config.get(
                "tie_word_embeddings", outer_config.get("tie_word_embeddings", False)
            )
        )
        config.config_dtype = text_config.get(
            "dtype", outer_config.get("dtype", "bfloat16")
        )
        config.enable_fp32_lm_head = False
        config.has_positional_encoding = False
        config.attn_config.rope_config.index_factor = 1
        config.position_ids_style = 0
        config.qk_norm = False

        config.special_tokens.bos_token_id = int(
            outer_config.get("bos_token_id", text_config.get("bos_token_id", -1))
        )
        config.special_tokens.eos_token_id = int(
            outer_config.get("eos_token_id", text_config.get("eos_token_id", 0))
        )
        config.special_tokens.pad_token_id = int(
            outer_config.get("pad_token_id", text_config.get("pad_token_id", 0))
        )
        config.special_tokens.stop_words_id_list = [
            [config.special_tokens.eos_token_id]
        ]

    @classmethod
    def _parse_attention_config(
        cls, text_config: Dict[str, Any], config: KimiK3ModelConfig
    ) -> None:
        head_num = int(cls._required(text_config, "num_attention_heads"))
        qk_nope_head_dim = int(cls._required(text_config, "qk_nope_head_dim"))
        qk_rope_head_dim = int(cls._required(text_config, "qk_rope_head_dim"))

        config.attn_config.head_num = head_num
        config.attn_config.kv_head_num = int(
            text_config.get("num_key_value_heads", head_num)
        )
        config.attn_config.size_per_head = qk_nope_head_dim + qk_rope_head_dim
        config.attn_config.q_lora_rank = int(text_config.get("q_lora_rank") or 0)
        config.attn_config.kv_lora_rank = int(
            cls._required(text_config, "kv_lora_rank")
        )
        config.attn_config.nope_head_dim = qk_nope_head_dim
        # This is a physical 64-d suffix in K3.  The runtime config below marks
        # it as no-RoPE even though existing MLA APIs retain the historical name.
        config.attn_config.rope_head_dim = qk_rope_head_dim
        config.attn_config.v_head_dim = int(cls._required(text_config, "v_head_dim"))
        config.attn_config.use_mla = True
        config.attn_config.is_causal = True
        config.attn_config.rope_config.style = 0
        config.attn_config.rope_config.base = 10000
        config.attn_config.rope_config.dim = qk_rope_head_dim
        config.attn_config.rope_config.offset = qk_nope_head_dim

        linear_config = cls._required(text_config, "linear_attn_config")
        if not isinstance(linear_config, dict):
            raise ValueError("linear_attn_config must be an object")
        linear_heads = int(cls._required(linear_config, "num_heads"))
        if linear_heads != head_num:
            raise ValueError(
                "KDA and MLA head counts must match for hybrid TP: "
                f"KDA={linear_heads}, MLA={head_num}"
            )
        config.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP32
        config.linear_attention_config.conv_state_dtype = DataType.TYPE_BF16
        linear_head_dim = int(cls._required(linear_config, "head_dim"))
        config.linear_attention_config.linear_key_head_dim = linear_head_dim
        config.linear_attention_config.linear_value_head_dim = linear_head_dim
        config.linear_attention_config.linear_num_key_heads = linear_heads
        config.linear_attention_config.linear_num_value_heads = linear_heads
        config.linear_attention_config.replay_vector_gate = True
        config.linear_attention_config.linear_conv_kernel_dim = int(
            linear_config.get("short_conv_kernel_size", 4)
        )

    @classmethod
    def _parse_moe_config(
        cls, text_config: Dict[str, Any], config: KimiK3ModelConfig
    ) -> None:
        if text_config.get("hidden_act") != "situ":
            raise ValueError(
                "Kimi K3 requires hidden_act='situ', got "
                f"{text_config.get('hidden_act')!r}"
            )
        if text_config.get("moe_router_activation_func", "sigmoid") != "sigmoid":
            raise ValueError("Kimi K3 bring-up supports only sigmoid MoE routing")

        config.expert_num = int(cls._required(text_config, "num_experts"))
        config.moe_k = int(cls._required(text_config, "num_experts_per_token"))
        if config.moe_k <= 0 or config.moe_k > config.expert_num:
            raise ValueError(
                f"invalid Kimi K3 top-k {config.moe_k} for {config.expert_num} experts"
            )
        config.moe_inter_size = int(cls._required(text_config, "moe_intermediate_size"))
        config.moe_n_group = int(text_config.get("num_expert_group", 1))
        config.moe_topk_group = int(text_config.get("topk_group", 1))
        config.scoring_func = 1  # sigmoid
        config.routed_scaling_factor = float(
            text_config.get("routed_scaling_factor", 1.0)
        )
        config.has_moe_norm = bool(text_config.get("moe_renormalize", True))
        config.moe_normalize_expert_scale = config.has_moe_norm
        config.moe_style = 2  # shared + routed experts

        num_shared_experts = int(text_config.get("num_shared_experts", 0))
        config.inter_size = max(1, num_shared_experts) * config.moe_inter_size
        first_dense = int(text_config.get("first_k_dense_replace", 0))
        moe_frequency = int(text_config.get("moe_layer_freq", 1))
        if first_dense < 0 or moe_frequency <= 0:
            raise ValueError(
                "first_k_dense_replace must be >= 0 and moe_layer_freq must be > 0"
            )
        config.moe_layer_index = [
            layer_idx
            for layer_idx in range(config.num_layers)
            if layer_idx >= first_dense and layer_idx % moe_frequency == 0
        ]

    @classmethod
    def _parse_hybrid_attention_config(
        cls, text_config: Dict[str, Any], config: KimiK3ModelConfig
    ) -> None:
        linear_config = cls._required(text_config, "linear_attn_config")
        kda_layers = cls._one_based_layer_set(
            linear_config.get("kda_layers", []), "kda_layers", config.num_layers
        )
        mla_layers = cls._one_based_layer_set(
            linear_config.get("full_attn_layers", []),
            "full_attn_layers",
            config.num_layers,
        )
        overlap = kda_layers & mla_layers
        if overlap:
            raise ValueError(
                f"KDA and MLA layer schedules overlap at 1-based layers {sorted(overlap)}"
            )
        expected = set(range(1, config.num_layers + 1))
        missing = expected - kda_layers - mla_layers
        if missing:
            raise ValueError(
                "KDA/MLA schedules must cover every text layer; missing 1-based "
                f"layers {sorted(missing)}"
            )

        config.hybrid_attention_config.enable_hybrid_attention = True
        # The main Cache builder separates MLA and KDA by the "full" and
        # "linear" descriptor tags assigned in _post_build_model_config.
        layer_types: List[HybridAttentionType] = []
        for layer_1based in range(1, config.num_layers + 1):
            layer_types.append(
                HybridAttentionType.LINEAR
                if layer_1based in kda_layers
                else HybridAttentionType.NONE
            )
        config.hybrid_attention_config.hybrid_attention_types = layer_types

    @staticmethod
    def _one_based_layer_set(
        values: Iterable[Any], name: str, num_layers: int
    ) -> Set[int]:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must contain integer 1-based layer ids")
        try:
            raw_layers = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must contain integer 1-based layer ids") from exc
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_layers
        ):
            raise ValueError(f"{name} must contain integer 1-based layer ids")
        layers = set(raw_layers)
        invalid = sorted(layer for layer in layers if layer < 1 or layer > num_layers)
        if invalid:
            raise ValueError(
                f"{name} contains out-of-range 1-based layers {invalid}; "
                f"valid range is [1, {num_layers}]"
            )
        return layers

    @classmethod
    def _parse_kimi_runtime_config(
        cls, text_config: Dict[str, Any], config: KimiK3ModelConfig
    ) -> None:
        quant = cls._required(text_config, "quantization_config")
        groups = quant.get("config_groups", {})
        if (
            quant.get("format") != "mxfp4-pack-quantized"
            or not groups
            or any(
                group.get("weights", {}).get("num_bits") != 4
                or group.get("weights", {}).get("group_size") != 32
                for group in groups.values()
            )
        ):
            raise ValueError("K3 requires checkpoint-native group-32 MXFP4 experts")
        linear_config = cls._required(text_config, "linear_attn_config")
        attn_res_block_size = int(cls._required(text_config, "attn_res_block_size"))
        if attn_res_block_size <= 0:
            raise ValueError("attn_res_block_size must be > 0")

        # SiTU betas are model-defining constants (K3 uses 4.0 / 25.0); a wrong
        # default silently corrupts the activation, so require them explicitly.
        # ``activation_situ_linear_beta`` may still be an explicit null, which
        # legitimately disables the up-projection tanh clamp.
        linear_beta = cls._required(text_config, "activation_situ_linear_beta")
        gate_lower_bound = cls._required(linear_config, "gate_lower_bound")
        if gate_lower_bound is None or float(gate_lower_bound) >= 0:
            raise ValueError("K3 requires a negative bounded-KDA gate_lower_bound")
        if text_config.get("hidden_act") != "situ":
            raise ValueError("K3 requires SiTU activation")
        if float(cls._required(text_config, "activation_situ_beta")) <= 0:
            raise ValueError("SiTU gate beta must be positive")
        if linear_beta is not None and float(linear_beta) <= 0:
            raise ValueError("SiTU linear beta must be positive or null")
        config.k3_runtime_config = KimiK3RuntimeConfig(
            dense_intermediate_size=int(
                cls._required(text_config, "intermediate_size")
            ),
            routed_expert_hidden_size=int(
                cls._required(text_config, "routed_expert_hidden_size")
            ),
            num_shared_experts=int(text_config.get("num_shared_experts", 0)),
            activation_situ_beta=float(
                cls._required(text_config, "activation_situ_beta")
            ),
            activation_situ_linear_beta=(
                None if linear_beta is None else float(linear_beta)
            ),
            attn_res_block_size=attn_res_block_size,
            latent_moe_use_norm=bool(text_config.get("latent_moe_use_norm", False)),
            mla_use_nope=bool(cls._required(text_config, "mla_use_nope")),
            mla_use_output_gate=bool(cls._required(text_config, "mla_use_output_gate")),
            kda_gate_lower_bound=(
                None if gate_lower_bound is None else float(gate_lower_bound)
            ),
            kda_use_full_rank_gate=bool(linear_config.get("use_full_rank_gate", False)),
        )

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self):
        if isinstance(self, KimiK3Mtp):
            from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel

            model_cls = KimiK3MtpModel
        else:
            from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model

            model_cls = KimiK3Model

        config = self.model_config
        runtime = self.runtime_config
        hw = self.hw_kernel_config
        tp = int(self.parallelism_config.tp_size)
        role = self.parallelism_config.role_type
        prefill = 0
        if role != RoleType.DECODE:
            # Direct model construction has no scheduler; retain its explicit budget.
            prefill = max(config.max_seq_len, config.moe_prefill_max_tokens_per_rank)
            if runtime is not None:
                scheduler = runtime.fifo_scheduler_config
                prefill = (
                    scheduler.prefill_chunk_size
                    if scheduler.prefill_chunk_size > 0
                    else max(config.max_seq_len, scheduler.max_batch_tokens_size)
                )
            if hw.enable_cuda_graph and role == RoleType.PDFUSION:
                prefill = max([prefill, *hw.generation_prefill_capture_token_buckets])

        if runtime is not None:
            scheduler = runtime.fifo_scheduler_config
            prefill_warmup = role != RoleType.DECODE
            if runtime.use_batch_decode_scheduler:
                prefill_warmup = (
                    runtime.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type
                    != 0
                )
            if (
                runtime.warm_up
                and prefill_warmup
                and (scheduler.prefill_chunk_size <= 0 or runtime.warm_up_with_loss)
                and not config.mm_model_config.is_multimodal
                and not self.parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate
            ):
                # Full-length warmup bypasses chunking; max_seq_len covers both
                # ordinary and speculative warmup without a separate reserve rule.
                prefill = max(
                    prefill, config.max_seq_len * scheduler.max_context_batch_size
                )

        # SP pads complete requests. Cover decode, verify and MTP update, including
        # capture buckets larger than the serving batch limit. K3 has no query CP.
        batch = self.max_generate_batch_size
        if hw.enable_cuda_graph:
            batch = max([batch, *hw.decode_capture_batch_sizes])
        decode = 0
        for width in (1, max(int(config.gen_num_per_cycle) + 1, 1)):
            request_alignment = tp // gcd(tp, width)
            decode = max(
                decode,
                (batch + request_alignment - 1)
                // request_alignment
                * request_alignment
                * width,
            )
        # Resident prompts were tokenized by BaseModel and execute without chunks.
        prompt = max(
            map(len, self.kv_cache_config.multi_task_prompt_tokens.values()), default=0
        )
        max_moe_tokens = (max(prefill, decode, prompt, 1) + tp - 1) // tp * tp

        # Release inactive blocks left by checkpoint loading before the expert
        # weight transform allocates its large temporary buffer.
        if not isinstance(self, KimiK3Mtp):
            torch.cuda.empty_cache()

        self.py_model = model_cls(
            self.model_config,
            self.parallelism_config,
            self.weight,
            max_generate_batch_size=self.max_generate_batch_size,
            max_moe_tokens=max_moe_tokens,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
            moe_config=self.moe_config,
        )
        # Release rank-dependent inactive blocks before the native engine
        # allocates KV cache and workspaces.
        if not isinstance(self, KimiK3Mtp):
            torch.cuda.empty_cache()
        return self.py_model

    @staticmethod
    def get_weight_cls():
        return KimiK3Weight


class KimiK3Mtp(KimiK3):
    """The independent, recurrent K3 MTP layer from a draft-only checkpoint."""

    @classmethod
    def speculative_weight_alias_names(cls, target_model, draft_model_config):
        if type(target_model) is not KimiK3:
            raise TypeError("Native K3 MTP requires a K3 target owner")
        for field in ("vocab_size", "hidden_size", "data_type", "enable_fp32_lm_head"):
            if getattr(target_model.model_config, field) != getattr(
                draft_model_config, field
            ):
                raise ValueError(f"K3 target/draft cannot share incompatible {field}")
        media_id = target_model.model_config.mm_related_params.special_token_ids.get(
            "image_token_index"
        )
        if media_id is not None:
            if (
                not isinstance(media_id, int)
                or isinstance(media_id, bool)
                or not 0 <= media_id < draft_model_config.input_vocab_size
            ):
                raise ValueError("K3 target media token must be within the vocabulary")
            draft_ids = draft_model_config.mm_related_params.special_token_ids
            draft_runtime = draft_model_config.k3_runtime_config
            for draft_media_id in (
                draft_ids.get("image_token_index"),
                draft_runtime.media_placeholder_token_id,
            ):
                if draft_media_id is not None and (
                    not isinstance(draft_media_id, int)
                    or isinstance(draft_media_id, bool)
                    or draft_media_id != media_id
                ):
                    raise ValueError("K3 target/draft media token IDs must match")
            draft_ids["image_token_index"] = media_id
            draft_model_config.k3_runtime_config = replace(
                draft_runtime, media_placeholder_token_id=media_id
            )
        return (W.embedding, W.lm_head)

    @classmethod
    def _from_config_json(cls, config_json, ckpt_path=""):
        text = config_json.get("text_config", {})
        if config_json.get("model_type") != "kimi_k3":
            raise ValueError("K3 MTP requires a kimi_k3 checkpoint")
        source_layer = text.get("num_hidden_layers")
        if type(source_layer) is not int or source_layer < 1:
            raise ValueError("K3 MTP requires a positive num_hidden_layers")
        if text.get("num_nextn_predict_layers") != 1:
            raise ValueError("K3 MTP currently supports one recurrent nextn layer")
        schedule = text.get("linear_attn_config", {})
        # Checkpoint attention schedules use one-based layer numbers.
        schedule_layer = source_layer + 1
        if schedule_layer not in schedule.get(
            "full_attn_layers", []
        ) or schedule_layer in schedule.get("kda_layers", []):
            raise ValueError(f"K3 MTP checkpoint layer {source_layer} must be full MLA")
        config = KimiK3ModelConfig()
        config.ckpt_path = ckpt_path
        config.tokenizer_path = ckpt_path
        config.model_type = "kimi_k3_mtp"
        config.is_mtp = True
        cls._parse_basic_config(config_json, text, config)
        cls._parse_attention_config(text, config)
        cls._parse_moe_config(text, config)
        cls._parse_kimi_runtime_config(text, config)
        if source_layer not in config.moe_layer_index:
            # The target schedule excludes MTP; evaluate its source layer explicitly.
            if source_layer < int(
                text.get("first_k_dense_replace", 0)
            ) or source_layer % int(text.get("moe_layer_freq", 1)):
                raise ValueError("K3 MTP source layer must be MoE")
        config.num_layers = 1
        config.moe_layer_index = [0]
        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.NONE
        ]
        media_token_id = config_json.get("media_placeholder_token_id")
        if media_token_id is not None and (
            not isinstance(media_token_id, int)
            or isinstance(media_token_id, bool)
            or not 0 <= media_token_id < config.input_vocab_size
        ):
            raise ValueError(
                "K3 MTP media_placeholder_token_id must be a vocabulary ID"
            )
        config.k3_runtime_config = replace(
            config.k3_runtime_config,
            attn_res_block_size=0,
            mtp_source_layer=source_layer,
            media_placeholder_token_id=media_token_id,
        )
        config.mm_model_config.is_multimodal = False
        if media_token_id is not None:
            # The draft uses the media token's vocabulary embedding without
            # constructing its own ViT or consuming external image features.
            config.mm_related_params.special_token_ids["image_token_index"] = (
                media_token_id
            )
        if not config.k3_runtime_config.mla_use_nope:
            raise ValueError("K3 MTP requires NoPE MLA")
        return config

    @staticmethod
    def get_weight_cls():
        return KimiK3MtpWeight


register_model("kimi_k3_mtp", KimiK3Mtp, ["KimiK3MTPModel"])

register_model(
    "kimi_k3",
    KimiK3,
    ["KimiK3ForConditionalGeneration"],
)
