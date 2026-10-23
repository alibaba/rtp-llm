import json
from pathlib import Path
from typing import Dict, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.qwen4_exp_config import (
    build_qwen4_quant_config,
    freeze_runtime_config,
    validate_qwen4_runtime,
)
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Moe
from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    build_qwen4_exp_kv_cache_spec_descs,
)
from rtp_llm.models.qwen4_exp.qwen4_exp_weight import Qwen4ExpWeight
from rtp_llm.ops import HybridAttentionType

_LAYER_TYPE_TO_HYBRID_ATTENTION: Dict[str, HybridAttentionType] = {
    "linear_attention": HybridAttentionType.LINEAR,
    # NONE means "not hybrid", i.e. a complete attention layer.
    "full_attention": HybridAttentionType.NONE,
}


class Qwen4Exp(Qwen35Moe):
    """Qwen3.8-Flash-Next (``Qwen4ExpForConditionalGeneration``).

    Shares the Qwen3.5 multimodal shell and GDN linear-attention layers. Gated
    residual and checkpoint-required PLE/QSA paths are always connected.
    """

    @staticmethod
    def get_weight_cls():
        return Qwen4ExpWeight

    def support_cuda_graph(self) -> bool:
        # Capability only; the common graph setting and exact-batch metadata
        # decide capture. CPU PLE rows are prepared before capture/replay.
        return getattr(self, "model_config", None) is not None

    def load(self, skip_python_model: bool = False):
        config = getattr(self, "model_config", None)
        validate_qwen4_runtime(config, self.parallelism_config)
        freeze_runtime_config(config, self.parallelism_config)
        return super().load(skip_python_model=skip_python_model)

    def _py_model_class(self):
        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel

        return Qwen4ExpModel

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        config.model_type = "qwen4_exp"
        root = Path(ckpt_path)
        config_json = json.loads((root / "config.json").read_text())
        source_quant = config_json.get("quantization_config")
        config._qwen4_fp8_source = bool(source_quant)
        if source_quant:
            from rtp_llm.models.qwen4_exp.moe_fp8 import validate_moe_fp8_checkpoint

            index = json.loads((root / "model.safetensors.index.json").read_text())
            validate_moe_fp8_checkpoint(config_json, index["weight_map"], root)
        cls._configure_precision_recipe(config)
        return config

    @staticmethod
    def _configure_precision_recipe(config: ModelConfig) -> None:
        config._qwen4_full_fp8 = bool(getattr(config, "_qwen4_fp8_source", False))
        config.quant_config = build_qwen4_quant_config(
            config.quantization, config._qwen4_full_fp8
        )
        config._qwen4_full_fp8_nvfp4 = bool(
            config.quant_config is not None
            and config.quant_config.get_method() == "modelopt_fp4"
        )
        if config.quant_config is not None:
            quant = config.quant_config
            config.quant_algo.setQuantAlgo(
                quant.get_algo(), quant.bits, quant.group_size()
            )

    @classmethod
    def _parse_extra_text_config(cls, config_json: dict, config: ModelConfig) -> None:
        # hc_mult is the engine-wide name for the internal residual stream width.
        # MTP's cross-model input width is configured independently below. The
        # low-rank dim is inferred from the weights.
        config.hc_mult = config_json["hc_count"]
        # The released MTP head consumes the target residual before the final
        # hyper-connection mixer.  Each row therefore retains all hc branches;
        # the draft applies fc_hidden independently to every branch.
        config.mtp_input_hidden_size = config.hidden_size * config.hc_mult
        # Linear-attention output gate activation; upstream falls back to hidden_act.
        config.linear_attn_norm_activation = config_json.get(
            "output_gate_type"
        ) or config_json.get("hidden_act", "silu")
        config._qwen4_ple_layer_ids = list(config_json.get("ple_layer_ids", []))
        config._qwen4_ngram_size = int(config_json.get("ngram_size", 0))
        config._qwen4_ple_conv_kernel_size = int(
            config_json.get("ple_conv_kernel_size", 0)
        )
        config._qwen4_split_ngram_parts = int(config_json.get("split_ngram_parts", 0))
        config._qwen4_ngram_vocab_size_base = int(
            config_json.get("ngram_vocab_size_base", 0)
        )
        config._qwen4_ple_embed_dim = int(config_json.get("ple_embed_dim", 0))
        config._qwen4_heads_per_ngram = int(config_json.get("heads_per_ngram", 8))
        config._extra_weight_bytes = 0.0
        config._tp_sharded_extra_weight_bytes = 0.0
        config._replicated_extra_weight_bytes = 0.0
        config._extra_weight_param_count = 0
        cls._set_ple_weight_size_estimate(config)
        # QSA indexer. ``indexer_topk`` counts *blocks*, not tokens: selection is
        # block-wise over ``compress_ratio``-sized groups, so budget/ratio blocks
        # expand back to budget tokens plus the always-kept partial tail.
        head_dim = int(config_json.get("indexer_head_dim", 0))
        ratio = int(config_json.get("indexer_compress_ratio", 0))
        config._qwen4_indexer_head_dim = head_dim
        config._qwen4_indexer_compress_ratio = ratio
        config._qwen4_indexer_budget = int(config_json.get("indexer_budget", 0))
        config._qwen4_indexer_kv_heads = int(config_json.get("indexer_kv_heads", 0))
        if head_dim > 0:
            if ratio <= 0:
                raise ValueError(
                    f"qwen4_exp indexer_head_dim={head_dim} enables the indexer but "
                    f"indexer_compress_ratio is {ratio}"
                )
            config.attn_config.is_sparse = True
            config.attn_config.use_sparse_gqa_fmha = True
            config.attn_config.indexer_head_dim = head_dim
            config.attn_config.indexer_head_num = int(config_json["indexer_n_heads"])
            config.attn_config.indexer_topk = config._qwen4_indexer_budget // ratio

    @staticmethod
    def _set_ple_weight_size_estimate(config: ModelConfig) -> None:
        """Account for the TP-sharded BF16 PLE tensors omitted by generic formulas."""
        ngram_size = int(config._qwen4_ngram_size)
        vocab_base = int(config._qwen4_ngram_vocab_size_base)
        embed_dim = int(config._qwen4_ple_embed_dim)
        heads_per_ngram = int(config._qwen4_heads_per_ngram)
        ple_layers = len(config._qwen4_ple_layer_ids)
        if min(ngram_size - 1, vocab_base, embed_dim, heads_per_ngram, ple_layers) <= 0:
            raise ValueError("qwen4_exp PLE weight-size geometry is incomplete")
        ngram_heads = (ngram_size - 1) * heads_per_ngram
        if embed_dim % ngram_heads:
            raise ValueError(
                f"qwen4_exp ple_embed_dim={embed_dim} is not divisible by "
                f"ngram_heads={ngram_heads}"
            )

        # The checkpoint chooses a nearby prime for every head and pads the
        # concatenated table to split_ngram_parts. The generic estimator is
        # intentionally approximate, so use the config's nominal base here;
        # for the released checkpoint the difference is below 0.001%.
        table_params = vocab_base * embed_dim
        hc_hidden = int(config.hc_mult) * int(config.hidden_size)
        auxiliary_bf16_params = (
            hc_hidden * embed_dim  # key projection
            + int(config.hidden_size) * embed_dim  # value projection
            + hc_hidden * int(config._qwen4_ple_conv_kernel_size)
            + 3 * hc_hidden  # norm key/query/conv
        )
        table_params *= ple_layers
        auxiliary_bf16_params *= ple_layers
        metadata_params = ple_layers * (ngram_size + 2 * ngram_heads)
        config._extra_weight_param_count = (
            table_params + auxiliary_bf16_params + metadata_params
        )
        table_bytes = float(
            table_params * (1 if getattr(config, "_qwen4_full_fp8", False) else 2)
        )
        replicated_bytes = float(auxiliary_bf16_params * 2 + metadata_params * 8)
        config._extra_weight_bytes = table_bytes + replicated_bytes
        # Size/parameter APIs describe the full checkpoint independently of
        # loader residency. PLE tables reside in pinned host memory; auxiliary
        # weights remain replicated on the execution device.
        if config.enable_qwen4_ple:
            config._replicated_extra_weight_bytes = replicated_bytes

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        # Generic precision initialization reads source checkpoint metadata;
        # finalize the complete source-to-runtime recipe after that stage.
        cls._configure_precision_recipe(model_config)
        if not model_config.is_mtp:
            cls._set_ple_weight_size_estimate(model_config)
        if (
            type(model_config.qwen4_ple_prefetch_mb) is not int
            or model_config.qwen4_ple_prefetch_mb <= 0
        ):
            raise ValueError("qwen4_ple_prefetch_mb must be a positive integer")
        ple_layer_ids = (
            getattr(model_config, "_qwen4_ple_layer_ids", [])
            if model_config.enable_qwen4_ple
            else []
        )
        indexer_head_dim = (
            getattr(model_config, "_qwen4_indexer_head_dim", 0)
            if model_config.enable_qwen4_qsa
            else 0
        )
        if not ple_layer_ids and not indexer_head_dim:
            super()._post_build_model_config(model_config)
            return
        # ple_layer_ids are 1-based (upstream indexes them as layer_idx + 1).
        ple_layer_indices = [layer_id - 1 for layer_id in ple_layer_ids]
        model_config.kv_cache_spec_descs = build_qwen4_exp_kv_cache_spec_descs(
            model_config,
            ple_layer_indices,
            model_config._qwen4_ple_conv_kernel_size,
            model_config._qwen4_ngram_size,
            indexer_head_dim,
            getattr(model_config, "_qwen4_indexer_compress_ratio", 0),
        )

    @classmethod
    def _parse_hybrid_attention_config(cls, config_json: dict, config: ModelConfig):
        layer_types = config_json.get("layer_types")
        if not layer_types:
            super()._parse_hybrid_attention_config(config_json, config)
            return

        if len(layer_types) != config.num_layers:
            raise ValueError(
                f"qwen4_exp layer_types has {len(layer_types)} entries but the model "
                f"has {config.num_layers} layers"
            )

        hybrid_layer_types: List[HybridAttentionType] = []
        for idx, layer_type in enumerate(layer_types):
            if layer_type not in _LAYER_TYPE_TO_HYBRID_ATTENTION:
                raise ValueError(
                    f"qwen4_exp layer_types[{idx}] is {layer_type!r}, expected one of "
                    f"{sorted(_LAYER_TYPE_TO_HYBRID_ATTENTION)}"
                )
            hybrid_layer_types.append(_LAYER_TYPE_TO_HYBRID_ATTENTION[layer_type])

        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.hybrid_attention_types = hybrid_layer_types


register_model("qwen4_exp", Qwen4Exp, ["Qwen4ExpForConditionalGeneration"])
