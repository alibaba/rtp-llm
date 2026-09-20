import logging
import os
from typing import Dict, List

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Moe
from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    build_qwen4_exp_kv_cache_spec_descs,
)
from rtp_llm.models.qwen4_exp.qwen4_exp_weight import Qwen4ExpWeight
from rtp_llm.ops import DataType, HybridAttentionType, KvCacheDataType, RoleType
from rtp_llm.utils.util import str_to_bool

_LAYER_TYPE_TO_HYBRID_ATTENTION: Dict[str, HybridAttentionType] = {
    "linear_attention": HybridAttentionType.LINEAR,
    # NONE means "not hybrid", i.e. a complete attention layer.
    "full_attention": HybridAttentionType.NONE,
}

_SKELETON_WARNING = (
    "qwen4_exp is incomplete: output will be wrong. The PLE / n-gram layer is "
    "available only through its restricted correctness path, while QSA supports a "
    "restricted text-only MTP path in addition to prefill and ordinary decode. See docs/design/"
    "qwen3.8_flash_next_support_design.md appendix C."
)

_EXPERIMENTAL_SERVING_ENV = "RTP_LLM_ENABLE_QWEN4_EXP_EXPERIMENTAL"
_PLE_ENV = "RTP_LLM_ENABLE_QWEN4_EXP_PLE"
_QSA_ENV = "RTP_LLM_ENABLE_QWEN4_EXP_QSA"


class Qwen4Exp(Qwen35Moe):
    """Qwen3.8-Flash-Next (``Qwen4ExpForConditionalGeneration``).

    Shares the Qwen3.5 multimodal shell and GDN linear-attention layers. Gated
    residual plus restricted PLE/QSA correctness paths are connected. Serving
    remains disabled unless the model-level experimental gate is explicit.
    """

    @staticmethod
    def get_weight_cls():
        return Qwen4ExpWeight

    def support_cuda_graph(self) -> bool:
        # The input-hidden width is now explicit, but Qwen4's graph contract is
        # still incomplete: draft-prefill capacity handling infers HC output
        # semantics from hc_mult, and QSA/PLE graph state is not supported.
        return False

    @staticmethod
    def _experimental_serving_enabled() -> bool:
        """Return whether the known-incomplete serving path may run.

        The guard is evaluated by ``load`` rather than config parsing so config
        and checkpoint-manifest tests remain usable. It runs before any weight
        tensor is materialised, including the 102-GB PLE table.
        """
        return str_to_bool(os.environ.get(_EXPERIMENTAL_SERVING_ENV, "false"))

    def load(self, skip_python_model: bool = False):
        if not self._experimental_serving_enabled():
            raise RuntimeError(
                "qwen4_exp serving is incomplete and disabled by default: the "
                "restricted PLE/QSA correctness paths are not production-ready. Set "
                f"{_EXPERIMENTAL_SERVING_ENV}=true only for isolated development."
            )
        # Use the already-parsed config instead of re-reading the environment:
        # changing the process environment after config construction must not
        # change which model implementation is loaded.
        config = getattr(self, "model_config", None)
        if getattr(config, "enable_qwen4_qsa", False):
            if getattr(config, "data_type", None) != DataType.TYPE_BF16:
                raise RuntimeError(
                    "qwen4_exp QSA currently requires act_type=BF16; refusing "
                    "before indexer weights are loaded"
                )
            quant_config = getattr(config, "quant_config", None)
            is_quanted = getattr(quant_config, "is_quanted", None)
            if quant_config is not None and (
                not callable(is_quanted) or bool(is_quanted())
            ):
                raise RuntimeError(
                    "qwen4_exp QSA does not support quantized indexer weights"
                )
            attn_config = getattr(config, "attn_config", None)
            if not bool(getattr(attn_config, "is_sparse", False)) or not bool(
                getattr(attn_config, "use_sparse_gqa_fmha", False)
            ):
                raise RuntimeError(
                    "qwen4_exp QSA routing flags are inconsistent; dense fallback "
                    "is forbidden"
                )
            kv_cache_dtype = getattr(
                attn_config, "kv_cache_dtype", KvCacheDataType.BASE
            )
            if kv_cache_dtype != KvCacheDataType.BASE:
                raise RuntimeError(
                    "qwen4_exp QSA paged attention requires the base BF16 KV cache"
                )
            parallelism = getattr(self, "parallelism_config", None)
            role_type = getattr(parallelism, "role_type", RoleType.PDFUSION)
            if role_type in (RoleType.PREFILL, RoleType.DECODE):
                raise RuntimeError(
                    "qwen4_exp QSA does not support PD-separated PREFILL/DECODE "
                    "roles; refusing before indexer weights are loaded"
                )
            cp = getattr(parallelism, "prefill_cp_config", None)
            if cp is not None and (cp.is_enabled() or cp.is_prefill_enabled()):
                raise RuntimeError(
                    "qwen4_exp QSA does not support context/prefill parallelism; "
                    "refusing before indexer weights are loaded"
                )
        if getattr(config, "enable_qwen4_ple", False) and (
            getattr(config, "data_type", None) != DataType.TYPE_BF16
        ):
            raise RuntimeError(
                "qwen4_exp PLE currently requires act_type=BF16; refusing to "
                "load its rank-local n-gram table for an incompatible dtype"
            )
        if getattr(config, "enable_qwen4_ple", False):
            parallelism = getattr(self, "parallelism_config", None)
            role_type = getattr(parallelism, "role_type", RoleType.PDFUSION)
            if role_type in (RoleType.PREFILL, RoleType.DECODE):
                raise RuntimeError(
                    "qwen4_exp PLE does not support PD-separated PREFILL/DECODE "
                    "roles; refusing before its rank-local n-gram table is loaded"
                )
            cp = getattr(parallelism, "prefill_cp_config", None)
            if cp is not None and (cp.is_enabled() or cp.is_prefill_enabled()):
                raise RuntimeError(
                    "qwen4_exp PLE does not support context/prefill parallelism; "
                    "refusing before its rank-local n-gram table is loaded"
                )
            # Match ModelDeployWeightInfo/LoadConfig exactly. In particular,
            # ALL_GATHER context parallelism changes attention TP to one; using
            # raw tp_size here would let the 102-GB table start loading before
            # its descriptor rejected the effective topology.
            tp_size = (
                int(parallelism.get_attn_tp_size()) if parallelism is not None else 0
            )
            shard_count = int(getattr(config, "_qwen4_split_ngram_parts", 0))
            if tp_size <= 1 or shard_count <= 0 or shard_count % tp_size:
                raise RuntimeError(
                    "qwen4_exp PLE requires effective attention TP > 1 and an "
                    "even rank-local shard split, got "
                    f"split_ngram_parts={shard_count}, attn_tp_size={tp_size}"
                )
        logging.warning(
            "%s=true bypasses the qwen4_exp incomplete-model safety gate",
            _EXPERIMENTAL_SERVING_ENV,
        )
        return super().load(skip_python_model=skip_python_model)

    def _py_model_class(self):
        from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel

        return Qwen4ExpModel

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = super()._create_config(ckpt_path)
        logging.warning(_SKELETON_WARNING)
        return config

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
        # Keep unfinished subsystems disabled independently from the model-level
        # experimental gate. This prevents dense-fallback development from
        # allocating unused side pools or loading the 102-GB PLE table.
        config.enable_qwen4_ple = str_to_bool(os.environ.get(_PLE_ENV, "false"))
        config.enable_qwen4_qsa = str_to_bool(os.environ.get(_QSA_ENV, "false"))
        config.attn_config.use_sparse_gqa_fmha = config.enable_qwen4_qsa
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
            config.attn_config.is_sparse = config.enable_qwen4_qsa
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
        table_bytes = float(table_params * 2)
        replicated_bytes = float(auxiliary_bf16_params * 2 + metadata_params * 8)
        config._extra_weight_bytes = table_bytes + replicated_bytes
        # Size/parameter APIs describe the full checkpoint independently of
        # feature gates. Resident loader memory only includes PLE when its
        # execution path is explicitly enabled.
        if config.enable_qwen4_ple:
            config._tp_sharded_extra_weight_bytes = table_bytes
            config._replicated_extra_weight_bytes = replicated_bytes

    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
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
        # More than one region per layer routes cache config through the DSv4
        # independent-pool path (HybridPoolConfigCreator).
        hybrid = model_config.hybrid_attention_config
        hybrid.enable_independent_kv_cache_pools = True

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
