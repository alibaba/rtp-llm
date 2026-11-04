"""K3.1 configuration adapter over the shared K3 decoder and weight loader."""

from dataclasses import dataclass, replace
import copy
import json
import logging

from rtp_llm.config.dsa_config import DsaConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.kimi_k3.kimi_k3 import (
    KimiK3, KimiK3Mtp, KimiK3ModelConfig, KimiK3RuntimeConfig,
)
from rtp_llm.models.kimi_k3.kimi_k31_weight import KimiK31Weight, KimiK31MtpWeight
from rtp_llm.ops import (
    DataType,
    HybridAttentionType,
    KVCacheSpecDesc,
    KVCacheSpecType,
    KvCacheDataType,
    OpaqueBlockEntryCountMode,
    CPRotateMethod,
)


@dataclass(frozen=True)
class KimiK31RuntimeConfig(KimiK3RuntimeConfig):
    dsa: DsaConfig = None


class KimiK31ModelConfig(KimiK3ModelConfig):
    _python_fields = KimiK3ModelConfig._python_fields | {"indexer_projection_quant_config"}
    indexer_projection_quant_config = None

    def _bind_indexer_precision(self):
        enabled = self.k3_runtime_config.dsa.indexer_use_fp8
        if not enabled and not self.is_mtp:
            raise ValueError("K3.1 target requires FP8 Indexer; BF16 is draft-only")
        if enabled and self.model_type == "kimi_k31_mtp":
            raise ValueError("K3.1 synthetic draft Indexer must remain BF16")
        if enabled and self.attn_config.mla_fp8_compute:
            raise ValueError("FP8 Indexer requires BF16 SFA compute")
        self.indexer_projection_quant_config = Fp8BlockWiseQuantConfig() if enabled else None

    def init_precision_config(self, kv_cache_config, act_type):
        super().init_precision_config(kv_cache_config, act_type)
        if not self.is_mtp:
            self.attn_config.kv_cache_dtype = (
                KvCacheDataType.FP8
                if kv_cache_config is not None and kv_cache_config.fp8_kv_cache
                else KvCacheDataType.BASE
            )
        self._bind_indexer_precision()
        logging.info(
            "K3.1 precision: projections=%s indexer=%s sfa_compute=BF16 "
            "sfa_cache=%s absorb_bmm=BF16 kda_recurrent=FP32 conv_state=BF16",
            "FP8" if self.attention_projection_quant_config is not None else "BF16",
            "FP8" if self.indexer_projection_quant_config is not None else "BF16",
            "fp8_ds_mla" if self.attn_config.kv_cache_dtype == KvCacheDataType.FP8 else "BF16",
        )

    def apply_override_args(self, json_model_override_args):
        args = json.loads(json_model_override_args)
        if "indexer_use_fp8" in args:
            enabled = args["indexer_use_fp8"]
            if type(enabled) is not bool:
                raise ValueError("indexer_use_fp8 override must be a boolean")
            if not enabled and not self.is_mtp:
                raise ValueError("K3.1 target requires FP8 Indexer; BF16 is draft-only")
            if enabled and self.model_type == "kimi_k31_mtp":
                raise ValueError("K3.1 synthetic draft Indexer must remain BF16")
            if enabled and self.attn_config.mla_fp8_compute:
                raise ValueError("FP8 Indexer requires BF16 SFA compute")
            self.k3_runtime_config = replace(
                self.k3_runtime_config,
                dsa=replace(self.k3_runtime_config.dsa, indexer_use_fp8=enabled),
            )
            self._bind_indexer_precision()
        super().apply_override_args(json_model_override_args)


class KimiK31(KimiK3):
    HF_MODEL_TYPE = "kimi_k31"
    MODEL_TYPE = "kimi_k31"
    CONFIG_CLASS = KimiK31ModelConfig

    @staticmethod
    def get_weight_cls():
        return KimiK31Weight

    @classmethod
    def _dsa_config(cls, text):
        return DsaConfig.from_kimi_config(text, allow_mtp=cls.MODEL_TYPE == "kimi_k31_mtp")

    @classmethod
    def _parse_attention_config(cls, text, config):
        super()._parse_attention_config(text, config)
        dsa = cls._dsa_config(text)
        config.attn_config.is_sparse = True
        config.attn_config.indexer_head_num = dsa.index_num_heads
        config.attn_config.indexer_head_dim = dsa.index_head_dim
        config.attn_config.indexer_topk = dsa.index_topk

    @classmethod
    def _parse_kimi_runtime_config(cls, text, config):
        super()._parse_kimi_runtime_config(text, config)
        config.k3_runtime_config = KimiK31RuntimeConfig(
            **vars(config.k3_runtime_config),
            dsa=replace(cls._dsa_config(text), indexer_use_fp8=cls.MODEL_TYPE != "kimi_k31_mtp"),
        )

    @classmethod
    def _post_build_model_config(cls, config):
        super()._post_build_model_config(config)
        dsa = config.k3_runtime_config.dsa
        descs = config.kv_cache_spec_descs
        for layer, layer_descs in enumerate(descs):
            for desc in layer_descs:
                if desc.cache_type == KVCacheSpecType.MLA:
                    desc.dtype = (
                        DataType.TYPE_FP8_E4M3
                        if config.attn_config.kv_cache_dtype == KvCacheDataType.FP8
                        else DataType.TYPE_BF16
                    )
                elif desc.cache_type == KVCacheSpecType.LINEAR:
                    desc.dtype = DataType.TYPE_BF16
            if not dsa.has_layer(layer):
                continue
            if config.hybrid_attention_config.hybrid_attention_types[layer] == HybridAttentionType.LINEAR:
                raise ValueError("DSA Indexer cache cannot be attached to a KDA layer")
            existing = next((desc for desc in layer_descs if desc.tag == "indexer_kv"), None)
            if existing is not None:
                # Runtime precision overrides also update an existing descriptor;
                # never leave BF16 capacity attached to an FP8 cache codec.
                existing.entry_elems = dsa.index_cache_entry_bytes
                continue
            desc = KVCacheSpecDesc()
            desc.tag = "indexer_kv"
            desc.cache_type = KVCacheSpecType.OPAQUE_KV
            # The opaque ABI is byte addressed for BF16 as well as FP8.
            desc.entry_dtype = DataType.TYPE_UINT8
            desc.entry_elems = dsa.index_cache_entry_bytes
            desc.entry_count_mode = OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED
            desc.compression_ratio = 1
            layer_descs.append(desc)
        config.kv_cache_spec_descs = descs


register_model("kimi_k31", KimiK31, ["KimiK31ForConditionalGeneration"])


class KimiK31Mtp(KimiK3Mtp):
    """Trained K3 MLA nextn layer paired with a K3.1 sparse target."""

    MODEL_TYPE = "kimi_k31_mtp"
    TARGET_MODEL_CLASS = KimiK31

    @classmethod
    def _from_config_json(cls, config_json, ckpt_path=""):
        fixture = config_json.get("rtp_llm_fixture", {})
        if fixture.get("synthetic") or fixture.get("trained") is False:
            raise ValueError("K3.1 MTP requires trained native K3 MLA weights; synthetic fallback is disabled")
        return super()._from_config_json(config_json, ckpt_path)

    @classmethod
    def speculative_weight_alias_names(cls, target_model, draft_model_config):
        # Validate the pairing and special-token contract, but retain the
        # trained draft's embedding and shared head. Shape equality cannot
        # establish weight identity across independently trained checkpoints.
        super().speculative_weight_alias_names(target_model, draft_model_config)
        return ()

    @staticmethod
    def speculative_parallelism_config(parallelism):
        draft = copy.deepcopy(parallelism)
        # ParallelismConfig's legacy pickle omits local_rank. Keep device and
        # cache-store ownership when making the draft's private config.
        draft.local_rank = parallelism.local_rank
        draft.prefill_cp_config.method = CPRotateMethod.DISABLED
        draft.prefill_cp_config.preserve_tp_layout = False
        # KV ownership remains Page-RR; only the query split is disabled.
        return draft

    @staticmethod
    def get_weight_cls():
        return KimiK31MtpWeight


register_model("kimi_k31_mtp", KimiK31Mtp, ["KimiK31MTPModel"])
