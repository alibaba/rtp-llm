"""K3.1 adds replicated Indexer weights to the existing K3 MLA manifest."""

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3MtpWeight, KimiK3Weight
from rtp_llm.models.kimi_k3.nvfp4_config import uses_fp8_output
from rtp_llm.ops import RoleType
from rtp_llm.utils.model_weight import W, transpose


class KimiK31Weight(KimiK3Weight):
    def __init__(self, model_config, parallelism_config, *args, **kwargs):
        cp = parallelism_config.prefill_cp_config
        self.prefill_query_cp = (
            parallelism_config.role_type == RoleType.PREFILL
            and cp.is_enabled()
            and cp.preserve_tp_layout
        )
        super().__init__(model_config, parallelism_config, *args, **kwargs)

    def _global_weights(self):
        weights = super()._global_weights()
        if self.prefill_query_cp:
            weights = [
                (
                    CustomAtomicWeight(
                        weight.name,
                        weight.weights,
                        process_fun=weight.process_fun,
                        data_type=weight.data_type,
                    )
                    if weight.name == W.embedding
                    else weight
                )
                for weight in weights
            ]
        return weights

    def _mla_projection_parallelism(self):
        return (
            (1, 0) if self.prefill_query_cp else super()._mla_projection_parallelism()
        )

    def _replicate_mla_query_heads(self):
        return self.prefill_query_cp or self.role_type == RoleType.DECODE

    @staticmethod
    def _attention_fp8_weight_cls():
        from rtp_llm.models.kimi_k3.fp8_weight import KimiK31LoadAttentionFp8Weight

        return KimiK31LoadAttentionFp8Weight

    def _nvfp4_projection(self, source):
        modules = getattr(self.model_config, "k31_nvfp4_modules", ())
        if not modules:
            return source
        from rtp_llm.models.kimi_k3.nvfp4_weight import (
            ATTENTION_PROJECTION_MODULES,
            KimiK31Nvfp4Weight,
        )

        module = ATTENTION_PROJECTION_MODULES.get(source.name)
        if module not in modules:
            return source
        return KimiK31Nvfp4Weight(source, module=module)

    def _attention_projection_weights(self, attention):
        return [self._nvfp4_projection(source) for source in attention]

    def _attention_fp8_quant_config(self, source):
        quant = super()._attention_fp8_quant_config(source)
        modules = getattr(self.model_config, "k31_nvfp4_modules", ())
        if quant is None and modules:
            from rtp_llm.models.kimi_k3.nvfp4_weight import ATTENTION_PROJECTION_MODULES

            if uses_fp8_output(ATTENTION_PROJECTION_MODULES.get(source.name), modules):
                return Fp8BlockWiseQuantConfig()
        return quant

    # Verified against the weight index at 546f79b5ee100f74d7ef432f0c078d92dc92bab6.
    _INDEXER_SUFFIXES = (
        "self_attn.indexer.wq_b.weight",
        "self_attn.indexer.wk.weight",
        "self_attn.indexer.k_norm.weight",
        "self_attn.indexer.k_norm.bias",
        "self_attn.indexer.weights_proj.weight",
    )
    _MLA_SUFFIXES = KimiK3Weight._MLA_SUFFIXES + _INDEXER_SUFFIXES

    def _mla_weights(self):
        attention = super()._mla_weights()
        indexer = [
            self._linear(W.mla_indexer_qb_w, self._INDEXER_SUFFIXES[0]),
            self._linear(W.mla_indexer_k_w, self._INDEXER_SUFFIXES[1]),
            self._custom(W.mla_indexer_k_norm_w, self._INDEXER_SUFFIXES[2]),
            self._custom(W.mla_indexer_k_norm_b, self._INDEXER_SUFFIXES[3]),
            self._custom(
                W.mla_indexer_weights_proj_w,
                self._INDEXER_SUFFIXES[4],
                process_fun=transpose,
                data_type=torch.float32,
            ),
        ]
        quant = getattr(self.model_config, "indexer_projection_quant_config", None)
        if quant is not None:
            from rtp_llm.models.kimi_k3.fp8_weight import KimiK31LoadIndexerFp8Weight

            indexer = [
                (
                    KimiK31LoadIndexerFp8Weight(source, quant)
                    if source.name in KimiK31LoadIndexerFp8Weight.w8a8_weight_list
                    else source
                )
                for source in indexer
            ]
        return attention + indexer


# Preserve the import name without adding K3.1 Indexer tensors to the draft.
KimiK31MtpWeight = KimiK3MtpWeight
