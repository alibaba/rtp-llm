import logging
from typing import Callable, Dict, List

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops.compute_ops import PyAttentionInputs
from rtp_llm.utils.model_weight import W

# Lists to store registered implementations
PREFILL_MHA_IMPS: List[type[FMHAImplBase]] = []
DECODE_MHA_IMPS: List[type[FMHAImplBase]] = []
PREFILL_MLA_IMPS: List[type[FMHAImplBase]] = []
DECODE_MLA_IMPS: List[type[FMHAImplBase]] = []


def get_mla_impl(
    config: GptInitModelParameters, weight: ModelWeights, attn_inputs: PyAttentionInputs
) -> FMHAImplBase:
    mla_impls = PREFILL_MLA_IMPS if attn_inputs.is_prefill else DECODE_MLA_IMPS
    for impl in mla_impls:
        cos_sin_cache = weight.get_global_weight(W.rope_cos_sin_cache)
        instance = impl(
            config,
            attn_inputs,
            weight.weights,
            cos_sin_cache,
        )
        if instance.support():
            return instance
    raise Exception(f"can not find mla type")


def get_fmha_impl(
    config: GptInitModelParameters, weight: ModelWeights, attn_inputs: PyAttentionInputs
) -> FMHAImplBase:
    mha_impls = PREFILL_MHA_IMPS if attn_inputs.is_prefill else DECODE_MHA_IMPS
    for impl in mha_impls:
        instance = impl(config, attn_inputs)
        if instance.support():
            return instance
    raise Exception(f"can not find mha type")


class AttnImplFactory(object):
    """Factory class for creating FMHA implementations based on attention_type."""

    # FMHA implementation registry - maps attention_type to impl method
    FMHA_IMPL_REGISTRY: Dict[
        str,
        Callable[
            [GptInitModelParameters, ModelWeights, PyAttentionInputs], FMHAImplBase
        ],
    ] = {
        "mha": get_fmha_impl,
        "mla": get_mla_impl,
    }

    @classmethod
    def get_fmha_impl(
        cls,
        config: GptInitModelParameters,
        weight: ModelWeights,
        attn_inputs: PyAttentionInputs,
    ) -> FMHAImplBase:
        key_str = "mla" if config.use_mla else "mha"
        fmha_impl_method = cls.FMHA_IMPL_REGISTRY[key_str]
        instance = fmha_impl_method(config, weight, attn_inputs)
        logging.debug(f"get fmha impl: {instance.fmha_type()}")
        return instance

    @classmethod
    def get_fmha_impl_method(cls, attention_type: str) -> str:
        """
        Get the appropriate FMHA implementation method based on attention_type.

        Args:
            attention_type: String identifying the attention type

        Returns:
            Method name to call for getting FMHA implementation

        Raises:
            ValueError: If attention_type is not supported
        """
        if attention_type not in cls.FMHA_IMPL_REGISTRY:
            available_types = list(cls.FMHA_IMPL_REGISTRY.keys())
            raise ValueError(
                f"Unsupported attention type '{attention_type}'. Available types: {available_types}"
            )

        return cls.FMHA_IMPL_REGISTRY[attention_type]

    @classmethod
    def register_fmha_impl(cls, attention_type: str, impl_method: str):
        """
        Register a new FMHA implementation method for an attention type.

        Args:
            attention_type: String key for the attention type
            impl_method: Method name to call for getting FMHA implementation
        """
        cls.FMHA_IMPL_REGISTRY[attention_type] = impl_method

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported attention types."""
        return list(cls.FMHA_IMPL_REGISTRY.keys())
