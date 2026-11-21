"""Legacy descriptor names are imported only when requested.

TIPC is an independent transport used by the serving RPC. Importing it must
not instantiate the old checkpoint loader or any old model descriptors.
"""

from importlib import import_module

_EXPORTS = {
    "AttnAtomicWeight": "attn_weight",
    "AttnConfig": "attn_weight",
    "MlaAttnAtomicWeight": "attn_weight",
    "MlaConfig": "attn_weight",
    "LoadCompressedW4A8Int4PerGroupQuantWeight": "compressed_w4a8_int4_per_channel_weight",
    "CompressedW8A8Int8PerChannelWeight": "compressed_w8a8_int8_per_channel_weight",
    "LoadQuantDynamicPerTensorFp8Weight": "dynamic_fp8_quant_weight",
    "FfnAtomicWeight": "ffn_weight",
    "FfnConfig": "ffn_weight",
    "FfnWeight": "ffn_weight",
    "MoeAtomicWeight": "ffn_weight",
    "MoeConfig": "ffn_weight",
    "GroupWiseWeight": "group_wise_quant_weight",
    "MixedFp4Weight": "mixed_fp4_quant_weight",
    "OmniQuantWeightInfo": "omni_quant_weight",
    "OnlineModelOptFp4MoeWeight": "online_modelopt_fp4_quant_weight",
    "PerBlockFp8Weight": "per_block_fp8_quant_weight",
    "PerChannelFp8Weight": "per_channel_fp8_quant_weight",
    "PerGroupFp4Weight": "per_group_fp4_quant_weight",
    "PerTensorInt8QuantWeight": "per_tensor_int8_quant_weight",
    "SmoothQuantWeightInfo": "smooth_quant_weight",
    "Fp8PerTensorCompressedWeight": "static_fp8_quant_weight",
    "LoadW4a8Int4PerChannelQuantWeight": "w4a8_int4_per_channel_quant_weight",
    "WeightOnlyPerColWeight": "weight_only_quant_weight",
}
_EXPORTS.update(
    dict.fromkeys(
        (
            "OnlineMegaMoeFp4Weight",
            "OnlineMegaMoeFp4FromFp8Weight",
            "OnlineModelOptFp4HybridFp8AttnWeight",
        ),
        "online_modelopt_fp4_quant_weight",
    )
)
_ALIASES = {
    "OnlineMegaMoEFp4Weight": "OnlineMegaMoeFp4Weight",
    "OnlineMegaMoEFp8ToFp4Weight": "OnlineMegaMoeFp4FromFp8Weight",
}
__all__ = list(_EXPORTS) + list(_ALIASES)


def __getattr__(name):
    name = _ALIASES.get(name, name)
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module("." + module, __name__), name)
    globals()[name] = value
    return value
