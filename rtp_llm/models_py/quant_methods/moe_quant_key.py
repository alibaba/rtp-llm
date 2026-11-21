"""Resolve the loader's expert-weight representation from main's config API."""


def moe_checkpoint_method_key(config):
    # get_method() identifies the runtime algorithm, not the on-disk layout:
    # for example exactly the same FP8 algorithm accepts BF16 or prequantized
    # sources. The NewLoader must select packing from both facts.
    method = config.get_method()
    prequantized = config.is_quanted()
    formats = {
        "FP8": "fp8",
        "FP8_DYNAMIC_PER_TENSOR": "fp8",
        "FP8_PER_TENSOR_COMPRESSED": "fp8",
        "FP8_PER_BLOCK": "fp8_block",
        "FP8_PER_CHANNEL_COMPRESSED": "fp8_per_channel",
        "FP8_PER_CHANNEL_QUARK": "fp8_per_channel",
        "modelopt_fp4": "modelopt_fp4",
    }
    if method in formats:
        base = formats[method]
        return base if prequantized else base + "_online"
    if method == "QuarkMXFP4":
        if not prequantized:
            raise ValueError("Quark MXFP4 needs a prequantized expert checkpoint")
        return "mxfp4_quark"
    return method
