import argparse


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer, got {value!r}"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return parsed


def init_quantization_group_args(parser, quantization_config):
    ##############################################################################################################
    # Quantization Configuration
    ##############################################################################################################
    quantization_group = parser.add_argument_group("Quantization Configuration")
    quantization_group.add_argument(
        "--int8_mode",
        env_name="INT8_MODE",
        bind_to=(quantization_config, "int8_mode"),
        type=int,
        default=0,
        help="权重类型是否使用int8模式",
    )
    quantization_group.add_argument(
        "--quantization",
        env_name="QUANTIZATION",
        bind_to=(quantization_config, "quantization"),
        type=str,
        default=None,
        help="",
    )
    quantization_group.add_argument(
        "--w8a8_quant_chunk_rows",
        env_name="W8A8_QUANT_CHUNK_ROWS",
        bind_to=(quantization_config, "w8a8_quant_chunk_rows"),
        type=_positive_int,
        default=1024,
        help="Rows loaded per W8A8 online quantization chunk.",
    )
