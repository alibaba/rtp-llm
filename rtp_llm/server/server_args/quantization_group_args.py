import argparse

from rtp_llm.server.server_args.util import str2bool


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
    quantization_group.add_argument(
        "--enable_w4a16_sm120_dense_ffn",
        env_name="ENABLE_W4A16_SM120_DENSE_FFN",
        bind_to=(quantization_config, "enable_w4a16_sm120_dense_ffn"),
        type=str2bool,
        default=False,
        help=("是否开启 sm120 中小 batch 的 w4a16 ffn 量化来加速 gemm"),
    )
    quantization_group.add_argument(
        "--enable_nvfp4_dense_ffn",
        env_name="ENABLE_NVFP4_DENSE_FFN",
        bind_to=(quantization_config, "enable_nvfp4_dense_ffn"),
        type=str2bool,
        default=False,
        help=(
            "是否在基础量化之上将 dense FFN 权重在线量化为 NVFP4"
            "(叠加式开关，注意力等其他权重仍按基础量化加载)"
        ),
    )
