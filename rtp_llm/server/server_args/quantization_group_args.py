from rtp_llm.server.server_args.util import str2bool


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
        "--enable_w4a16_sm120_dense_ffn",
        env_name="ENABLE_W4A16_SM120_DENSE_FFN",
        bind_to=(quantization_config, "enable_w4a16_sm120_dense_ffn"),
        type=str2bool,
        default=False,
        help=("是否开启 sm120 中小 batch 的 w4a16 ffn 量化来加速 gemm"),
    )
