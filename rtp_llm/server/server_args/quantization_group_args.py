def init_quantization_group_args(parser, quantization_config):
    ##############################################################################################################
    # Quantization Configuration
    ##############################################################################################################
    quantization_group = parser.add_argument_group("Quantization Configuration")
    quantization_group.add_argument(
        "--int8_mode",
        env_name="INT8_MODE",
        bind_to=(quantization_config, 'int8_mode'),
        type=int,
        default=0,
        help="权重类型是否使用int8模式",
    )
    quantization_group.add_argument(
        "--quantization",
        env_name="QUANTIZATION",
        bind_to=(quantization_config, 'quantization'),
        type=str,
        default=None,
        help=""
    )
