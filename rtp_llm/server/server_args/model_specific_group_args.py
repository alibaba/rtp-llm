def init_model_specific_group_args(parser, model_specific_config):
    ##############################################################################################################
    # 模型特定配置
    ##############################################################################################################
    model_specific_group = parser.add_argument_group("模型特定配置")

    model_specific_group.add_argument(
        "--max_lora_model_size",
        env_name="MAX_LORA_MODEL_SIZE",
        bind_to=(model_specific_config, 'max_lora_model_size'),
        type=int,
        default=-1,
        help="指定 LoRA 模型的最大允许大小。",
    )
