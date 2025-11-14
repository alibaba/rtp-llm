from rtp_llm.server.server_args.util import str2bool


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
    
    model_specific_group.add_argument(
        "--load_python_model",
        env_name="LOAD_PYTHON_MODEL",
        bind_to=(model_specific_config, 'load_python_model'),
        type=str2bool,
        default=False,
        help="是否加载 Python 模型。设置为 True 启用 Python 模型，False 使用传统的 C++ GptModel。",
    )
