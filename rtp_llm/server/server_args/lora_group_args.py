from rtp_llm.server.server_args.util import str2bool


def init_lora_group_args(parser, lora_config):
    ##############################################################################################################
    # Lora Configuration
    ##############################################################################################################
    lora_group = parser.add_argument_group("Lora Configuration")
    lora_group.add_argument(
        "--lora_info",
        env_name="LORA_INFO",
        bind_to=(lora_config, 'lora_info'),
        type=str,
        default="{}",
        help="Lora的信息"
    )
    lora_group.add_argument(
        "--merge_lora",
        env_name="MERGE_LORA",
        bind_to=(lora_config, 'merge_lora'),
        type=str2bool,
        default=True,
        help="启动时将单个 LoRA 合并到所有请求使用的权重。关闭此项或配置多个适配器时，仅加载基座模型；原生引擎不支持按请求切换 LoRA。",
    )
