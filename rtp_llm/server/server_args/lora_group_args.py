from rtp_llm.server.server_args.util import str2bool


def init_lora_group_args(parser):
    ##############################################################################################################
    # Lora Configuration
    ##############################################################################################################
    lora_group = parser.add_argument_group("Lora Configuration")
    lora_group.add_argument(
        "--lora_info", env_name="LORA_INFO", type=str, default="{}", help="Lora的信息"
    )
    lora_group.add_argument(
        "--merge_lora",
        env_name="MERGE_LORA",
        type=str2bool,
        default=True,
        help="Lora合并",
    )
