from rtp_llm.server.server_args.util import str2bool


def init_generate_group_args(parser, generate_env_config):
    ##############################################################################################################
    # Generate Configuration
    ##############################################################################################################
    generate_group = parser.add_argument_group("Generate Configuration")
    generate_group.add_argument(
        "--think_end_tag",
        env_name="THINK_END_TAG",
        bind_to=(generate_env_config, 'think_end_tag'),
        type=str,
        default="</think>\n\n",
        help="深度思考模式的结束标签",
    )
    generate_group.add_argument(
        "--think_end_token_id",
        env_name="THINK_END_TOKEN_ID",
        bind_to=(generate_env_config, 'think_end_token_id'),
        type=int,
        default=-1,
        help="深度思考模式的结束标签的 TOKEN_ID",
    )
    generate_group.add_argument(
        "--think_mode",
        env_name="THINK_MODE",
        bind_to=(generate_env_config, 'think_mode'),
        type=int,
        default=0,
        help="深度思考模式是否开启",
    )
    generate_group.add_argument(
        "--force_stop_words",
        env_name="FORCE_STOP_WORDS",
        bind_to=(generate_env_config, 'force_stop_words'),
        type=str2bool,
        default=False,
        help="是否开启使用环境变量强制指定模型的STOP WORDS",
    )
    generate_group.add_argument(
        "--stop_words_list",
        env_name="STOP_WORDS_LIST",
        bind_to=(generate_env_config, 'stop_words_list'),
        type=str,
        default=None,
        help="STOP_WORDS的TokenID列表",
    )
    generate_group.add_argument(
        "--stop_words_str",
        env_name="STOP_WORDS_STR",
        bind_to=(generate_env_config, 'stop_words_str'),
        type=str,
        default=None,
        help="STOP_WORDS的string明文",
    )
    generate_group.add_argument(
        "--think_start_tag",
        env_name="THINK_START_TAG",
        bind_to=(generate_env_config, 'think_start_tag'),
        type=str,
        default="<think>\\n",
        help="深度思考模式的起始标签",
    )
    generate_group.add_argument(
        "--generation_config_path",
        env_name="GENERATION_CONFIG_PATH",
        bind_to=(generate_env_config, 'generation_config_path'),
        type=str,
        default=None,
        help="生成配置路径",
    )
