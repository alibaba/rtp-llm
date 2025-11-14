from rtp_llm.server.server_args.util import str2bool


def init_misc_group_args(parser, misc_config):
    ##############################################################################################################
    # Miscellaneous 配置
    ##############################################################################################################
    misc_group = parser.add_argument_group("Miscellaneous")

    misc_group.add_argument(
        "--disable_pdl",
        env_name="DISABLE_PDL",
        bind_to=(misc_config.misc_config, 'disable_pdl'),
        type=str2bool,
        default=True,
        help="是否禁用PDL",
    )

    misc_group.add_argument(
        "--aux_string",
        env_name="AUX_STRING",
        bind_to=(misc_config.misc_config, 'aux_string'),
        type=str,
        default="",
        help="管控环境变量字符串",
    )

    misc_group.add_argument(
        "--oss_endpoint",
        env_name="OSS_ENDPOINT",
        bind_to=(misc_config, 'oss_endpoint'),
        type=str,
        default=None,
        help="OSS端点",
    )
    misc_group.add_argument(
        "--dashscope_api_key",
        env_name="DASHSCOPE_API_KEY",
        bind_to=(misc_config, 'dashscope_api_key'),
        type=str,
        default="EMPTY",
        help="Dashscope API Key",
    )
    misc_group.add_argument(
        "--dashscope_http_url",
        env_name="DASHSCOPE_HTTP_URL",
        bind_to=(misc_config, 'dashscope_http_url'),
        type=str,
        default=None,
        help="Dashscope HTTP URL",
    )
    misc_group.add_argument(
        "--dashscope_websocket_url",
        env_name="DASHSCOPE_WEBSOCKET_URL",
        bind_to=(misc_config, 'dashscope_websocket_url'),
        type=str,
        default=None,
        help="Dashscope WebSocket URL",
    )
    misc_group.add_argument(
        "--openai_api_key",
        env_name="OPENAI_API_KEY",
        bind_to=(misc_config, 'openai_api_key'),
        type=str,
        default="EMPTY",
        help="OpenAI API Key",
    )
