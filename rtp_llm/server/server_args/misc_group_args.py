from rtp_llm.server.server_args.util import str2bool


def init_misc_group_args(parser):
    ##############################################################################################################
    # Miscellaneous 配置
    ##############################################################################################################
    misc_group = parser.add_argument_group("Miscellaneous")

    misc_group.add_argument(
        "--disable_pdl",
        env_name="DISABLE_PDL",
        type=str2bool,
        default=True,
        help="是否禁用PDL",
    )

    misc_group.add_argument(
        "--aux_string",
        env_name="AUX_STRING",
        type=str,
        default="",
        help="管控环境变量字符串",
    )

    misc_group.add_argument(
        "--disable_access_log",
        env_name="DISABLE_ACCESS_LOG",
        type=str2bool,
        default=False,
        help="默认禁用请求的 access log",
    )
