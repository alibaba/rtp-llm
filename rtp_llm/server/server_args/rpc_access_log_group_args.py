from rtp_llm.server.server_args.util import str2bool


def init_rpc_access_log_group_args(parser):
    ##############################################################################################################
    # RPC Access Log 配置
    ##############################################################################################################
    rpc_access_log_group = parser.add_argument_group("RPC Access Log")

    rpc_access_log_group.add_argument(
        "--enable_rpc_access_log",
        env_name="ENABLE_RPC_ACCESS_LOG",
        type=str2bool,
        default=True,
        help="是否启用RPC访问日志记录",
    )

    rpc_access_log_group.add_argument(
        "--access_log_interval",
        env_name="ACCESS_LOG_INTERVAL",
        type=int,
        default=1,
        help="RPC访问日志记录间隔（每N次请求记录一次）",
    )

    rpc_access_log_group.add_argument(
        "--log_plaintext",
        env_name="LOG_PLAINTEXT",
        type=str2bool,
        default=True,
        help="是否以明文格式记录日志（True为明文，False为protobuf格式）",
    )
