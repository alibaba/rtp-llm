from rtp_llm.server.server_args.util import str2bool


def init_gang_group_args(parser, distribute_config):
    ##############################################################################################################
    # Gang Configuration
    ##############################################################################################################
    gang_group = parser.add_argument_group("Gang Configuration")
    gang_group.add_argument(
        "--fake_gang_env",
        env_name="FAKE_GANG_ENV",
        bind_to=(distribute_config, "fake_gang_env"),
        type=str2bool,
        default=False,
        help="在多机启动时的fake行为",
    )
    gang_group.add_argument(
        "--gang_annocation_path",
        env_name="GANG_ANNOCATION_PATH",
        bind_to=(distribute_config, "gang_annocation_path"),
        type=str,
        default="/etc/podinfo/annotations",
        help="GANG信息的路径",
    )
    gang_group.add_argument(
        "--gang_config_string",
        env_name="GANG_CONFIG_STRING",
        bind_to=(distribute_config, "gang_config_string"),
        type=str,
        default=None,
        help="GAG信息的字符串表达",
    )
    gang_group.add_argument(
        "--zone_name",
        env_name="ZONE_NAME",
        bind_to=(distribute_config, "zone_name"),
        type=str,
        default="",
        help="角色名",
    )
    gang_group.add_argument(
        "--distribute_config_file",
        env_name="DISTRIBUTE_CONFIG_FILE",
        bind_to=(distribute_config, "distribute_config_file"),
        type=str,
        default=None,
        help="分布式的配置文件路径",
    )
    gang_group.add_argument(
        "--dist_comm_timeout",
        env_name="DIST_COMM_TIMEOUT",
        bind_to=(distribute_config, "dist_comm_timeout"),
        type=int,
        default=300,
        help="心跳检测的超时时间",
    )
    gang_group.add_argument(
        "--gang_sleep_time",
        env_name="GANG_SLEEP_TIME",
        bind_to=(distribute_config, "gang_sleep_time"),
        type=int,
        default=10,
        help="心跳检测的间隔时间",
    )
    gang_group.add_argument(
        "--gang_timeout_min",
        env_name="GANG_TIMEOUT_MIN",
        bind_to=(distribute_config, "gang_timeout_min"),
        type=int,
        default=30,
        help="心跳超时的最小时间",
    )
    gang_group.add_argument(
        "--remote_server_port",
        env_name="REMOTE_SERVER_PORT",
        bind_to=(distribute_config, "remote_server_port"),
        type=int,
        default=0,
        help="远端服务端口",
    )
