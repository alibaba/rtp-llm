def init_server_group_args(parser, server_config):
    ##############################################################################################################
    # Server Configuration
    ##############################################################################################################
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument(
        "--frontend_server_count",
        env_name="FRONTEND_SERVER_COUNT",
        bind_to=(server_config, "frontend_server_count"),
        type=int,
        default=4,
        help="前端服务器启动进程数量",
    )
    server_group.add_argument(
        "--start_port",
        env_name="START_PORT",
        bind_to=(server_config, "start_port"),
        type=int,
        default=8088,
        help="服务启动端口",
    )
    server_group.add_argument(
        "--timeout_keep_alive",
        env_name="TIMEOUT_KEEP_ALIVE",
        bind_to=(server_config, "timeout_keep_alive"),
        type=int,
        default=5,
        help="健康检查的超时时间",
    )
    server_group.add_argument(
        "--frontend_server_id",
        env_name="FRONTEND_SERVER_ID",
        bind_to=(server_config, "frontend_server_id"),
        type=int,
        default=0,
        help="前端服务器序号",
    )
    server_group.add_argument(
        "--worker_info_port_num",
        env_name="WORKER_INFO_PORT_NUM",
        bind_to=(server_config, "worker_info_port_num"),
        type=int,
        default=10,
        help="worker的总的端口的数量",
    )
    server_group.add_argument(
        "--shutdown_timeout",
        env_name="SHUTDOWN_TIMEOUT",
        bind_to=(server_config, "shutdown_timeout"),
        type=int,
        default=50,
        help="Process manager shutdown timeout in seconds. Set to -1 to wait indefinitely for processes to finish (no force kill)",
    )
    server_group.add_argument(
        "--monitor_interval",
        env_name="MONITOR_INTERVAL",
        bind_to=(server_config, "monitor_interval"),
        type=int,
        default=1,
        help="Process manager monitor interval in seconds",
    )
