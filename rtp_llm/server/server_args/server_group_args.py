def init_server_group_args(parser):
    ##############################################################################################################
    # Server Configuration
    ##############################################################################################################
    server_group = parser.add_argument_group("Server Configuration")
    server_group.add_argument(
        "--frontend_server_count",
        env_name="FRONTEND_SERVER_COUNT",
        type=int,
        default=4,
        help="前端服务器启动进程数量",
    )
    server_group.add_argument(
        "--start_port",
        env_name="START_PORT",
        type=int,
        default=8088,
        help="服务启动端口",
    )
    server_group.add_argument(
        "--timeout_keep_alive",
        env_name="TIMEOUT_KEEP_ALIVE",
        type=int,
        default=5,
        help="健康检查的超时时间",
    )
    server_group.add_argument(
        "--frontend_server_id",
        env_name="FRONTEND_SERVER_ID",
        type=int,
        default=0,
        help="前端服务器序号",
    )
