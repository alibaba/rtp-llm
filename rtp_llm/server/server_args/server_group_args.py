from rtp_llm.server.server_args.util import str2bool


def init_server_group_args(parser, server_config, distribute_config):
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
        "--vit_server_count",
        env_name="VIT_SERVER_COUNT",
        bind_to=(server_config, "vit_server_count"),
        type=int,
        default=1,
        help="VIT服务器启动进程数量",
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
        "--vit_server_id",
        env_name="VIT_SERVER_ID",
        bind_to=(server_config, "vit_server_id"),
        type=int,
        default=0,
        help="VIT服务器序号",
    )
    server_group.add_argument(
        "--worker_info_port_num",
        env_name="WORKER_INFO_PORT_NUM",
        bind_to=[
            (server_config, "worker_info_port_num"),
            (distribute_config, "worker_info_port_num"),
        ],
        type=int,
        default=9,
        help="rank 端口块步进；启用 DashSc gRPC 的非 VIT 服务最小为 9",
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
        "--frontend_pre_stop_drain_seconds",
        env_name="FRONTEND_PRE_STOP_DRAIN_SECONDS",
        bind_to=(server_config, "frontend_pre_stop_drain_seconds"),
        type=float,
        default=120.0,
        help="Frontend pre-stop unavailable/drain window in seconds before graceful shutdown.",
    )
    server_group.add_argument(
        "--dash_sc_grpc_pre_stop_drain_seconds",
        env_name="DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS",
        bind_to=(server_config, "dash_sc_grpc_pre_stop_drain_seconds"),
        type=float,
        default=120.0,
        help="DashSc gRPC pre-stop unavailable/drain window in seconds before graceful shutdown.",
    )
    server_group.add_argument(
        "--pre_stop_drain_headroom_seconds",
        env_name="RTP_LLM_PRE_STOP_DRAIN_HEADROOM_SECONDS",
        bind_to=(server_config, "pre_stop_drain_headroom_seconds"),
        type=float,
        default=-1.0,
        help="Shutdown budget reserved after pre-stop drain. Negative means auto.",
    )
    server_group.add_argument(
        "--pre_stop_drain_signal",
        env_name="RTP_LLM_PRE_STOP_DRAIN_SIGNAL",
        bind_to=(server_config, "pre_stop_drain_signal"),
        type=str2bool,
        default=True,
        help="Whether the parent sends SIGUSR1 for pre-stop drain before SIGTERM.",
    )
    server_group.add_argument(
        "--backend_post_frontend_drain_seconds",
        env_name="RTP_LLM_BACKEND_POST_FRONTEND_DRAIN_SECONDS",
        bind_to=(server_config, "backend_post_frontend_drain_seconds"),
        type=float,
        default=-1.0,
        help="Parent wait window before backend shutdown after frontend drain. Negative derives from frontend/dash_sc drain config.",
    )
    server_group.add_argument(
        "--monitor_interval",
        env_name="MONITOR_INTERVAL",
        bind_to=(server_config, "monitor_interval"),
        type=int,
        default=1,
        help="Process manager monitor interval in seconds",
    )
