def init_worker_group_args(parser):
    ##############################################################################################################
    # Worker Configuration
    ##############################################################################################################
    worker_group = parser.add_argument_group("Worker Configuration")
    worker_group.add_argument(
        "--worker_info_port_num",
        env_name="WORKER_INFO_PORT_NUM",
        type=int,
        default=7,
        help="worker的总的端口的数量",
    )
    worker_group.add_argument(
        "--shutdown-timeout",
        env_name="SHUTDOWN_TIMEOUT",
        type=int,
        default=50,
        help="Process manager shutdown timeout in seconds. Set to -1 to wait indefinitely for processes to finish (no force kill)",
    )
    worker_group.add_argument(
        "--monitor-interval",
        env_name="MONITOR_INTERVAL",
        type=int,
        default=1,
        help="Process manager monitor interval in seconds",
    )
