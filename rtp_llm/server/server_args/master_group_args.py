def init_master_group_args(parser, master_config):
    master_group = parser.add_argument_group("Master Configuration")

    master_group.add_argument(
        "--master_queue_reject_threshold",
        env_name="MASTER_QUEUE_REJECT_THRESHOLD",
        bind_to=(master_config, "master_queue_reject_threshold"),
        type=int,
        default=100000,
        help="Master queue reject threshold",
    )

    master_group.add_argument(
        "--master_default_timeout_ms",
        env_name="MASTER_DEFAULT_TIMEOUT_MS",
        bind_to=(master_config, "master_default_timeout_ms"),
        type=int,
        default=3600000,
        help="Master default timeout in milliseconds",
    )

    master_group.add_argument(
        "--master_max_connect_pool_size",
        env_name="MASTER_MAX_CONNECT_POOL_SIZE",
        bind_to=(master_config, "master_max_connect_pool_size"),
        type=int,
        default=100000,
        help="Master max connect pool size",
    )

    master_group.add_argument(
        "--master_session_timeout_s",
        env_name="MASTER_SESSION_TIMEOUT_S",
        bind_to=(master_config, "master_session_timeout_s"),
        type=float,
        default=-1,
        help="Master HTTP session total timeout in seconds. -1: auto (3600 when queue mode, 0.5 otherwise); else use this value.",
    )
