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
        help="Master 调度 per-request 默认超时（毫秒），用作 per-request 未传 timeout 时的默认值；"
        "<=0 表示不设超时（链路不超时）；per-request 显式 timeout 优先级更高",
    )
