def init_pd_separation_group_args(parser, pd_separation_config):
    ##############################################################################################################
    # PD分离的配置
    ##############################################################################################################
    pd_separation_group = parser.add_argument_group("pd_separation")
    pd_separation_group.add_argument(
        "--load_cache_timeout_ms",
        env_name="LOAD_CACHE_TIMEOUT_MS",
        bind_to=(pd_separation_config, "load_cache_timeout_ms"),
        type=int,
        default=5000,
        help="KV cache 加载超时（毫秒），共享默认 5000；P2P 部署可显式设置 900000（15min）。",
    )

    pd_separation_group.add_argument(
        "--max_rpc_timeout_ms",
        env_name="MAX_RPC_TIMEOUT_MS",
        bind_to=(pd_separation_config, "max_rpc_timeout_ms"),
        type=int,
        default=2 * 3600 * 1000,  # 2h
        help="RPC 调用最大超时（毫秒），用作 per-request 未传 timeout 时的默认值；"
        "<=0 表示不设 deadline（链路不超时）；"
        "per-request generate_config.timeout_ms 优先级更高",
    )
