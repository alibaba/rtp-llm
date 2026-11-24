from rtp_llm.server.server_args.util import str2bool


def init_pd_separation_group_args(parser, pd_separation_config):
    ##############################################################################################################
    # PD分离的配置
    ##############################################################################################################
    pd_separation_group = parser.add_argument_group("pd_separation")
    pd_separation_group.add_argument(
        "--prefill_retry_times",
        env_name="PREFILL_RETRY_TIMES",
        bind_to=(pd_separation_config, "prefill_retry_times"),
        type=int,
        default=0,
        help="Prefill 多模态瞬时错误的额外重试次数，0 表示不重试。",
    )
    pd_separation_group.add_argument(
        "--prefill_retry_timeout_ms",
        env_name="PREFILL_RETRY_TIMEOUT_MS",
        bind_to=(pd_separation_config, "prefill_retry_timeout_ms"),
        type=int,
        default=0,
        help="Prefill 多模态重试预算（毫秒），0 表示不另设预算；不会延长请求 deadline。",
    )

    pd_separation_group.add_argument(
        "--load_cache_timeout_ms",
        env_name="LOAD_CACHE_TIMEOUT_MS",
        bind_to=(pd_separation_config, "load_cache_timeout_ms"),
        type=int,
        default=5000,
        help="PD KV 单次通信超时（毫秒），默认 5000。等待 Prefill 就绪使用请求 deadline。",
    )

    pd_separation_group.add_argument(
        "--max_rpc_timeout_ms",
        env_name="MAX_RPC_TIMEOUT_MS",
        bind_to=(pd_separation_config, "max_rpc_timeout_ms"),
        type=int,
        default=2 * 3600 * 1000,  # 2h
        help="RPC 调用最大超时（毫秒），用作 per-request 未传 timeout 时的默认值；"
        "<=0 表示不设置默认生成期限；PD 交接阶段仍有独立的资源清理期限；"
        "per-request generate_config.timeout_ms 优先级更高",
    )

    pd_separation_group.add_argument(
        "--decode_linear_tail_only",
        env_name="DECODE_LINEAR_TAIL_ONLY",
        bind_to=(pd_separation_config, "decode_linear_tail_only"),
        type=str2bool,
        default=False,
        help="PD decode handoff keeps only active linear states while retaining full-attention prefix reuse",
    )
