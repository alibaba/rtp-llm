from rtp_llm.server.server_args.util import str2bool


def init_concurrent_group_args(parser, concurrency_config):
    ##############################################################################################################
    # Concurrency 控制
    ##############################################################################################################
    concurrent_group = parser.add_argument_group("Concurrent")
    concurrent_group.add_argument(
        "--concurrency_with_block",
        env_name="CONCURRENCY_WITH_BLOCK",
        bind_to=(concurrency_config, 'concurrency_with_block'),
        type=str2bool,
        default=False,
        help="控制并发请求的阻塞行为。通常设置为 '1' (启用阻塞) 或 '0' (禁用阻塞)。",
    )
    concurrent_group.add_argument(
        "--concurrency_limit",
        env_name="CONCURRENCY_LIMIT",
        bind_to=(concurrency_config, 'concurrency_limit'),
        type=int,
        default=32,
        help="设置系统允许的最大并发请求数量。",
    )
    # bind_to uses a root-config string path (py_env_configs.
    # max_generate_batch_size_override) so the "explicitly provided"
    # tri-state survives the later legacy derivation inside
    # EngineConfig.create (max_generate_batch_size = concurrency_limit).
    concurrent_group.add_argument(
        "--max_generate_batch_size",
        env_name="MAX_GENERATE_BATCH_SIZE",
        bind_to="max_generate_batch_size_override",
        type=int,
        default=None,
        help=(
            "调度器侧 decode 运行批上限；超出的请求在调度器内排队，而不是被前端拒绝"
            "（区别于 --concurrency_limit：它作用在 HTTP 前端，超限默认直接拒绝）。"
            "不设置（默认）= 跟随 --concurrency_limit。"
        ),
    )
