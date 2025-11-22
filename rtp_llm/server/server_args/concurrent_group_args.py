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
