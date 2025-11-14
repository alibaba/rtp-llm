from rtp_llm.server.server_args.util import str2bool


def init_threefs_group_args(parser, kv_cache_config):
    ##############################################################################################################
    # 3FS 配置
    ##############################################################################################################
    threefs_group = parser.add_argument_group("3FS")
    threefs_group.add_argument(
        "--enable_3fs",
        env_name="ENABLE_3FS",
        bind_to=(kv_cache_config, 'enable_3fs'),
        type=str2bool,
        default=False,
        help="是否启用 3FS 存储 KVCache. 打开此开关需要先打开 REUSE_CACHE",
    )
    threefs_group.add_argument(
        "--threefs_match_timeout_ms",
        env_name="THREEFS_MATCH_TIMEOUT_MS",
        bind_to=(kv_cache_config, 'match_timeout_ms'),
        type=int,
        default=1000,
        help="所有 RANK 从远端匹配 KVCache 的超时时间, 单位为毫秒",
    )
    threefs_group.add_argument(
        "--threefs_rpc_get_cache_timeout_ms",
        env_name="THREEFS_RPC_GET_CACHE_TIMEOUT_MS",
        bind_to=(kv_cache_config, 'rpc_get_cache_timeout_ms'),
        type=int,
        default=3000,
        help="所有 RANK 从远端拉取 KVCache 的超时时间, 单位为毫秒",
    )
    threefs_group.add_argument(
        "--threefs_rpc_put_cache_timeout_ms",
        env_name="THREEFS_RPC_PUT_CACHE_TIMEOUT_MS",
        bind_to=(kv_cache_config, 'rpc_put_cache_timeout_ms'),
        type=int,
        default=3000,
        help="所有 RANK 向远端存储 KVCache 的超时时间, 单位为毫秒",
    )
    threefs_group.add_argument(
        "--threefs_read_timeout_ms",
        env_name="THREEFS_READ_TIMEOUT_MS",
        bind_to=(kv_cache_config, 'threefs_read_timeout_ms'),
        type=int,
        default=1000,
        help="3FS 读 KVCache 的超时时间, 单位为毫秒",
    )
    threefs_group.add_argument(
        "--threefs_write_timeout_ms",
        env_name="THREEFS_WRITE_TIMEOUT_MS",
        bind_to=(kv_cache_config, 'threefs_write_timeout_ms'),
        type=int,
        default=2000,
        help="3FS 写 KVCache 的超时时间, 单位为毫秒",
    )
    threefs_group.add_argument(
        "--threefs_read_iov_size",
        env_name="THREEFS_READ_IOV_SIZE",
        bind_to=(kv_cache_config, 'threefs_read_iov_size'),
        type=int,
        default=1 << 32,
        help="3FS 读 IOV 大小, 单位为字节",
    )
    threefs_group.add_argument(
        "--threefs_write_iov_size",
        env_name="THREEFS_WRITE_IOV_SIZE",
        bind_to=(kv_cache_config, 'threefs_write_iov_size'),
        type=int,
        default=1 << 32,
        help="3FS 写 IOV 大小, 单位为字节",
    )
