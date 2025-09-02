from rtp_llm.server.server_args.util import str2bool


def init_kv_cache_group_args(parser):
    ##############################################################################################################
    # KV Cache 相关配置
    ##############################################################################################################
    kv_cache_group = parser.add_argument_group("KVCache")
    kv_cache_group.add_argument(
        "--reuse_cache",
        env_name="REUSE_CACHE",
        type=str2bool,
        default=False,
        help="控制是否激活KV Cache的重用机制。设置为 True 启用 , False 关闭",
    )
    kv_cache_group.add_argument(
        "--multi_task_prompt",
        env_name="MULTI_TASK_PROMPT",
        type=str,
        default=None,
        help="指定一个多任务提示（multi-task prompt），为一个路径，系统会读取路径指定的多任务json文件。默认为空",
    )
    kv_cache_group.add_argument(
        "--multi_task_prompt_str",
        env_name="MULTI_TASK_PROMPT_STR",
        type=str,
        default=None,
        help="指定一个多任务提示字符串（multi-task prompt string），为多任务纯json字符串，类似于系统提示词。默认为空 ",
    )
    kv_cache_group.add_argument(
        "--int8_kv_cache",
        env_name="INT8_KV_CACHE",
        type=int,
        default=0,
        help="是否开启INT8的KV_CACHE",
    )
    kv_cache_group.add_argument(
        "--fp8_kv_cache",
        env_name="FP8_KV_CACHE",
        type=int,
        default=0,
        help="是否开启FP8的KV_CACHE",
    )
    kv_cache_group.add_argument(
        "--kv_cache_mem_mb",
        env_name="KV_CACHE_MEM_MB",
        type=int,
        default=-1,
        help="KV_CACHE的大小",
    )
    kv_cache_group.add_argument(
        "--seq_size_per_block",
        env_name="SEQ_SIZE_PER_BLOCK",
        type=str,
        default=None,
        help="单独一个KV_CACHE的Block里面token的数量",
    )
    kv_cache_group.add_argument(
        "--test_block_num",
        env_name="TEST_BLOCK_NUM",
        type=int,
        default=0,
        help="在测试时强制指定BLOCK的数量",
    )
    kv_cache_group.add_argument(
        "--enable_3fs",
        env_name="ENABLE_3FS",
        type=str2bool,
        default=False,
        help="是否启用 3FS 存储 KVCache. 打开此开关需要先打开 REUSE_CACHE",
    )
    kv_cache_group.add_argument(
        "--match_timeout_ms",
        env_name="MATCH_TIMEOUT_MS",
        type=int,
        default=1000,
        help="所有 RANK 从远端匹配 KVCache 的超时时间, 单位为毫秒",
    )
    kv_cache_group.add_argument(
        "--rpc_get_cache_timeout_ms",
        env_name="RPC_GET_CACHE_TIMEOUT_MS",
        type=int,
        default=3000,
        help="所有 RANK 从远端拉取 KVCache 的超时时间, 单位为毫秒",
    )
    kv_cache_group.add_argument(
        "--rpc_put_cache_timeout_ms",
        env_name="RPC_PUT_CACHE_TIMEOUT_MS",
        type=int,
        default=3000,
        help="所有 RANK 向远端存储 KVCache 的超时时间, 单位为毫秒",
    )
    kv_cache_group.add_argument(
        "--threefs_read_timeout_ms",
        env_name="THREEFS_READ_TIMEOUT_MS",
        type=int,
        default=1000,
        help="3FS 读 KVCache 的超时时间, 单位为毫秒",
    )
    kv_cache_group.add_argument(
        "--threefs_write_timeout_ms",
        env_name="THREEFS_WRITE_TIMEOUT_MS",
        type=int,
        default=2000,
        help="3FS 写 KVCache 的超时时间, 单位为毫秒",
    )
    kv_cache_group.add_argument(
        "--max_block_size_per_item",
        env_name="MAX_BLOCK_SIZE_PER_ITEM",
        type=int,
        default=16,
        help="KVCache 分块存储每个 item 最大容纳 block 的数量",
    )
    kv_cache_group.add_argument(
        "--threefs_read_iov_size",
        env_name="THREEFS_READ_IOV_SIZE",
        type=int,
        default=1 << 32,
        help="3FS 读 IOV 大小, 单位为字节",
    )
    kv_cache_group.add_argument(
        "--threefs_write_iov_size",
        env_name="THREEFS_WRITE_IOV_SIZE",
        type=int,
        default=1 << 32,
        help="3FS 写 IOV 大小, 单位为字节",
    )
