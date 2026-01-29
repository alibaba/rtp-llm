from rtp_llm.server.server_args.util import str2bool


def init_kv_cache_group_args(parser, kv_cache_config):
    ##############################################################################################################
    # KV Cache 相关配置
    ##############################################################################################################
    kv_cache_group = parser.add_argument_group("KVCache")
    kv_cache_group.add_argument(
        "--reuse_cache",
        env_name="REUSE_CACHE",
        bind_to=(kv_cache_config, "reuse_cache"),
        type=str2bool,
        default=False,
        help="控制是否激活KV Cache的重用机制, 默认开启显存重用, 其他cache重用需手动开启。设置为 True 启用 , False 关闭",
    )
    kv_cache_group.add_argument(
        "--enable_device_cache",
        env_name="ENABLE_DEVICE_CACHE",
        bind_to=(kv_cache_config, "enable_device_cache"),
        type=str2bool,
        default=True,
        help="控制是否启用显存Cache的重用机制, 默认开启。设置为 True 启用 , False 关闭",
    )
    kv_cache_group.add_argument(
        "--reserve_block_ratio",
        env_name="RESERVE_BLOCK_RATIO",
        bind_to=(kv_cache_config, "reserve_block_ratio"),
        type=int,
        default=5,
        help="KV cache 预留 block 的百分比（仅对首次分配/空 batch_kv_resource 生效），用于保护正在运行的 stream 后续增量申请。",
    )
    kv_cache_group.add_argument(
        "--multi_task_prompt",
        env_name="MULTI_TASK_PROMPT",
        bind_to=(kv_cache_config, "multi_task_prompt"),
        type=str,
        default=None,
        help="指定一个多任务提示（multi-task prompt），为一个路径，系统会读取路径指定的多任务json文件。默认为空",
    )
    kv_cache_group.add_argument(
        "--multi_task_prompt_str",
        env_name="MULTI_TASK_PROMPT_STR",
        bind_to=(kv_cache_config, "multi_task_prompt_str"),
        type=str,
        default=None,
        help="指定一个多任务提示字符串（multi-task prompt string），为多任务纯json字符串，类似于系统提示词。默认为空 ",
    )
    kv_cache_group.add_argument(
        "--int8_kv_cache",
        env_name="INT8_KV_CACHE",
        bind_to=(kv_cache_config, "int8_kv_cache"),
        type=int,
        default=0,
        help="是否开启INT8的KV_CACHE",
    )
    kv_cache_group.add_argument(
        "--fp8_kv_cache",
        env_name="FP8_KV_CACHE",
        bind_to=(kv_cache_config, "fp8_kv_cache"),
        type=int,
        help="是否开启FP8的KV_CACHE",
    )
    # compatible with old version
    kv_cache_group.add_argument(
        "--blockwise_use_fp8_kv_cache",
        env_name="BLOCKWISE_USE_FP8_KV_CACHE",
        bind_to=(kv_cache_config, "fp8_kv_cache"),
        type=int,
        help="是否开启FP8的KV_CACHE",
    )
    kv_cache_group.add_argument(
        "--kv_cache_mem_mb",
        env_name="KV_CACHE_MEM_MB",
        bind_to=(kv_cache_config, "kv_cache_mem_mb"),
        type=int,
        default=-1,
        help="KV_CACHE的大小",
    )
    kv_cache_group.add_argument(
        "--seq_size_per_block",
        env_name="SEQ_SIZE_PER_BLOCK",
        bind_to=(kv_cache_config, "seq_size_per_block"),
        type=int,
        default=64,
        help="单独一个KV_CACHE的Block里面token的数量",
    )
    kv_cache_group.add_argument(
        "--linear_step",
        env_name="LINEAR_STEP",
        bind_to=(kv_cache_config, "linear_step"),
        type=int,
        default=1,
        help="线性注意力（Linear Attention）缓存重用的步长：每隔 linear_step 个 block 额外保留一个 block（>=1）。",
    )
    kv_cache_group.add_argument(
        "--test_block_num",
        env_name="TEST_BLOCK_NUM",
        bind_to=(kv_cache_config, "test_block_num"),
        type=int,
        default=0,
        help="在测试时强制指定BLOCK的数量",
    )
    kv_cache_group.add_argument(
        "--enable_memory_cache",
        env_name="ENABLE_MEMORY_CACHE",
        bind_to=(kv_cache_config, "enable_memory_cache"),
        type=str2bool,
        default=False,
        help="内存 KVCache 开关. 当开启时, 需要显示通过 MEMORY_CACHE_SIZE_MB 设置内存大小",
    )
    kv_cache_group.add_argument(
        "--memory_cache_size_mb",
        env_name="MEMORY_CACHE_SIZE_MB",
        bind_to=(kv_cache_config, "memory_cache_size_mb"),
        type=int,
        default=0,
        help="单个RANK Memory Cache 的大小, 单位为MB",
    )
    kv_cache_group.add_argument(
        "--memory_cache_sync_timeout_ms",
        env_name="MEMORY_CACHE_SYNC_TIMEOUT_MS",
        bind_to=(kv_cache_config, "memory_cache_sync_timeout_ms"),
        type=int,
        default=10000,
        help="Memory Cache 多TP同步的超时时间, 单位为毫秒",
    )
