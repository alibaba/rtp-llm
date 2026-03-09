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
        "--enable_remote_cache",
        env_name="ENABLE_REMOTE_CACHE",
        bind_to=(kv_cache_config, "enable_remote_cache"),
        type=str2bool,
        default=False,
        help="控制是否启用Remote Cache的机制。设置为 True 启用 , False 关闭",
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
    kv_cache_group.add_argument(
        "--write_cache_sync",
        env_name="WRITE_CACHE_SYNC",
        bind_to=(kv_cache_config, "write_cache_sync"),
        type=str2bool,
        default=False,
        help="KVCache 同步写入开关. 当开启时, 在写入 Cache 时会等待写入完成. 默认关闭(即异步写入), Smoke 测试时需开启",
    )

    # Remote connector configuration arguments
    kv_cache_group.add_argument(
        "--reco_enable_vipserver",
        env_name="RECO_ENABLE_VIPSERVER",
        bind_to=(kv_cache_config, "reco_enable_vipserver"),
        type=str2bool,
        default=False,
        help="是否启用kvcm的VIPServer",
    )
    kv_cache_group.add_argument(
        "--reco_vipserver_domain",
        env_name="RECO_VIPSERVER_DOMAIN",
        bind_to=(kv_cache_config, "reco_vipserver_domain"),
        type=str,
        default="",
        help="kvcm VIPServer域名",
    )
    kv_cache_group.add_argument(
        "--reco_server_address",
        env_name="RECO_SERVER_ADDRESS",
        bind_to=(kv_cache_config, "reco_server_address"),
        type=str,
        default="",
        help="kvcm server地址",
    )
    kv_cache_group.add_argument(
        "--reco_instance_group",
        env_name="RECO_INSTANCE_GROUP",
        bind_to=(kv_cache_config, "reco_instance_group"),
        type=str,
        default="default",
        help="instance_group名称",
    )
    kv_cache_group.add_argument(
        "--reco_meta_channel_retry_time",
        env_name="RECO_META_CHANNEL_RETRY_TIME",
        bind_to=(kv_cache_config, "reco_meta_channel_retry_time"),
        type=int,
        default=3,
        help="grpc重试次数",
    )
    kv_cache_group.add_argument(
        "--reco_meta_channel_connection_timeout",
        env_name="RECO_META_CHANNEL_CONNECTION_TIMEOUT",
        bind_to=(kv_cache_config, "reco_meta_channel_connection_timeout"),
        type=int,
        default=6000,
        help="超时时间",
    )
    kv_cache_group.add_argument(
        "--reco_meta_channel_call_timeout",
        env_name="RECO_META_CHANNEL_CALL_TIMEOUT",
        bind_to=(kv_cache_config, "reco_meta_channel_call_timeout"),
        type=int,
        default=100,
        help="超时时间",
    )
    kv_cache_group.add_argument(
        "--reco_storage_thread_num",
        env_name="RECO_STORAGE_THREAD_NUM",
        bind_to=(kv_cache_config, "reco_storage_thread_num"),
        type=int,
        default=4,
        help="kvcm SdkWrapper中任务处理线程数量",
    )
    kv_cache_group.add_argument(
        "--reco_storage_queue_size",
        env_name="RECO_STORAGE_QUEUE_SIZE",
        bind_to=(kv_cache_config, "reco_storage_queue_size"),
        type=int,
        default=2000,
        help="kvcm SdkWrapper中线程池队列大小",
    )
    kv_cache_group.add_argument(
        "--reco_put_timeout_ms",
        env_name="RECO_PUT_TIMEOUT_MS",
        bind_to=(kv_cache_config, "reco_put_timeout_ms"),
        type=int,
        default=2000,
        help="PUT操作超时时间（毫秒）",
    )
    kv_cache_group.add_argument(
        "--reco_get_timeout_ms",
        env_name="RECO_GET_TIMEOUT_MS",
        bind_to=(kv_cache_config, "reco_get_timeout_ms"),
        type=int,
        default=2000,
        help="GET操作超时时间（毫秒）",
    )
    kv_cache_group.add_argument(
        "--reco_model_sdk_config",
        env_name="RECO_MODEL_SDK_CONFIG",
        bind_to=(kv_cache_config, "reco_model_sdk_config"),
        type=str,
        default='[{"type":"local","sdk_log_level":"DEBUG"}]',
        help="SDK 配置",
    )
    kv_cache_group.add_argument(
        "--reco_model_user_data",
        env_name="RECO_MODEL_USER_DATA",
        bind_to=(kv_cache_config, "reco_model_user_data"),
        type=str,
        default="",
        help="模型用户数据",
    )
    kv_cache_group.add_argument(
        "--reco_model_extra_info",
        env_name="RECO_MODEL_EXTRA_INFO",
        bind_to=(kv_cache_config, "reco_model_extra_info"),
        type=str,
        default="",
        help="模型额外信息",
    )
    kv_cache_group.add_argument(
        "--reco_instance_id_salt",
        env_name="RECO_INSTANCE_ID_SALT",
        bind_to=(kv_cache_config, "reco_instance_id_salt"),
        type=str,
        default="",
        help="实例 ID salt值",
    )
    kv_cache_group.add_argument(
        "--reco_asyncwrapper_thread_num",
        env_name="RECO_ASYNCWRAPPER_THREAD_NUM",
        bind_to=(kv_cache_config, "reco_asyncwrapper_thread_num"),
        type=int,
        default=16,
        help="异步包装器线程数量",
    )
    kv_cache_group.add_argument(
        "--reco_asyncwrapper_queue_size",
        env_name="RECO_ASYNCWRAPPER_QUEUE_SIZE",
        bind_to=(kv_cache_config, "reco_asyncwrapper_queue_size"),
        type=int,
        default=1000,
        help="异步包装器队列大小",
    )
    kv_cache_group.add_argument(
        "--reco_get_broadcast_timeout",
        env_name="RECO_GET_BROADCAST_TIMEOUT",
        bind_to=(kv_cache_config, "reco_get_broadcast_timeout"),
        type=int,
        default=2000,
        help="GET广播超时时间（毫秒）",
    )
    kv_cache_group.add_argument(
        "--reco_put_broadcast_timeout",
        env_name="RECO_PUT_BROADCAST_TIMEOUT",
        bind_to=(kv_cache_config, "reco_put_broadcast_timeout"),
        type=int,
        default=2000,
        help="PUT广播超时时间（毫秒）",
    )
    kv_cache_group.add_argument(
        "--reco_client_config",
        env_name="RECO_CLIENT_CONFIG",
        bind_to=(kv_cache_config, "reco_client_config"),
        type=int,
        default=2000,
    )
    kv_cache_group.add_argument(
        "--enable_tiered_memory_cache",
        env_name="ENABLE_TIERED_MEMORY_CACHE",
        bind_to=(kv_cache_config, "enable_tiered_memory_cache"),
        type=str2bool,
        default=False,
        help="分层 cache 开关。开启后，stream 释放时只全量写 remote，再按 GPU 空闲 block 阈值将冷 block 淘汰到 memory。",
    )
    kv_cache_group.add_argument(
        "--device_cache_min_free_blocks",
        env_name="DEVICE_CACHE_MIN_FREE_BLOCKS",
        bind_to=(kv_cache_config, "device_cache_min_free_blocks"),
        type=int,
        default=0,
        help="分层 cache 模式下 GPU 侧至少保留的空闲 block 数；当空闲 block 低于该阈值时，会把冷 block 从 GPU 淘汰到 memory。"
        "不填或填 0 时自动计算为 min(max_context_batch_size * max_seq_len, max_batch_tokens_size) / seq_size_per_block。",
    )
