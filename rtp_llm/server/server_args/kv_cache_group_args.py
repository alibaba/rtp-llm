import logging
from ipaddress import IPv6Address

from rtp_llm.server.server_args.util import (
    bounded_int,
    non_negative_int64,
    positive_int32,
    str2bool,
)

KV_CACHE_EVENT_MAX_QUEUE_CAPACITY = 2**20
KV_CACHE_EVENT_MAX_REPORT_BATCH_SIZE = 2**14
KV_CACHE_EVENT_MAX_SNAPSHOT_KEYS = 1_000_000
KV_CACHE_EVENT_MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024


def _valid_port(port: str) -> bool:
    # std::from_chars in the C++ validator accepts ASCII decimal digits only.
    # Keep Python startup validation identical so configuration cannot pass one
    # entry point and fail after the engine starts through another.
    return port.isascii() and port.isdigit() and 1 <= int(port) <= 65535


def _valid_authority(authority: str, *, allow_bracketed_ipv6: bool) -> bool:
    if not authority or any(
        ord(char) <= 0x20 or ord(char) == 0x7F or char in "/?#@\\%"
        for char in authority
    ):
        return False

    if authority.startswith("["):
        close = authority.find("]")
        if (
            not allow_bracketed_ipv6
            or close <= 1
            or "[" in authority[1:]
            or "]" in authority[close + 1 :]
        ):
            return False
        host = authority[1:close]
        try:
            IPv6Address(host)
        except ValueError:
            return False
        suffix = authority[close + 1 :]
        return not suffix or (suffix.startswith(":") and _valid_port(suffix[1:]))

    if "[" in authority or "]" in authority or authority.count(":") > 1:
        return False
    if ":" in authority:
        host, port = authority.rsplit(":", 1)
        if not _valid_port(port):
            return False
    else:
        host = authority
    return bool(host) and not (
        host.startswith(".") or host.endswith(".") or ".." in host
    )


def _has_valid_percent_encoding(value: str) -> bool:
    """Reject malformed escapes before libcurl gets a different URL."""

    index = 0
    while index < len(value):
        if value[index] != "%":
            index += 1
            continue
        if index + 2 >= len(value) or any(
            char not in "0123456789abcdefABCDEF"
            for char in value[index + 1 : index + 3]
        ):
            return False
        index += 3
    return True


def _valid_manager_endpoint(endpoint: str) -> bool:
    # Must remain equivalent to detail::isValidKVCacheEventEndpoint(). Both
    # sides consume config/test/kv_cache_event_validation_cases.inc in tests.
    if any(ord(char) <= 0x20 or ord(char) >= 0x7F for char in endpoint):
        return False
    for scheme in ("http://", "https://"):
        if endpoint.startswith(scheme):
            remainder = endpoint[len(scheme) :]
            authority = remainder.split("/", 1)[0]
            return (
                _valid_authority(authority, allow_bracketed_ipv6=True)
                and not any(char in "?#\\" for char in remainder)
                and _has_valid_percent_encoding(remainder)
            )
    return False


def _valid_host_ip_port(host_ip_port: str) -> bool:
    # Must remain equivalent to detail::isValidKVCacheEventHostIpPort().
    # Keep this stricter than KVCM's current "non-empty and no #" check. The
    # value is embedded in both a path component and URI authority, where
    # non-ASCII, percent escapes, and delimiters would otherwise be ambiguous.
    return (
        _valid_authority(host_ip_port, allow_bracketed_ipv6=False)
        and host_ip_port.isascii()
        and "%" not in host_ip_port
    )


def _valid_kvcm_identity(identity: str) -> bool:
    """Match detail::isValidKVCacheEventIdentity()."""

    return (
        bool(identity)
        and identity.isascii()
        and all(0x20 < ord(char) < 0x7F for char in identity)
    )


def validate_kv_cache_event_config(parser, kv_cache_config) -> None:
    if kv_cache_config.kv_cache_event_publisher_type != "kvcm":
        return
    required = {
        "KV_CACHE_EVENT_MANAGER_ENDPOINT": (
            kv_cache_config.kv_cache_event_manager_endpoint
        ),
        "KV_CACHE_EVENT_INSTANCE_GROUP or RECO_INSTANCE_GROUP": (
            kv_cache_config.kv_cache_event_instance_group
            or kv_cache_config.reco_instance_group
        ),
        "KV_CACHE_EVENT_INSTANCE_ID": kv_cache_config.kv_cache_event_instance_id,
        "KV_CACHE_EVENT_HOST_IP_PORT": kv_cache_config.kv_cache_event_host_ip_port,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error("kvcm KV cache event publisher requires: " + ", ".join(missing))
    if not _valid_manager_endpoint(kv_cache_config.kv_cache_event_manager_endpoint):
        parser.error(
            "KV_CACHE_EVENT_MANAGER_ENDPOINT must be an http(s) URL with a "
            "valid authority and no credentials, query, fragment, or whitespace"
        )
    if not _valid_host_ip_port(kv_cache_config.kv_cache_event_host_ip_port):
        parser.error(
            "KV_CACHE_EVENT_HOST_IP_PORT must be a hostname or IPv4 address "
            "with an optional port in the range 1..65535; IPv6 is unsupported"
        )
    effective_group = (
        kv_cache_config.kv_cache_event_instance_group
        or kv_cache_config.reco_instance_group
    )
    if not _valid_kvcm_identity(effective_group):
        parser.error(
            "KV_CACHE_EVENT_INSTANCE_GROUP or RECO_INSTANCE_GROUP must be "
            "non-empty printable ASCII without whitespace"
        )
    if not _valid_kvcm_identity(kv_cache_config.kv_cache_event_instance_id):
        parser.error(
            "KV_CACHE_EVENT_INSTANCE_ID must be non-empty printable ASCII "
            "without whitespace"
        )
    if (
        not kv_cache_config.kv_cache_event_instance_group
        and kv_cache_config.reco_instance_group == "default"
    ):
        logging.warning(
            "KV cache event publisher is using the placeholder reco instance "
            "group 'default'; set KV_CACHE_EVENT_INSTANCE_GROUP or "
            "RECO_INSTANCE_GROUP explicitly to avoid cross-deployment grouping"
        )


def init_kv_cache_group_args(parser, kv_cache_config):
    # Validation belongs to the parser lifecycle rather than only setup_args(),
    # so every caller that installs this argument group gets the same contract.
    parser.register_post_parse_validator(
        lambda: validate_kv_cache_event_config(parser, kv_cache_config)
    )

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
        "--kv_cache_event_publisher_type",
        env_name="KV_CACHE_EVENT_PUBLISHER_TYPE",
        bind_to=(kv_cache_config, "kv_cache_event_publisher_type"),
        strict_config_binding=True,
        type=str,
        choices=["none", "log", "kvcm"],
        default="none",
        empty_env_as_unset=True,
        strict_env_choice=True,
        emit_string_from_env=True,
        help=(
            "HBM KV cache 事件输出：none 关闭，log 输出验证日志，kvcm 直连 KVCM。"
            "kvcm 模式必须同时配置 manager endpoint、instance id 和 host ip:port，"
            "instance group 可从 reco_instance_group 继承。"
        ),
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_manager_endpoint",
        env_name="KV_CACHE_EVENT_MANAGER_ENDPOINT",
        bind_to=(kv_cache_config, "kv_cache_event_manager_endpoint"),
        strict_config_binding=True,
        type=str,
        default="",
        empty_env_as_unset=True,
        emit_string_from_env=True,
        help="KVCM Meta HTTP endpoint，例如 http://127.0.0.1:56020。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_instance_group",
        env_name="KV_CACHE_EVENT_INSTANCE_GROUP",
        bind_to=(kv_cache_config, "kv_cache_event_instance_group"),
        strict_config_binding=True,
        type=str,
        default="",
        empty_env_as_unset=True,
        emit_string_from_env=True,
        help="KVCM instance group；为空时复用 reco_instance_group。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_instance_id",
        env_name="KV_CACHE_EVENT_INSTANCE_ID",
        bind_to=(kv_cache_config, "kv_cache_event_instance_id"),
        strict_config_binding=True,
        type=str,
        default="",
        empty_env_as_unset=True,
        emit_string_from_env=True,
        help="稳定的 KVCM instance id，不能使用进程 PID。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_host_ip_port",
        env_name="KV_CACHE_EVENT_HOST_IP_PORT",
        bind_to=(kv_cache_config, "kv_cache_event_host_ip_port"),
        strict_config_binding=True,
        type=str,
        default="",
        empty_env_as_unset=True,
        emit_string_from_env=True,
        help="当前 DP replica 的 tp_rank=0 Cache 协调端点；pp_size>1 时该功能禁用。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_queue_capacity",
        env_name="KV_CACHE_EVENT_QUEUE_CAPACITY",
        bind_to=(kv_cache_config, "kv_cache_event_queue_capacity"),
        strict_config_binding=True,
        type=bounded_int(1, KV_CACHE_EVENT_MAX_QUEUE_CAPACITY),
        default=100000,
        empty_env_as_unset=True,
        help=(
            "Publisher 非阻塞有界队列容量，范围 1..1048576；"
            "资源上限防止可选事件导出器耗尽进程内存。"
        ),
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_report_batch_size",
        env_name="KV_CACHE_EVENT_REPORT_BATCH_SIZE",
        bind_to=(kv_cache_config, "kv_cache_event_report_batch_size"),
        strict_config_binding=True,
        type=bounded_int(1, KV_CACHE_EVENT_MAX_REPORT_BATCH_SIZE),
        default=1000,
        empty_env_as_unset=True,
        help="单次 ReportEvent 的最大增量事件数，范围 1..16384。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_flush_interval_ms",
        env_name="KV_CACHE_EVENT_FLUSH_INTERVAL_MS",
        bind_to=(kv_cache_config, "kv_cache_event_flush_interval_ms"),
        strict_config_binding=True,
        type=positive_int32,
        default=20,
        empty_env_as_unset=True,
        help="增量 batch 最长等待时间，必须大于 0 ms。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_heartbeat_interval_ms",
        env_name="KV_CACHE_EVENT_HEARTBEAT_INTERVAL_MS",
        bind_to=(kv_cache_config, "kv_cache_event_heartbeat_interval_ms"),
        strict_config_binding=True,
        type=positive_int32,
        default=1000,
        empty_env_as_unset=True,
        help="KVCM 节点心跳周期，必须大于 0 ms。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_request_timeout_ms",
        env_name="KV_CACHE_EVENT_REQUEST_TIMEOUT_MS",
        bind_to=(kv_cache_config, "kv_cache_event_request_timeout_ms"),
        strict_config_binding=True,
        type=positive_int32,
        default=1500,
        empty_env_as_unset=True,
        help="KVCM 注册、心跳和增量 HTTP 请求超时，必须大于 0 ms。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_snapshot_timeout_ms",
        env_name="KV_CACHE_EVENT_SNAPSHOT_TIMEOUT_MS",
        bind_to=(kv_cache_config, "kv_cache_event_snapshot_timeout_ms"),
        strict_config_binding=True,
        type=positive_int32,
        default=30000,
        empty_env_as_unset=True,
        help="KVCM 全量 snapshot HTTP 请求超时，必须大于 0 ms；全量请求可能包含大量 key。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_retry_interval_ms",
        env_name="KV_CACHE_EVENT_RETRY_INTERVAL_MS",
        bind_to=(kv_cache_config, "kv_cache_event_retry_interval_ms"),
        strict_config_binding=True,
        type=positive_int32,
        default=500,
        empty_env_as_unset=True,
        help="注册或发送失败后的重试间隔，必须大于 0 ms。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_snapshot_interval_ms",
        env_name="KV_CACHE_EVENT_SNAPSHOT_INTERVAL_MS",
        bind_to=(kv_cache_config, "kv_cache_event_snapshot_interval_ms"),
        strict_config_binding=True,
        type=positive_int32,
        default=300000,
        empty_env_as_unset=True,
        help="HBM 全量 snapshot 周期，必须大于 0 ms；瞬态异常和服务端 advisory 会触发 snapshot。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_log_max_keys",
        env_name="KV_CACHE_EVENT_LOG_MAX_KEYS",
        bind_to=(kv_cache_config, "kv_cache_event_log_max_keys"),
        strict_config_binding=True,
        type=non_negative_int64,
        default=8,
        empty_env_as_unset=True,
        help="LogPublisher 每个 batch 最多输出的 key 样本数，必须大于等于 0；0 表示不输出样本 key。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_snapshot_max_keys",
        env_name="KV_CACHE_EVENT_SNAPSHOT_MAX_KEYS",
        bind_to=(kv_cache_config, "kv_cache_event_snapshot_max_keys"),
        strict_config_binding=True,
        type=bounded_int(1, KV_CACHE_EVENT_MAX_SNAPSHOT_KEYS),
        default=1000000,
        empty_env_as_unset=True,
        help="发布器内存镜像允许的最大逻辑 key 数，范围 1..1000000；超限时熔断事件导出。",
    )
    kv_cache_group.add_argument(
        "--kv_cache_event_snapshot_max_bytes",
        env_name="KV_CACHE_EVENT_SNAPSHOT_MAX_BYTES",
        bind_to=(kv_cache_config, "kv_cache_event_snapshot_max_bytes"),
        strict_config_binding=True,
        type=bounded_int(1, KV_CACHE_EVENT_MAX_SNAPSHOT_BYTES),
        default=256 * 1024 * 1024,
        empty_env_as_unset=True,
        help="单个 KVCM 全量 snapshot JSON 的最大字节数，范围 1..268435456；超限时熔断事件导出。",
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
        default=0,
        help="单独一个KV_CACHE的Block里面token的数量, 0表示使用平台默认值(CUDA:64, PPU:256, ROCm:16)",
    )
    kv_cache_group.add_argument(
        "--kernel_seq_size_per_block",
        env_name="KERNEL_SEQ_SIZE_PER_BLOCK",
        bind_to=(kv_cache_config, "kernel_seq_size_per_block"),
        type=int,
        default=0,
        help="Attention算子使用的kernel block大小（token数量）。0表示与seq_size_per_block相同。",
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
        "--ssm_state_dtype",
        env_name="SSM_STATE_DTYPE",
        bind_to=(kv_cache_config, "ssm_state_dtype"),
        type=str,
        choices=["bf16", "fp32"],
        default="bf16",
        help="线性注意力 SSM state 的数据类型。默认 bf16，可选 fp32 和 bf16。",
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
        "--enable_memory_cache_sm_copy",
        env_name="ENABLE_MEMORY_CACHE_SM_COPY",
        bind_to=(kv_cache_config, "enable_memory_cache_sm_copy"),
        type=str2bool,
        default=False,
        help="内存 Cache 拷贝是否启用 split-KV SM scatter/gather（CUDA 上满足布局条件时）。默认 False；True 时满足条件可走 SM copy。",
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
        "--enable_memory_cache_disk",
        env_name="ENABLE_MEMORY_CACHE_DISK",
        bind_to=(kv_cache_config, "enable_memory_cache_disk"),
        type=str2bool,
        default=False,
        help="控制是否启用磁盘 KV cache。默认关闭。",
    )
    kv_cache_group.add_argument(
        "--memory_cache_disk_paths",
        env_name="MEMORY_CACHE_DISK_PATHS",
        bind_to=(kv_cache_config, "memory_cache_disk_paths"),
        type=str,
        default="",
        help="磁盘 KV cache 路径；多个路径的格式由磁盘 cache 实现解析。",
    )
    kv_cache_group.add_argument(
        "--memory_cache_disk_size_mb",
        env_name="MEMORY_CACHE_DISK_SIZE_MB",
        bind_to=(kv_cache_config, "memory_cache_disk_size_mb"),
        type=int,
        default=0,
        help="单个 rank 的磁盘 KV cache 容量，单位 MB。",
    )
    kv_cache_group.add_argument(
        "--memory_cache_disk_buffered_io",
        env_name="MEMORY_CACHE_DISK_BUFFERED_IO",
        bind_to=(kv_cache_config, "memory_cache_disk_buffered_io"),
        type=str2bool,
        default=True,
        help="磁盘 KV cache 是否使用 buffered I/O。默认开启。",
    )
    kv_cache_group.add_argument(
        "--memory_cache_disk_sync_timeout_ms",
        env_name="MEMORY_CACHE_DISK_SYNC_TIMEOUT_MS",
        bind_to=(kv_cache_config, "memory_cache_disk_sync_timeout_ms"),
        type=int,
        default=30000,
        help="磁盘 KV cache 同步超时，单位毫秒。",
    )
    kv_cache_group.add_argument(
        "--enable_gpu_prefix_tree",
        env_name="ENABLE_GPU_PREFIX_TREE",
        bind_to=(kv_cache_config, "enable_gpu_prefix_tree"),
        type=str2bool,
        default=False,
        help="控制是否启用 GPU prefix-tree cache 策略。默认关闭。",
    )
    kv_cache_group.add_argument(
        "--enable_prefix_tree_memory_cache",
        env_name="ENABLE_PREFIX_TREE_MEMORY_CACHE",
        bind_to=(kv_cache_config, "enable_prefix_tree_memory_cache"),
        type=str2bool,
        default=False,
        help="控制 memory cache 是否使用 prefix-tree 策略。默认关闭。",
    )
    kv_cache_group.add_argument(
        "--enable_legacy_memory_connector_fallback",
        env_name="ENABLE_LEGACY_MEMORY_CONNECTOR_FALLBACK",
        bind_to=(kv_cache_config, "enable_legacy_memory_connector_fallback"),
        type=str2bool,
        default=True,
        help="新 memory cache 路径不可用时是否回退 legacy connector。默认开启。",
    )
    kv_cache_group.add_argument(
        "--prefix_tree_memory_state_swa_pool_ratio",
        env_name="PREFIX_TREE_MEMORY_STATE_SWA_POOL_RATIO",
        bind_to=(kv_cache_config, "prefix_tree_memory_state_swa_pool_ratio"),
        type=int,
        default=0,
        help="Prefix-tree memory cache 中 state/SWA pool 的容量比例。",
    )
    kv_cache_group.add_argument(
        "--enable_independent_group_eviction",
        env_name="ENABLE_INDEPENDENT_GROUP_EVICTION",
        bind_to=(kv_cache_config, "enable_independent_group_eviction"),
        type=str2bool,
        default=False,
        help="控制各 KV cache group 是否独立驱逐。默认关闭。",
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
        emit_string_from_env=True,
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
        default=1500,
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
        default=12000,
        help="PUT操作超时时间（毫秒）",
    )
    kv_cache_group.add_argument(
        "--reco_get_timeout_ms",
        env_name="RECO_GET_TIMEOUT_MS",
        bind_to=(kv_cache_config, "reco_get_timeout_ms"),
        type=int,
        default=12000,
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
        default=15000,
        help="GET广播超时时间（毫秒）",
    )
    kv_cache_group.add_argument(
        "--reco_put_broadcast_timeout",
        env_name="RECO_PUT_BROADCAST_TIMEOUT",
        bind_to=(kv_cache_config, "reco_put_broadcast_timeout"),
        type=int,
        default=15000,
        help="PUT广播超时时间（毫秒）",
    )
    kv_cache_group.add_argument(
        "--reco_client_config",
        env_name="RECO_CLIENT_CONFIG",
        bind_to=(kv_cache_config, "reco_client_config"),
        type=str,
        default="",
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
