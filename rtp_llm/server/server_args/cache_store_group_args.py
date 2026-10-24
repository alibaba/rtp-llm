import argparse
import logging

from rtp_llm.ops import CacheStoreConfig
from rtp_llm.server.server_args.util import str2bool

RDMA_DEVICE_HEALTH_FAULT_HANDLERS = ["LOG", "ABORT"]


def _rdma_device_health_range(field_name: str):
    return getattr(CacheStoreConfig, f"{field_name}_range")


def _rdma_device_health_int(field_name: str):
    min_value, max_value = _rdma_device_health_range(field_name)

    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as error:
            raise argparse.ArgumentTypeError(
                f"{field_name} must be an integer, got {value!r}"
            ) from error
        if parsed < min_value or parsed > max_value:
            raise argparse.ArgumentTypeError(
                f"{field_name} must be in [{min_value}, {max_value}], got {parsed}"
            )
        return parsed

    return parse


def validate_rdma_device_health_config(cache_store_config) -> None:
    min_intervals = CacheStoreConfig.rdma_device_health_min_confirm_probe_intervals
    probe_interval_ms = cache_store_config.rdma_device_health_probe_interval_ms
    for field_name in (
        "rdma_device_health_device_fault_confirm_ms",
        "rdma_device_health_unavailable_threshold_ms",
    ):
        value = getattr(cache_store_config, field_name)
        if value < min_intervals * probe_interval_ms:
            raise ValueError(
                f"{field_name} ({value}) must be at least "
                f"{min_intervals} * rdma_device_health_probe_interval_ms ({probe_interval_ms})"
            )
    unavailable_threshold_ms = (
        cache_store_config.rdma_device_health_unavailable_threshold_ms
    )
    fault_window_ms = cache_store_config.rdma_device_health_fault_window_ms
    if unavailable_threshold_ms > fault_window_ms:
        raise ValueError(
            f"rdma_device_health_unavailable_threshold_ms ({unavailable_threshold_ms}) must not exceed "
            f"rdma_device_health_fault_window_ms ({fault_window_ms})"
        )

    # 相邻恢复至少间隔两轮探测（UNAVAILABLE、AVAILABLE）；窗口包含两端点，故上限加一。
    # 查询及上报耗时只会降低实际次数；超过理论上限时提示，不拒绝启动或改写阈值。
    max_recoveries = fault_window_ms // (2 * probe_interval_ms) + 1
    flap_warn_threshold = cache_store_config.rdma_device_health_flap_warn_threshold
    if flap_warn_threshold > max_recoveries:
        logging.warning(
            "[RDMA_DEVICE_HEALTH_CONFIG] rdma_device_health_flap_warn_threshold (%d) "
            "exceeds the theoretical maximum recoveries (%d) within "
            "rdma_device_health_fault_window_ms (%d) at "
            "rdma_device_health_probe_interval_ms (%d); flapping warnings cannot "
            "trigger with this configuration. Startup continues without changing the threshold.",
            flap_warn_threshold,
            max_recoveries,
            fault_window_ms,
            probe_interval_ms,
        )


def init_cache_store_group_args(parser, cache_store_config):
    ##############################################################################################################
    # Cache Store 配置
    ##############################################################################################################
    cache_store_group = parser.add_argument_group("Cache Store")
    cache_store_group.add_argument(
        "--cache_store_rdma_mode",
        env_name="CACHE_STORE_RDMA_MODE",
        bind_to=(cache_store_config, "cache_store_rdma_mode"),
        type=str2bool,
        default=False,
        help="控制 cache store 是否使用 RDMA 模式。",
    )

    cache_store_group.add_argument(
        "--wrr_available_ratio",
        env_name="WRR_AVAILABLE_RATIO",
        bind_to=(cache_store_config, "wrr_available_ratio"),
        type=int,
        default=80,
        help="为 WRR (Weighted Round Robin) 负载均衡器设置的可用性阈值百分比 (0-100)，数值越低越容易启用动态权重分配，但可能降低全局负载均衡准确性。",
    )

    cache_store_group.add_argument(
        "--rank_factor",
        env_name="RANK_FACTOR",
        bind_to=(cache_store_config, "rank_factor"),
        type=int,
        default=0,
        choices=[0, 1],
        help="指定 WRR 负载均衡器使用的排序因子。`0` 表示基于 KV_CACHE 使用情况排序，`1` 表示基于正在处理的请求数 (ONFLIGHT_REQUESTS) 排序。",
    )

    cache_store_group.add_argument(
        "--cache_store_thread_count",
        env_name="CACHE_STORE_THREAD_COUNT",
        bind_to=(cache_store_config, "thread_count"),
        type=int,
        default=32,
        help="为 cache store 线程池设置线程数量。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_connect_timeout_ms",
        env_name="CACHE_STORE_RDMA_CONNECT_TIMEOUT_MS",
        bind_to=(cache_store_config, "rdma_connect_timeout_ms"),
        type=int,
        default=250,
        help="为 cache store RDMA 连接设置超时时间，单位为毫秒。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_qp_count_per_connection",
        env_name="CACHE_STORE_RDMA_QP_COUNT_PER_CONNECTION",
        bind_to=(cache_store_config, "rdma_qp_count_per_connection"),
        type=int,
        default=2,
        help="为 cache store RDMA 连接设置每个连接的底层QP数量。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_io_thread_count",
        env_name="CACHE_STORE_RDMA_IO_THREAD_COUNT",
        bind_to=(cache_store_config, "rdma_io_thread_count"),
        type=int,
        default=4,
        help="为 cache store RDMA 通信层设置 IO 线程数量。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_worker_thread_count",
        env_name="CACHE_STORE_RDMA_WORKER_THREAD_COUNT",
        bind_to=(cache_store_config, "rdma_worker_thread_count"),
        type=int,
        default=2,
        help="为 cache store RDMA 通信层设置 worker 线程数量。",
    )

    cache_store_group.add_argument(
        "--cache_store_rdma_max_block_pairs_per_connection",
        env_name="CACHE_STORE_RDMA_MAX_BLOCK_PAIRS_PER_CONNECTION",
        bind_to=(cache_store_config, "rdma_max_block_pairs_per_connection"),
        type=int,
        default=0,
        help="限制单个 RDMA 连接读操作中包含的 block_pair 数量，0 表示不拆分。",
    )

    cache_store_group.add_argument(
        "--cache_store_messager_io_thread_count",
        env_name="CACHE_STORE_MESSAGER_IO_THREAD_COUNT",
        bind_to=(cache_store_config, "messager_io_thread_count"),
        type=int,
        default=2,
        help="为 cache store P2P messager 通信层设置 IO 线程数量。",
    )
    cache_store_group.add_argument(
        "--cache_store_messager_worker_thread_count",
        env_name="CACHE_STORE_MESSAGER_WORKER_THREAD_COUNT",
        bind_to=(cache_store_config, "messager_worker_thread_count"),
        type=int,
        default=32,
        help="为 cache store P2P messager 通信层设置 worker 线程数量。",
    )
    cache_store_group.add_argument(
        "--cache_store_rdma_transfer_wait_timeout_ms",
        env_name="CACHE_STORE_RDMA_TRANSFER_WAIT_TIMEOUT_MS",
        bind_to=(cache_store_config, "rdma_transfer_wait_timeout_ms"),
        type=int,
        default=180000,
        help="RDMA 传输完成最大等待超时时间（毫秒），默认 180 秒。",
    )

    cache_store_group.add_argument(
        "--p2p_read_steal_before_deadline_ms",
        env_name="P2P_READ_STEAL_BEFORE_DEADLINE_MS",
        bind_to=(cache_store_config, "p2p_read_steal_before_deadline_ms"),
        type=int,
        default=250,
        help="Decode read：距 deadline 小于该毫秒数时从 recv store steal，阻止新 transfer 匹配。",
    )
    cache_store_group.add_argument(
        "--p2p_read_return_before_deadline_ms",
        env_name="P2P_READ_RETURN_BEFORE_DEADLINE_MS",
        bind_to=(cache_store_config, "p2p_read_return_before_deadline_ms"),
        type=int,
        default=100,
        help="Decode read 与 Prefill send：transfer 须在 deadline 前该毫秒数内完成（与对端对齐）。",
    )
    cache_store_group.add_argument(
        "--p2p_transfer_not_done_resource_hold_ms",
        env_name="P2P_TRANSFER_NOT_DONE_RESOURCE_HOLD_MS",
        bind_to=(cache_store_config, "p2p_transfer_not_done_resource_hold_ms"),
        type=int,
        default=10000,
        help="Scheduler：TRANSFER_NOT_DONE 后延迟 done 以保留显存安全窗口（毫秒）。",
    )

    cache_store_group.add_argument(
        "--p2p_resource_store_timeout_check_interval_ms",
        env_name="P2P_RESOURCE_STORE_TIMEOUT_CHECK_INTERVAL_MS",
        bind_to=(cache_store_config, "p2p_resource_store_timeout_check_interval_ms"),
        type=int,
        default=100,
        help="P2P decode 侧资源 store 周期扫描超时资源的间隔（毫秒）。",
    )
    cache_store_group.add_argument(
        "--p2p_layer_cache_buffer_store_timeout_ms",
        env_name="P2P_LAYER_CACHE_BUFFER_STORE_TIMEOUT_MS",
        bind_to=(cache_store_config, "p2p_layer_cache_buffer_store_timeout_ms"),
        type=int,
        default=100000,
        help="P2P LayerCacheBufferStore 条目保留时长（毫秒），默认 100s。",
    )
    cache_store_group.add_argument(
        "--p2p_cancel_broadcast_timeout_ms",
        env_name="P2P_CANCEL_BROADCAST_TIMEOUT_MS",
        bind_to=(cache_store_config, "p2p_cancel_broadcast_timeout_ms"),
        type=int,
        default=1000,
        help="P2P Scheduler 广播 CANCEL 时的 gRPC 超时（毫秒）。",
    )
    cache_store_group.add_argument(
        "--cache_store_tcp_anet_rpc_thread_num",
        env_name="CACHE_STORE_TCP_ANET_RPC_THREAD_NUM",
        bind_to=(cache_store_config, "cache_store_tcp_anet_rpc_thread_num"),
        type=int,
        default=3,
        help="P2P TCP 控制面 ANetRPCServer threadNum（与 ArpcServerWrapper.threadNum 语义一致）。",
    )
    cache_store_group.add_argument(
        "--cache_store_tcp_anet_rpc_queue_num",
        env_name="CACHE_STORE_TCP_ANET_RPC_QUEUE_NUM",
        bind_to=(cache_store_config, "cache_store_tcp_anet_rpc_queue_num"),
        type=int,
        default=100,
        help="P2P TCP 控制面 ANetRPCServer queueNum。",
    )
    probe_interval_range = _rdma_device_health_range(
        "rdma_device_health_probe_interval_ms"
    )
    device_fault_confirm_range = _rdma_device_health_range(
        "rdma_device_health_device_fault_confirm_ms"
    )
    fault_window_range = _rdma_device_health_range("rdma_device_health_fault_window_ms")
    unavailable_threshold_range = _rdma_device_health_range(
        "rdma_device_health_unavailable_threshold_ms"
    )
    flap_threshold_range = _rdma_device_health_range(
        "rdma_device_health_flap_warn_threshold"
    )
    min_confirm_intervals = (
        CacheStoreConfig.rdma_device_health_min_confirm_probe_intervals
    )
    cache_store_group.add_argument(
        "--rdma_device_health_enabled",
        env_name="RDMA_DEVICE_HEALTH_ENABLED",
        bind_to=(cache_store_config, "rdma_device_health_enabled"),
        type=str2bool,
        default=True,
        help="CacheStore / P2P RDMA 初始化成功后启用健康监控；监控启动返回失败不改变已成功的 RDMA 初始化结果。",
    )
    cache_store_group.add_argument(
        "--rdma_device_health_fault_handler",
        env_name="RDMA_DEVICE_HEALTH_FAULT_HANDLER",
        bind_to=(cache_store_config, "rdma_device_health_fault_handler"),
        type=str,
        choices=RDMA_DEVICE_HEALTH_FAULT_HANDLERS,
        default="LOG",
        help=(
            "全 RDMA 设备硬故障确认后的处理：LOG 每个总体故障期记录一次，ABORT 在首次确认时退出。"
            "累计故障期阈值只产生限频 WARN；设备确认优先于同轮 WARN，后续 UNAVAILABLE 轮仍可告警。"
        ),
    )
    cache_store_group.add_argument(
        "--rdma_device_health_probe_interval_ms",
        env_name="RDMA_DEVICE_HEALTH_PROBE_INTERVAL_MS",
        bind_to=(cache_store_config, "rdma_device_health_probe_interval_ms"),
        type=_rdma_device_health_int("rdma_device_health_probe_interval_ms"),
        default=1000,
        help="RDMA 设备健康探测间隔，单位毫秒，范围 {}-{}。".format(
            *probe_interval_range
        ),
    )
    cache_store_group.add_argument(
        "--rdma_device_health_device_fault_confirm_ms",
        env_name="RDMA_DEVICE_HEALTH_DEVICE_FAULT_CONFIRM_MS",
        bind_to=(cache_store_config, "rdma_device_health_device_fault_confirm_ms"),
        type=_rdma_device_health_int("rdma_device_health_device_fault_confirm_ms"),
        default=3000,
        help=(
            "全设备移除或硬查询错误的 UNAVAILABLE 轮启动确认计时；再次出现该状态且计时达到阈值才确认。"
            "UNKNOWN 无成功查询的明确健康设备证据时保留起点，经过时间继续计入，但当轮不确认。"
            "AVAILABLE、DEGRADED、成功查询中有明确 device HEALTHY 证据的 UNKNOWN，或非全设备硬故障的 UNAVAILABLE 清除计时。"
            "单位毫秒，范围 {}-{}，至少覆盖 {} 个探测间隔。"
        ).format(
            *device_fault_confirm_range, min_confirm_intervals
        ),
    )
    cache_store_group.add_argument(
        "--rdma_device_health_fault_window_ms",
        env_name="RDMA_DEVICE_HEALTH_FAULT_WINDOW_MS",
        bind_to=(cache_store_config, "rdma_device_health_fault_window_ms"),
        type=_rdma_device_health_int("rdma_device_health_fault_window_ms"),
        default=600000,
        help=(
            "累计总体故障期时长与抖动告警共用的滑动窗口。故障期从首次 UNAVAILABLE 到下次 AVAILABLE，"
            "包含中间 UNKNOWN/DEGRADED。单位毫秒，范围 {}-{}。"
        ).format(
            *fault_window_range
        ),
    )
    cache_store_group.add_argument(
        "--rdma_device_health_unavailable_threshold_ms",
        env_name="RDMA_DEVICE_HEALTH_UNAVAILABLE_THRESHOLD_MS",
        bind_to=(cache_store_config, "rdma_device_health_unavailable_threshold_ms"),
        type=_rdma_device_health_int("rdma_device_health_unavailable_threshold_ms"),
        default=120000,
        help=(
            "窗口内总体故障期累计阈值，包含中间 UNKNOWN/DEGRADED；仅 UNAVAILABLE 轮评估，只 WARN、不退出。"
            "设备确认优先于同轮 WARN；告警至少相隔一个窗口，恢复不重置限频。"
            "单位毫秒，范围 {}-{}，至少覆盖 {} 个探测间隔且不得超过窗口。"
        ).format(
            *unavailable_threshold_range, min_confirm_intervals
        ),
    )
    cache_store_group.add_argument(
        "--rdma_device_health_flap_warn_threshold",
        env_name="RDMA_DEVICE_HEALTH_FLAP_WARN_THRESHOLD",
        bind_to=(cache_store_config, "rdma_device_health_flap_warn_threshold"),
        type=_rdma_device_health_int("rdma_device_health_flap_warn_threshold"),
        default=10,
        help=(
            "AVAILABLE 关闭故障期时，窗口内恢复次数达到阈值且距上次抖动 WARN 至少一个窗口才告警，范围 {}-{}。"
            "恢复次数理论上限为 fault_window_ms // (2 * probe_interval_ms) + 1；"
            "阈值超过上限时启动仅提示 WARN，不拒绝启动或自动调整阈值。"
        ).format(*flap_threshold_range),
    )
