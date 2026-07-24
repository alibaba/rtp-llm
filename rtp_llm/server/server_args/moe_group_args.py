import argparse
import math

from rtp_llm.server.server_args.util import str2bool


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("value must be a number") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
    return parsed


def _sampling_interval(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed > 2**31 - 1:
        raise argparse.ArgumentTypeError(
            "sampling interval must fit in a signed 32-bit integer"
        )
    return max(1, parsed)


def init_moe_group_args(parser, moe_config, eplb_config, deep_ep_config):
    ##############################################################################################################
    # MOE 特性
    ##############################################################################################################
    moe_group = parser.add_argument_group("MOE 专家并行")
    moe_group.add_argument(
        "--moe_runtime_mem_log",
        env_name="MOE_RUNTIME_MEM_LOG",
        type=str2bool,
        default=False,
        help="运行期 non-torch 显存诊断，默认关闭；会在每个 MoE layer forward 查询显存。",
    )
    moe_group.add_argument(
        "--moe_runtime_slot_log",
        env_name="MOE_RUNTIME_SLOT_LOG",
        type=str2bool,
        default=False,
        help=(
            "运行期 EP slot 分布诊断，默认关闭；采样时执行 Group.DP all_reduce 和 CPU "
            "同步。启动时校验 DP 组内开关与采样间隔，capture 状态仍须保持一致。"
        ),
    )
    moe_group.add_argument(
        "--moe_runtime_slot_min_slots",
        env_name="MOE_RUNTIME_SLOT_MIN_SLOTS",
        type=_nonnegative_int,
        default=0,
        help="输出 EP slot 分布日志所需的最小全局 slot 数，默认 0。",
    )
    moe_group.add_argument(
        "--moe_runtime_slot_log_interval",
        env_name="MOE_RUNTIME_SLOT_LOG_INTERVAL",
        type=_sampling_interval,
        default=100,
        help=(
            "每多少次 eligible MoE layer forward 采样一次 slot 分布，默认 100；"
            "设为 1 表示每次采样，0 会按 1 处理。"
        ),
    )
    moe_group.add_argument(
        "--moe_skew_mult",
        env_name="MOE_SKEW_MULT",
        type=_nonnegative_float,
        default=1.5,
        help="MoE warmup skew 的乘法余量，默认 1.5；所有 EP rank 应一致设置。",
    )
    moe_group.add_argument(
        "--moe_skew_add",
        env_name="MOE_SKEW_ADD",
        type=_nonnegative_float,
        default=0.1,
        help="MoE warmup skew 的加法余量，默认 0.1；所有 EP rank 应一致设置。",
    )
    moe_group.add_argument(
        "--use_deepep_moe",
        env_name="USE_DEEPEP_MOE",
        bind_to=(deep_ep_config, "use_deepep_moe"),
        type=str2bool,
        help="设置为 `True` 以启用 DeepEP 来处理 MoE 模型的 expert 部分。默认值为 None，允许自动配置。",
    )

    moe_group.add_argument(
        "--use_deepep_internode",
        env_name="USE_DEEPEP_INTERNODE",
        bind_to=(deep_ep_config, "use_deepep_internode"),
        type=str2bool,
        help="设置为 `True` 以启用 DeepEP 来优化跨节点 (inter-node) 通信。默认值为 None，允许自动配置。",
    )

    moe_group.add_argument(
        "--use_deepep_low_latency",
        env_name="USE_DEEPEP_LOW_LATENCY",
        bind_to=(deep_ep_config, "use_deepep_low_latency"),
        type=str2bool,
        help="设置为 `True` 以启用 DeepEP 的低延迟模式。默认值为 None，允许自动配置。",
    )

    moe_group.add_argument(
        "--use_deepep_p2p_low_latency",
        env_name="USE_DEEPEP_P2P_LOW_LATENCY",
        bind_to=(moe_config, "use_deepep_p2p_low_latency"),
        type=str2bool,
        default=False,
        help="设置为 `True` 以启用 DeepEP 的点对点 (P2P) 低延迟模式。",
    )

    moe_group.add_argument(
        "--deep_ep_num_sm",
        env_name="DEEP_EP_NUM_SM",
        bind_to=(moe_config, "deep_ep_num_sm"),
        type=int,
        default=0,
        help="为 DeepEPBuffer 设置 SM (Streaming Multiprocessor) 数量。设置为 `0` 将使用系统默认配置。",
    )

    moe_group.add_argument(
        "--use_mori_ep",
        env_name="USE_MORI_EP",
        bind_to=(deep_ep_config, "use_mori_ep"),
        type=str2bool,
        help="设置为 `True` 以启用 MoriEP 来处理 MoE 模型的 expert 部分。默认值为 None，允许自动配置。",
    )

    moe_group.add_argument(
        "--fake_balance_expert",
        env_name="FAKE_BALANCE_EXPERT",
        bind_to=(moe_config, "fake_balance_expert"),
        type=str2bool,
        default=False,
        help="设置为 `True` 时，为 MoE 模型中的 expert 启用伪均衡 (fake balancing) 机制。用于测试或模拟特定均衡行为。",
    )

    moe_group.add_argument(
        "--eplb_control_step",
        env_name="EPLB_CONTROL_STEP",
        bind_to=(eplb_config, "eplb_control_step"),
        type=int,
        default=100,
        help="为 EPLB (Expert Placement Load Balancing) 控制器指定控制周期或步骤参数。这可能影响专家的负载均衡调整的频率或粒度。",
    )

    moe_group.add_argument(
        "--eplb_test_mode",
        env_name="EPLB_TEST_MODE",
        bind_to=(eplb_config, "eplb_test_mode"),
        type=str2bool,
        default=False,
        help="设置为 `True` 时，为 ExpertBalancer 组件启用测试模式。用于调试或特定的测试场景。",
    )

    moe_group.add_argument(
        "--eplb_balance_layer_per_step",
        env_name="EPLB_BALANCE_LAYER_PER_STEP",
        bind_to=(eplb_config, "eplb_balance_layer_per_step"),
        type=int,
        default=1,
        help="设置 eplb 每次更新的层数。",
    )

    moe_group.add_argument(
        "--eplb_mode",
        env_name="EPLB_MODE",
        bind_to=(eplb_config, "eplb_mode"),
        type=str,
        default="NONE",
        help="专家并行的负载均衡模式",
    )
    moe_group.add_argument(
        "--eplb_update_time",
        env_name="EPLB_UPDATE_TIME",
        bind_to=(eplb_config, "eplb_update_time"),
        type=int,
        default=5000,
        help="专家并行复杂均衡的更新时间",
    )
    moe_group.add_argument(
        "--redundant_expert",
        env_name="REDUNDANT_EXPERT",
        bind_to=(eplb_config, "redundant_expert"),
        type=int,
        default=0,
        help="冗余专家个数",
    )
    moe_group.add_argument(
        "--balance_method",
        env_name="BALANCE_METHOD",
        bind_to=(eplb_config, "balance_method"),
        type=str,
        default="mix",
        help="负载均衡的方法",
    )
    moe_group.add_argument(
        "--eplb_force_repack",
        env_name="EPLB_FORCE_REPACK",
        bind_to=(eplb_config, "eplb_force_repack"),
        type=int,
        default=0,
        help="EPLB_FORCE_REPACK",
    )
    moe_group.add_argument(
        "--eplb_stats_window_size",
        env_name="EPLB_STATS_WINDOW_SIZE",
        bind_to=(eplb_config, "eplb_stats_window_size"),
        type=int,
        default=10,
        help="负载均衡的统计窗口大小",
    )
    moe_group.add_argument(
        "--masked_max_token_num",
        env_name="MASKED_MAX_TOKEN_NUM",
        bind_to=(moe_config, "masked_max_token_num"),
        type=int,
        default=256,
        help="非deepep low latency场景下使用deepgemm masked的最大token数目, 默认为256",
    )
    moe_group.add_argument(
        "--use_all_gather",
        env_name="USE_ALL_GATHER",
        bind_to=(moe_config, "use_all_gather"),
        type=str2bool,
        default=True,
        help="是否使用 all_gather 进行通信。",
    )
    moe_group.add_argument(
        "--moe_strategy",
        env_name="MOE_STRATEGY",
        bind_to=(moe_config, "moe_strategy"),
        type=str,
        choices=[
            "auto",
            "no_auant_ep_low_latency",
            "no_auant_cpp",
            "no_auant_dp_normal",
            "fp8_per_block_no_dp_masked",
            "fp8_per_block_no_dp",
            "fp8_per_block_ep_low_latency",
            "fp8_per_block_ep_normal",
            "fp8_per_block_pure_cp",
            "fp8_per_block_pure_dp",
            "fp8_per_tensor_no_dp",
            "fp8_per_tensor_ep_low_latency",
            "fp8_per_tensor_ep_normal",
            "w4a8_int4_per_channel_no_dp",
            "w4a8_int4_per_channel_ep_low_latency",
            "w4a8_int4_per_channel_ep_normal",
            "fp4_ep_low_latency",
            "fp4_ep_normal",
            "fp4_no_dp",
        ],
        default="auto",
        help="指定moe strategy, 默认为auto",
    )
    moe_group.add_argument(
        "--fp4_moe_op",
        env_name="FP4_MOE_OP",
        bind_to=(moe_config, "fp4_moe_op"),
        type=str,
        choices=["auto", "trtllm", "cutedsl"],
        default="auto",
        help="指定 FP4 MOE算子。可选值: auto (自动选择), trtllm (使用 TensorRT-LLM), cutedsl (使用 CuTe DSL)。",
    )
