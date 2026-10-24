from rtp_llm.server.server_args.util import str2bool
from rtp_llm.utils.backend_registry import run_backend_registrations

MOE_STRATEGY_CHOICES = (
    "auto",
    "no_auant_ep_low_latency",
    "no_quant_cpp",
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
    "mega_moe",
    "mega_moe_se",
    "grouped_fp4",
    "local_loop",
    "rocm_ep_normal",
    "rocm_ep_low_latency",
    "rocm_fp8_per_channel_no_dp",
    "rocm_mxfp4_no_dp",
)


def init_moe_group_args(parser, moe_config, eplb_config, deep_ep_config):
    ##############################################################################################################
    # MOE 特性
    ##############################################################################################################
    moe_group = parser.add_argument_group("MOE 专家并行")
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
        "--eplb_mode",
        env_name="EPLB_MODE",
        bind_to=(eplb_config, "eplb_mode"),
        type=str,
        default="NONE",
        help=(
            "专家负载均衡模式。STATS 采样但不迁移，支持在线切换到 EPLB/ALL。"
            "EPLB 和 ALL 均采样并重排专家；搬迁缓冲区在首次重排时初始化。"
            "非 NONE 模式要求模型实现 EPLB、完整专家 EP=WORLD、PP=1 且关闭 FFN 分离；"
            "仅上报负载指标请用 NONE 并设置 eplb_metrics_report_interval。"
        ),
    )
    moe_group.add_argument(
        "--eplb_update_time",
        env_name="EPLB_UPDATE_TIME",
        bind_to=(eplb_config, "eplb_update_time"),
        type=int,
        default=0,
        help="重排周期（有效 model step 数）；0 关闭重排，首次在周期的 1/4 后执行。",
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
        "--eplb_stats_window_size",
        env_name="EPLB_STATS_WINDOW_SIZE",
        bind_to=(eplb_config, "eplb_stats_window_size"),
        type=int,
        default=100,
        help="重排计数窗口；非负，开启重排时必须在 1 到 eplb_update_time 之间。",
    )
    moe_group.add_argument(
        "--eplb_rearrange_after_prefill",
        env_name="EPLB_REARRANGE_AFTER_PREFILL",
        bind_to=(eplb_config, "eplb_rearrange_after_prefill"),
        type=str2bool,
        default=False,
        help="PDFUSION 的 prefill 计数入窗，并在下一个 decode step 重排；默认跳过其 prefill。",
    )
    moe_group.add_argument(
        "--eplb_rearrange_p2p_groups",
        env_name="EPLB_REARRANGE_P2P_GROUPS",
        bind_to=(eplb_config, "eplb_rearrange_p2p_groups"),
        type=int,
        default=1,
        help="跨 NVLink 域重排的 P2P 分组数，必须大于 0。",
    )
    moe_group.add_argument(
        "--eplb_metrics_report_interval",
        env_name="EPLB_METRICS_REPORT_INTERVAL",
        bind_to=(eplb_config, "eplb_metrics_report_interval"),
        type=int,
        default=-1,
        help=(
            "每 W 次大模型 prefill/decode forward 同步汇报 DeepInfer 口径的 moe_lb；"
            "投机解码仅统计 target verify，不统计 draft；"
            "默认 -1 按角色选择 prefill 100 步、decode/fusion 1000 步，"
            "0 关闭，其他值必须大于 0；独立于 EPLB_MODE。"
            "不支持的设备或并行拓扑会告警并跳过监控；dense 保留空窗口 moe_lb=1。"
        ),
    )
    moe_group.add_argument(
        "--eplb_metrics_snapshot_max_files",
        env_name="EPLB_METRICS_SNAPSHOT_MAX_FILES",
        bind_to=(eplb_config, "eplb_metrics_snapshot_max_files"),
        type=int,
        default=0,
        help=(
            "每个模型实例循环覆盖保留的专家激活量窗口快照数；0 关闭（默认），"
            "开启后写入 LOG_PATH/eplb_snapshot，LOG_PATH 默认 logs。"
        ),
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
        # Backend hooks append out-of-tree strategy names to this parser.
        # Give each parser its own mutable list instead of sharing the tuple.
        choices=list(MOE_STRATEGY_CHOICES),
        default="auto",
        help=(
            "指定 MoE strategy，默认为 auto。具体策略的适用性由当前模型配置、"
            "并行拓扑和运行时后端共同校验。"
        ),
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

    # Out-of-tree backends ship MoE strategies the public parser must not
    # advertise, so let them extend the accepted choices here instead.
    run_backend_registrations("moe_strategy_choices", repeatable=True, parser=parser)
