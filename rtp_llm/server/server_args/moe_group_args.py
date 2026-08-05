from rtp_llm.server.server_args.util import greater_than_one_float, str2bool

# The role and execution-path gates are separate from the warmup gate itself --
# MoEConfigAdapter only enables the skew for RoleType::PREFILL, and it lives on
# the models_py execution path, so a deployment running the C++ model path never
# constructs it.
_SKEW_SCOPE_HELP = (
    "仅在以下全部满足时生效：CUDA 构建、PD 分离的 PREFILL role、实际执行了 forward warmup"
    "（见 --warm_up 的跳过条件）、且使用 models_py 执行路径；DECODE 保持模型自然路由。"
    "任一不满足时本参数静默无效，运行期显存预留请改用 --runtime_mem_no_warmup_floor_mb 与"
    " --reserver_runtime_mem_mb 调整。"
)


def init_moe_group_args(parser, moe_config, eplb_config, deep_ep_config):
    ##############################################################################################################
    # MOE 特性
    ##############################################################################################################
    moe_group = parser.add_argument_group("MOE 专家并行")
    # MoeConfig (C++ ConfigModules.h) is the single source of truth for this
    # default: the value below feeds both `default=` and the help text, so a C++
    # change cannot leave the documented default behind. The Python-side mirror in
    # rtp_llm/utils/pre_import_config.py is pinned to the same value by
    # tests/test_warmup_bindings.py.
    default_skew_mult = moe_config.moe_skew_mult
    moe_group.add_argument(
        "--moe_skew_mult",
        env_name="MOE_SKEW_MULT",
        bind_to=(moe_config, "moe_skew_mult"),
        type=greater_than_one_float,
        default=default_skew_mult,
        help=(
            f"PD PREFILL warmup 中热点 rank(rank 0) 承载的 token 量相对均值的倍数，默认 {default_skew_mult:g}。"
            + _SKEW_SCOPE_HELP
            + "所有 PREFILL EP rank 应一致设置。"
            "最终占比 skew_fraction = min(1, moe_skew_mult / ep_size)：热点 token 全部路由到 "
            "rank 0，其余 token 均分到非 rank 0 的专家上，因此 rank 0 的实际负载占比精确等于该值"
            "（top_k 超过 rank 0 本地专家数时会按比例放大热点行数补偿稀释，上限为整个 batch；"
            "实际槽占比见 [MOE_WARMUP] 日志的 rank0_slot_share）。"
            "取值必须严格大于 1，否则 warmup 退化为均匀路由、测出的峰值不含专家不均衡余量。"
            "热点 rank 的 warmup 峰值最高、可用显存最低，经 KVCacheManager 的 block_num min 归约后"
            "成为全簇 KV cache 上界，因此调高本项会等比收紧全簇 KV cache。"
        ),
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
