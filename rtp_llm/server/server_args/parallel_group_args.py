from rtp_llm.ops import CPRotateMethod
from rtp_llm.server.server_args.util import str2_cp_rotate_method, str2bool


def init_parallel_group_args(
    parser, parallelism_config, ffn_disaggregate_config, prefill_cp_config
):
    ##############################################################################################################
    # Parallelism and Distributed Setup Configuration
    ##############################################################################################################
    parallel_group = parser.add_argument_group(
        "Parallelism and Distributed Setup Configuration"
    )
    parallel_group.add_argument(
        "--tp_size",
        env_name="TP_SIZE",
        bind_to=(parallelism_config, "tp_size"),
        type=int,
        default=1,
        help="指定用于张量并行度。",
    )
    parallel_group.add_argument(
        "--ep_size",
        env_name="EP_SIZE",
        bind_to=(parallelism_config, "ep_size"),
        type=int,
        default=1,
        help="定义用于专家并行（Expert Parallelism）的模型（专家）实例数量。未设置时默认 1，多进程时由配置阶段按 TP*DP*PP 推导。",
    )
    parallel_group.add_argument(
        "--dp_size",
        env_name="DP_SIZE",
        bind_to=(parallelism_config, "dp_size"),
        type=int,
        default=1,
        help="设置数据并行（Data Parallelism）的副本数量或组大小。",
    )
    parallel_group.add_argument(
        "--world_size",
        env_name="WORLD_SIZE",
        bind_to=(parallelism_config, "world_size"),
        type=int,
        default=1,
        help="分布式设置中使用的GPU总数。通常情况下，`WORLD_SIZE = TP_SIZE * DP_SIZE`",
    )
    parallel_group.add_argument(
        "--world_rank",
        env_name="WORLD_RANK",
        bind_to=(parallelism_config, "world_rank"),
        type=int,
        default=0,
        help="当前进程/GPU在分布式系统中的全局唯一编号（从0到 `WORLD_SIZE - 1`）。",
    )
    parallel_group.add_argument(
        "--local_world_size",
        env_name="LOCAL_WORLD_SIZE",
        bind_to=(parallelism_config, "local_world_size"),
        type=int,
        default=1,
        help="在多节点分布式设置中，当前节点（Node）上使用的GPU设备数量。未设置时默认 1，多进程时由配置阶段按 min(cuda 数, world_size) 推导。",
    )
    parallel_group.add_argument(
        "--ffn_sp_size",
        env_name="FFN_SP_SIZE",
        bind_to=(parallelism_config, "ffn_sp_size"),
        type=int,
        default=1,
        help="FFN层序列并行大小。",
    )
    parallel_group.add_argument(
        "--enable_ffn_disaggregate",
        env_name="ENABLE_FFN_DISAGGREGATE",
        bind_to=(ffn_disaggregate_config, "enable_ffn_disaggregate"),
        type=str2bool,
        default=False,
        help="启用AF分离功能。",
    )
    parallel_group.add_argument(
        "--cp_rotate_method",
        env_name="CP_ROTATE_METHOD",
        bind_to=(prefill_cp_config, "method"),
        type=str2_cp_rotate_method,
        default=CPRotateMethod.DISABLED,
        help="指定用于上下文并行通信方法。可选值: ALL_GATHER, ALL_GATHER_WITH_OVERLAP, ALLTOALL",
    )
    parallel_group.add_argument(
        "--comm_buffer_size",
        env_name="COMM_BUFFER_SIZE",
        bind_to=(prefill_cp_config, "comm_buffer_size"),
        type=int,
        default=512 * 1024 * 1024,
        help="指定用于上下文并行通信的缓冲区大小，单位为字节。默认值为 512MB。",
    )
    parallel_group.add_argument(
        "--pd_sep_enable_pcp",
        env_name="PD_SEP_ENABLE_PCP",
        bind_to=(prefill_cp_config, "pd_sep_enable_pcp"),
        type=str2bool,
        default=False,
        help="PD分离的PREFILL节点是否启动了上下文并行，默认值为False，表示不启动。",
    )
