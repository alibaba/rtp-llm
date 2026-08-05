from rtp_llm.server.server_args.util import str2bool


def init_engine_group_args(parser, runtime_config):
    ##############################################################################################################
    # Engine Configuration
    # Fields merged from EngineConfig to RuntimeConfig (warm_up, warm_up_with_loss)
    ##############################################################################################################
    engine_group = parser.add_argument_group("Engine Configuration")
    engine_group.add_argument(
        "--warm_up",
        env_name="WARM_UP",
        bind_to=(runtime_config, "warm_up"),
        type=str2bool,
        default=True,
        help=(
            "是否在服务启动时执行 forward warmup，Python 服务入口默认开启。"
            "以下任一条件成立时即使开启也会跳过 warmup，改走 no-warmup 定容路径："
            "非 CUDA 构建（ROCm/PPU/CPU 均不执行本套显存 trace）；"
            "role 不是 PD 分离的 PREFILL/DECODE（PDFUSION 会额外打 WARNING 说明不支持）；"
            "multimodal 模型；启用了 ffn disaggregation（--enable_ffn_disaggregate）。"
            "未执行 warmup 时，运行时显存预留为 configured / sampler / no_warmup_floor / "
            "safety_ratio×总显存 四者的最大值（与升级前语义一致）；此时可调的旋钮是"
            " --runtime_mem_no_warmup_floor_mb 与 --reserver_runtime_mem_mb。"
        ),
    )
    engine_group.add_argument(
        "--warm_up_with_loss",
        env_name="WARM_UP_WITH_LOSS",
        bind_to=(runtime_config, "warm_up_with_loss"),
        type=str2bool,
        default=False,
        help="在服务启动时是否开启损失去预热",
    )
