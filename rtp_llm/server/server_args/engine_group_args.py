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
            "是否在服务启动时执行 forward warmup，默认开启。CUDA PREFILL/DECODE 使用实测峰值"
            "辅助 KV cache 定容；PDFUSION 保留 warmup forward 但使用 no-warmup 公式；"
            "VIT/FRONTEND、多模态、FFN 分离及非 CUDA 平台不执行。设为 false 可完全关闭；"
            "可用 --reserver_runtime_mem_mb 调整固定预留。"
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
