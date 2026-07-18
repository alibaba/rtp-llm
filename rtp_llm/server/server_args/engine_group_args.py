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
        help="在服务启动时是否开启预热",
    )
    engine_group.add_argument(
        "--warm_up_with_loss",
        env_name="WARM_UP_WITH_LOSS",
        bind_to=(runtime_config, "warm_up_with_loss"),
        type=str2bool,
        default=False,
        help="在服务启动时是否开启损失去预热",
    )
    engine_group.add_argument(
        "--enable_sleep_mode",
        "--enable-sleep-mode",
        env_name="ENABLE_SLEEP_MODE",
        bind_to=(
            (runtime_config, "enable_sleep_mode")
            if hasattr(runtime_config, "enable_sleep_mode")
            else None
        ),
        type=str2bool,
        default=False,
        help="是否开启 sleep/wake_up 生命周期管理接口，默认关闭",
    )
    engine_group.add_argument(
        "--sleep_mode_level",
        "--sleep-mode-level",
        env_name="SLEEP_MODE_LEVEL",
        bind_to=(
            (runtime_config, "sleep_mode_level")
            if hasattr(runtime_config, "sleep_mode_level")
            else None
        ),
        type=int,
        choices=[1, 2],
        default=1,
        help="本进程启动时选定的 sleep level（torch_memory_saver 在加载时就绑定权重 region 的 "
        "cpu_backup，无法按请求切换）。1=sleep 时权重备份到 pinned host（唤醒快，常驻 host 内存）；"
        "2=sleep 时丢弃权重（释放 GPU+host，零落盘），唤醒时由 model loader 从原始 checkpoint "
        "流式原地 copy_ 重载（不写磁盘）。/sleep 请求的 level 必须与此值一致，默认 1。仅接受 1/2："
        "level 0（state-preserving）尚未实现，非法值在启动期即被 argparse 拒绝，避免 Python 权重层与 "
        "C++ RuntimeConfig 对 level 认知分歧",
    )
