from rtp_llm.server.server_args.util import str2bool


def init_scheduler_group_args(parser):
    ##############################################################################################################
    # 调度器配置
    ##############################################################################################################
    scheduler_group = parser.add_argument_group("Scheduler")
    scheduler_group.add_argument(
        "--use_batch_decode_scheduler",
        env_name="USE_BATCH_DECODE_SCHEDULER",
        type=str2bool,
        default=False,
        help="若为 True，则启用一个专门为decode阶段优化的特化调度器。此调度器在 decode 期间以固定大小的 batch 处理请求。若为 False，系统将使用一个 FIFO-based的默认调度器，默认调度器采用continuous batching。",
    )
