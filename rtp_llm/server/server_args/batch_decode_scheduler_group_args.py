def init_batch_decode_scheduler_group_args(parser):
    ##############################################################################################################
    # BatchDecode 调度器配置
    ##############################################################################################################
    batch_decode_scheduler_group = parser.add_argument_group("BatchDecode Scheduler")
    batch_decode_scheduler_group.add_argument(
        "--batch_decode_scheduler_batch_size",
        env_name="BATCH_DECODE_SCHEDULER_BATCH_SIZE",
        type=int,
        default=1,
        help="当 USE_BATCH_DECODE_SCHEDULER 为 True 时，decode 阶段单次处理迭代中将组合在一起的请求数量。增加此值可以提高系统的整体 throughput，但代价是单个请求的 latency 可能会增加。减小此值可以降低 latency，但可能无法充分利用硬件资源。约束：整数 > 0。仅当 USE_BATCH_DECODE_SCHEDULER 为 True 时有效。",
    )
