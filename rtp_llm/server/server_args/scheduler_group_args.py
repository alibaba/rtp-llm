from rtp_llm.server.server_args.util import str2bool


def init_scheduler_group_args(parser, runtime_config):
    ##############################################################################################################
    # 调度器配置
    ##############################################################################################################
    scheduler_group = parser.add_argument_group("Scheduler")
    scheduler_group.add_argument(
        "--use_batch_decode_scheduler",
        env_name="USE_BATCH_DECODE_SCHEDULER",
        bind_to=(runtime_config, 'use_batch_decode_scheduler'),
        type=str2bool,
        default=False,
        help="若为 True，则启用一个专门为decode阶段优化的特化调度器。此调度器在 decode 期间以固定大小的 batch 处理请求。若为 False，系统将使用一个 FIFO-based的默认调度器，默认调度器采用continuous batching。",
    )
    scheduler_group.add_argument(
        "--use_gather_batch_scheduler",
        env_name="USE_GATHER_BATCH_SCHEDULER",
        bind_to=(runtime_config, 'use_gather_batch_scheduler'),
        type=str2bool,
        default=False,
        help="若为 True，则启用 gather batch scheduler。",
    )
    scheduler_group.add_argument(
        "--pre_allocate_op_mem",
        env_name="PRE_ALLOCATE_OP_MEM",
        bind_to=(runtime_config, 'pre_allocate_op_mem'),
        type=str2bool,
        default=True,
        help="是否预分配操作内存。",
    )
    scheduler_group.add_argument(
        "--max_block_size_per_item",
        env_name="MAX_BLOCK_SIZE_PER_ITEM",
        bind_to=(runtime_config, 'max_block_size_per_item'),
        type=int,
        default=16,
        help="每个 item 的最大 block 大小。",
    )
