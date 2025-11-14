from rtp_llm.server.server_args.util import str2bool


def init_batch_decode_scheduler_group_args(parser, batch_decode_scheduler_config):
    ##############################################################################################################
    # Batch Decode Scheduler 配置
    ##############################################################################################################
    batch_decode_scheduler_group = parser.add_argument_group("Batch Decode Scheduler")
    
    batch_decode_scheduler_group.add_argument(
        "--batch_decode_scheduler_batch_size",
        env_name="BATCH_DECODE_SCHEDULER_BATCH_SIZE",
        bind_to=[(batch_decode_scheduler_config, 'batch_decode_scheduler_batch_size')],
        type=int,
        default=1,
        help="Batch decode scheduler 的 batch size",
    )
    
    batch_decode_scheduler_group.add_argument(
        "--batch_decode_scheduler_warmup_type",
        env_name="BATCH_DECODE_SCHEDULER_WARMUP_TYPE",
        bind_to=[(batch_decode_scheduler_config, 'batch_decode_scheduler_warmup_type')],
        type=int,
        default=0,
        help="Batch decode scheduler 的 warmup 类型：0 表示使用 decode warmup，其他值表示使用 prefill warmup",
    )
