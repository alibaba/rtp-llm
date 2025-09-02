def init_engine_group_args(parser):
    ##############################################################################################################
    # Engine Configuration
    ##############################################################################################################
    engine_group = parser.add_argument_group("Engine Configuration")
    engine_group.add_argument(
        "--warm_up",
        env_name="WARM_UP",
        type=int,
        default=1,
        help="在服务启动时是否开启预热",
    )
    engine_group.add_argument(
        "--warm_up_with_loss",
        env_name="WARM_UP_WITH_LOSS",
        type=int,
        default=0,
        help="在服务启动时是否开启损失去预热",
    )
    engine_group.add_argument(
        "--max_seq_len",
        env_name="MAX_SEQ_LEN",
        type=int,
        default=0,
        help="输入输出的最大长度",
    )
