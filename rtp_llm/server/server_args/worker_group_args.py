def init_worker_group_args(parser):
    ##############################################################################################################
    # Worker Configuration
    ##############################################################################################################
    worker_group = parser.add_argument_group("Worker Configuration")
    worker_group.add_argument(
        "--worker_info_port_num",
        env_name="WORKER_INFO_PORT_NUM",
        type=int,
        default=8,
        help="worker的总的端口的数量",
    )
