def init_worker_group_args(parser, worker_config):
    ##############################################################################################################
    # Worker Configuration
    ##############################################################################################################
    worker_group = parser.add_argument_group("Worker Configuration")
    worker_group.add_argument(
        "--worker_info_port_num",
        env_name="WORKER_INFO_PORT_NUM",
        bind_to=(worker_config, 'worker_info_port_num'),
        type=int,
        default=7,
        help="worker的总的端口的数量",
    )
