from rtp_llm.server.server_args.util import str2bool


def init_load_group_args(parser, load_config):
    ##############################################################################################################
    # Load Configuration
    ##############################################################################################################
    load_group = parser.add_argument_group("Load Configuration")
    load_group.add_argument(
        "--phy2log_path",
        env_name="PHY2LOG_PATH",
        bind_to=(load_config, 'phy2log_path'),
        type=str,
        default="",
        help="python日志输出路径",
    )
    load_group.add_argument(
        "--converter_num_per_gpu",
        env_name="CONVERTER_NUM_PER_GPU",
        bind_to=(load_config, 'converter_num_per_gpu'),
        type=int,
        default=4,
        help="每个GPU做多少个转化",
    )
    load_group.add_argument(
        "--tokenizers_parallelism",
        env_name="TOKENIZERS_PARALLELISM",
        bind_to=(load_config, 'tokenizers_parallelism'),
        type=str2bool,
        default=False,
        help="分词器并行度",
    )
    load_group.add_argument(
        "--load_ckpt_num_process",
        env_name="LOAD_CKPT_NUM_PROCESS",
        bind_to=(load_config, 'load_ckpt_num_process'),
        type=int,
        default=0,
        help="加载Checkpoint的进程数量",
    )
