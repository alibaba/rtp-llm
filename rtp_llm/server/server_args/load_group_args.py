from rtp_llm.server.server_args.util import str2bool


def init_load_group_args(parser):
    ##############################################################################################################
    # Load Configuration
    ##############################################################################################################
    load_group = parser.add_argument_group("Load Configuration")
    load_group.add_argument(
        "--phy2log_path",
        env_name="PHY2LOG_PATH",
        type=str,
        default="",
        help="python日志输出路径",
    )
    load_group.add_argument(
        "--converter_num_per_gpu",
        env_name="CONVERTER_NUM_PER_GPU",
        type=int,
        default=4,
        help="每个GPU做多少个转化",
    )
    load_group.add_argument(
        "--tokenizers_parallelism",
        env_name="TOKENIZERS_PARALLELISM",
        type=str2bool,
        default=False,
        help="分词器并行度",
    )
    load_group.add_argument(
        "--load_ckpt_num_process",
        env_name="LOAD_CKPT_NUM_PROCESS",
        type=int,
        default=0,
        help="加载Checkpoint的进程数量",
    )
    load_group.add_argument(
        "--load_method",
        env_name="LOAD_METHOD",
        type=str,
        default="auto",
        help="模型权重加载方法",
    )
    load_group.add_argument(
        "--use_fast_tokenizer",
        env_name="USE_FAST_TOKENIZER",
        type=str2bool,
        default=True,
        help="模型权重Tokenizer是否开启Fast方式",
    )
