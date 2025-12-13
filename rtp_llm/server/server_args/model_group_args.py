from rtp_llm.server.server_args.util import str2bool


def init_model_group_args(parser, model_args):
    ##############################################################################################################
    # Model Configuration
    ##############################################################################################################
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--extra_data_path",
        env_name="EXTRA_DATA_PATH",
        bind_to=(model_args, 'extra_data_path'),
        type=str,
        default="",
        help="额外的数据路径",
    )
    model_group.add_argument(
        "--local_extra_data_path",
        env_name="LOCAL_EXTRA_DATA_PATH",
        bind_to=(model_args, 'local_extra_data_path'),
        type=str,
        default="",
        help="本地额外数据路径",
    )
    model_group.add_argument(
        "--tokenizer_path",
        env_name="TOKENIZER_PATH",
        bind_to=(model_args, 'tokenizer_path'),
        type=str,
        default="",
        help="分词器的路径",
    )
    model_group.add_argument(
        "--act_type",
        env_name="ACT_TYPE",
        bind_to=(model_args, 'act_type'),
        type=str,
        default=None,
        help="计算使用的数据类型",
    )
    model_group.add_argument(
        "--mla_ops_type",
        env_name="MLA_OPS_TYPE",
        bind_to=(model_args, 'mla_ops_type'),
        type=str,
        default=None,
        help="Multi Latent Attention的操作类型（将自动转换为枚举）",
    )
    model_group.add_argument(
        "--task_type",
        env_name="TASK_TYPE",
        bind_to=(model_args, 'task_type'),
        type=str,
        default=None,
        help="任务类型（将自动转换为枚举）"
    )
    model_group.add_argument(
        "--model_type",
        env_name="MODEL_TYPE",
        bind_to=(model_args, 'model_type'),
        type=str,
        default=None,
        help="模型类型"
    )
    model_group.add_argument(
        "--checkpoint_path",
        env_name="CHECKPOINT_PATH",
        bind_to=(model_args, 'ckpt_path'),
        type=str,
        help="Checkpoint路径",
    )
    model_group.add_argument(
        "--ptuning_path",
        env_name="PTUNING_PATH",
        bind_to=(model_args, 'ptuning_path'),
        type=str,
        default="",
        help="PTuning路径",
    )
    model_group.add_argument(
        "--json_model_override_args",
        env_name="JSON_MODEL_OVERRIDE_ARGS",
        bind_to=(model_args, 'json_model_override_args'),
        type=str,
        default="{}",
        help="A dictionary in JSON string format used to override default model configurations.",
    )
    model_group.add_argument(
        "--max_seq_len",
        env_name="MAX_SEQ_LEN",
        bind_to=(model_args, 'max_seq_len'),
        type=int,
        default=None,
        help="最大序列长度",
    )
