from rtp_llm.server.server_args.util import str2bool


def init_model_group_args(parser):
    ##############################################################################################################
    # Model Configuration
    ##############################################################################################################
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--extra_data_path",
        env_name="EXTRA_DATA_PATH",
        type=str,
        default=None,
        help="额外的数据路径",
    )
    model_group.add_argument(
        "--local_extra_data_path",
        env_name="LOCAL_EXTRA_DATA_PATH",
        type=str,
        default=None,
        help="本地额外数据路径",
    )
    model_group.add_argument(
        "--tokenizer_path",
        env_name="TOKENIZER_PATH",
        type=str,
        default=None,
        help="分词器的路径",
    )
    model_group.add_argument(
        "--act_type",
        env_name="ACT_TYPE",
        type=str,
        default="FP16",
        help="计算使用的数据类型",
    )
    model_group.add_argument(
        "--use_float32",
        env_name="USE_FLOAT32",
        type=str2bool,
        default=False,
        help="是否使用FP32",
    )
    model_group.add_argument(
        "--original_checkpoint_path",
        env_name="ORIGINAL_CHECKPOINT_PATH",
        type=str,
        default=None,
        help="原始的checkpoint的路径",
    )
    model_group.add_argument(
        "--mla_ops_type",
        env_name="MLA_OPS_TYPE",
        type=str,
        default="AUTO",
        help="Multi Latent Attention的操作类型",
    )
    model_group.add_argument(
        "--parallel_batch",
        env_name="PARALLEL_BATCH",
        type=int,
        default=0,
        help="Batch推理时采用串行还是并行",
    )
    model_group.add_argument(
        "--ft_plugin_path",
        env_name="FT_PLUGIN_PATH",
        type=str,
        default=None,
        help="插件路径",
    )
    model_group.add_argument(
        "--weight_type",
        env_name="WEIGHT_TYPE",
        type=str,
        default=None,
        help="模型权重类型",
    )
    model_group.add_argument(
        "--task_type", env_name="TASK_TYPE", type=str, default=None, help="任务类型"
    )
    model_group.add_argument(
        "--model_type", env_name="MODEL_TYPE", type=str, default=None, help="模型类型"
    )
    model_group.add_argument(
        "--checkpoint_path",
        env_name="CHECKPOINT_PATH",
        type=str,
        default=None,
        help="Checkpoint路径",
    )
    model_group.add_argument(
        "--oss_endpoint",
        env_name="OSS_ENDPOINT",
        type=str,
        default=None,
        help="OSS端点",
    )
    model_group.add_argument(
        "--ptuning_path",
        env_name="PTUNING_PATH",
        type=str,
        default=None,
        help="PTuning路径",
    )
    model_group.add_argument(
        "--dashscope_api_key",
        env_name="DASHSCOPE_API_KEY",
        type=str,
        default="EMPTY",
        help="Dashscope API Key",
    )
    model_group.add_argument(
        "--dashscope_http_url",
        env_name="DASHSCOPE_HTTP_URL",
        type=str,
        default=None,
        help="Dashscope HTTP URL",
    )
    model_group.add_argument(
        "--dashscope_websocket_url",
        env_name="DASHSCOPE_WEBSOCKET_URL",
        type=str,
        default=None,
        help="Dashscope WebSocket URL",
    )
    model_group.add_argument(
        "--openai_api_key",
        env_name="OPENAI_API_KEY",
        type=str,
        default="EMPTY",
        help="OpenAI API Key",
    )
    model_group.add_argument(
        "--json_model_override_args",
        env_name="JSON_MODEL_OVERRIDE_ARGS",
        type=str,
        default="{}",
        help="A dictionary in JSON string format used to override default model configurations.",
    )
