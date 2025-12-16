from rtp_llm.server.server_args.util import str2bool


def init_speculative_decoding_group_args(parser, sp_config):
    ##############################################################################################################
    # 投机采样配置
    ##############################################################################################################
    speculative_decoding_group = parser.add_argument_group("投机采样")
    speculative_decoding_group.add_argument(
        "--sp_model_type",
        env_name="SP_MODEL_TYPE",
        bind_to=(sp_config, "model_type"),
        type=str,
        default="",
        help='指定 speculative decoding 的草稿模型类型。例如："mixtbstars-mtp", "deepseek-v3-mtp"。',
    )

    speculative_decoding_group.add_argument(
        "--sp_type",
        env_name="SP_TYPE",
        bind_to=(sp_config, "type"),
        type=str,
        default="",
        help='控制是否启用 speculative decoding 。"vanilla" 不启用，"mtp" 启用 ',
    )

    speculative_decoding_group.add_argument(
        "--sp_min_token_match",
        env_name="SP_MIN_TOKEN_MATCH",
        bind_to=(sp_config, "sp_min_token_match"),
        type=int,
        default=2,
        help="为 speculative decoding 设置最小 token 匹配长度。",
    )

    speculative_decoding_group.add_argument(
        "--sp_max_token_match",
        env_name="SP_MAX_TOKEN_MATCH",
        bind_to=(sp_config, "sp_max_token_match"),
        type=int,
        default=2,
        help="为 speculative decoding 设置最大 token 匹配长度。",
    )

    speculative_decoding_group.add_argument(
        "--tree_decode_config",
        env_name="TREE_DECODE_CONFIG",
        bind_to=(sp_config, "tree_decode_config"),
        type=str,
        default="",
        help="Tree decode的配置文件名，定义了从前缀词到候选Token的映射。",
    )
    speculative_decoding_group.add_argument(
        "--sp_act_type",
        env_name="SP_ACT_TYPE",
        type=str,
        default=None,
        help="小模型的计算使用的类型",
    )
    speculative_decoding_group.add_argument(
        "--sp_quantization",
        env_name="SP_QUANTIZATION",
        bind_to=(sp_config, "quantization"),
        type=str,
        default=None,
        help="",
    )
    speculative_decoding_group.add_argument(
        "--sp_checkpoint_path",
        env_name="SP_CHECKPOINT_PATH",
        bind_to=(sp_config, "checkpoint_path"),
        type=str,
        default=None,
        help="",
    )

    speculative_decoding_group.add_argument(
        "--gen_num_per_cycle",
        env_name="GEN_NUM_PER_CIRCLE",
        bind_to=(sp_config, "gen_num_per_cycle"),
        type=int,
        default=1,
        help="每一轮 speculative execution（推测式生成）中，最多生成多少个 token。",
    )

    speculative_decoding_group.add_argument(
        "--force_stream_sample",
        env_name="FORCE_STREAM_SAMPLE",
        bind_to=(sp_config, "force_stream_sample"),
        type=str2bool,
        default=False,
        help="投机采样强制使用流式采样",
    )

    speculative_decoding_group.add_argument(
        "--force_score_context_attention",
        env_name="FORCE_SCORE_CONTEXT_ATTENTION",
        bind_to=(sp_config, "force_score_context_attention"),
        type=str2bool,
        default=True,
        help="投机采样强制score阶段使用context attention",
    )

    speculative_decoding_group.add_argument(
        "--use_new_sp_engine",
        env_name="USE_NEW_SP_ENGINE",
        bind_to=(sp_config, "use_new_sp_engine"),
        type=str2bool,
        default=False,
        help="是否使用新的投机采样引擎",
    )
