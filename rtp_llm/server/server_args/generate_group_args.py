from rtp_llm.server.server_args.util import str2bool


def init_generate_group_args(parser, generate_env_config):
    ##############################################################################################################
    # Generate Configuration
    ##############################################################################################################
    generate_group = parser.add_argument_group("Generate Configuration")
    generate_group.add_argument(
        "--think_end_tag",
        env_name="THINK_END_TAG",
        bind_to=(generate_env_config, "think_end_tag"),
        type=str,
        default="</think>\n\n",
        help="深度思考模式的结束标签",
    )
    generate_group.add_argument(
        "--think_end_token_id",
        env_name="THINK_END_TOKEN_ID",
        bind_to=(generate_env_config, "think_end_token_id"),
        type=int,
        default=-1,
        help="深度思考模式的结束标签的 TOKEN_ID",
    )
    generate_group.add_argument(
        "--think_mode",
        env_name="THINK_MODE",
        bind_to=(generate_env_config, "think_mode"),
        type=int,
        default=0,
        help="深度思考模式是否开启",
    )
    generate_group.add_argument(
        "--force_stop_words",
        env_name="FORCE_STOP_WORDS",
        bind_to=(generate_env_config, "force_stop_words"),
        type=str2bool,
        default=False,
        help="是否开启使用环境变量强制指定模型的STOP WORDS",
    )
    generate_group.add_argument(
        "--stop_words_list",
        env_name="STOP_WORDS_LIST",
        bind_to=(generate_env_config, "stop_words_list"),
        type=str,
        default=None,
        help="STOP_WORDS的TokenID列表",
    )
    generate_group.add_argument(
        "--stop_words_str",
        env_name="STOP_WORDS_STR",
        bind_to=(generate_env_config, "stop_words_str"),
        type=str,
        default=None,
        help="STOP_WORDS的string明文",
    )
    generate_group.add_argument(
        "--think_start_tag",
        env_name="THINK_START_TAG",
        bind_to=(generate_env_config, "think_start_tag"),
        type=str,
        default="<think>\\n",
        help="深度思考模式的起始标签",
    )
    generate_group.add_argument(
        "--think_terminate_token_id",
        env_name="THINK_TERMINATE_TOKEN_ID",
        bind_to=(generate_env_config, "think_terminate_token_id"),
        type=int,
        default=1,
        help="思考阶段被特殊 token 立刻终止时的 token id(DSV4 默认为 1);<=0 表示禁用该路径",
    )
    generate_group.add_argument(
        "--generation_config_path",
        env_name="GENERATION_CONFIG_PATH",
        bind_to=(generate_env_config, "generation_config_path"),
        type=str,
        default=None,
        help="生成配置路径",
    )
    generate_group.add_argument(
        "--xgrammar_compile_cache_size",
        env_name="XGRAMMAR_COMPILE_CACHE_SIZE",
        bind_to=(generate_env_config, "xgrammar_compile_cache_size"),
        type=int,
        default=1024,
        help="XGrammar frontend compile cache LRU capacity per process",
    )
    generate_group.add_argument(
        "--xgrammar_compile_thread_num",
        env_name="XGRAMMAR_COMPILE_THREAD_NUM",
        bind_to=(generate_env_config, "xgrammar_compile_thread_num"),
        type=int,
        default=4,
        help="XGrammar frontend compile worker thread count",
    )
    generate_group.add_argument(
        "--xgrammar_precompile_list",
        env_name="XGRAMMAR_PRECOMPILE_LIST",
        bind_to=(generate_env_config, "xgrammar_precompile_list"),
        type=str,
        default=None,
        help="Optional JSONL file with response_format objects to precompile",
    )
    generate_group.add_argument(
        "--xgrammar_enable_stream_partial_json",
        env_name="XGRAMMAR_ENABLE_STREAM_PARTIAL_JSON",
        bind_to=(generate_env_config, "xgrammar_enable_stream_partial_json"),
        type=str2bool,
        default=True,
        help="Enable prefix-valid partial JSON checks for streaming structured output",
    )
