def init_jit_group_args(parser, jit_config):
    ##############################################################################################################
    # JIT Configuration
    ##############################################################################################################
    jit_group = parser.add_argument_group("JIT Configuration")
    jit_group.add_argument(
        "--local_jit_cache_dir",
        env_name="LOCAL_JIT_CACHE_DIR",
        bind_to=(jit_config, "local_jit_cache_dir"),
        type=str,
        default="~/.cache/rtp_llm/jit",
        help="JIT本地cache目录",
    )
