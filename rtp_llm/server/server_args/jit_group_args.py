def init_jit_group_args(parser, jit_config):
    ##############################################################################################################
    # JIT Configuration
    ##############################################################################################################
    jit_group = parser.add_argument_group("JIT Configuration")
    jit_group.add_argument(
        "--local_jit_dir",
        env_name="LOCAL_JIT_DIR",
        bind_to=(jit_config, "local_jit_dir"),
        type=str,
        default="",
        help="统一JIT本地cache根目录（自动追加版本子目录）；为空时使用/tmp/rtp-llm/.jit_cache。"
        "SCR可配置为/dev/shm/rtp-llm/.jit_cache，须可写且允许加载动态库",
    )
    jit_group.add_argument(
        "--remote_jit_dir",
        env_name="REMOTE_JIT_DIR",
        bind_to=(jit_config, "remote_jit_dir"),
        type=str,
        default="",
        help="JIT远程cache目录，必须是已存在的本地绝对路径或可通过FUSE挂载的远端URI；为空时仅使用统一的本地cache",
    )
