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
        default="./.jit_cache",
        help="JIT本地cache目录",
    )
    jit_group.add_argument(
        "--remote_jit_dir",
        env_name="REMOTE_JIT_DIR",
        bind_to=(jit_config, "remote_jit_dir"),
        type=str,
        default="",
        help="JIT远程cache挂载目录，必须是已存在的绝对路径",
    )
    jit_group.add_argument(
        "--jit_remote_timeout_s",
        env_name="JIT_REMOTE_TIMEOUT_S",
        bind_to=(jit_config, "jit_remote_timeout_s"),
        type=float,
        default=30,
        help="JIT远程cache上传/下载超时时间，单位秒",
    )
