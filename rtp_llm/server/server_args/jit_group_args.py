def init_jit_group_args(parser, jit_config):
    ##############################################################################################################
    # JIT Configuration
    ##############################################################################################################
    jit_group = parser.add_argument_group("JIT Configuration")
    jit_group.add_argument(
        "--remote_jit_dir",
        env_name="REMOTE_JIT_DIR",
        bind_to=(jit_config, "remote_jit_dir"),
        type=str,
        default="",
        help="JIT远程v1快照根（可信绝对路径或FUSE URI）；为空仍使用固定/tmp本地树，预设组件cache环境变量可退出对应组件托管",
    )
    jit_group.add_argument(
        "--jit_cache_setup_timeout_s",
        env_name="JIT_CACHE_SETUP_TIMEOUT_S",
        bind_to=(jit_config, "jit_cache_setup_timeout_s"),
        type=int,
        default=180,
        help="JIT缓存恢复预算秒数，超时转为本地冷编译",
    )
