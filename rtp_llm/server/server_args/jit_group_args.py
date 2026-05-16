def init_jit_group_args(parser, jit_config):
    ##############################################################################################################
    # JIT Configuration
    ##############################################################################################################
    jit_group = parser.add_argument_group("JIT Configuration")
    jit_group.add_argument(
        "--remote_jit_dir",
        env_name="REMOTE_JIT_DIR",
        bind_to=(jit_config, 'remote_jit_dir'),
        type=str,
        default="",
        help="JIT 远程 cache 目录：以 RW 方式 fuse，并把 DG_JIT_CACHE_DIR / TRITON_CACHE_DIR / TILELANG_CACHE_DIR 全部覆盖到 fuse 出来的目录",
    )
    jit_group.add_argument(
        "--remote_jit_read_dir",
        env_name="REMOTE_JIT_READ_DIR",
        bind_to=(jit_config, 'remote_jit_read_dir'),
        type=str,
        default="",
        help="JIT 远程只读 cache 目录：以 RO 方式 fuse，把内容拷贝到 ${HIPPO_APP_WORKDIR}/jit_cache，再把三个 JIT env 都设到该本地目录",
    )
