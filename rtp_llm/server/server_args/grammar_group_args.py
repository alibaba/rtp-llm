from rtp_llm.server.server_args.util import str2bool


def init_grammar_group_args(parser, grammar_config, grammar_admission_config):
    grammar_group = parser.add_argument_group("Grammar Configuration")
    grammar_group.add_argument(
        "--grammar_backend",
        env_name="GRAMMAR_BACKEND",
        bind_to=(grammar_config, "grammar_backend"),
        type=str,
        default="xgrammar",
        help="Grammar backend type: xgrammar or none",
    )
    grammar_group.add_argument(
        "--constrained_json_disable_any_whitespace",
        env_name="CONSTRAINED_JSON_DISABLE_ANY_WHITESPACE",
        bind_to=(grammar_config, "constrained_json_disable_any_whitespace"),
        type=str2bool,
        default=False,
        help="Disable xgrammar any-whitespace mode for JSON schema constraints",
    )
    grammar_group.add_argument(
        "--grammar_num_workers",
        env_name="GRAMMAR_NUM_WORKERS",
        bind_to=(grammar_config, "num_workers"),
        type=int,
        default=8,
        help="xgrammar compiler worker count",
    )
    grammar_group.add_argument(
        "--grammar_compile_timeout_ms",
        env_name="GRAMMAR_COMPILE_TIMEOUT_MS",
        bind_to=(grammar_config, "compile_timeout_ms"),
        type=int,
        default=50,
        help=(
            "Engine-side wall-clock budget for one grammar compile; <=0 restores the unbounded synchronous "
            "compile. A compile that exceeds this keeps running in the background, so the caller is told to "
            "retry and the retry is served from cache. Kept short because the caller is an enqueue thread: "
            "waiting there only adds latency to a request whose retry hits the cache anyway. Note the "
            "frontend admission sandbox (--grammar_admission_compile_timeout_s) applies its own, larger "
            "budget."
        ),
    )
    grammar_group.add_argument(
        "--grammar_compile_concurrency",
        env_name="GRAMMAR_COMPILE_CONCURRENCY",
        bind_to=(grammar_config, "compile_concurrency"),
        type=int,
        default=16,
        help=(
            "Engine-side grammar compiles running at once. Each compile internally fans out over "
            "--grammar_num_workers threads, so this multiplies CPU usage. It caps what used to be "
            "unbounded: before this guard every caller compiled inline on its own thread."
        ),
    )
    grammar_group.add_argument(
        "--grammar_compile_queue_size",
        env_name="GRAMMAR_COMPILE_QUEUE_SIZE",
        bind_to=(grammar_config, "compile_queue_size"),
        type=int,
        default=64,
        help=(
            "Queued engine-side grammar compiles; further ones are rejected. Soft bound: a slot is freed "
            "when a worker picks the compile up, so up to queue_size + concurrency can be outstanding."
        ),
    )
    grammar_group.add_argument(
        "--grammar_compiler_cache_bytes",
        env_name="GRAMMAR_COMPILER_CACHE_BYTES",
        bind_to=(grammar_config, "compiler_cache_bytes"),
        type=int,
        default=2 * 1024 * 1024 * 1024,
        help=(
            "Byte ceiling for engine-side cached compiled grammars, using xgrammar's own memory estimate. "
            "Applied both to xgrammar's compiler cache and to the engine verdict cache; the two share the "
            "same compiled grammars, so the per-rank bound is about 4/3 of this, and a DP8 node holds eight "
            "times that. Least-recently-used grammars are dropped first; a single grammar larger than the "
            "ceiling is served but not cached. <=0 is unlimited. The frontend admission sandbox has its own "
            "cache, capped by --grammar_admission_compiler_cache_bytes."
        ),
    )
    grammar_group.add_argument(
        "--grammar_admission_queue_timeout_s",
        env_name="DS_LLM_GRAMMAR_QUEUE_TIMEOUT_S",
        bind_to=(grammar_admission_config, "queue_timeout_s"),
        type=float,
        default=30.0,
        help="Maximum wait for an idle grammar sandbox worker.",
    )
    grammar_group.add_argument(
        "--grammar_admission_compile_timeout_s",
        env_name="DS_LLM_GRAMMAR_COMPILE_TIMEOUT_S",
        bind_to=(grammar_admission_config, "compile_timeout_s"),
        type=float,
        default=30.0,
        help="Grammar compile timeout after a sandbox worker is checked out.",
    )
    grammar_group.add_argument(
        "--grammar_admission_sandbox_pool_size",
        env_name="DS_LLM_GRAMMAR_SANDBOX_POOL_SIZE",
        bind_to=(grammar_admission_config, "sandbox_pool_size"),
        type=int,
        default=0,
        help="Grammar sandbox worker count; 0 selects an automatic size.",
    )
    grammar_group.add_argument(
        "--grammar_admission_sandbox_process_memory_limit_mb",
        env_name="DS_LLM_GRAMMAR_SANDBOX_PROCESS_MEMORY_LIMIT_MB",
        bind_to=(grammar_admission_config, "sandbox_process_memory_limit_mb"),
        type=int,
        default=1024,
        help="Per-worker address-space headroom in MiB; <=0 disables the cap.",
    )
    grammar_group.add_argument(
        "--grammar_admission_compiler_cache_bytes",
        env_name="DS_LLM_GRAMMAR_COMPILER_CACHE_BYTES",
        bind_to=(grammar_admission_config, "compiler_cache_bytes"),
        type=int,
        default=1024 * 1024 * 1024,
        help="Byte cap on each admission xgrammar compiler cache; <=0 is unlimited.",
    )
    grammar_group.add_argument(
        "--grammar_admission_result_cache_max_entries",
        env_name="DS_LLM_GRAMMAR_RESULT_CACHE_MAX_ENTRIES",
        bind_to=(grammar_admission_config, "result_cache_max_entries"),
        type=int,
        default=2048,
        help="Maximum cached deterministic grammar validation results; 0 disables it.",
    )
