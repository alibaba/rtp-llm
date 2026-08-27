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
        default=0,
        help=(
            "Threads one grammar compile fans out over. A compile spends nearly all of its time in a phase "
            "that parallelises almost perfectly, so this is the main lever on compile latency. <=0 derives it "
            "from the CPU this rank actually owns, namely the process CPU affinity split across the ranks "
            "sharing the node, clamped so a small cpuset stays usable and a large one does not run past the "
            "point where extra threads lose to lock and memory contention. Also feeds the frontend admission "
            "sandbox's automatic pool sizing, where a wider fanout per compile means fewer grammars validated "
            "in parallel, up to a ceiling on the number of sandbox processes; "
            "--grammar_admission_sandbox_pool_size overrides that."
        ),
    )
    grammar_group.add_argument(
        "--grammar_compile_timeout_ms",
        env_name="GRAMMAR_COMPILE_TIMEOUT_MS",
        bind_to=(grammar_config, "compile_timeout_ms"),
        type=int,
        default=2000,
        help=(
            "Engine-side wall-clock budget a caller waits for one grammar compile; <=0 restores the unbounded "
            "synchronous compile. A compile that exceeds this keeps running in the background, so the budget "
            "does not bound CPU cost, only how long the caller waits before the request is rejected as "
            "overloaded. Sized to cover a legitimately expensive grammar rather than to fail fast, because "
            "the compiled grammar is cached per rank: a rejected request that retries onto another rank pays "
            "the compile again. Note the frontend admission sandbox "
            "(--grammar_admission_compile_timeout_s) applies its own, larger budget."
        ),
    )
    grammar_group.add_argument(
        "--grammar_compile_concurrency",
        env_name="GRAMMAR_COMPILE_CONCURRENCY",
        bind_to=(grammar_config, "compile_concurrency"),
        type=int,
        default=1,
        help=(
            "Engine-side grammar compiles running at once. This times --grammar_num_workers is the compile "
            "thread budget, which should stay within the CPU the rank owns or compiles will contend with the "
            "engine's own threads. For a fixed budget, spending it on fanout rather than on concurrency "
            "finishes each compile sooner instead of finishing several slowly together, which is why this "
            "defaults to serial. It also caps what used to be unbounded: before this guard every caller "
            "compiled inline on its own thread."
        ),
    )
    grammar_group.add_argument(
        "--grammar_compile_queue_size",
        env_name="GRAMMAR_COMPILE_QUEUE_SIZE",
        bind_to=(grammar_config, "compile_queue_size"),
        type=int,
        default=2,
        help=(
            "Queued engine-side grammar compiles; further ones are rejected. Kept shallow because compiles "
            "run serially: only the entries near the head can still finish inside the compile timeout, and "
            "anything behind them outlives its caller and runs only to warm the cache for a retry. Soft "
            "bound: a slot is freed when a worker picks the compile up rather than when it finishes, and the "
            "queue is only refused once it is over size, so a little more than queue_size + concurrency can "
            "be outstanding."
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
            "Applied both to xgrammar's compiler cache and to the engine verdict cache; the two hold the same "
            "compiled grammars, so the resident total is not the sum of the two, and every rank owns a "
            "backend, so a node holds one per rank. Least-recently-used grammars are dropped first. The "
            "ceiling is soft by one entry: a grammar too large to fit under it on its own is cached anyway, "
            "because dropping it would recompile it on every request. <=0 is unlimited. The frontend "
            "admission sandbox has its own cache, capped by --grammar_admission_compiler_cache_bytes."
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
