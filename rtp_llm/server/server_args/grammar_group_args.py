import argparse

from rtp_llm.server.server_args.util import str2bool


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def init_grammar_group_args(parser, grammar_config, grammar_admission_config):
    grammar_group = parser.add_argument_group("Grammar Configuration")
    grammar_group.add_argument(
        "--constrained_json_disable_any_whitespace",
        env_name="CONSTRAINED_JSON_DISABLE_ANY_WHITESPACE",
        bind_to=(grammar_config, "constrained_json_disable_any_whitespace"),
        type=str2bool,
        default=False,
        help="Disable xgrammar any-whitespace mode for JSON schema constraints",
    )
    grammar_group.add_argument(
        "--grammar_terminate_without_stop_token",
        env_name="GRAMMAR_TERMINATE_WITHOUT_STOP_TOKEN",
        bind_to=(grammar_config, "terminate_without_stop_token"),
        type=str2bool,
        default=False,
        help=(
            "Terminate xgrammar matchers as soon as the root grammar is complete, "
            "without waiting for a model-generated stop token. This is a service-level "
            "policy applied to every grammar request."
        ),
    )
    grammar_group.add_argument(
        "--grammar_num_workers",
        env_name="GRAMMAR_NUM_WORKERS",
        bind_to=(grammar_config, "num_workers"),
        type=int,
        default=0,
        help=(
            "Threads one grammar compile fans out over. <=0 derives the value from "
            "the process CPU affinity split across ranks on this node."
        ),
    )
    grammar_group.add_argument(
        "--grammar_compile_timeout_ms",
        env_name="GRAMMAR_COMPILE_TIMEOUT_MS",
        bind_to=(grammar_config, "compile_timeout_ms"),
        type=_positive_int,
        default=2000,
        help=(
            "Positive engine-side wait budget for one grammar compile. A timed-out "
            "compile continues and may warm the cache; non-positive values are rejected."
        ),
    )
    grammar_group.add_argument(
        "--grammar_compile_concurrency",
        env_name="GRAMMAR_COMPILE_CONCURRENCY",
        bind_to=(grammar_config, "compile_concurrency"),
        type=_positive_int,
        default=1,
        help="Positive maximum number of engine-side grammar compiles running concurrently.",
    )
    grammar_group.add_argument(
        "--grammar_compile_queue_size",
        env_name="GRAMMAR_COMPILE_QUEUE_SIZE",
        bind_to=(grammar_config, "compile_queue_size"),
        type=_positive_int,
        default=2,
        help="Positive maximum number of queued engine-side compiles before overload rejection.",
    )
    grammar_group.add_argument(
        "--grammar_compiler_cache_bytes",
        env_name="GRAMMAR_COMPILER_CACHE_BYTES",
        bind_to=(grammar_config, "compiler_cache_bytes"),
        type=int,
        default=2 * 1024 * 1024 * 1024,
        help=(
            "Total byte cap shared by xgrammar's compiler cache and the engine verdict "
            "LRU (split evenly between them); <=0 makes both caches unlimited."
        ),
    )
    grammar_group.add_argument(
        "--grammar_admission_queue_timeout_s",
        env_name="GRAMMAR_ADMISSION_QUEUE_TIMEOUT_S",
        bind_to=(grammar_admission_config, "queue_timeout_s"),
        type=float,
        default=30.0,
        help="Maximum wait for an idle grammar sandbox worker.",
    )
    grammar_group.add_argument(
        "--grammar_admission_compile_timeout_s",
        env_name="GRAMMAR_ADMISSION_COMPILE_TIMEOUT_S",
        bind_to=(grammar_admission_config, "compile_timeout_s"),
        type=float,
        default=30.0,
        help="Grammar compile timeout after a sandbox worker is checked out.",
    )
    grammar_group.add_argument(
        "--grammar_admission_sandbox_pool_size",
        env_name="GRAMMAR_ADMISSION_SANDBOX_POOL_SIZE",
        bind_to=(grammar_admission_config, "sandbox_pool_size"),
        type=int,
        default=0,
        help="Grammar sandbox worker count; 0 selects an automatic size.",
    )
    grammar_group.add_argument(
        "--grammar_admission_sandbox_process_memory_limit_mb",
        env_name="GRAMMAR_ADMISSION_SANDBOX_PROCESS_MEMORY_LIMIT_MB",
        bind_to=(grammar_admission_config, "sandbox_process_memory_limit_mb"),
        type=int,
        default=1024,
        help="Per-worker address-space headroom in MiB; <=0 disables the cap.",
    )
    grammar_group.add_argument(
        "--grammar_admission_compiler_cache_bytes",
        env_name="GRAMMAR_ADMISSION_COMPILER_CACHE_BYTES",
        bind_to=(grammar_admission_config, "compiler_cache_bytes"),
        type=int,
        default=1024 * 1024 * 1024,
        help="Byte cap on each admission xgrammar compiler cache; <=0 is unlimited.",
    )
    grammar_group.add_argument(
        "--grammar_admission_result_cache_max_entries",
        env_name="GRAMMAR_ADMISSION_RESULT_CACHE_MAX_ENTRIES",
        bind_to=(grammar_admission_config, "result_cache_max_entries"),
        type=int,
        default=2048,
        help="Maximum cached deterministic grammar validation results; 0 disables it.",
    )

    # Deprecated DS_LLM_-prefixed env aliases for the six admission knobs above.
    # Early DSv4 rollouts configured these names; the canonical envs now follow
    # the repo-wide convention (flag name upper-cased). default=None keeps an
    # unset alias from clobbering the canonical value, and an explicitly-passed
    # CLI canonical flag always beats a stale alias env
    # (EnvArgumentParser._apply_config_bindings two-pass rule).
    _deprecated_admission_env_aliases = [
        ("queue_timeout_s", "DS_LLM_GRAMMAR_QUEUE_TIMEOUT_S", float),
        ("compile_timeout_s", "DS_LLM_GRAMMAR_COMPILE_TIMEOUT_S", float),
        ("sandbox_pool_size", "DS_LLM_GRAMMAR_SANDBOX_POOL_SIZE", int),
        (
            "sandbox_process_memory_limit_mb",
            "DS_LLM_GRAMMAR_SANDBOX_PROCESS_MEMORY_LIMIT_MB",
            int,
        ),
        ("compiler_cache_bytes", "DS_LLM_GRAMMAR_COMPILER_CACHE_BYTES", int),
        (
            "result_cache_max_entries",
            "DS_LLM_GRAMMAR_RESULT_CACHE_MAX_ENTRIES",
            int,
        ),
    ]
    for _field, _env, _type in _deprecated_admission_env_aliases:
        grammar_group.add_argument(
            f"--grammar_admission_{_field}_ds_llm_alias",
            env_name=_env,
            bind_to=(grammar_admission_config, _field),
            type=_type,
            default=None,
            help=f"[deprecated] {_env} 的旧 env 别名，请改用 --grammar_admission_{_field}。",
        )
