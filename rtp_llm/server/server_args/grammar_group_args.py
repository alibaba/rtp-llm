import argparse

from rtp_llm.server.server_args.util import str2bool


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return ivalue


def init_grammar_group_args(parser, grammar_config):
    grammar_group = parser.add_argument_group("Grammar Configuration")
    grammar_group.add_argument(
        "--grammar_backend",
        env_name="GRAMMAR_BACKEND",
        bind_to=(grammar_config, "grammar_backend"),
        type=str,
        default="xgrammar",
        choices=["xgrammar", "none", ""],
        help="Grammar backend type: xgrammar or none",
    )
    grammar_group.add_argument(
        "--constrained_json_disable_any_whitespace",
        env_name="CONSTRAINED_JSON_DISABLE_ANY_WHITESPACE",
        bind_to=(grammar_config, "constrained_json_disable_any_whitespace"),
        type=str2bool,
        default=False,
        help="Disable any-whitespace mode for constrained JSON decoding",
    )
    grammar_group.add_argument(
        "--grammar_compile_timeout_ms",
        env_name="GRAMMAR_COMPILE_TIMEOUT_MS",
        bind_to=(grammar_config, "compile_timeout_ms"),
        type=_positive_int,
        default=60000,
        help=(
            "Wall-clock timeout (ms) a single grammar-compile request may sit "
            "in the GrammarCompiler queue before being force-failed with "
            "GENERATE_TIMEOUT. Raise under sustained queue pressure, lower "
            "to fail fast on large schemas."
        ),
    )
    grammar_group.add_argument(
        "--grammar_mask_wait_timeout_ms",
        env_name="GRAMMAR_MASK_WAIT_TIMEOUT_MS",
        bind_to=(grammar_config, "mask_wait_timeout_ms"),
        type=_positive_int,
        default=5000,
        help=(
            "Per-tick budget (ms) for the sampler to wait on walk_thread "
            "advancing the grammar matcher before flagging the stream as "
            "errored and routing through FINISHED. Lower to fail fast on "
            "stuck workers; raise under heavy GIL contention."
        ),
    )
    grammar_group.add_argument(
        "--grammar_num_workers",
        env_name="GRAMMAR_NUM_WORKERS",
        bind_to=(grammar_config, "num_workers"),
        type=_positive_int,
        default=grammar_config.num_workers,
        help=(
            "Size of the native C++ compile worker pool inside GrammarCompiler. "
            "Each worker is a std::thread that pops a queue entry and calls "
            "directly into XGrammarBackend::compileNow — no GIL, no "
            "Python ThreadPoolExecutor, no per-task py::module_ trampoline. "
            "The same value is also forwarded to xgrammar as "
            "max_compiler_threads for FSM construction. Raise under sustained "
            "queue pressure or when pathological schemas can hang workers; "
            "C++ clamps invalid values to at least 1."
        ),
    )
