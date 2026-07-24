from rtp_llm.server.server_args.util import str2bool


def init_grammar_group_args(parser, grammar_config):
    grammar_group = parser.add_argument_group("Grammar Configuration")
    grammar_group.add_argument(
        "--constrained_json_disable_any_whitespace",
        env_name="CONSTRAINED_JSON_DISABLE_ANY_WHITESPACE",
        bind_to=(grammar_config, "constrained_json_disable_any_whitespace"),
        type=str2bool,
        default=False,
        help="Disable any-whitespace mode for constrained JSON decoding",
    )
    grammar_group.add_argument(
        "--grammar_num_workers",
        env_name="GRAMMAR_NUM_WORKERS",
        bind_to=(grammar_config, "num_workers"),
        type=int,
        default=grammar_config.num_workers,
        help=(
            "Forwarded to the grammar compiler as max_compiler_threads, "
            "which parallelizes FSM construction (NFA->DFA) within a single "
            "compile. This knob affects intra-compile parallelism, not request-level "
            "concurrency. Raise "
            "for large/complex schemas; C++ clamps invalid values to at least 1."
        ),
    )
    grammar_group.add_argument(
        "--grammar_compiler_cache_bytes",
        env_name="GRAMMAR_COMPILER_CACHE_BYTES",
        bind_to=(grammar_config, "compiler_cache_bytes"),
        type=int,
        default=grammar_config.compiler_cache_bytes,
        help=(
            "Byte cap on the internal compiled-grammar cache. Set <=0 "
            "for unlimited."
        ),
    )
