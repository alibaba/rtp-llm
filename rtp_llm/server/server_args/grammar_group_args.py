from rtp_llm.server.server_args.util import str2bool


def init_grammar_group_args(parser, grammar_config):
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
        default=8,
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
        default=512 * 1024 * 1024,
        help=(
            "Byte cap on the internal compiled-grammar cache. Set <=0 " "for unlimited."
        ),
    )
