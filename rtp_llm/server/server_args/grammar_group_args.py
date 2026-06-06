from rtp_llm.server.server_args.util import str2bool


def init_grammar_group_args(parser, grammar_config):
    ##############################################################################################################
    # Grammar / Constrained Decoding Configuration
    ##############################################################################################################
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
        help="Disable any-whitespace mode for constrained JSON decoding",
    )
    grammar_group.add_argument(
        "--reasoning_parser",
        env_name="REASONING_PARSER",
        bind_to=(grammar_config, "reasoning_parser"),
        type=str,
        default="",
        help=(
            "Reasoning detector key. Pass a detector name from "
            "ReasoningParser.DetectorMap (e.g. qwen3, deepseek-r1, "
            "deepseek-v3, glm45, kimi, kimi_k2, qwen3-thinking, step3) to "
            "enable the grammar reasoner wrapper. Empty/omit to disable. The "
            "engine resolves the matching think_end_token at startup; if the "
            "tokenizer cannot encode it as a single token the reasoner "
            "silently falls back to plain xgrammar."
        ),
    )
    grammar_group.add_argument(
        "--grammar_compile_timeout_ms",
        env_name="GRAMMAR_COMPILE_TIMEOUT_MS",
        bind_to=(grammar_config, "compile_timeout_ms"),
        type=int,
        default=60000,
        help=(
            "Wall-clock timeout (ms) a single grammar-compile request may sit "
            "in the GrammarManager queue before being force-failed with "
            "GENERATE_TIMEOUT. Raise under sustained queue pressure, lower "
            "to fail fast on large schemas."
        ),
    )
    grammar_group.add_argument(
        "--grammar_num_workers",
        env_name="GRAMMAR_NUM_WORKERS",
        bind_to=(grammar_config, "num_workers"),
        type=int,
        default=32,
        help=(
            "Size of the native C++ compile worker pool inside GrammarManager. "
            "Each worker is a std::thread that pops a queue entry and calls "
            "directly into XGrammarBackendCpp::compileNow — no GIL, no "
            "Python ThreadPoolExecutor, no per-task py::module_ trampoline. "
            "The same value is also forwarded to xgrammar as "
            "max_compiler_threads for FSM construction. The main reason to "
            "keep this large (32 default) is fault tolerance: a pathological "
            "schema (recursive $ref, ReDoS regex, xgrammar bug) can hang one "
            "worker indefinitely — compile_timeout_ms only marks the queue "
            "entry as failed, it does NOT interrupt the running compileNow. "
            "With N workers, the system survives N-1 concurrent hangs."
        ),
    )
