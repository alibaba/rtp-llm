from rtp_llm.server.server_args.util import str2bool


def init_grammar_group_args(parser, grammar_config):
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
