def init_reasoning_group_args(parser, reasoning_config):
    reasoning_group = parser.add_argument_group("Reasoning Configuration")
    reasoning_group.add_argument(
        "--reasoning_parser",
        env_name="REASONING_PARSER",
        bind_to=(reasoning_config, "reasoning_parser"),
        type=str,
        default="",
        help=(
            "Reasoning detector key. Pass a detector name from "
            "ReasoningParser.DetectorMap (e.g. qwen3, deepseek-r1, "
            "deepseek-v3, glm45, kimi, kimi_k2, qwen3-thinking, step3) to "
            "enable the OpenAI renderer's reasoning split and, when grammar "
            "is also enabled, the grammar reasoner gating. Empty/omit to "
            "disable. The engine resolves the matching think_start_token / "
            "think_end_token at startup; if the tokenizer cannot encode "
            "them as single tokens the reasoner silently falls back."
        ),
    )
