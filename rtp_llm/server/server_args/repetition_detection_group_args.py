from rtp_llm.server.server_args.util import str2bool


def init_repetition_detection_group_args(parser, repetition_detection_config):
    group = parser.add_argument_group("Repetition Detection")

    group.add_argument(
        "--output_repetition_monitor",
        env_name="RTP_LLM_OUTPUT_REPETITION_MONITOR",
        bind_to=(repetition_detection_config, "output_repetition_monitor"),
        type=str2bool,
        default=True,
    )
    group.add_argument(
        "--output_repetition_min_repeats",
        env_name="RTP_LLM_OUTPUT_REPETITION_MIN_REPEATS",
        bind_to=(repetition_detection_config, "output_repetition_min_repeats"),
        type=int,
        default=3,
    )
    group.add_argument(
        "--output_repetition_min_dup_tokens",
        env_name="RTP_LLM_OUTPUT_REPETITION_MIN_DUP_TOKENS",
        bind_to=(repetition_detection_config, "output_repetition_min_dup_tokens"),
        type=int,
        default=32,
    )
    group.add_argument(
        "--output_repetition_max_period",
        env_name="RTP_LLM_OUTPUT_REPETITION_MAX_PERIOD",
        bind_to=(repetition_detection_config, "output_repetition_max_period"),
        type=int,
        default=512,
    )
    group.add_argument(
        "--noncontig_repeat_min_span_tokens",
        env_name="RTP_LLM_NONCONTIG_REPEAT_MIN_SPAN_TOKENS",
        bind_to=(repetition_detection_config, "noncontig_repeat_min_span_tokens"),
        type=int,
        default=32,
    )
    group.add_argument(
        "--noncontig_repeat_min_occurrences",
        env_name="RTP_LLM_NONCONTIG_REPEAT_MIN_OCCURRENCES",
        bind_to=(repetition_detection_config, "noncontig_repeat_min_occurrences"),
        type=int,
        default=3,
    )
    group.add_argument(
        "--noncontig_repeat_max_span_tokens",
        env_name="RTP_LLM_NONCONTIG_REPEAT_MAX_SPAN_TOKENS",
        bind_to=(repetition_detection_config, "noncontig_repeat_max_span_tokens"),
        type=int,
        default=256,
    )

    group.add_argument(
        "--tool_call_loop_monitor",
        env_name="RTP_LLM_TOOL_CALL_LOOP_MONITOR",
        bind_to=(repetition_detection_config, "tool_call_loop_monitor"),
        type=str2bool,
        default=True,
        help="Enable token-only tool-call loop monitoring when marker strings are configured.",
    )
    group.add_argument(
        "--tool_call_loop_threshold",
        env_name="RTP_LLM_TOOL_CALL_LOOP_THRESHOLD",
        bind_to=(repetition_detection_config, "tool_call_loop_threshold"),
        type=int,
        default=5,
        help="Number of repeated tool-call spans required before reporting a loop.",
    )
    group.add_argument(
        "--tool_call_loop_max_span_tokens",
        env_name="RTP_LLM_TOOL_CALL_LOOP_MAX_SPAN_TOKENS",
        bind_to=(repetition_detection_config, "tool_call_loop_max_span_tokens"),
        type=int,
        default=16384,
        help="Maximum generated token span recorded for one completed tool call.",
    )
    group.add_argument(
        "--tool_call_loop_begin_marker",
        env_name="RTP_LLM_TOOL_CALL_LOOP_BEGIN_MARKER",
        bind_to=(repetition_detection_config, "tool_call_loop_begin_marker"),
        type=str,
        default="",
        help="Tool-call begin marker string. Tool-call loop monitoring runs only "
        "when both begin and end markers are configured.",
    )
    group.add_argument(
        "--tool_call_loop_end_marker",
        env_name="RTP_LLM_TOOL_CALL_LOOP_END_MARKER",
        bind_to=(repetition_detection_config, "tool_call_loop_end_marker"),
        type=str,
        default="",
        help="Tool-call end marker string. Tool-call loop monitoring runs only "
        "when both begin and end markers are configured.",
    )
