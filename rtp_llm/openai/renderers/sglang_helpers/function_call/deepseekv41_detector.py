"""V4.1 DSML names, retaining the shared incremental parser state machine."""

import re

from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv4_detector import (
    DeepSeekV4Detector,
)


class DeepSeekV41Detector(DeepSeekV4Detector):
    def __init__(self, encoding_module=None, thinking_mode: str = "chat"):
        super().__init__(encoding_module, thinking_mode)
        prefix = "\uff5cDSML\uff5c "
        self.bot_token = f"<{prefix}calls>"
        self.eot_token = f"</{prefix}calls>"
        self.invoke_token = f"<{prefix}invoke"
        self.invoke_end_token = f"</{prefix}invoke>"
        self.parameter_regex = rf'<{prefix}parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</{prefix}parameter>'
        self.partial_parameter_regex = (
            rf'<{prefix}parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*)$'
        )
        self.function_calls_regex = rf"<{prefix}calls>(.*?)</{prefix}calls>"
        self.invoke_regex = (
            rf'<{prefix}invoke\s+name="([^"]+)"\s*>(.*?)(</{prefix}invoke>|$)'
        )
        self.prefix_parameter_end_call = ["</", "\uff5cDSML\uff5c ", "parameter"]

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text or self.invoke_token in text

    def _parse_parameters_from_xml(
        self, invoke_content, allow_partial=False, param_config=None
    ):
        parameters = {}
        end = 0
        for match in re.finditer(self.parameter_regex, invoke_content, re.DOTALL):
            parameters[match.group(1)] = (match.group(3), match.group(2))
            end = match.end()
        if allow_partial:
            remainder = invoke_content[end:]
            closing_tag = "</\uff5cDSML\uff5c parameter>"
            # Strip only a tag prefix. str.rstrip(tag) would also strip
            # ordinary trailing parameter characters such as 'e' in 'value'.
            for size in range(min(len(remainder), len(closing_tag) - 1), 0, -1):
                if remainder.endswith(closing_tag[:size]):
                    remainder = remainder[:-size]
                    break
            match = re.search(self.partial_parameter_regex, remainder, re.DOTALL)
            if match and match.group(3):
                parameters[match.group(1)] = (match.group(3), match.group(2))
        return self._convert_raw_parameters(parameters, param_config or {})

    def tool_call_structural_tag(self, tools, *, stop_after_first):
        # The V4 builtin emits JSON within invoke. V4.1's official decoder
        # requires parameter tags; do not silently reuse that grammar.
        raise ValueError(
            "V4.1 forced tool_choice requires its parameter-tag grammar backend"
        )
