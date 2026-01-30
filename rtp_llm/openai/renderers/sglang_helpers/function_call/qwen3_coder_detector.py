import ast
import html
import json
import logging
import re
from typing import Any, Dict, List, Tuple

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import Tool
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.ebnf_composer import (
    EBNFComposer,
)

logger = logging.getLogger(__name__)


def _safe_val(raw: str) -> Any:
    raw = html.unescape(raw.strip())
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return raw


def _convert_param_value(
    param_value: str, param_name: str, param_config: Dict[str, Any], func_name: str
) -> Any:
    """
    根据工具配置中的参数类型智能转换参数值
    """
    # 处理 null 值
    if param_value.lower() == "null":
        return None

    # 获取参数类型配置
    if param_name not in param_config:
        logger.warning(
            f"Parameter '{param_name}' not defined in tool '{func_name}', "
            f"using fallback parsing"
        )
        return _safe_val(param_value)

    param_spec = param_config[param_name]
    if isinstance(param_spec, dict) and "type" in param_spec:
        param_type = str(param_spec["type"]).strip().lower()
    else:
        # 如果没有类型信息，使用原始解析
        return _safe_val(param_value)

    # 根据类型进行转换
    try:
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value

        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
            or param_type == "integer"
        ):
            try:
                return int(param_value)
            except ValueError:
                logger.warning(
                    f"Cannot convert '{param_value}' to integer for parameter "
                    f"'{param_name}' in tool '{func_name}', using string"
                )
                return param_value

        elif (
            param_type.startswith("num")
            or param_type.startswith("float")
            or param_type == "number"
        ):
            try:
                float_val = float(param_value)
                # 如果是整数，返回整数类型
                return int(float_val) if float_val.is_integer() else float_val
            except ValueError:
                logger.warning(
                    f"Cannot convert '{param_value}' to number for parameter "
                    f"'{param_name}' in tool '{func_name}', using string"
                )
                return param_value

        elif param_type in ["boolean", "bool", "binary"]:
            param_value_lower = param_value.lower()
            if param_value_lower in ["true", "false"]:
                return param_value_lower == "true"
            else:
                logger.warning(
                    f"Invalid boolean value '{param_value}' for parameter "
                    f"'{param_name}' in tool '{func_name}', defaulting to False"
                )
                return False

        elif param_type in ["object", "dict", "array", "list"]:
            try:
                return json.loads(param_value)
            except json.JSONDecodeError:
                logger.warning(
                    f"Cannot parse JSON for parameter '{param_name}' in tool "
                    f"'{func_name}', trying alternative parsing"
                )
                try:
                    return ast.literal_eval(param_value)
                except Exception:
                    logger.warning(
                        f"Cannot parse object for parameter '{param_name}' "
                        f"in tool '{func_name}', using string"
                    )
                    return param_value

        else:
            # 未知类型，尝试智能解析
            logger.info(
                f"Unknown type '{param_type}' for parameter '{param_name}', using fallback"
            )
            return _safe_val(param_value)

    except Exception as e:
        logger.warning(
            f"Error converting parameter '{param_name}' in tool '{func_name}': {e}, "
            f"using fallback parsing"
        )
        return _safe_val(param_value)


def _get_param_config(func_name: str, tools: List[Tool]) -> Dict[str, Any]:
    """
    从工具列表中获取指定函数的参数配置
    """
    for tool in tools:
        if (
            hasattr(tool, "function")
            and hasattr(tool.function, "name")
            and tool.function.name == func_name
        ):
            if hasattr(tool.function, "parameters"):
                params = tool.function.parameters
                if isinstance(params, dict):
                    # 支持两种格式：
                    # 1. {"properties": {"param1": {"type": "string"}}}
                    # 2. {"param1": {"type": "string"}}
                    if "properties" in params:
                        return params["properties"]
                    else:
                        return params

    logger.warning(f"Tool '{func_name}' not found in tools list")
    return {}


class Qwen3CoderDetector(BaseFormatDetector):
    """
    Detector for Qwen 3 models.
    Assumes function call format:
        <tool_call>
        <function=execute_bash>
        <parameter=command>
        pwd && ls
        </parameter>
        </function>
        </tool_call>
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )
        self._buf: str = ""
        # 添加索引计数器来追踪工具调用
        self._current_tool_index: int = 0
        # 追踪上一次解析是否结束于完整的tool_call
        self._prev_ended_with_tool_call: bool = False

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buf += new_text
        normal = ""
        calls: List[ToolCallItem] = []
        processed_tool_call_this_chunk = False
        while True:
            if self.tool_call_start_token not in self._buf:
                normal += self._buf
                self._buf = ""
                break
            s = self._buf.find(self.tool_call_start_token)
            if s > 0:
                normal += self._buf[:s]
                self._buf = self._buf[s:]
            e = self._buf.find(self.tool_call_end_token)
            if e == -1:
                break
            block = self._buf[: e + len(self.tool_call_end_token)]
            self._buf = self._buf[e + len(self.tool_call_end_token) :]
            calls.extend(self._parse_block(block, tools))
            processed_tool_call_this_chunk = True

        # Strip leading newlines if previous chunk ended with a complete tool call
        # This handles the case where speculative decoding creates chunks like:
        # ["xxx</tool_call>", "\n", "<tool_call>yyy"] or ["xxx</tool_call>", "\n<tool_call>yyy"]
        if self._prev_ended_with_tool_call and normal:
            normal = normal.strip("\n")

        # Update flag for next iteration
        self._prev_ended_with_tool_call = processed_tool_call_this_chunk

        return StreamingParseResult(normal_text=normal, calls=calls)

    def _extract(self, text: str, tools: List[Tool]) -> Tuple[str, List[ToolCallItem]]:
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        cursor = 0
        while True:
            s = text.find(self.tool_call_start_token, cursor)
            if s == -1:
                normal_parts.append(text[cursor:])
                break
            normal_parts.append(text[cursor:s])
            e = text.find(self.tool_call_end_token, s)
            if e == -1:
                normal_parts.append(text[s:])
                break
            block = text[s : e + len(self.tool_call_end_token)]
            cursor = e + len(self.tool_call_end_token)
            calls.extend(self._parse_block(block, tools))
        return "".join(normal_parts), calls

    def _parse_block(self, block: str, tools: List[Tool]) -> List[ToolCallItem]:
        res: List[ToolCallItem] = []
        for m in self.tool_call_function_regex.findall(block):
            txt = m[0] if m[0] else m[1]
            if ">" not in txt:
                continue
            idx = txt.index(">")
            fname = txt[:idx].strip()
            body = txt[idx + 1 :]

            # 获取该函数的参数配置
            param_config = _get_param_config(fname, tools)

            params: Dict[str, Any] = {}
            for pm in self.tool_call_parameter_regex.findall(body):
                ptxt = pm[0] if pm[0] else pm[1]
                if ">" not in ptxt:
                    continue
                pidx = ptxt.index(">")
                pname = ptxt[:pidx].strip()
                pval = ptxt[pidx + 1 :].lstrip("\n").rstrip("\n")

                # 使用类型感知的参数转换
                params[pname] = _convert_param_value(pval, pname, param_config, fname)

            raw = {"name": fname, "arguments": params}
            try:
                # 调用父类方法解析
                parsed_calls = self.parse_base_json(raw, tools)
                # 手动设置正确的 tool_index（父类注释要求的）
                for call in parsed_calls:
                    call.tool_index = self._current_tool_index
                    self._current_tool_index += 1
                res.extend(parsed_calls)
            except Exception:
                logger.warning("invalid tool call for %s dropped", fname)
        return res

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def build_ebnf(self, tools: List[Tool]):
        return EBNFComposer.build_ebnf(
            tools,
            individual_call_start_token=self.tool_call_start_token.replace("\n", "\\n"),
            individual_call_end_token=self.tool_call_end_token.replace("\n", "\\n"),
            tool_call_separator="\\n",
            function_format="xml",
            call_rule_fmt='"<function={name}>\\n" {arguments_rule} "\\n</function>"',
            key_value_rule_fmt='"<parameter={key}>\\n" {valrule} "\\n</parameter>"',
            key_value_separator="\\n",
        )
