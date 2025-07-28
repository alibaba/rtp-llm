from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen25_detector import (
    Qwen25Detector,
)
from rtp_llm.openai.renderers.tool_base_renderer import ToolBaseRenderer


class QwenToolRenderer(ToolBaseRenderer):
    """QwenToolRenderer 使用 Qwen25Detector 进行工具调用解析"""

    def _create_detector(self) -> BaseFormatDetector:
        return Qwen25Detector()

    def in_think_mode(self, request: ChatCompletionRequest):
        if request.disable_thinking():
            return False
        return super().in_think_mode(request)

    async def _update_single_status(
        self,
        status: StreamStatus,
        output: GenerateOutput,
        max_new_tokens: int,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
    ) -> OutputDelta:
        if status.finish_reason != None:  # type: ignore
            return await self._create_empty_delta(status.output.aux_info)  # type: ignore
        status.update_output(  # type: ignore
            output,
            self._clean_output_ids,
            functools.partial(self._check_finish_reason, max_new_tokens=max_new_tokens),
            self._remove_stop_word_ids,
        )
        decoded_prev_token = self.tokenizer.decode(status.prev_token_id)  # type: ignore
        decoded_string = self.tokenizer.decode(status.tokens_to_decode)  # type: ignore
        # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
        if is_streaming:
            if len(decoded_string) > 0 and "\uFFFD" == decoded_string[-1]:
                return await self._create_empty_delta(output.aux_info)
        else:
            while (len(decoded_string) > 0) and ("\uFFFD" == decoded_string[-1]):
                decoded_string = decoded_string[:-1]
        status.delta_output_string = decoded_string[len(decoded_prev_token) :]

        # <qwen_tool_renderer>, 其他部分同父类custom_renderer保持一致
        if isinstance(status, QwenToolStreamStatus) and status.request.tools:
            tool_delta = await self.process_tool_calls(status, output, is_streaming)
            # tool_delta为None代表继续默认逻辑处理
            if tool_delta is not None:
                status.update_result()
                return tool_delta
        # </qwen_tool_renderer>

        if is_truncated(status.delta_output_string, stop_words_str, is_streaming):
            status.finish_reason = FinisheReason.stop
            return await self._create_empty_delta(output.aux_info)
        if not is_truncated(
            status.delta_output_string, stop_word_slice_list, is_streaming
        ):
            status.update_result()
            delta = OutputDelta(
                output_str=status.delta_output_string,
                logprobs=await self._generate_log_probs(status, output),
                input_length=output.aux_info.input_len,
                output_length=output.aux_info.output_len,
                reuse_length=output.aux_info.reuse_len,
            )
            status.delta_output_string = ""
            return delta
        else:
            return await self._create_empty_delta(output.aux_info)

    async def process_tool_calls(
        self,
        status: QwenToolStreamStatus,
        output: GenerateOutput,
        is_streaming: bool,
    ) -> Optional[OutputDelta]:
        status.tool_call_message_extract_strategy = (
            ToolCallMessageExtractStrategy.from_extra_configs(status.request)
        )

        return await (
            self._handle_streaming_case(status, output)
            if is_streaming
            else self._handle_no_streaming_case(status, output)
        )

    async def _handle_no_streaming_case(
        self,
        status: QwenToolStreamStatus,
        output: GenerateOutput,
    ) -> OutputDelta:
        # before: status.delta_output_string = "你好\n<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "北京, China", "unit": "celsius"}}</tool_call>"
        # after: status.delta_output_string = "你好\n"
        tool_calls, status.delta_output_string = (
            self._extract_tool_calls_from_complete_message(
                status.delta_output_string,
                status.tool_call_message_extract_strategy,
            )
        )
        if tool_calls:
            status.generating_tool_call = True
        return OutputDelta(
            output_str=DeltaMessage(
                tool_calls=tool_calls,
            ),
            logprobs=await self._generate_log_probs(status, output),
            input_length=output.aux_info.input_len,
            output_length=output.aux_info.output_len,
            reuse_length=output.aux_info.reuse_len,
        )

    def _extract_tool_calls_from_complete_message(
        self,
        original_text: str,
        tool_call_message_extract_strategy: ToolCallMessageExtractStrategy = ToolCallMessageExtractStrategy.DEFAULT,
        tool_call_begin_tag: str = "<tool_call>",
        tool_call_end_tag: str = "</tool_call>",
    ) -> Tuple[Optional[List[ToolCall]], str]:
        """
        从文本中提取所有被 <tool_call> </tool_call> 标签包围的内容,并解析成 ToolCall 对象列表,
        同时从原文本中删除这些标签及其内容

        Args:
            status: 包含 tool_call 标签的状态对象

        Returns:
            List[ToolCall]: ToolCall 对象列表
        """
        tool_calls: List[ToolCall] = []

        parts = original_text.split(tool_call_begin_tag)

        # 保存第一部分(标签之前的内容)
        result_text = parts[0]

        # 处理剩余部分

        index = 0
        for part in parts[1:]:
            if tool_call_end_tag not in part:
                # 认为由<tool_call>开始但是没有</tool_call>结束的是一种failure, skip_on_failure情况下, 就不给予解析
                # 例如 <tool_call>{"name": "get_current_temperature", "arguments": {"location": "北京, China", "unit": "celsius"}}<tool_call>{"name": "get_current_temperature", "arguments": {"location": "上海, China", "unit": "celsius"}}</tool_call>的北京就不允许解析
                if (
                    tool_call_message_extract_strategy
                    == ToolCallMessageExtractStrategy.SKIP_ON_FAILURE
                ):
                    continue
                result_text += tool_call_begin_tag + part
                continue

            tool_parts = part.split(tool_call_end_tag)
            content = tool_parts[0].strip()

            if content:
                try:
                    # 解析 JSON 内容
                    function_name, function_args = self._extract_name_args(content)
                    # 创建 FunctionCall 对象
                    function_call = FunctionCall(
                        name=function_name, arguments=function_args
                    )
                    # 创建 ToolCall 对象
                    tool_call = ToolCall(
                        index=index,
                        id=self._generate_random_call_id(),
                        type="function",
                        function=function_call,
                    )
                    index += 1
                    tool_calls.append(tool_call)
                except Exception as e:
                    logging.error(
                        f"Extract function call from complete message error: {e}"
                    )
                    # begin/end tag之中不是一个正确的json, 也认为是一种failure
                    # 例如<tool_call>{"name": "get_current_temperature", "arguments": error_args}</tool_call>
                    if (
                        tool_call_message_extract_strategy
                        == ToolCallMessageExtractStrategy.SKIP_ON_FAILURE
                    ):
                        continue

                    result_text += tool_call_begin_tag + part
                    continue
            # 如果有剩余文本，添加到结果中
            if len(tool_parts) > 1 and len(tool_parts[1].strip()) > 0:
                result_text += tool_parts[1]

        if tool_calls:
            return tool_calls, result_text
        else:
            return None, result_text

    async def _handle_streaming_case(
        self,
        status: QwenToolStreamStatus,
        output: GenerateOutput,
    ) -> Optional[OutputDelta]:
        # 如果是qwen2.5之前的qwen, 可能会提前stream出<tool_call这样的内容
        if (
            "<tool_call>" in status.tool_call_responded_string
            or status.delta_output_string == "<tool_call>"
        ):
            status.tool_call_responded_string += status.delta_output_string
            status.delta_output_string = ""
            if "</tool_call>" not in status.tool_call_responded_string:
                return await self._create_empty_delta(output.aux_info)
            else:
                # 提取和处理工具调用
                function_call, status.tool_call_responded_string = (
                    self._extract_function_call_from_streaming_message(
                        status.tool_call_responded_string,
                        status.tool_call_message_extract_strategy,
                    )
                )
                # 设置工具调用状态
                if not function_call:
                    # 如果没有提取到工具调用，则把完整的原始文本作为delta输出
                    status.delta_output_string = status.tool_call_responded_string
                    status.tool_call_responded_string = ""
                else:
                    status.generating_tool_call = True
                    # 创建工具调用的delta输出
                    delta = OutputDelta(
                        output_str=DeltaMessage(
                            tool_calls=[
                                ToolCall(
                                    index=status.tool_call_index,
                                    id=self._generate_random_call_id(),
                                    type="function",
                                    function=function_call,
                                )
                            ]
                        ),
                        logprobs=await self._generate_log_probs(status, output),
                        input_length=output.aux_info.input_len,
                        output_length=output.aux_info.output_len,
                        reuse_length=output.aux_info.reuse_len,
                    )
                    status.tool_call_index += 1
                    return delta

        """
        处理特殊换行情况, 例如
        <tool_call>
        {"name": "get_current_temperature", "arguments": {"location": "北京, China", "unit": "celsius"}}
        </tool_call>\n<tool_call>
        {"name": "get_current_temperature", "arguments": {"location": "上海, China", "unit": "celsius"}}
        </tool_call>
        """

        if status.generating_tool_call and status.delta_output_string == "\n":
            return await self._create_empty_delta(output.aux_info)

        # 返回None的意义是让_update_single_status按照默认逻辑继续返回status.delta_output_string的delta
        return None

    def _extract_function_call_from_streaming_message(
        self,
        text: str,
        tool_call_message_extract_strategy: ToolCallMessageExtractStrategy = ToolCallMessageExtractStrategy.DEFAULT,
    ) -> Tuple[Optional[FunctionCall], str]:
        tool_call_name_args_str = text[
            text.index("<tool_call>") + len("<tool_call>") : text.index("</tool_call>")
        ]

        # 更新tool_call_responded_string
        extracted_text = (
            text[: text.index("<tool_call>")]
            + text[text.index("</tool_call>") + len("</tool_call>") :]
        )

        # 解析工具调用参数
        try:
            function_name, function_args = self._extract_name_args(
                tool_call_name_args_str
            )
            return (
                FunctionCall(name=function_name, arguments=function_args),
                extracted_text,
            )
        except Exception as e:
            # json提取失败的时候, 返回原始文本
            logging.error(f"qwen tool extract function call error: {str(e)}")
            if (
                tool_call_message_extract_strategy
                == ToolCallMessageExtractStrategy.SKIP_ON_FAILURE
            ):
                return None, ""
            return None, text

    def _generate_random_call_id(self, length: int = 24) -> str:
        """生成随机调用ID"""
        import secrets
        import string

        characters = string.ascii_letters + string.digits
        random_string = "".join(secrets.choice(characters) for _ in range(length))
        return "call_" + random_string

    def _extract_name_args(self, text: str):
        try:
            data = json.loads(text)
            function_name = data["name"]
            # 将 arguments 转换为标准 JSON 字符串格式
            function_args = json.dumps(data["arguments"], ensure_ascii=False)
            return function_name, function_args
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Missing required field: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unknown error: {str(e)}")

    async def _flush_buffer(
        self,
        buffer_list: List[StreamStatus],
        stop_words_str: List[str],
        is_streaming: bool,
        think_status: ThinkStatus,
    ):
        output_items: List[OutputDelta] = []
        for buffer in buffer_list:
            # 解被截断的bad_case
            # "response":"<tool_call>\n{\"name\": \"get_average_month"
            if (
                isinstance(buffer, QwenToolStreamStatus)
                and buffer.tool_call_responded_string
                and "<tool_call>" in buffer.tool_call_responded_string
                and buffer.tool_call_message_extract_strategy
                == ToolCallMessageExtractStrategy.DEFAULT
            ):
                buffer.delta_output_string += buffer.tool_call_responded_string

            if buffer.output is None:
                raise Exception("last output should not be None")
            aux_info = buffer.output.aux_info
            trunc_string = truncate_response_with_stop_words(
                buffer.delta_output_string, stop_words_str, is_streaming
            )
            output_items.append(
                OutputDelta(
                    trunc_string,
                    await self._generate_log_probs(buffer, buffer.output),
                    aux_info.input_len,
                    aux_info.output_len,
                    aux_info.reuse_len,
                )
            )
        return await self._generate_stream_response(output_items, think_status)


register_renderer("qwen_tool", QwenToolRenderer)
register_renderer("qwen_3_tool", QwenToolRenderer)
