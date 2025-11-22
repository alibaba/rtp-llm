from typing import List, Tuple

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPartTypeEnum,
    RendererInfo,
    RoleEnum,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RenderedInputs,
    RendererParams,
)


class ChatGlm4Renderer(CustomChatRenderer):
    def __init__(
        self, 
        tokenizer: BaseTokenizer, 
        renderer_params: RendererParams,
        generate_env_config,
        render_config=None,
        ckpt_path=None,
        misc_config=None,
        vit_config=None,
    ):
        super().__init__(tokenizer, renderer_params, generate_env_config, render_config, ckpt_path, misc_config, vit_config)

    def get_renderer_info(self) -> RendererInfo:
        renderer_info = super().get_renderer_info()
        return renderer_info

    def build_single_message(
        self, role: str, metadata: str, message: str, prefix_message_list: List[str]
    ) -> Tuple[List[int], str]:
        assert role in ["system", "user", "assistant", "observation"], role
        # chatglm4 tokenizer.tokenizer -> tiktoken.Encoding
        # rtp_llm/frontend/tokenizer_factory/tokenizers/tokenization_chatglm4.py:32
        role_tokens = [
            self.tokenizer.convert_tokens_to_ids(f"<|{role}|>")
        ] + self.tokenizer.tokenizer.tokenizer.encode(
            f"{metadata}\n", disallowed_special=()
        )
        prefix_message_tokens = []
        prefix_message = ""
        if prefix_message_list is not None:
            prefix_message_tokens = self.tokenizer.convert_tokens_to_ids(
                prefix_message_list
            )
            prefix_message = "".join(prefix_message_list)
        message_tokens = self.tokenizer.tokenizer.encode(message, disallowed_special=())
        tokens: List[int] = role_tokens + prefix_message_tokens + message_tokens

        return tokens, str(f"<|{role}|>{metadata}\n{prefix_message + message}")

    def handle_single_conversation(self, conversation: List[ChatMessage]):
        input_ids = [
            self.tokenizer.convert_tokens_to_ids("[gMASK]"),
            self.tokenizer.convert_tokens_to_ids("<sop>"),
        ]
        input_message = "[gMASK]<sop>"
        input_image = []
        for item in conversation:
            if item.role == RoleEnum.tool:
                tools = item.tool_calls
                assert tools is not None, "tools should not be None"
                content = "你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。"
                for tool in tools:
                    if tool.type == "function":
                        function = tool.function
                        content += (
                            f"\n\n## {function.name}\n\n{function.model_dump_json()}"
                        )
                        content += (
                            "\n在调用上述函数时，请使用 Json 格式表示调用的参数。"
                        )
                    elif tool.type == "python":
                        content += "\n\n## python\n\n当你向 `python` 发送包含 Python 代码的消息时，该代码将会在一个有状态的 Jupyter notebook 环境中执行。\n`python` 返回代码执行的输出，或在执行 60 秒后返回超时。\n`/mnt/data` 将会持久化存储你的文件。在此会话中，`python` 无法访问互联网。不要使用 `python` 进行任何网络请求或者在线 API 调用，这些在线内容的访问将不会成功。"
                    elif tool.type == "simple_browser":
                        content += "\n\n## simple_browser\n\n你可以使用 `simple_browser` 工具。该工具支持以下函数：\n`search(query: str, recency_days: int)`：使用搜索引擎进行查询并显示结果，可以使用 `recency_days` 参数控制搜索内容的时效性。\n`mclick(ids: list[int])`：获取一系列指定 id 的页面内容。每次调用时，须选择3-10个页面。选择多个角度的页面，同时尽可能选择可信任的信息来源。考虑到部分页面是无法加载的，你也可以多打开一些可能有用的页面而不用担心内容过多。\n`open_url(url: str)`：打开指定的 URL。\n\n使用 `【{引用 id}†{引用文本}】` 来引用内容。\n\n操作步骤：1. 使用 `search` 来获得信息列表; 2. 使用 `mclick` 来获取指定 ID 页面的内容; 3. 根据获得的内容进行回复。在回复中应当引用信息来源。\n 如果用户提供了 URL，也可以用 `open_url` 直接打开页面。\n如果初次搜索结果没有找到合适的信息，也可以再次使用 `search` 进行搜索。"
                    elif tool.type == "cogview":
                        content += "\n\n## cogview\n\n如果用户的请求中包含了对图像的描述，你可以使用 `cogview` 来生成图像并展示给用户。你需要向 `cogview` 发送图像描述，规则：\n- 发送给 `cogview` 的消息必须使用英语。用户的中文描述必须完全翻译为英语。\n- 应当尽可能详细地描述图像生成的需求，需求描述约 100 英文单词。\n- 保持用户原始描述的意图。不要虚构内容或者没见过的人物。\n- 如无特殊说明，所在地为中国，持有中国立场并遵循中国社会主义价值观。"
                    else:
                        raise NotImplementedError(f"Unknown tool type {tool.type}")
                id, message = self.build_single_message("system", "", content)
                input_ids.extend(id)
                input_message += message
            # no metadata yet
            message_prefix = None
            message = ""
            if isinstance(item.content, str):
                message += item.content
            elif isinstance(item.content, list):
                for content_part in item.content:
                    if content_part.type == ContentPartTypeEnum.text:
                        assert isinstance(content_part.text, str)
                        message += content_part.text
                    elif content_part.type == ContentPartTypeEnum.image_url:
                        assert (
                            len(input_image) == 0 and message_prefix is None
                        ), "Multiple images are not supported"
                        input_image.append(content_part.image_url.url)
                        message_prefix = [
                            "<|begin_of_image|>",
                            "<|endoftext|>",
                            "<|end_of_image|>",
                        ]
            else:
                raise Exception(f"unkown chat message: {item.model_dump()}")

            if message or message_prefix:
                part_ids, part_messages = self.build_single_message(
                    item.role, "", message, message_prefix
                )
                input_ids.extend(part_ids)
                input_message += part_messages

        input_ids.extend([self.tokenizer.convert_tokens_to_ids("<|assistant|>")])
        input_message += "<|assistant|>"

        return input_ids, input_message, input_image

    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        input_ids, input_message, input_image = self.handle_single_conversation(
            request.messages
        )
        return RenderedInputs(
            input_ids=input_ids, rendered_prompt=input_message, input_urls=input_image
        )


register_renderer("chatglm4", ChatGlm4Renderer)
register_renderer("chatglm4v", ChatGlm4Renderer)
