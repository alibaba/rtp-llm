from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import CustomChatRenderer, RenderedInputs


class KimiK3Renderer(CustomChatRenderer):
    """Use the checkpoint's Python XTML encoder for text conversations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_extra_stop_word_ids(
            [self.tokenizer.encode(marker, add_special_tokens=False) for marker in (
                "<|end_of_msg|>", "<|close|>response<|sep|>"
            )]
        )

    def render_chat(self, request):
        data = request.model_dump(exclude_none=True)
        if data.get("tools") or data.get("functions"):
            raise ValueError("Kimi K3 text renderer does not support tool calls")
        messages = data["messages"]
        for message in messages:
            if message.get("tool_calls") or message.get("role") == "tool":
                raise ValueError("Kimi K3 text renderer does not support tool history")
            content = message.get("content", "")
            if isinstance(content, list):
                if any(part.get("type") != "text" for part in content):
                    raise ValueError("Kimi K3 text renderer requires text-only messages")
                message["content"] = "".join(part["text"] for part in content)
        # Resolve thinking exactly as main's output parser and GenerateConfig do.
        # Tokenize directly: re-encoding a rendered string would turn user text
        # resembling XTML structure into privileged control tokens.
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            thinking=self.in_think_mode(request),
        )
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            raise ValueError("Kimi K3 tokenizer must return a non-empty token ID list")
        return RenderedInputs(input_ids=ids)


register_renderer("kimi_k3", KimiK3Renderer)
