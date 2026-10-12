"""Mean pooling, task prefixes and Matryoshka output for EmbeddingGemma 2."""

import json
import os
from typing import Optional

import torch
from torch.nn import functional as F

from rtp_llm.models.downstream_modules.custom_module import CustomHandler, CustomModule
from rtp_llm.models.downstream_modules.embedding.api_datatype import (
    ChatMessage,
    ContentPart,
    OpenAIEmbeddingRequest,
)
from rtp_llm.models.downstream_modules.embedding.dense_embedding_module import (
    DenseEmbeddingRenderer,
)


class EmbeddingGemma2Request(OpenAIEmbeddingRequest):
    dimensions: Optional[int] = None
    prompt_name: Optional[str] = None


class EmbeddingGemma2Handler(CustomHandler):
    def forward(self, input_ids, hidden_states, input_lengths):
        # Inputs are packed with no padding. Include every valid token, including
        # task instructions, matching the checkpoint's include_prompt=true.
        lengths = tuple(int(n) for n in input_lengths.tolist())
        if not lengths or min(lengths) <= 0 or sum(lengths) != hidden_states.shape[0]:
            raise ValueError("EmbeddingGemma2 pooling requires valid packed lengths")
        chunks = hidden_states.split(lengths)
        # SentenceTransformers pooling preserves the model activation dtype.
        pooled = torch.stack(
            [x.sum(dim=0) / x.new_tensor(n) for x, n in zip(chunks, lengths)]
        )
        return F.normalize(pooled, p=2, dim=-1)


class EmbeddingGemma2Renderer(DenseEmbeddingRenderer):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        path = os.path.join(config.ckpt_path, "config_sentence_transformers.json")
        self.prompts = {}
        if os.path.exists(path):
            with open(path) as reader:
                self.prompts = json.load(reader).get("prompts", {})
        self.embedding_dim = config.mm_related_params.config[
            "embeddinggemma2_embedding_dim"
        ]

    def render_request(self, request_json):
        if "left" in request_json:
            return super().render_request(request_json)
        request = EmbeddingGemma2Request(**request_json)
        if request.dimensions is not None and request.dimensions not in (
            128,
            256,
            512,
            self.embedding_dim,
        ):
            raise ValueError("EmbeddingGemma2 dimensions must be 128, 256, 512 or 768")
        if request.prompt_name is not None and request.prompt_name not in self.prompts:
            raise ValueError(
                f"Unknown EmbeddingGemma2 prompt_name: {request.prompt_name}"
            )
        return request

    def create_input(self, request):
        if isinstance(request, EmbeddingGemma2Request):
            prefix = request.prompt or self.prompts.get(request.prompt_name, "")
            texts = [request.input] if isinstance(request.input, str) else request.input
            if isinstance(request.input, ContentPart):
                texts = [ChatMessage(role="user", content=[request.input])]
            elif isinstance(request.input, ChatMessage):
                texts = [request.input]
            elif (
                isinstance(texts, list)
                and texts
                and all(isinstance(item, ContentPart) for item in texts)
            ):
                texts = [ChatMessage(role="user", content=texts)]
            if (
                isinstance(texts, list)
                and texts
                and all(isinstance(item, ChatMessage) for item in texts)
            ):
                if prefix:
                    texts = [ChatMessage(role="system", content=prefix)] + texts
                return self.generator.generate(
                    texts,
                    truncate=False,
                    tokenizer_config=request.extra_configs.tokenizer_config,
                )
            if (
                not isinstance(texts, list)
                or not texts
                or not all(isinstance(text, str) for text in texts)
            ):
                raise ValueError(
                    "EmbeddingGemma2 input must be text, content parts or messages"
                )
            return self.generator.generate(
                [prefix + text for text in texts],
                truncate=False,
                tokenizer_config=request.extra_configs.tokenizer_config,
            )
        return self.generator.generate(request.left + request.right, truncate=False)

    def embedding_func(self, request, res, input_length, input_tokens):
        dimensions = getattr(request, "dimensions", None)
        if dimensions is not None and dimensions != res.shape[-1]:
            res = F.normalize(res[:dimensions], p=2, dim=-1)
        return res.tolist()


class EmbeddingGemma2Module(CustomModule):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.handler = EmbeddingGemma2Handler(config)
        self.renderer = EmbeddingGemma2Renderer(config, tokenizer)
