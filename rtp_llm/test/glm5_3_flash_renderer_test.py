import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.chatglm45_renderer import Glm53FlashRenderer
from rtp_llm.utils.base_model_datatypes import MMUrlType


class _Tokenizer:
    def __len__(self):
        return 256

    @staticmethod
    def encode(prompt):
        return list(range(len(prompt)))


class Glm53FlashRendererTest(unittest.TestCase):
    def test_endpoint_merge_preserves_explicit_extra_sampling_config(self):
        renderer = self._template_renderer()
        endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
        endpoint.tokenizer = _Tokenizer()
        endpoint.chat_renderer = renderer
        endpoint.stop_words_str_list = []
        endpoint.stop_words_id_list = []
        endpoint.generate_env_config = GenerateEnvConfig()
        for outer in (None, 0.7):
            kwargs = {} if outer is None else {"top_p": outer}
            request = ChatCompletionRequest(
                messages=[{"role": "user", "content": "hello"}],
                extra_configs={"top_p": 0.6},
                **kwargs,
            )
            original = request.extra_configs.model_dump()
            config = endpoint._extract_generation_config(request)
            self.assertEqual(request.extra_configs.model_dump(), original)
            renderer.apply_chat_completion_constraints(request, config)
            self.assertEqual(config.top_p, 0.6 if outer is None else outer)

    def test_checkpoint_top_p_default_and_explicit_overrides(self):
        cases = (
            ({}, 0.95),
            ({"top_p": None}, 0.95),
            ({"top_p": 1.0}, 1.0),
            ({"top_p": 0.7}, 0.7),
            ({"extra_configs": {"top_p": 0.6}}, 0.6),
            ({"top_p": 0.7, "extra_configs": {"top_p": 0.6}}, 0.7),
        )
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "generation_config.json").write_text(
                json.dumps({"top_p": 0.95})
            )
            renderer = self._template_renderer()
            renderer.ckpt_path = directory
            for kwargs, expected in cases:
                with self.subTest(request=kwargs):
                    request = ChatCompletionRequest(
                        messages=[{"role": "user", "content": "hello"}], **kwargs
                    )
                    config = GenerateConfig()
                    if request.top_p is not None:
                        config.top_p = request.top_p
                    renderer.apply_chat_completion_constraints(request, config)
                    self.assertEqual(config.top_p, expected)

    def test_checkpoint_without_top_p_keeps_api_default(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "generation_config.json").write_text("{}")
            renderer = self._template_renderer()
            renderer.ckpt_path = directory
            config = GenerateConfig()
            renderer.apply_chat_completion_constraints(
                ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}]),
                config,
            )
            self.assertEqual(config.top_p, 1.0)

    def test_missing_generation_config_keeps_api_default(self):
        with tempfile.TemporaryDirectory() as directory:
            renderer = self._template_renderer()
            renderer.ckpt_path = directory
            config = GenerateConfig()
            renderer.apply_chat_completion_constraints(
                ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}]),
                config,
            )
            self.assertEqual(config.top_p, 1.0)

    def _renderer(self):
        renderer = Glm53FlashRenderer.__new__(Glm53FlashRenderer)
        renderer.tokenizer = _Tokenizer()
        renderer._build_prompt = lambda request: "".join(
            part.text or ""
            for message in request.messages
            if isinstance(message.content, list)
            for part in message.content
        )
        return renderer

    def _template_renderer(self):
        renderer = Glm53FlashRenderer.__new__(Glm53FlashRenderer)
        renderer.tokenizer = _Tokenizer()
        renderer.chat_template = (
            "{% if add_generation_prompt %}<|assistant|><think>{% endif %}"
        )
        return renderer

    def test_tool_string_fields_are_json_quoted_and_escaped(self):
        renderer = self._template_renderer()
        renderer.chat_template = (
            '{% for tool in tools %}{"name": {{ tool.function.name | tojson }},'
            ' "description": {{ tool.function.description | tojson(ensure_ascii=False) }},'
            ' "parameters": {{ tool.function.parameters | tojson }}}{% endfor %}'
        )
        tool = {
            "name": "bash",
            "description": 'Execute a "quoted" command\n中文',
            "parameters": {"type": "object", "properties": {}},
        }
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": tool}],
        )

        rendered = renderer.render_chat(request)

        self.assertEqual(json.loads(rendered.rendered_prompt), tool)
        self.assertIn("中文", rendered.rendered_prompt)

    def test_image_content_is_forwarded_to_vit(self):
        renderer = self._renderer()
        request = ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "/tmp/image.jpg"},
                        },
                        {"type": "text", "text": "describe it"},
                    ],
                }
            ]
        )

        rendered = renderer.render_chat(request)

        self.assertEqual(
            rendered.rendered_prompt,
            "<|begin_of_image|><|image|><|end_of_image|>\ndescribe it",
        )
        self.assertEqual(
            [item.url for item in rendered.multimodal_inputs], ["/tmp/image.jpg"]
        )
        self.assertEqual(
            [item.mm_type for item in rendered.multimodal_inputs], [MMUrlType.IMAGE]
        )
        self.assertEqual(len(rendered.input_ids), len(rendered.rendered_prompt))

    def test_video_content_is_forwarded_to_vit(self):
        renderer = self._renderer()
        request = ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "/tmp/video.mp4", "fps": 2.0},
                        }
                    ],
                }
            ]
        )

        rendered = renderer.render_chat(request)

        self.assertEqual(
            rendered.rendered_prompt,
            "<|begin_of_video|><|video|><|end_of_video|>\n",
        )
        self.assertEqual(
            [item.url for item in rendered.multimodal_inputs], ["/tmp/video.mp4"]
        )
        self.assertEqual(
            [item.mm_type for item in rendered.multimodal_inputs], [MMUrlType.VIDEO]
        )
        self.assertAlmostEqual(
            rendered.multimodal_inputs[0].mm_preprocess_config.fps, 2.0
        )

    def test_checkpoint_forced_think_mode_is_always_enabled(self):
        renderer = self._renderer()
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

        self.assertTrue(renderer.in_think_mode(request))

    def test_explicit_disable_closes_checkpoint_forced_think_prefix(self):
        for request_kwargs in (
            {"enable_thinking": False},
            {"chat_template_kwargs": {"enable_thinking": False}},
        ):
            with self.subTest(request_kwargs=request_kwargs):
                renderer = self._template_renderer()
                request = ChatCompletionRequest(
                    messages=[{"role": "user", "content": "hello"}],
                    **request_kwargs,
                )

                rendered = renderer.render_chat(request)
                parser = renderer._create_reasoning_parser(request)

                self.assertTrue(request.disable_thinking())
                self.assertTrue(renderer.in_think_mode(request))
                self.assertEqual(
                    rendered.rendered_prompt,
                    "<|assistant|><think></think>",
                )
                self.assertIsNotNone(parser)
                reasoning, content = parser.parse_non_stream("Hello!")
                self.assertEqual(reasoning, "")
                self.assertEqual(content, "Hello!")

    def test_default_generation_keeps_checkpoint_forced_think_prefix(self):
        renderer = self._template_renderer()
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

        rendered = renderer.render_chat(request)
        parser = renderer._create_reasoning_parser(request)

        self.assertEqual(rendered.rendered_prompt, "<|assistant|><think>")
        self.assertIsNotNone(parser)
        reasoning, content = parser.parse_non_stream(
            "I should answer briefly.</think>Hello!"
        )
        self.assertEqual(reasoning, "I should answer briefly.")
        self.assertEqual(content, "Hello!")

    def test_default_think_parser_splits_implicit_reasoning(self):
        renderer = self._renderer()
        renderer._build_prompt = lambda request: "<think>"
        request = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

        parser = renderer._create_reasoning_parser(request)
        self.assertIsNotNone(parser)
        reasoning, content = parser.parse_non_stream(
            "I should answer briefly.</think>Hello!"
        )
        self.assertEqual(reasoning, "I should answer briefly.")
        self.assertEqual(content, "Hello!")


if __name__ == "__main__":
    unittest.main()
