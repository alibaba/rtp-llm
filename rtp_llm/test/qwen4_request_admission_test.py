import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.qwen35_renderer import Qwen4Renderer, Qwen35Renderer
from rtp_llm.openai.renderers.qwen_vl_renderer import Qwen2VLRenderer
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor


class Qwen4RequestAdmissionTest(unittest.TestCase):
    def test_thinking_mode_disabled_reaches_template(self):
        renderer = Qwen35Renderer.__new__(Qwen35Renderer)
        for kwargs in ({"thinking_mode": "disabled"}, {"enable_thinking": False}):
            request = ChatCompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "hello"}],
                chat_template_kwargs=kwargs,
            )
            with patch.object(
                Qwen2VLRenderer, "_render_messages", return_value=None
            ) as render:
                renderer._render_messages(request, False)
                self.assertIs(render.call_args.args[-1], False)

    def test_image_is_rejected_before_rendering_or_rpc(self):
        renderer = Qwen4Renderer.__new__(Qwen4Renderer)
        request = ChatCompletionRequest(
            model="test",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.invalid/image.png"},
                        }
                    ],
                }
            ],
        )
        with patch.object(
            Qwen35Renderer,
            "render_chat",
            side_effect=AssertionError("must reject before preprocessing"),
        ):
            with self.assertRaisesRegex(FtRuntimeException, "text inputs only"):
                renderer.render_chat(request)
        visitor = BackendRPCServerVisitor.__new__(BackendRPCServerVisitor)
        visitor._qsa_text_only = True
        with self.assertRaisesRegex(FtRuntimeException, "text inputs only"):
            visitor._validate_input(SimpleNamespace(mm_inputs=[object()]))

    def test_production_pipeline_propagates_text_only_admission(self):
        from rtp_llm.pipeline.pipeline import Pipeline

        model = SimpleNamespace(model_type="qwen4_exp", enable_qwen4_qsa=True)
        with patch("rtp_llm.pipeline.pipeline.BackendRPCServerVisitor") as visitor:
            pipeline = Pipeline(None, None, [], 128, 128, None, model_config=model)
        self.assertIs(visitor.call_args.kwargs["model_config"], model)
        with self.assertRaisesRegex(FtRuntimeException, "text inputs only"):
            pipeline.pipeline_async("image", urls=["https://example.invalid/image.png"])
        module = "rtp_llm.server.backend_rpc_server_visitor."
        with patch(module + "ModelRpcClient"), patch(module + "HostService"), patch(
            module + "MasterClient"
        ), patch(module + "get_v41_cache_key_seed", return_value=0), patch.object(
            BackendRPCServerVisitor, "get_backend_role_list", return_value=[]
        ):
            real = BackendRPCServerVisitor(
                128,
                128,
                SimpleNamespace(max_rpc_timeout_ms=100, decode_entrance=False),
                [],
                model_config=model,
            )
        with self.assertRaisesRegex(FtRuntimeException, "text inputs only"):
            real._validate_input(SimpleNamespace(mm_inputs=[object()]))

    def test_text_request_keeps_shared_renderer(self):
        renderer = Qwen4Renderer.__new__(Qwen4Renderer)
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        )
        expected = object()
        with patch.object(Qwen35Renderer, "render_chat", return_value=expected):
            self.assertIs(renderer.render_chat(request), expected)


if __name__ == "__main__":
    unittest.main()
