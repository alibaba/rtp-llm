import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.qwen35_renderer import Qwen4Renderer, Qwen35Renderer
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor


class Qwen4RequestAdmissionTest(unittest.TestCase):
    def test_image_is_rejected_before_rendering(self):
        renderer = Qwen4Renderer.__new__(Qwen4Renderer)
        image = {
            "type": "image_url",
            "image_url": {"url": "https://example.invalid/image.png"},
        }
        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": [image]}],
        )
        with patch.object(
            Qwen35Renderer,
            "render_chat",
            side_effect=AssertionError("must reject before preprocessing"),
        ):
            with self.assertRaisesRegex(FtRuntimeException, "text inputs only"):
                renderer.render_chat(request)

    def test_pipeline_and_rpc_reject_images(self):
        from rtp_llm.pipeline.pipeline import Pipeline

        module = "rtp_llm.server.backend_rpc_server_visitor."
        with patch(module + "ModelRpcClient"), patch(module + "HostService"), patch(
            module + "MasterClient"
        ), patch(module + "get_v41_cache_key_seed", return_value=0), patch.object(
            BackendRPCServerVisitor, "get_backend_role_list", return_value=[]
        ):
            pipeline = Pipeline(
                None,
                SimpleNamespace(max_rpc_timeout_ms=100, decode_entrance=False),
                [],
                128,
                128,
                None,
                model_config=SimpleNamespace(
                    model_type="qwen4_exp", enable_qwen4_qsa=True
                ),
            )
        with self.assertRaisesRegex(FtRuntimeException, "text inputs only"):
            pipeline.pipeline_async("image", urls=["https://example.invalid/image.png"])
        with self.assertRaisesRegex(FtRuntimeException, "text inputs only"):
            pipeline.backend_rpc_server_visitor._validate_input(
                SimpleNamespace(mm_inputs=[object()])
            )

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
