"""Real DP2/EP2 inference: default-on late ViT in ratio and cadence modes."""

import base64
import io
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class LocalMMRatioCadenceSmokeTest(unittest.TestCase):
    def test_default_late_prefill_matches_eager_with_concurrent_images(self):
        os.environ.pop("RTP_LLM_LOCAL_MM_LATE_PREFILL", None)
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        queries = []
        for color, expected in (("red", "红色"), ("green", "绿色"), ("blue", "蓝色")):
            buffer = io.BytesIO()
            Image.new("RGB", (224, 224), color).save(buffer, format="PNG")
            queries.append(
                (
                    {
                        "model": "qwen35_moe",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/png;base64,"
                                            + base64.b64encode(
                                                buffer.getvalue()
                                            ).decode()
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": "图片是什么颜色？只回答红色、绿色或蓝色，不要解释。",
                                    },
                                ],
                            }
                        ],
                        "temperature": 0,
                        "max_tokens": 32,
                        "stream": False,
                    },
                    expected,
                )
            )
        text_query = {
            "model": "qwen35_moe",
            "messages": [{"role": "user", "content": "请只回答：测试正常"}],
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
        }
        reference = None
        for mode in ("eager", "ratio", "cadence", "adaptive"):
            env = {
                "LOAD_PYTHON_MODEL": "1",
                "FT_SERVER_TEST": "1",
                "LOG_LEVEL": "INFO",
                "PDFUSION_SCHEDULER_MODE": "ratio",
                "PDFUSION_COORD_MODE": (
                    "cadence" if mode in ("cadence", "adaptive") else "off"
                ),
                "PDFUSION_CADENCE_ADAPTIVE": "1" if mode == "adaptive" else "0",
                "DECODE_PREFILL_RATIO": "7",
                "MAX_CONTEXT_BATCH_SIZE": "1",
                "MAX_BATCH_TOKENS_SIZE": "4096",
                "PDFUSION_SCHEDULE_TRACE": "1",
                "MM_CACHE_GPU_MAX_BYTES": "0",
                "MM_CACHE_CPU_MAX_BYTES": "0",
            }
            if mode == "eager":
                env["RTP_LLM_LOCAL_MM_LATE_PREFILL"] = "0"
            server = MagaServerManager(
                env_args=env, role_name=mode, smoke_args_str=os.environ["SMOKE_ARGS"]
            )

            def request(item):
                query, expected = item
                with requests.Session() as session:
                    session.trust_env = False
                    response = session.post(
                        f"http://127.0.0.1:{server.port}/v1/chat/completions",
                        json=query,
                        timeout=180,
                    )
                response.raise_for_status()
                body = response.json()
                self.assertNotIn("error", body)
                content = body["choices"][0]["message"]["content"].strip()
                self.assertEqual(content, expected, body)
                self.assertEqual(body["choices"][0]["finish_reason"], "stop", body)
                return {
                    "content": content,
                    "prompt_tokens": body["usage"]["prompt_tokens"],
                }

            try:
                self.assertTrue(
                    server.start_server(
                        model_path=os.environ["RTP_MM_E2E_CHECKPOINT"],
                        model_type="qwen35_moe",
                        timeout=1600,
                    )
                )
                request((text_query, "测试正常"))
                with ThreadPoolExecutor(max_workers=6) as pool:
                    results = list(pool.map(request, queries * 2))
                request((text_query, "测试正常"))
                (output / f"{mode}.results.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2)
                )
                if reference is None:
                    reference = results
                else:
                    self.assertEqual(results, reference)
            finally:
                server.stop_server()
            logs = "\n".join(
                path.read_text(errors="replace")
                for path in (output / f"{mode}_logs").rglob("*.log")
            )
            if mode != "eager":
                self.assertIn("LOCAL_MM_LATE_PREFILL prepare batch=", logs)
            self.assertNotIn("CUDA out of memory", logs)
            print(
                f"SMOKE_ACCEPT mode={mode} requests=8 concurrent=6 exact=True",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
