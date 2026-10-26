"""Compare eager and length-only LOCAL PDFUSION on two real DP ranks."""

import copy
import json
import os
import time
import unittest
from pathlib import Path

import requests

from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class LocalMMLatePrefillSmokeTest(unittest.TestCase):
    def run_mode(self, late, queries):
        label = "late" if late else "eager"
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        server = MagaServerManager(
            env_args={
                "LOAD_PYTHON_MODEL": "1",
                "RTP_LLM_LOCAL_MM_LATE_PREFILL": "1" if late else "0",
                "FT_SERVER_TEST": "1",
                "LOG_LEVEL": "INFO",
            },
            role_name=label,
            process_file_name=f"{label}.process.log",
            smoke_args_str=os.environ["SMOKE_ARGS"],
        )
        session = requests.Session()
        session.trust_env = False
        results = []
        try:
            self.assertTrue(
                server.start_server(
                    model_path=os.environ["RTP_MM_E2E_CHECKPOINT"],
                    model_type="qwen35_moe",
                    timeout=1600,
                )
            )
            for index, query in enumerate(queries):
                started = time.monotonic()
                print(f"SMOKE_REQUEST_START mode={label} request={index}", flush=True)
                response = session.post(
                    f"http://127.0.0.1:{server.port}/v1/chat/completions",
                    json=query,
                    # Preserve the original 8192-token decode cap; only the
                    # harness wait budget is larger for full-video acceptance.
                    timeout=1800,
                )
                (output / f"{label}.{index}.json").write_text(response.text)
                response.raise_for_status()
                body = response.json()
                self.assertNotIn("error", body)
                self.assertTrue(body["choices"])
                content = body["choices"][0]["message"]["content"]
                self.assertTrue(content and content.strip(), body)
                if index == 0:
                    self.assertEqual(content.strip(), "测试正常")
                    self.assertEqual(body["choices"][0]["finish_reason"], "stop")
                results.append((body["choices"], body["usage"]["prompt_tokens"]))
                print(
                    f"SMOKE_REQUEST_DONE mode={label} request={index} "
                    f"seconds={time.monotonic() - started:.3f} usage={body['usage']}",
                    flush=True,
                )
        finally:
            session.close()
            server.stop_server()
        return results

    def test_text_then_four_videos_and_repeated_cache_hits(self):
        video_dir = Path(os.environ["RTP_MM_TEST_VIDEO_DIR"])
        queries = [
            {
                "model": "qwen35_moe",
                "messages": [{"role": "user", "content": "请只回答：测试正常"}],
                "temperature": 0,
                "max_tokens": 32,
                "stream": False,
            }
        ]
        names = ("video.mp4", "video_02.mp4", "video_03.mp4", "video_04.mp4")
        expected_config = {
            "fps": 6,
            "min_frames": -1,
            "max_frames": 180,
            "min_pixels": 2500000,
            "max_pixels": 73728000,
            "resized_width": 1280,
            "resized_height": 704,
        }
        for index, name in enumerate(names, 1):
            query = json.loads((video_dir / f"request_{index:02d}.json").read_text())
            media_count = 0
            for message in query["messages"]:
                for content in (
                    message["content"] if isinstance(message["content"], list) else []
                ):
                    if content["type"] == "video_url":
                        self.assertEqual(content["preprocess_config"], expected_config)
                        content["video_url"]["url"] = str(video_dir / name)
                        media_count += 1
            self.assertEqual(media_count, 1)
            # Keep original text, preprocessing and decode limits unchanged.
            queries.append(query)
        queries.extend(copy.deepcopy(queries[1:]))
        eager = self.run_mode(False, queries)
        late = self.run_mode(True, queries)
        self.assertEqual(len(eager), len(late))
        for index, (before, after) in enumerate(zip(eager, late)):
            with self.subTest(request=index):
                self.assertEqual(before, after)
                print(
                    f"SMOKE_ACCEPT request={index} prompt_tokens={before[1]} exact=True",
                    flush=True,
                )


if __name__ == "__main__":
    unittest.main()
