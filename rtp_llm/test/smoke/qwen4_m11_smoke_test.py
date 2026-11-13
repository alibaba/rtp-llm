"""Manual text smoke for the full Qwen4-Max M11 MXFP4 checkpoint."""

import json
import os
import unittest
from pathlib import Path

import requests


class Qwen4M11SmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checkpoint = os.environ.get("QWEN4_M11_CHECKPOINT")
        if not checkpoint:
            raise RuntimeError("Set QWEN4_M11_CHECKPOINT to the local M11 checkpoint")
        config = json.loads((Path(checkpoint) / "config.json").read_text())
        if config.get("quantization_config", {}).get("quant_method") != "mxfp4":
            raise ValueError("The M11 smoke requires a native MXFP4 checkpoint")
        cls.text_config = config["text_config"]
        cls.session = requests.Session()
        cls.session.trust_env = False
        cls.addClassCleanup(cls.session.close)
        from rtp_llm.test.utils.maga_server_manager import MagaServerManager

        server = MagaServerManager(
            env_args={"FT_SERVER_TEST": "1"},
            role_name="qwen4_m11_sm10x",
            smoke_args_str=os.environ["SMOKE_ARGS"],
        )
        cls.addClassCleanup(server.stop_server)
        if not server.start_server(
            model_path=checkpoint, model_type="qwen4_exp", timeout=1800
        ):
            raise RuntimeError("Qwen4-Max M11 server did not become healthy")
        cls.server_url = f"http://127.0.0.1:{server.port}/"

    def generate(self, prompt):
        output_tokens = 12
        response = self.session.post(
            self.server_url,
            json={
                "prompt": prompt,
                "generate_config": {
                    "max_new_tokens": output_tokens,
                    "ignore_eos": True,
                    "top_k": 1,
                    "random_seed": 42,
                    "reuse_cache": False,
                    "return_output_ids": True,
                    "aux_info": True,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        actual = response.json()
        self.assertNotIn("error_code", actual, actual)
        self.assertIs(actual["finished"], True)
        self.assertEqual(len(actual["output_ids"]), 1)
        ids = actual["output_ids"][0]
        self.assertEqual(len(ids), output_tokens)
        self.assertTrue(
            all(0 <= token < self.text_config["vocab_size"] for token in ids)
        )
        self.assertEqual(actual["aux_info"]["speculative_draft_rounds"], 0)
        return actual

    def chat(self, prompt):
        response = self.session.post(
            self.server_url + "v1/chat/completions",
            json={
                "model": "qwen4_exp",
                "messages": [{"role": "user", "content": prompt}],
                "enable_thinking": False,
                "temperature": 0,
                "max_tokens": 48,
                "extra_configs": {"reuse_cache": False},
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop", result)
        answer = choice["message"]["content"].strip()
        self.assertTrue(answer, result)
        return answer, result["usage"]["prompt_tokens"]

    def test_prefill_and_decode(self):
        prompt = "Complete the sequence: 1, 2, 3, 4,"
        self.assertEqual(
            self.generate(prompt)["output_ids"], self.generate(prompt)["output_ids"]
        )

    def test_chinese_answer(self):
        answer, _ = self.chat("北京是哪个国家的首都？请用一个完整的中文句子回答。")
        for word in ("北京", "首都"):
            self.assertIn(word, answer)
        self.assertRegex(answer, r"中国|中华人民共和国")
        print(f"Chinese answer: {answer}", flush=True)

    def test_qsa_beyond_indexer_budget(self):
        prompt = "The archive keyword is BLUE LANTERN.\n" + (
            "The archive records books, maps, letters, and journeys.\n" * 350
        )
        answer, prompt_tokens = self.chat(
            prompt + "Reply with only the archive keyword."
        )
        self.assertGreater(prompt_tokens, self.text_config["indexer_budget"])
        self.assertEqual(answer.strip("\"'`*.").upper(), "BLUE LANTERN", answer)
        print(f"QSA answer ({prompt_tokens} prompt tokens): {answer}", flush=True)


if __name__ == "__main__":
    unittest.main()
