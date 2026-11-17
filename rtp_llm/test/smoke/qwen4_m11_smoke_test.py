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
        from rtp_llm.config.py_config_modules import ServerConfig
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
        backend = ServerConfig()
        backend.start_port = (
            server.port
        )  # DP rank 0 owns every request in the cache case.
        cls.cache_role_addrs = [
            {
                "role": "PDFUSION",
                "ip": "127.0.0.1",
                "http_port": backend.http_port,
                "grpc_port": backend.rpc_server_port,
            }
        ]

    def generate(self, prompt, *, output_tokens=12, reuse_cache=False, role_addrs=None):
        response = self.session.post(
            self.server_url,
            json={
                "prompt": prompt,
                "generate_config": {
                    "max_new_tokens": output_tokens,
                    "ignore_eos": True,
                    "top_k": 1,
                    "random_seed": 42,
                    "reuse_cache": reuse_cache,
                    "return_input_ids": role_addrs is not None,
                    "return_output_ids": True,
                    "aux_info": True,
                    **({"role_addrs": role_addrs} if role_addrs is not None else {}),
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

    def test_prefix_cache_restore(self):
        page_size = 128
        prefix = (
            "Archive seal register. Copy the five-word seal for the requested route.\n\n"
            "NORTH: silver lantern by the river\n"
            "SOUTH: amber compass under the bridge\n"
            "WEST: copper bell above the gate\n\n"
        )
        prefix += "".join(
            f"Record {index:04d}: the archive stores books, maps, and letters.\n"
            for index in range(256)
        )
        prefix += "\nRequested route: EAST\nSeal: bronze feather over the hill\n\n"
        route = self.cache_role_addrs
        seed = self.generate(
            prefix + "Requested route: WEST\nSeal:",
            output_tokens=1,
            reuse_cache=True,
            role_addrs=route,
        )
        self.assertEqual(seed["aux_info"]["reuse_len"], 0)
        seed_ids = seed["input_ids"][0]
        for direction, expected in (
            ("NORTH", "silver lantern by the river"),
            ("SOUTH", "amber compass under the bridge"),
        ):
            with self.subTest(direction=direction):
                prompt = prefix + f"Requested route: {direction}\nSeal:"
                reference = self.generate(prompt, output_tokens=5, role_addrs=route)
                self.assertEqual(reference["aux_info"]["reuse_len"], 0)
                self.assertEqual(reference["response"].strip(), expected)
                reference_ids = reference["input_ids"][0]
                common = next(
                    (
                        i
                        for i, (left, right) in enumerate(zip(seed_ids, reference_ids))
                        if left != right
                    ),
                    min(len(seed_ids), len(reference_ids)),
                )
                reuse = common // page_size * page_size
                self.assertGreater(reuse, self.text_config["indexer_budget"])
                hit = self.generate(
                    prompt, output_tokens=5, reuse_cache=True, role_addrs=route
                )
                self.assertEqual(hit["aux_info"]["reuse_len"], reuse)
                self.assertEqual(hit["response"].strip(), expected)
                self.assertEqual(hit["input_ids"], reference["input_ids"])
                self.assertEqual(hit["output_ids"], reference["output_ids"])
                print(
                    f"{direction}: reused {reuse} tokens, response={expected!r}",
                    flush=True,
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
