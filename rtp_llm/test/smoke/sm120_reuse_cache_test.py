"""Compare long Qwen3 FP8 requests with and without local prefix reuse."""

import json
import logging
import os
import unittest
from pathlib import Path

import requests
from transformers import AutoTokenizer

from rtp_llm.test.utils.maga_server_manager import MagaServerManager

MODEL_PATH = (
    "/mnt/nas1/hf/models--Qwen--Qwen3-1.7B/snapshots/"
    "0060bc56d46589041c1048efd1a397421b1142b5"
)
BLOCK_SIZE = 2048


class SM120ReuseCacheTest(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True, local_files_only=True
        )
        self.server = MagaServerManager(
            env_args={"LOAD_PYTHON_MODEL": "1", "FRONTEND_SERVER_COUNT": "1"},
            role_name="reuse_cache",
            smoke_args_str=os.environ["SMOKE_ARGS"],
        )
        self.addCleanup(self.server.stop_server)
        self.assertTrue(
            self.server.start_server(model_path=MODEL_PATH, model_type="qwen_3"),
            "Qwen3 dense FP8 server failed to start",
        )
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)

    def make_prompts(self):
        records = []
        # Distinct rows exercise real long-context tokenization and cache blocks.
        while True:
            start = len(records)
            records.extend(
                f"Record {i:04d}: the archive stores books, maps, and letters.\n"
                for i in range(start, start + 32)
            )
            prefix = (
                "Read the archive below and return only the requested code word.\n"
                "The NORTH code is MAPLE. The SOUTH code is CEDAR.\n" + "".join(records)
            )
            if len(self.tokenizer.encode(prefix)) >= 3 * BLOCK_SIZE + 127:
                break
        prompts = []
        for direction in ("NORTH", "SOUTH"):
            prompts.append(
                self.tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": prefix
                            + f"\nWhat is the {direction} code? Reply with only the code word.",
                        }
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        token_ids = [self.tokenizer.encode(p) for p in prompts]
        common_len = next(
            (i for i, (a, b) in enumerate(zip(*token_ids)) if a != b),
            min(map(len, token_ids)),
        )
        self.assertGreaterEqual(common_len, 3 * BLOCK_SIZE)
        self.assertLess(max(map(len, token_ids)) + 32, 16384)
        return prompts, common_len

    def request(self, label, prompt, reuse_cache, expected_word):
        query = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": 16,
                "top_k": 1,
                "random_seed": 42,
                "reuse_cache": reuse_cache,
                "return_input_ids": True,
                "return_output_ids": True,
                "aux_info": True,
            },
        }
        response = self.session.post(
            f"http://127.0.0.1:{self.server.port}/", json=query, timeout=300
        )
        (self.output_dir / f"{label}.http.txt").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        (self.output_dir / f"{label}.json").write_text(
            json.dumps({"query": query, "actual": actual}, indent=2, ensure_ascii=False)
        )
        self.assertNotIn("error_code", actual, actual)
        self.assertIs(actual.get("finished"), True, actual)
        self.assertEqual(actual["response"].strip(), expected_word, label)
        output_ids = actual["output_ids"]
        self.assertEqual(len(output_ids), 1, label)
        self.assertGreater(len(output_ids[0]), 0, label)
        self.assertLessEqual(len(output_ids[0]), 16, label)
        self.assertTrue(
            all(type(t) is int and 0 <= t < len(self.tokenizer) for t in output_ids[0]),
            label,
        )
        aux = actual["aux_info"]
        self.assertGreater(aux["input_len"], 3 * BLOCK_SIZE, label)
        self.assertEqual(len(actual["input_ids"][0]), aux["input_len"], label)
        self.assertIs(type(aux["reuse_len"]), int, label)
        self.assertGreaterEqual(aux["reuse_len"], 0, label)
        self.assertLess(aux["reuse_len"], aux["input_len"], label)
        logging.info(
            "%s: input_len=%s reuse_len=%s output_ids=%s response=%r",
            label,
            aux["input_len"],
            aux["reuse_len"],
            output_ids,
            actual["response"],
        )
        return actual

    def assert_same_output(self, reference, actual):
        for key in ("input_ids", "output_ids", "response"):
            self.assertEqual(reference[key], actual[key], key)

    def assert_cache_hit(self, actual, max_reuse):
        reused = actual["aux_info"]["reuse_len"]
        self.assertGreaterEqual(reused, 2 * BLOCK_SIZE)
        self.assertLessEqual(reused, max_reuse)
        self.assertEqual(reused % BLOCK_SIZE, 0)

    def test_long_prefix_reuse_matches_uncached_tokens(self):
        prompts, common_len = self.make_prompts()
        north_ref = self.request("north_uncached", prompts[0], False, "MAPLE")
        south_ref = self.request("south_uncached", prompts[1], False, "CEDAR")
        for reference in (north_ref, south_ref):
            self.assertEqual(reference["aux_info"]["reuse_len"], 0)

        cold = self.request("north_cache_fill", prompts[0], True, "MAPLE")
        self.assertEqual(cold["aux_info"]["reuse_len"], 0)
        self.assert_same_output(north_ref, cold)

        repeat = self.request("north_repeat", prompts[0], True, "MAPLE")
        self.assert_cache_hit(repeat, repeat["aux_info"]["input_len"] - 1)
        self.assert_same_output(north_ref, repeat)

        fork = self.request("south_shared_prefix", prompts[1], True, "CEDAR")
        self.assert_cache_hit(fork, common_len)
        self.assert_same_output(south_ref, fork)

        repeat_fork = self.request("south_repeat", prompts[1], True, "CEDAR")
        self.assert_cache_hit(repeat_fork, repeat_fork["aux_info"]["input_len"] - 1)
        self.assert_same_output(south_ref, repeat_fork)

        # A populated cache must still respect the per-request bypass switch.
        bypass = self.request(
            "north_bypass_populated_cache", prompts[0], False, "MAPLE"
        )
        self.assertEqual(bypass["aux_info"]["reuse_len"], 0)
        self.assert_same_output(north_ref, bypass)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
