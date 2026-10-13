"""MiniCPM-V-4.7 reuse-cache three-scenario coverage (skill 2026-10-14 clause).

Scenario S1: same-prefix different-suffix continuation -> KV prefix hit.
Scenario S2: multimodal partial match -> same image with a different text tail
             hits the image-token KV (partial content match).
Scenario S3: mixed batch -> unrelated prompts interleaved with prefix-sharing
             ones hit and miss respectively.

Each scenario asserts the KV reuse behavior through aux_info.reuse_len /
usage.prompt_tokens_details.cached_tokens and asserts the cached outputs stay
identical to the no-cache baseline (sm120_reuse_cache_test precedent).
"""

import json
import logging
import os
import unittest
from pathlib import Path

import requests
from transformers import AutoTokenizer

from rtp_llm.test.utils.maga_server_manager import MagaServerManager

MODEL_PATH = "/data0/liukan.lk/models/MiniCPM-V-4.7-1B"
BLOCK_SIZE = 2048
# Resolve the smoke image from the test runfiles so the URL works regardless
# of the server process cwd (custom tests do not inherit the entry.py cwd).
IMAGE_URL = os.path.join(
    os.environ.get("TEST_SRCDIR", ""),
    "rtp_llm",
    "rtp_llm",
    "test",
    "smoke",
    "data",
    "model",
    "qwen_vl",
    "1.jpeg",
)


class MiniCPMV47ReuseCacheTest(unittest.TestCase):
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
            self.server.start_server(model_path=MODEL_PATH, model_type="minicpmv4_7"),
            "MiniCPM-V-4.7 1B server failed to start",
        )
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        # (port, reuse_flag) -> response dict; the reuse flag rides the
        # generate_config so the same server serves both baselines and
        # cache-on runs.
        self.responses = {}

    def chat(self, name, messages, reuse=True, max_tokens=16):
        self._last_extra = {"reuse_cache": bool(reuse)} if not reuse else {}
        resp = self.session.post(
            f"http://127.0.0.1:{self.server.port}/v1/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_k": 1,
                "top_p": 0,
                "extra_configs": self._last_extra,
            },
            timeout=600,
        )
        resp.raise_for_status()
        body = resp.json()
        self.responses[(name, reuse)] = body
        dump = self.output_dir / "reuse_responses.json"
        try:
            existing = json.loads(dump.read_text()) if dump.exists() else {}
        except Exception:
            existing = {}
        existing[name] = {
            "reuse_len": (body.get("aux_info") or {}).get("reuse_len"),
            "cached_tokens": ((body.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
            "image_tokens": ((body.get("usage") or {}).get("prompt_tokens_details") or {}).get("image_tokens"),
            "prompt_tokens": (body.get("usage") or {}).get("prompt_tokens"),
            "content": body["choices"][0]["message"]["content"][:80],
        }
        dump.write_text(json.dumps(existing, indent=1))
        return body

    def reuse_of(self, body):
        aux = body.get("aux_info") or {}
        details = (body.get("usage") or {}).get("prompt_tokens_details") or {}
        candidates = [
            aux.get("reuse_len"),
            details.get("cached_tokens"),
        ]
        return max((v for v in candidates if isinstance(v, int)), default=0)

    def same_output(self, a, b):
        self.assertEqual(
            a["choices"][0]["message"]["content"],
            b["choices"][0]["message"]["content"],
            "cached output diverged from the no-cache baseline",
        )

    def make_prefix(self, lines=200):
        records = "".join(
            f"Record {i:04d}: the archive stores books, maps, and letters.\n"
            for i in range(lines)
        )
        return (
            "Read the archive below and return only the requested code word.\n"
            "The NORTH code is MAPLE. The SOUTH code is CEDAR.\n" + records
        )

    def test_s1_same_prefix_different_suffix(self):
        prefix = self.make_prefix()
        base = [
            {"role": "user", "content": prefix + "\nWhat is the NORTH code? Reply with only the code word."}
        ]
        north = [
            {"role": "user", "content": prefix + "\nWhat is the NORTH code? Reply with only the code word."},
            {"role": "assistant", "content": "MAPLE"},
            {"role": "user", "content": "Now what is the SOUTH code? Reply with only the code word."},
        ]
        south = [
            {"role": "user", "content": prefix + "\nWhat is the NORTH code? Reply with only the code word."},
            {"role": "assistant", "content": "MAPLE"},
            {"role": "user", "content": "Which archive record number is the highest? Reply with only the number."},
        ]
        # Baseline without reuse, then the same conversations with reuse on.
        base_ref = self.chat("s1_base_ref", base, reuse=False)
        base_cold = self.chat("s1_base_fill", base, reuse=True)
        self.reuse_of(base_ref) == 0 or self.assertEqual(self.reuse_of(base_ref), 0)
        self.assertEqual(self.reuse_of(base_cold), 0, "first fill must not hit cache")
        self.same_output(base_ref, base_cold)

        warm_north = self.chat("s1_north", north, reuse=True)
        warm_south = self.chat("s1_south", south, reuse=True)
        self.assertGreater(
            self.reuse_of(warm_north), 0, "S1 continuation must hit the shared prefix KV"
        )
        self.assertGreater(
            self.reuse_of(warm_south), 0, "S1 second suffix must hit the shared prefix KV"
        )
        # And the cached continuations agree with their no-cache baselines.
        north_ref = self.chat("s1_north_ref", north, reuse=False)
        south_ref = self.chat("s1_south_ref", south, reuse=False)
        self.same_output(north_ref, warm_north)
        self.same_output(south_ref, warm_south)

    def test_s2_multimodal_partial_match(self):
        image_messages = lambda tail: [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    {"type": "text", "text": tail},
                ],
            }
        ]
        first = image_messages("Describe this image in one short sentence.")
        second = image_messages("What is the dog in this image doing? One short sentence.")
        # Fill the image KV with the first query, then a different text tail
        # on the same image must partially hit (image tokens cached).
        first_cold = self.chat("s2_first", first, reuse=True, max_tokens=48)
        self.assertEqual(self.reuse_of(first_cold), 0)
        warm_second = self.chat("s2_second", second, reuse=True, max_tokens=48)
        self.assertGreater(
            self.reuse_of(warm_second), 0,
            "S2 same-image different-tail must hit the image-token KV",
        )
        second_ref = self.chat("s2_second_ref", second, reuse=False, max_tokens=48)
        self.same_output(second_ref, warm_second)

    def test_s3_mixed_batch_hit_and_miss(self):
        prefix = self.make_prefix(lines=200)
        shared = [
            {"role": "user", "content": prefix + "\nWhat is the SOUTH code? Reply with only the code word."}
        ]
        unrelated = [{"role": "user", "content": "Name one planet. One word only."}]
        self.chat("s3_fill", shared, reuse=True)
        warm = self.chat("s3_shared_again", shared, reuse=True)
        miss = self.chat("s3_unrelated", unrelated, reuse=True)
        self.assertGreater(self.reuse_of(warm), 0, "S3 repeated prompt must hit")
        self.assertEqual(self.reuse_of(miss), 0, "S3 unrelated prompt must miss")
        unrelated_ref = self.chat("s3_unrelated_ref", unrelated, reuse=False)
        self.same_output(unrelated_ref, miss)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
