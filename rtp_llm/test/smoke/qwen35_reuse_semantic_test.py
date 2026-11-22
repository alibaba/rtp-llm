"""Check Qwen3.5 cold and prefix-reused generation with or without MTP."""

import json
import math
import os
import unittest
from pathlib import Path

import requests

from rtp_llm.test.utils.maga_server_manager import MagaServerManager


FIXTURE = (
    Path(__file__).parent
    / "data/model/qwen3_next/q_r_next_fp8_tp2_mtp_reuse_cache.json"
)
BLOCK_SIZE = 128


class Qwen35ReuseSemanticTest(unittest.TestCase):
    def setUp(self):
        fixture = json.loads(FIXTURE.read_text())
        checkpoint_path = os.environ.get(
            "QWEN35_CHECKPOINT_PATH", fixture["model_path"]
        )
        smoke_args = os.environ["SMOKE_ARGS"].replace(
            fixture["model_path"], checkpoint_path
        )
        self.prompt = fixture["query_result"][0]["query"]["prompt"]
        user_prefix = "<|im_start|>user\n"
        assistant_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.assertTrue(self.prompt.startswith(user_prefix))
        self.assertTrue(self.prompt.endswith(assistant_suffix))
        self.user_text = self.prompt[len(user_prefix) : -len(assistant_suffix)]
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.server = MagaServerManager(
            env_args={"LOAD_PYTHON_MODEL": "1", "FRONTEND_SERVER_COUNT": "1"},
            role_name="qwen35_reuse",
            smoke_args_str=smoke_args,
        )
        self.addCleanup(self.server.stop_server)
        self.assertTrue(
            self.server.start_server(
                model_type=fixture["model_type"], model_path=checkpoint_path
            ),
            "Qwen3.5 server failed to start",
        )
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)

    def request(self, label, *, max_new_tokens, reuse_cache):
        query = {
            "prompt": self.prompt,
            "yield_generator": False,
            "generate_config": {
                "max_new_tokens": max_new_tokens,
                "top_k": 1,
                "top_p": 0,
                "temperature": 0,
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
        self.assertNotIn("error_code", actual, label)
        self.assertIs(actual.get("finished"), True, label)
        self.assertEqual(len(actual["output_ids"]), 1, label)
        self.assertEqual(actual["aux_info"]["output_len"], len(actual["output_ids"][0]))
        self.assertEqual(actual["aux_info"]["input_len"], len(actual["input_ids"][0]))
        self.assertGreater(len(actual["output_ids"][0]), 0, label)
        self.assertLessEqual(len(actual["output_ids"][0]), max_new_tokens, label)
        (self.output_dir / f"{label}.json").write_text(
            json.dumps({"query": query, "actual": actual}, ensure_ascii=False, indent=2)
        )
        return actual

    def request_openai(self, label, *, reuse_cache, exact_prompt=False):
        query = {
            "model": "qwen35_moe",
            "messages": [{"role": "user", "content": self.user_text}],
            "max_tokens": 2,
            "temperature": 1 if exact_prompt else 0,
            "top_p": 0,
            "seed": 42,
            "stream": False,
            "logprobs": True,
            "logprobs_mode": "original",
            "top_logprobs": 5,
            "extra_configs": {"top_k": 1, "reuse_cache": reuse_cache},
        }
        if exact_prompt:
            # Use the raw fixture's prompt bytes through the OpenAI renderer so
            # its original logprobs can be compared with the raw cache test.
            # Jinja drops a literal final newline from a template, so emit it
            # through an expression instead.
            self.assertTrue(self.prompt.endswith("\n"))
            query["user_template"] = self.prompt[:-1] + "{{ final_newline }}"
            query["chat_template_kwargs"] = {"final_newline": "\n"}
            rendered = self.session.post(
                f"http://127.0.0.1:{self.server.port}/v1/chat/render",
                json=query,
                timeout=300,
            )
            rendered.raise_for_status()
            rendered_ids = rendered.json()["input_ids"]
            (self.output_dir / f"{label}.render.json").write_text(
                json.dumps(rendered.json(), ensure_ascii=False, indent=2)
            )
            self.assertEqual(rendered_ids, self.reference_input_ids)
        response = self.session.post(
            f"http://127.0.0.1:{self.server.port}/v1/chat/completions",
            json=query,
            timeout=300,
        )
        (self.output_dir / f"{label}.http.txt").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        self.assertIn("choices", actual, label)
        (self.output_dir / f"{label}.json").write_text(
            json.dumps({"query": query, "actual": actual}, ensure_ascii=False, indent=2)
        )
        return actual

    def test_prefix_hit_preserves_target_distribution(self):
        reference = self.request("uncached_20", max_new_tokens=20, reuse_cache=False)
        self.assertEqual(reference["aux_info"]["reuse_len"], 0)
        self.reference_input_ids = reference["input_ids"][0]

        seed = self.request("seed_20", max_new_tokens=20, reuse_cache=True)
        self.assertEqual(seed["aux_info"]["reuse_len"], 0)
        self.assertEqual(seed["output_ids"], reference["output_ids"])

        mismatches = []
        raw_two = None
        for length in (1, 2, 3, 5, 20):
            uncached = self.request(
                f"uncached_{length}", max_new_tokens=length, reuse_cache=False
            )
            cached = self.request(
                f"cached_{length}", max_new_tokens=length, reuse_cache=True
            )
            self.assertEqual(uncached["aux_info"]["reuse_len"], 0, length)
            reused = cached["aux_info"]["reuse_len"]
            self.assertGreaterEqual(reused, 2 * BLOCK_SIZE, length)
            self.assertLess(reused, cached["aux_info"]["input_len"], length)
            self.assertEqual(reused % BLOCK_SIZE, 0, length)
            self.assertEqual(uncached["input_ids"], cached["input_ids"], length)

            expected_ids = uncached["output_ids"][0]
            actual_ids = cached["output_ids"][0]
            if length == 2:
                raw_two = (uncached, cached)
            if expected_ids != actual_ids or uncached["response"] != cached["response"]:
                first_difference = next(
                    (
                        i
                        for i, (expected, actual) in enumerate(
                            zip(expected_ids, actual_ids)
                        )
                        if expected != actual
                    ),
                    min(len(expected_ids), len(actual_ids)),
                )
                mismatches.append(
                    {
                        "max_new_tokens": length,
                        "first_different_token": first_difference,
                        "uncached_ids": expected_ids,
                        "cached_ids": actual_ids,
                    }
                )

        (self.output_dir / "comparison.json").write_text(
            json.dumps(mismatches, ensure_ascii=False, indent=2)
        )
        # The public logprobs response exposes the target's original top-token
        # probabilities at the first decode step without relying on draft scores.
        openai_cold = self.request_openai("openai_uncached_2", reuse_cache=False)
        openai_hot = self.request_openai("openai_cached_2", reuse_cache=True)
        cold_content = (openai_cold["choices"][0].get("logprobs") or {}).get(
            "content"
        )
        hot_content = (openai_hot["choices"][0].get("logprobs") or {}).get(
            "content"
        )
        self.assertEqual(len(cold_content), 2)
        self.assertEqual(cold_content, hot_content)
        self.assertEqual(
            openai_cold["usage"]["prompt_tokens"],
            openai_hot["usage"]["prompt_tokens"],
        )
        self.assertGreaterEqual(
            openai_hot["usage"]["prompt_tokens_details"]["cached_tokens"],
            2 * BLOCK_SIZE,
        )
        (self.output_dir / "openai_comparison.json").write_text(
            json.dumps(
                {
                    "cold_prompt_tokens": openai_cold.get("usage", {}).get(
                        "prompt_tokens"
                    ),
                    "hot_prompt_tokens": openai_hot.get("usage", {}).get(
                        "prompt_tokens"
                    ),
                    "cold_content": cold_content,
                    "hot_content": hot_content,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        exact_cold = self.request_openai(
            "openai_exact_uncached_2", reuse_cache=False, exact_prompt=True
        )
        exact_hot = self.request_openai(
            "openai_exact_cached_2", reuse_cache=True, exact_prompt=True
        )
        exact_cold_content = (exact_cold["choices"][0].get("logprobs") or {}).get(
            "content"
        )
        exact_hot_content = (exact_hot["choices"][0].get("logprobs") or {}).get(
            "content"
        )
        self.assertEqual(len(exact_cold_content), 2)
        self.assertEqual(len(exact_hot_content), 2)
        self.assertEqual(
            exact_cold["usage"]["prompt_tokens"], len(self.reference_input_ids)
        )
        self.assertEqual(
            exact_hot["usage"]["prompt_tokens"], len(self.reference_input_ids)
        )
        self.assertGreaterEqual(
            exact_hot["usage"]["prompt_tokens_details"]["cached_tokens"],
            2 * BLOCK_SIZE,
        )
        (self.output_dir / "openai_exact_comparison.json").write_text(
            json.dumps(
                {
                    "cold_content": exact_cold_content,
                    "hot_content": exact_hot_content,
                    "cold_prompt_tokens": exact_cold["usage"]["prompt_tokens"],
                    "hot_prompt_tokens": exact_hot["usage"]["prompt_tokens"],
                    "cold_reuse": (
                        exact_cold["usage"].get("prompt_tokens_details") or {}
                    ).get("cached_tokens", 0),
                    "hot_reuse": exact_hot["usage"]["prompt_tokens_details"].get(
                        "cached_tokens"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        # With BF16 KV reuse the target's two most likely tokens can swap when
        # their probabilities are nearly equal. A greedy string comparison is
        # then not a sound cache oracle: all later tokens follow different
        # contexts. Compare the original target distributions at that boundary
        # and accept only the known whitespace tie, not arbitrary drift.
        self.assertIsNotNone(raw_two)
        for raw, content in zip(raw_two, (exact_cold_content, exact_hot_content)):
            self.assertEqual(raw["response"], "".join(item["token"] for item in content))
        self.assertEqual(exact_cold_content[0]["token"], exact_hot_content[0]["token"])

        for cold, hot in zip(exact_cold_content, exact_hot_content):
            cold_top = {
                item["token"]: math.exp(item["logprob"])
                for item in cold["top_logprobs"]
            }
            hot_top = {
                item["token"]: math.exp(item["logprob"])
                for item in hot["top_logprobs"]
            }
            self.assertEqual(set(cold_top), set(hot_top))
            self.assertEqual(len(cold_top), len(cold["top_logprobs"]))
            self.assertEqual(len(hot_top), len(hot["top_logprobs"]))
            cold_tail = max(0.0, 1.0 - sum(cold_top.values()))
            hot_tail = max(0.0, 1.0 - sum(hot_top.values()))
            self.assertLess(cold_tail, 0.01)
            self.assertLess(hot_tail, 0.01)
            # Unknown tail tokens may be disjoint; this is an upper bound on
            # total variation, rather than treating both tails as one token.
            tv_upper = (
                sum(abs(cold_top[token] - hot_top[token]) for token in cold_top)
                + cold_tail
                + hot_tail
            ) / 2
            self.assertLess(tv_upper, 0.05)

        if mismatches:
            self.assertNotEqual(raw_two[0]["output_ids"], raw_two[1]["output_ids"])
            self.assertTrue(
                all(item["first_different_token"] == 1 for item in mismatches),
                mismatches,
            )
            cold_second, hot_second = exact_cold_content[1], exact_hot_content[1]
            self.assertNotEqual(cold_second["token"], hot_second["token"])
            self.assertTrue(cold_second["token"].isspace())
            self.assertTrue(hot_second["token"].isspace())
            for item in (cold_second, hot_second):
                top_two = sorted(
                    item["top_logprobs"], key=lambda entry: entry["logprob"], reverse=True
                )[:2]
                self.assertEqual(
                    {entry["token"] for entry in top_two},
                    {cold_second["token"], hot_second["token"]},
                )
                self.assertLess(top_two[0]["logprob"] - top_two[1]["logprob"], 0.1)


if __name__ == "__main__":
    unittest.main()
