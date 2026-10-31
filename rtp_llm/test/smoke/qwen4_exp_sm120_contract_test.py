"""Real Qwen4 SM120 generation, QSA tail, prefix reuse and MTP state checks.

Keep the existing chat goldens and run the state checks in the same server.
Cold-cache tokens from this run are the reference for prefix-cache restores;
fresh one-token prefills independently check speculative decode decisions.
"""

import json
import logging
import os
import re
import shlex
import unittest
from pathlib import Path

PAGE_SIZE = 128
OUTPUT_TOKENS = 160


def build_prompt_pair(tokenizer, token_budget, compress_ratio, render_prompt=None):
    if token_budget <= 0 or compress_ratio <= 1 or PAGE_SIZE % compress_ratio:
        raise ValueError("Expected page-aligned QSA geometry with a compressed tail")
    minimum = max(4096, token_budget + 2 * PAGE_SIZE)
    line = "The archive records books, maps, letters, and journeys.\n"
    prefix = "The archive keyword is MAP.\n" + line
    while len(tokenizer.encode(prefix, add_special_tokens=False)) < minimum:
        prefix += line
    for padding in range(2 * PAGE_SIZE):
        shared = prefix + " x" * padding
        first = shared + (
            "\nRepeat the archive keyword 200 times, separated by spaces. "
            "Output no explanation or other text."
        )
        if render_prompt is not None:
            first = render_prompt(first)
        tokens = tokenizer.encode(first, add_special_tokens=False)
        # Begin at the final token of a page. Decode crosses two page boundaries
        # and the incomplete compression block before the next complete block.
        if len(tokens) % PAGE_SIZE == PAGE_SIZE - 1:
            second = shared + (
                "\nRepeat the archive keyword 200 times, one per line. "
                "Output no explanation or other text."
            )
            if render_prompt is not None:
                second = render_prompt(second)
            return first, second
    raise AssertionError("Tokenizer could not construct the requested QSA page tail")


def validate_generation(actual, expected_ids, output_tokens, vocab_size, speculative):
    assert "error_code" not in actual, actual
    assert actual.get("finished") is True, actual
    assert actual["input_ids"] == [
        expected_ids
    ], "Server consumed different prompt tokens"
    tokens = actual["output_ids"]
    assert len(tokens) == 1 and len(tokens[0]) == output_tokens, tokens
    assert all(type(t) is int and 0 <= t < vocab_size for t in tokens[0]), tokens
    aux = actual["aux_info"]
    assert aux["input_len"] == len(expected_ids), aux
    assert aux["output_len"] == output_tokens, aux
    assert 0 <= aux["reuse_len"] < len(expected_ids), aux
    if speculative:
        rounds = aux["speculative_draft_rounds"]
        accepted = aux["speculative_accepted_tokens_per_pos"]
        assert rounds >= 2, "No repeated MTP draft/verify rounds"
        assert accepted and all(
            type(n) is int and 0 <= n <= rounds for n in accepted
        ), aux
        assert sum(accepted) > 0, "No speculative token was accepted"
    else:
        assert aux["speculative_draft_rounds"] == 0, "One-token prefill ran speculation"


class Qwen4SM120ContractTest(unittest.TestCase):
    def setUp(self):
        import requests
        from transformers import AutoTokenizer

        from rtp_llm.test.utils.maga_server_manager import MagaServerManager

        self.model_path = os.environ.get(
            "QWEN4_EXP_CHECKPOINT",
            "/mnt/fuse/smoke-test/hf/Qwen--Qwen3.8-Flash-Next-FP8--"
            "236dfdf285828023ca3bcd3f37366c58a3469b13",
        )
        raw = json.loads((Path(self.model_path) / "config.json").read_text())
        self.text_config = raw.get("text_config", raw)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        self.output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.observed = []
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        smoke_args = os.environ["SMOKE_ARGS"]
        tokens = shlex.split(smoke_args)
        self.graph_enabled = tokens[tokens.index("--enable_cuda_graph") + 1] == "1"
        self.server = MagaServerManager(
            env_args={
                "RTP_LLM_STREAM_ASYNC": "1",
                "LOG_LEVEL": "DEBUG",
                "FT_SERVER_TEST": "1",
            },
            role_name="qwen4_sm120_contract",
            smoke_args_str=smoke_args
            + " --sp_checkpoint_path "
            + shlex.quote(self.model_path),
        )
        self.addCleanup(self.server.stop_server)
        self.assertTrue(
            self.server.start_server(
                model_path=self.model_path, model_type="qwen4_exp", timeout=3600
            ),
            "Qwen4 server did not become healthy",
        )
        response = self.session.post(
            self.url("/set_log_level"), json={"log_level": "DEBUG"}, timeout=10
        )
        response.raise_for_status()
        self.log_offset = Path(self.server.log_file_path).stat().st_size

    def url(self, endpoint):
        return f"http://127.0.0.1:{self.server.port}{endpoint}"

    def post(self, label, endpoint, query):
        response = self.session.post(self.url(endpoint), json=query, timeout=600)
        (self.output_dir / f"{label}.http.txt").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        (self.output_dir / f"{label}.json").write_text(
            json.dumps({"query": query, "actual": actual}, ensure_ascii=False, indent=2)
        )
        return actual

    def check_chat_goldens(self, phase):
        from rtp_llm.openai.api_datatype import ChatCompletionResponse

        fixture = (
            Path(__file__).parent / "data/model/qwen4_exp/q_r_qwen4_exp_fp8_full.json"
        )
        for index, item in enumerate(json.loads(fixture.read_text())["query_result"]):
            with self.subTest(phase=phase, golden=index):
                actual = ChatCompletionResponse(
                    **self.post(
                        f"{phase}_golden_{index}", item["endpoint"], item["query"]
                    )
                )
                expected = ChatCompletionResponse(**item["result"])
                self.assertEqual(actual.choices, expected.choices)
                self.assertEqual(actual.extra_outputs, expected.extra_outputs)
                if not item.get("compare_config", {}).get("skip_usage", False):
                    self.assertEqual(actual.usage, expected.usage)

    def request(self, label, prompt, reuse, output_tokens=OUTPUT_TOKENS):
        expected_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        query = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": output_tokens,
                "ignore_eos": True,
                "top_k": 1,
                "top_p": 1.0,
                "random_seed": 42,
                "reuse_cache": reuse,
                "return_input_ids": True,
                "return_output_ids": True,
                "aux_info": True,
            },
        }
        actual = self.post(label, "/", query)
        validate_generation(
            actual,
            expected_ids,
            output_tokens,
            self.text_config["vocab_size"],
            output_tokens > 1,
        )
        if output_tokens > 1:
            # A high-confidence task with an answer stored before the long
            # prefix keeps FP8 prefill/decode rounding from turning this cache
            # check into a comparison of unconstrained creative completions.
            words = actual["response"].split()
            self.assertGreaterEqual(len(words), output_tokens // 4)
            self.assertEqual(set(words), {"MAP"})
        if not reuse:
            self.assertEqual(actual["aux_info"]["reuse_len"], 0)
        self.observed.append(
            {
                "label": label,
                "input_len": len(expected_ids),
                "aux_info": actual["aux_info"],
            }
        )
        return actual

    def assert_same_tokens(self, reference, actual):
        self.assertEqual(reference["input_ids"], actual["input_ids"])
        self.assertEqual(reference["output_ids"], actual["output_ids"])
        self.assertEqual(reference["response"], actual["response"])

    def check_fresh_prefill_reference(self, generated):
        input_ids = generated["input_ids"][0]
        output_ids = generated["output_ids"][0]
        # Check initial MTP decisions and both sides of the next page boundary.
        for step in (0, 1, 2, PAGE_SIZE, PAGE_SIZE + 1):
            prefix = input_ids + output_ids[:step]
            prompt = self.tokenizer.decode(prefix, skip_special_tokens=False)
            self.assertEqual(
                self.tokenizer.encode(prompt, add_special_tokens=False), prefix
            )
            fresh = self.request(
                f"fresh_prefill_{step}", prompt, False, output_tokens=1
            )
            self.assertEqual(fresh["output_ids"][0], [output_ids[step]])

    def test_goldens_qsa_cache_and_mtp(self):
        self.check_chat_goldens("before_state_checks")
        budget = int(self.text_config["indexer_budget"])
        ratio = int(self.text_config["indexer_compress_ratio"])

        def render_prompt(text):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        first, second = build_prompt_pair(
            self.tokenizer, budget, ratio, render_prompt=render_prompt
        )
        first_ids = self.tokenizer.encode(first, add_special_tokens=False)
        second_ids = self.tokenizer.encode(second, add_special_tokens=False)
        common = next(
            (
                i
                for i, (left, right) in enumerate(zip(first_ids, second_ids))
                if left != right
            ),
            min(len(first_ids), len(second_ids)),
        )
        self.assertGreaterEqual(common, 2 * PAGE_SIZE)
        self.assertGreater(len(first_ids), budget)
        self.assertNotEqual(len(first_ids) % ratio, 0)
        self.assertLess(max(len(first_ids), len(second_ids)) + OUTPUT_TOKENS, 32768)

        baseline = self.request("long_uncached", first, False)
        self.check_fresh_prefill_reference(baseline)
        # GDN decode retains only its active tail. Publish prompt checkpoints
        # before a long generation advances beyond those state pages.
        fill = self.request("long_cache_fill", first, True, output_tokens=1)
        self.assertEqual(fill["output_ids"][0], baseline["output_ids"][0][:1])
        warm = self.request("long_cache_hit", first, True)
        self.assertGreaterEqual(warm["aux_info"]["reuse_len"], PAGE_SIZE)
        self.assertEqual(warm["aux_info"]["reuse_len"] % PAGE_SIZE, 0)
        self.assert_same_tokens(baseline, warm)

        other = self.request("changed_suffix_uncached", second, False)
        fill = self.request("changed_suffix_cache_fill", second, True, output_tokens=1)
        self.assertEqual(fill["output_ids"][0], other["output_ids"][0][:1])
        reused = self.request("changed_suffix_cache_hit", second, True)
        self.assertGreaterEqual(reused["aux_info"]["reuse_len"], PAGE_SIZE)
        self.assert_same_tokens(other, reused)
        self.assert_same_tokens(
            baseline, self.request("long_after_changed_suffix", first, True)
        )
        self.assert_same_tokens(
            baseline, self.request("populated_cache_bypass", first, False)
        )

        # A short request after long speculative/cache activity detects leaked state.
        self.check_chat_goldens("after_state_checks")
        with Path(self.server.log_file_path).open("rb") as reader:
            reader.seek(self.log_offset)
            runtime = reader.read().decode(errors="replace")
        replay = re.findall(
            r"\[PyWrappedModel\] using CUDA graph forward, is_target_verify=1, is_prefill=[01], graph_bs=(\d+)",
            runtime,
        )
        if self.graph_enabled:
            self.assertIn(
                "1", replay, "No real MTP target CUDA graph replay after startup"
            )
        else:
            self.assertFalse(replay, "Eager target replayed an MTP CUDA graph")
        (self.output_dir / "qwen4_sm120_coverage.json").write_text(
            json.dumps(
                {"requests": self.observed, "target_graph_batches": replay}, indent=2
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
