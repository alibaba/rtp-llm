"""Real-model scenarios sharing one server per mixed FP8/NVFP4 TP2 profile.

Every cache restore scenario establishes its own fill/hit sequence. Uncached
requests provide references without relying on unittest method ordering.
"""

import json
import logging
import os
import re
import shlex
import sys
import unittest
from pathlib import Path

# The shared comparers import their peers through the smoke package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
        assert (
            aux["speculative_draft_rounds"] == 0
        ), "Non-speculative generation ran speculation"


def execution_modes(smoke_args):
    """Read the same explicit flags passed to the server, including plain decode."""
    tokens = shlex.split(smoke_args)

    def option(name, default):
        return tokens[tokens.index(name) + 1] if name in tokens else default

    return (
        option("--enable_cuda_graph", "0") == "1",
        option("--sp_type", "none") == "mtp",
    )


class Qwen4SM120ScenariosTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import requests
        from transformers import AutoTokenizer

        from rtp_llm.test.utils.maga_server_manager import MagaServerManager

        fixture = (
            Path(__file__).parent / "data/model/qwen4_exp/q_r_qwen4_exp_fp8_full.json"
        )
        cls.task_info = json.loads(fixture.read_text())
        cls.model_path = os.environ.get(
            "QWEN4_EXP_CHECKPOINT", cls.task_info["model_path"]
        )
        raw = json.loads((Path(cls.model_path) / "config.json").read_text())
        cls.text_config = raw.get("text_config", raw)
        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_path, trust_remote_code=True, local_files_only=True
        )
        cls.output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.observed = []
        cls.graph_batches = []
        cls.baseline = None
        cls.session = requests.Session()
        cls.session.trust_env = False
        cls.addClassCleanup(cls.session.close)
        smoke_args = os.environ["SMOKE_ARGS"]
        cls.graph_enabled, cls.mtp_enabled = execution_modes(smoke_args)
        if cls.mtp_enabled:
            smoke_args += " --sp_checkpoint_path " + shlex.quote(cls.model_path)
        cls.server = MagaServerManager(
            env_args={
                "RTP_LLM_STREAM_ASYNC": "1",
                "LOG_LEVEL": "DEBUG",
                "FT_SERVER_TEST": "1",
            },
            role_name="qwen4_sm120_scenarios",
            smoke_args_str=smoke_args,
        )
        cls.addClassCleanup(cls.server.stop_server)
        if not cls.server.start_server(
            model_path=cls.model_path, model_type="qwen4_exp", timeout=3600
        ):
            raise RuntimeError("Qwen4 server did not become healthy")
        response = cls.session.post(
            f"http://127.0.0.1:{cls.server.port}/set_log_level",
            json={"log_level": "DEBUG"},
            timeout=10,
        )
        response.raise_for_status()
        cls.addClassCleanup(cls.write_coverage)

    @classmethod
    def write_coverage(cls):
        (cls.output_dir / "qwen4_sm120_coverage.json").write_text(
            json.dumps(
                {
                    "profile": {"mtp": cls.mtp_enabled, "graph": cls.graph_enabled},
                    "requests": cls.observed,
                    "target_graph_batches": cls.graph_batches,
                },
                indent=2,
            )
        )

    def url(self, endpoint):
        return f"http://127.0.0.1:{self.server.port}{endpoint}"

    def post(self, label, endpoint, query):
        label = self._testMethodName + "_" + label
        response = self.session.post(self.url(endpoint), json=query, timeout=600)
        (self.output_dir / f"{label}.http.txt").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        (self.output_dir / f"{label}.json").write_text(
            json.dumps({"query": query, "actual": actual}, ensure_ascii=False, indent=2)
        )
        return actual

    def check_chat_goldens(self, phase):
        from smoke.openai_comparer import OpenaiComparer

        for index, item in enumerate(self.task_info["query_result"]):
            with self.subTest(phase=phase, golden=index):
                comparer = OpenaiComparer(None, item["endpoint"], item, None, False)
                actual = comparer.format_result(
                    self.post(
                        f"{phase}_golden_{index}", item["endpoint"], item["query"]
                    )
                )
                comparer.compare_result(comparer.format_result(item["result"]), actual)

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
            self.mtp_enabled and output_tokens > 1,
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

    def long_prompts(self):
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

        return first, second

    def uncached_baseline(self):
        if type(self).baseline is None:
            first, _ = self.long_prompts()
            type(self).baseline = self.request("long_uncached", first, False)
        return type(self).baseline

    def test_01_basic_generation(self):
        self.check_chat_goldens("before_state_checks")

    def test_02_long_context_qsa_tail(self):
        self.check_fresh_prefill_reference(self.uncached_baseline())

    def test_03_prefix_cache_restore(self):
        first, second = self.long_prompts()
        baseline = self.uncached_baseline()
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

    def test_04_speculative_execution_mode(self):
        actual = self.uncached_baseline()
        validate_generation(
            actual,
            actual["input_ids"][0],
            OUTPUT_TOKENS,
            self.text_config["vocab_size"],
            self.mtp_enabled,
        )

    def test_05_graph_execution_mode(self):
        # Observe this request only: warmup and earlier scenarios cannot satisfy it.
        offset = Path(self.server.log_file_path).stat().st_size
        first, _ = self.long_prompts()
        self.request("graph_probe", first, False)
        with Path(self.server.log_file_path).open("rb") as reader:
            reader.seek(offset)
            runtime = reader.read().decode(errors="replace")
        replay = re.findall(
            r"\[PyWrappedModel\] using CUDA graph forward, is_target_verify=1, is_prefill=[01], graph_bs=(\d+)",
            runtime,
        )
        type(self).graph_batches.extend(replay)
        if self.graph_enabled and self.mtp_enabled:
            self.assertIn(
                "1", replay, "No real MTP target CUDA graph replay after startup"
            )
        else:
            self.assertFalse(replay, "Eager target replayed an MTP CUDA graph")

    def test_06_generation_after_rejected_multimodal_request(self):
        from rtp_llm.config.exceptions import ExceptionType

        response = self.session.post(
            self.url("/v1/chat/completions"),
            json={
                "model": "qwen4_exp",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.invalid/image.png"
                                },
                            }
                        ],
                    }
                ],
                "max_tokens": 1,
            },
            timeout=30,
        )
        (self.output_dir / "multimodal_rejection.json").write_text(response.text)
        self.assertFalse(response.ok, response.text)
        error = response.json()
        self.assertEqual(
            error["error_code"], int(ExceptionType.UNSUPPORTED_OPERATION), error
        )
        self.assertIn(
            "Qwen3.8-Flash-Next QSA currently supports text inputs only",
            error["message"],
        )
        self.check_chat_goldens("after_rejected_multimodal_request")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
