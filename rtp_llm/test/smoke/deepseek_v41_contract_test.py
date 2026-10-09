"""GB200 V4.1 generation, Engram, graph replay and actual prefix-cache reuse.

Uses the shared smoke server lifecycle. Outputs are compared to an uncached
request from this run, so no unverified golden text or model weights are stored.
"""

import json
import logging
import math
import os
import re
import unittest
from pathlib import Path

import requests
from transformers import AutoTokenizer

from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class DeepSeekV41ContractTest(unittest.TestCase):
    pd_separation = False
    reuse_field = "reuse_len"
    graph_target_verify = 0
    graph_is_prefill = 0
    require_generation_scores = True
    compare_generation_logits = False

    def setUp(self):
        self.model_path = os.environ.get(
            "CHECKPOINT_PATH", "/mnt/nas1/hf/DeepSeek-V4.1-Flash"
        )
        raw = json.loads((Path(self.model_path) / "config.json").read_text())
        text = raw.get("text_config", raw)
        self.assertTrue(
            text.get("engram_layer_ids"), "This contract must exercise Engram layers"
        )
        self.vocab_size = int(text["vocab_size"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        self.output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_servers()
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        # Main initializes alog after the environment level; set the live
        # engine level after readiness, as the shared graph smoke does.
        for manager in {self.server, self.graph_server}:
            level = self.session.post(
                f"http://127.0.0.1:{manager.port}/set_log_level",
                json={"log_level": "DEBUG"},
                timeout=10,
            )
            level.raise_for_status()
            self.assertEqual(level.json(), {"status": "ok"})
        self.log_offset = Path(self.graph_server.log_file_path).stat().st_size
        startup = Path(self.graph_server.log_file_path).read_text(errors="replace")
        self.assertIn("DeepSeek V4.1 Engram initialized on layers", startup)
        self.assertIn("capture success for batch size", startup)
        self.observed = []

    def start_servers(self):
        self.server = MagaServerManager(
            env_args={
                "LOAD_PYTHON_MODEL": "1",
                "LOG_LEVEL": "DEBUG",
                "FT_SERVER_TEST": "1",
                "ROLE_TYPE": "PDFUSION",
                "DSV4_MOE_STRATEGY": "mega",
                "DSV4_USE_MEGA_MOE_SE": "0",
                "DSV4_USE_MEGA_MOE_FUSED": "0",
                "DSV41_SWA_BOUNDED_REPLAY": "0",
                "DSV41_CED": "0",
            },
            role_name="deepseek_v41_contract",
            smoke_args_str=self.server_args(),
        )
        self.addCleanup(self.server.stop_server)
        self.assertTrue(
            self.server.start_server(
                model_path=self.model_path, model_type="deepseek_v41", timeout=3600
            )
        )
        self.graph_server = self.server

    def server_args(self):
        return os.environ["SMOKE_ARGS"]

    def request(self, label, prompt, reuse):
        query = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": 24,
                "top_k": 1,
                "top_p": 1.0,
                "random_seed": 42,
                "reuse_cache": reuse,
                "can_use_pd_separation": self.pd_separation,
                "return_input_ids": True,
                "return_output_ids": True,
                "return_cum_log_probs": self.require_generation_scores,
                "aux_info": True,
            },
        }
        if self.compare_generation_logits:
            query["generate_config"].update(
                return_logits=True,
                select_tokens_id=[self.vocab_size * i // 16 for i in range(1, 16)],
            )
        response = self.session.post(
            f"http://127.0.0.1:{self.server.port}/", json=query, timeout=600
        )
        (self.output_dir / (label + ".http.txt")).write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        (self.output_dir / (label + ".json")).write_text(
            json.dumps({"query": query, "actual": actual}, ensure_ascii=False, indent=2)
        )
        self.assertNotIn("error_code", actual)
        self.assertIs(actual.get("finished"), True)
        self.assertIsInstance(actual.get("response"), str)
        self.assertTrue(actual["response"].strip(), label)
        tokens = actual["output_ids"]
        self.assertEqual(len(tokens), 1)
        self.assertTrue(0 < len(tokens[0]) <= 24)
        self.assertTrue(
            all(type(t) is int and 0 <= t < self.vocab_size for t in tokens[0])
        )
        aux = actual["aux_info"]
        self.assertEqual(len(actual["input_ids"][0]), aux["input_len"])
        self.assertTrue(0 <= aux["reuse_len"] < aux["input_len"])
        self.assertEqual(aux["pd_sep"], self.pd_separation)
        self.assertIn(self.reuse_field, aux)
        probabilities = aux.get("cum_log_probs")
        if self.require_generation_scores:
            self.assertTrue(
                probabilities, "Sampler must expose finite generation scores"
            )
        if probabilities:
            self.assertTrue(
                all(
                    isinstance(x, (int, float)) and math.isfinite(x)
                    for x in probabilities
                )
            )
        if self.compare_generation_logits:
            logits = actual.get("logits")
            self.assertEqual(len(logits), 1)
            self.assertEqual(len(logits[0]), 15)
            self.assertTrue(all(math.isfinite(value) for value in logits[0]))
        self.observed.append(
            {
                "label": label,
                "input_len": aux["input_len"],
                "reuse_len": aux["reuse_len"],
                "measured_reuse_field": self.reuse_field,
                "measured_reuse_len": aux[self.reuse_field],
                "output_ids": tokens,
            }
        )
        return actual

    def assert_same(self, reference, actual):
        for name in ("input_ids", "output_ids", "response"):
            self.assertEqual(reference[name], actual[name], name)
        if self.compare_generation_logits:
            self.assertEqual(reference["logits"], actual["logits"], "generation logits")
            self.assertEqual(
                reference["aux_info"]["cum_log_probs"],
                actual["aux_info"]["cum_log_probs"],
                "sampled generation scores",
            )

    def assert_decode_matches_full_prefill(self, generated):
        """A fresh full prefill independently checks the first decode steps."""
        prefix = list(generated["input_ids"][0])
        expected = []
        for step in range(min(3, len(generated["output_ids"][0]))):
            prompt = self.tokenizer.decode(prefix, skip_special_tokens=False)
            self.assertEqual(self.tokenizer.encode(prompt), prefix)
            query = {
                "prompt": prompt,
                "generate_config": {
                    "max_new_tokens": 1,
                    "top_k": 1,
                    "reuse_cache": False,
                    "can_use_pd_separation": self.pd_separation,
                    "return_input_ids": True,
                    "return_output_ids": True,
                },
            }
            response = self.session.post(
                f"http://127.0.0.1:{self.server.port}/", json=query, timeout=600
            )
            response.raise_for_status()
            actual = response.json()
            (self.output_dir / f"prefill_reference_{step}.json").write_text(
                json.dumps(
                    {"query": query, "actual": actual}, ensure_ascii=False, indent=2
                )
            )
            self.assertEqual(actual["input_ids"][0], prefix)
            self.assertEqual(len(actual["output_ids"][0]), 1)
            token = actual["output_ids"][0][0]
            expected.append(token)
            prefix.append(token)
            if token == self.tokenizer.eos_token_id:
                break
        self.assertGreaterEqual(len(expected), 2, "Prompt must exercise a decode step")
        self.assertEqual(generated["output_ids"][0][: len(expected)], expected)

    def test_graph_engram_and_prefix_reuse(self):
        short = "The capital of France is"
        first = self.request("short_uncached", short, False)
        self.assert_decode_matches_full_prefill(first)
        self.assert_same(first, self.request("short_repeat_uncached", short, False))
        records = []
        while True:
            i = len(records)
            records.append(
                f"Notebook entry {i}: the archive stores books, maps, and letters.\n"
            )
            prefix = "The following notebook records a journey.\n" + "".join(records)
            if len(self.tokenizer.encode(prefix)) >= 768:
                break
        prompt = prefix + "\nIn a few words, the journey was"
        self.assertLess(len(self.tokenizer.encode(prompt)) + 24, 4096)
        self._long_prefix_reuse_checks(prompt)

    def _long_prefix_reuse_checks(self, prompt):
        """Long-context prefix-reuse semantics; the CED PD contract overrides.

        Default (no P/D separation): a cache hit must reproduce the fresh
        generation token-exactly.
        """
        baseline = self.request("long_uncached", prompt, False)
        self.assertEqual(baseline["aux_info"][self.reuse_field], 0)
        cold = self.request("long_cache_fill", prompt, True)
        self.assert_same(baseline, cold)
        repeated = self.request("long_cache_hit", prompt, True)
        self.assert_same(baseline, repeated)
        self.assertGreater(
            repeated["aux_info"][self.reuse_field], cold["aux_info"][self.reuse_field]
        )
        self.assertGreaterEqual(repeated["aux_info"][self.reuse_field], 256)
        self.assert_same(baseline, self.request("long_second_cache_hit", prompt, True))
        bypass = self.request("populated_cache_bypass", prompt, False)
        self.assert_same(baseline, bypass)
        self.assertEqual(bypass["aux_info"][self.reuse_field], 0)
        self.assert_decode_graph_replayed_and_record_coverage()

    def assert_decode_graph_replayed_and_record_coverage(self):
        with open(self.graph_server.log_file_path, "rb") as reader:
            reader.seek(self.log_offset)
            runtime = reader.read().decode(errors="replace")
        replay = re.findall(
            rf"\[PyWrappedModel\] using CUDA graph forward, is_target_verify={self.graph_target_verify}, is_prefill={self.graph_is_prefill}, graph_bs=(\d+)",
            runtime,
        )
        self.assertIn("1", replay, "No observed decode graph replay after startup")
        (self.output_dir / "v41_coverage.json").write_text(
            json.dumps(
                {
                    "requests": self.observed,
                    "decode_graph_batches": sorted(set(replay)),
                    "engram_checkpoint_layers": True,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
