"""Exercise SM120 BF16 generation and prefix reuse with one Qwen3 server."""

import copy
import json
import logging
import os
import re
import unittest
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from rtp_llm.test.utils.maga_server_manager import MagaServerManager

FIXTURE = Path(__file__).parent / "data/model/qwen3/q_r_4b_bf16_sm120.json"
BLOCK_SIZE = 64


class SM120Qwen3Test(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text())
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.fixture["model_path"], local_files_only=True
        )
        self.server = MagaServerManager(
            env_args={
                "LOAD_PYTHON_MODEL": "1",
                "LOG_LEVEL": "DEBUG",
                "FT_SERVER_TEST": "1",
            },
            role_name="qwen3_bf16",
            smoke_args_str=os.environ["SMOKE_ARGS"],
        )
        self.addCleanup(self.server.stop_server)
        self.assertTrue(
            self.server.start_server(
                model_type=self.fixture["model_type"],
                model_path=self.fixture["model_path"],
            )
        )
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        # Exclude startup capture/warmup from runtime Graph evidence.
        self.log_offset = Path(self.server.log_file_path).stat().st_size
        self.requests = []

    def prompt(self, content):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def query(self, prompt, **config):
        return {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": 64,
                "top_k": 1,
                "reuse_cache": False,
                "can_use_pd_separation": False,
                "return_input_ids": True,
                "return_output_ids": True,
                "aux_info": True,
                **config,
            },
        }

    def request(self, label, query, endpoint="/", streaming=False):
        response = self.session.post(
            f"http://127.0.0.1:{self.server.port}{endpoint}",
            json=query,
            timeout=300,
            stream=streaming,
        )
        with response:
            response.raise_for_status()
            if streaming:
                events = [
                    line[5:].strip()
                    for line in response.iter_lines()
                    if line.startswith(b"data:")
                ]
                self.save(f"{label}.events.json", [event.decode() for event in events])
                self.assertTrue(events, label)
                self.assertEqual(events[-1], b"[done]", label)
                chunks = [json.loads(event) for event in events[:-1]]
                self.assertGreater(len(chunks), 1, label)
                actual = copy.deepcopy(chunks[-1])
                # Native SSE carries only each step's IDs, even when text is
                # cumulative. Compare the concatenated token stream to golden.
                actual["output_ids"] = [[t for c in chunks for t in c["output_ids"][0]]]
                if query["generate_config"].get("return_incremental"):
                    actual["response"] = "".join(c["response"] for c in chunks)
                else:
                    for before, after in zip(chunks, chunks[1:]):
                        self.assertTrue(
                            after["response"].startswith(before["response"])
                        )
                self.save(f"{label}.chunks.json", chunks)
            else:
                actual = response.json()
        self.save(f"{label}.json", {"query": query, "actual": actual})
        self.requests.append(label)
        self.assertNotIn("error_code", actual, label)
        self.assertNotIn("error", actual, label)
        return actual

    def save(self, name, value):
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False))

    def check_single(self, actual, max_tokens=64):
        self.assertIs(actual.get("finished"), True)
        ids = actual["output_ids"]
        self.assertEqual(len(ids), 1)
        self.assertGreater(len(ids[0]), 0)
        self.assertLessEqual(len(ids[0]), max_tokens)
        self.assertTrue(
            all(type(t) is int and 0 <= t < len(self.tokenizer) for t in ids[0])
        )
        aux = actual["aux_info"]
        self.assertIs(aux["pd_sep"], False)
        self.assertEqual(aux["output_len"], len(ids[0]))
        self.assertEqual(aux["input_len"], len(actual["input_ids"][0]))
        self.assertGreaterEqual(aux["reuse_len"], 0)
        self.assertLess(aux["reuse_len"], aux["input_len"])

    def check_same(self, reference, actual):
        self.check_single(actual)
        for field in ("response", "input_ids", "output_ids"):
            self.assertEqual(reference[field], actual[field], field)

    def run_cache(self):
        records = []
        while len(self.tokenizer.encode("".join(records))) < 2048 + 127:
            i = len(records)
            records.append(f"Record {i:04d}: the archive stores maps and letters.\n")
        prefix = (
            "Read the archive and return only the requested code word.\n"
            "The NORTH code is MAPLE. The SOUTH code is CEDAR.\n" + "".join(records)
        )
        prompts = [
            self.prompt(
                prefix + f"\nWhat is the {d} code? Reply with only the code word."
            )
            for d in ("NORTH", "SOUTH")
        ]
        encoded = [self.tokenizer.encode(p) for p in prompts]
        shared_len = next(i for i, (a, b) in enumerate(zip(*encoded)) if a != b)
        self.assertGreater(shared_len, 2048)
        self.assertLess(max(map(len, encoded)) + 64, 8192)
        references = []
        for i, word in enumerate(("MAPLE", "CEDAR")):
            result = self.request(f"cache_ref_{i}", self.query(prompts[i]))
            self.check_single(result)
            self.assertEqual(result["response"].strip(), word)
            self.assertEqual(result["aux_info"]["reuse_len"], 0)
            references.append(result)
        for label, idx, reuse, hit in (
            ("cache_cold", 0, True, False),
            ("cache_repeat", 0, True, True),
            ("cache_shared_suffix", 1, True, True),
            ("cache_repeat_suffix", 1, True, True),
            ("cache_bypass_populated", 0, False, False),
        ):
            actual = self.request(label, self.query(prompts[idx], reuse_cache=reuse))
            self.check_same(references[idx], actual)
            reused = actual["aux_info"]["reuse_len"]
            if hit:
                self.assertGreaterEqual(reused, 2 * BLOCK_SIZE, label)
                self.assertLessEqual(reused, shared_len, label)
                self.assertEqual(reused % BLOCK_SIZE, 0, label)
            else:
                self.assertEqual(reused, 0, label)

    def run_stream_and_stop(self, reference, query):
        for incremental in (False, True):
            streamed = copy.deepcopy(query)
            streamed["yield_generator"] = True
            streamed["generate_config"]["return_incremental"] = incremental
            actual = self.request(f"stream_{incremental}", streamed, streaming=True)
            self.check_same(reference, actual)
        for word, should_stop in ((" blue", True), ("a lighthouse", False)):
            stopped = copy.deepcopy(query)
            stopped["generate_config"]["stop_words_str"] = [word]
            actual = self.request(f"stop_{should_stop}", stopped)
            self.check_single(actual)
            if should_stop:
                self.assertEqual(
                    actual["response"].rstrip(),
                    reference["response"].split(word)[0].rstrip(),
                )
                self.assertLess(
                    len(actual["output_ids"][0]), len(reference["output_ids"][0])
                )
            else:
                self.check_same(reference, actual)

    def run_batch_and_seed(self, references, queries):
        batch = {
            "prompt_batch": [q["prompt"] for q in queries],
            "generate_config": queries[0]["generate_config"],
        }
        actual = self.request("batch_diverse", batch, "/batch_infer")
        self.assertEqual(len(actual["response_batch"]), len(references))
        for reference, item in zip(references, actual["response_batch"]):
            self.assertEqual(item["response"], reference["response"])
            self.assertIs(item["finished"], True)
            self.assertIs(item["aux_info"]["pd_sep"], False)
        multi = copy.deepcopy(queries[-1])
        multi["generate_config"]["num_return_sequences"] = 2
        actual = self.request("multi_sequence", multi)
        self.assertEqual(actual["response"], [references[-1]["response"]] * 2)
        self.assertIs(actual["finished"], True)
        sampled = self.query(
            self.prompt("Write one sentence about a forest."),
            top_k=100,
            random_seed=46,
            max_new_tokens=32,
        )
        first = self.request("seed_first", sampled)
        self.check_single(first, 32)
        self.check_same(first, self.request("seed_repeat", sampled))

    def run_logits_and_probs(self, reference, query):
        config = dict(
            query["generate_config"],
            return_logits=True,
            logits_index=2,
            is_streaming=True,
        )
        full = self.request(
            "logits_full",
            dict(query, generate_config=config, yield_generator=True),
            streaming=True,
        )
        self.check_same(reference, full)
        logits = torch.tensor(full["logits"])
        self.assertEqual(logits.ndim, 2)
        self.assertGreaterEqual(logits.shape[-1], len(self.tokenizer))
        self.assertTrue(torch.isfinite(logits).all().item())
        selected = [1, 2, reference["output_ids"][0][1]]
        config["select_tokens_id"] = selected
        sliced = self.request(
            "logits_selected",
            dict(query, generate_config=config, yield_generator=True),
            streaming=True,
        )
        self.check_same(reference, sliced)
        torch.testing.assert_close(
            torch.tensor(sliced["logits"]), logits[:, selected], rtol=1e-2, atol=1e-2
        )
        probs_query = copy.deepcopy(query)
        probs_query["generate_config"]["return_softmax_probs"] = True
        probs = self.request("softmax", probs_query)
        self.check_same(reference, probs)
        values = torch.tensor(probs["aux_info"]["softmax_probs"])
        self.assertEqual(values.numel(), len(reference["output_ids"][0]))
        self.assertTrue(torch.isfinite(values).all().item())
        self.assertTrue(((values >= 0) & (values <= 1)).all().item())
        self.assertAlmostEqual(
            values[1].item(),
            torch.softmax(logits[0], -1)[selected[-1]].item(),
            delta=0.01,
        )

    def run_openai(self, reference):
        query = {
            "messages": self.fixture["query_result"][2]["messages"],
            "max_tokens": 64,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "extra_configs": {
                "top_k": 1,
                "reuse_cache": False,
                "can_use_pd_separation": False,
                "return_logits": True,
                "logits_index": 2,
                "is_streaming": True,
                "select_tokens_id": [1, 2],
            },
        }
        actual = self.request("openai", query, "/v1/chat/completions")
        choice = actual["choices"][0]
        self.assertEqual(choice["message"]["content"], reference["response"])
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertIs(actual["aux_info"]["pd_sep"], False)
        values = torch.tensor(actual["extra_outputs"]["logits"])
        self.assertEqual(tuple(values.shape), (1, 2))
        self.assertTrue(torch.isfinite(values).all().item())
        native_query = self.query(
            self.prompt(query["messages"][0]["content"]),
            return_logits=True,
            logits_index=2,
            is_streaming=True,
            select_tokens_id=[1, 2],
        )
        native_query["yield_generator"] = True
        native = self.request("openai_logits_reference", native_query, streaming=True)
        self.check_same(reference, native)
        torch.testing.assert_close(
            values, torch.tensor(native["logits"]), rtol=1e-2, atol=1e-2
        )

    def check_graph(self):
        with open(self.server.log_file_path, "rb") as log:
            log.seek(self.log_offset)
            runtime = log.read().decode(errors="replace")
        matches = re.findall(
            r"\[PyWrappedModel\] using CUDA graph forward, is_target_verify=0, is_prefill=0, graph_bs=(\d+)",
            runtime,
        )
        self.save(
            "coverage.json",
            {"requests": self.requests, "decode_graph_batches": sorted(set(matches))},
        )
        self.assertIn("1", matches, "No runtime decode Graph replay for batch 1")
        self.assertIn("2", matches, "No runtime decode Graph replay for batch 2")

    def test_requests_share_one_server(self):
        # One test method owns the server; helper phases never restart it.
        with self.subTest(scenario="long_prefix_cache"):
            self.run_cache()
        references, queries = [], []
        for i, case in enumerate(self.fixture["query_result"]):
            query = self.query(self.prompt(case["messages"][0]["content"]))
            actual = self.request(case["name"], query)
            self.check_single(actual)
            self.assertEqual(
                actual["response"].strip().rstrip("."),
                case["expected_text"].rstrip("."),
            )
            self.save(
                f"smoke_actual/rtp_llm/test/smoke/data/model/qwen3/q_r_4b_bf16_sm120.query_{i}.json",
                {k: actual[k] for k in ("response", "output_ids")},
            )
            references.append(actual)
            queries.append(query)
        phases = (
            (
                "stream_and_stop",
                lambda: self.run_stream_and_stop(references[2], queries[2]),
            ),
            ("batch_and_seed", lambda: self.run_batch_and_seed(references, queries)),
            (
                "logits_and_probs",
                lambda: self.run_logits_and_probs(references[2], queries[2]),
            ),
            ("openai", lambda: self.run_openai(references[2])),
            ("decode_graph", self.check_graph),
        )
        for name, run in phases:
            with self.subTest(scenario=name):
                run()
        for case, actual in zip(self.fixture["query_result"], references):
            with self.subTest(golden=case["name"]):
                self.assertEqual(
                    {k: actual[k] for k in ("response", "output_ids")}, case["result"]
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
