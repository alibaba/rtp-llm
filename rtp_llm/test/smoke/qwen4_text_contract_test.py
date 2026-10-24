"""TP4 native FP8 text contract, 128-token cache and exact decode captures 1..16.

Runs eager/graph via QWEN4_SERVING_GRAPH=0/1. Capture coverage and actual
serving replay coverage are separate artifacts; neither proves an SLA.
"""

import argparse
import json
import logging
import os
import re
import shlex
import sys
import unicodedata
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import requests
from transformers import AutoTokenizer

from rtp_llm.test.smoke import qwen4_serving_contract_test as contract
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class Qwen4TextContractTest(contract.Qwen4ServingContractTest):
    image_question = "animal"  # Reuse functional counting, never token parity.
    cache_block_size = 128
    ep_size = 1

    def setUp(self):
        self.model_path = os.environ["QWEN4_EXP_CHECKPOINT"]
        self.assertTrue(
            (Path(self.model_path) / "model.safetensors.index.json").is_file()
        )
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        generation = json.loads(
            (Path(self.model_path) / "generation_config.json").read_text()
        )
        eos = generation.get("eos_token_id", [])
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos])
        self.eos_token_ids.add(self.tokenizer.eos_token_id)
        self.eos_token_ids.discard(None)
        devices = list(get_gpu_ids())
        self.assertGreaterEqual(len(devices), 4, "TP4 requires four GPUs")
        graph = os.environ.get("QWEN4_SERVING_GRAPH", "0")
        self.assertIn(graph, ("0", "1"))
        self.graph_enabled = graph == "1"
        weights = json.dumps(
            dict(
                full=4,
                linear=1,
                ple_conv_state=4,
                ple_ngram_ctx=4,
                indexer_kv=4,
                indexer_state=4,
            )
        )
        captures = ",".join(str(n) for n in range(1, 17))
        llm_args = (
            f"--tp_size 4 --ep_size {self.ep_size} --world_size 4 --act_type BF16 "
            "--quantization FP8_PER_BLOCK --use_local 1 --warm_up 0 "
            "--load_method scratch --use_deepep_moe 0 --enable_fp32_lm_head 0 "
            "--max_seq_len 8192 --max_context_batch_size 1 --concurrency_limit 16 "
            "--seq_size_per_block 128 --kernel_seq_size_per_block 128 "
            "--reuse_cache 1 --enable_device_cache 1 --enable_linear_cache_reclaim 1 "
            "--kv_cache_mem_mb 4096 --reserver_runtime_mem_mb 4096 "
            f"--enable_cuda_graph {graph} --decode_capture_config {captures} "
            f"--kv_cache_pool_block_weights {shlex.quote(weights)}"
        )
        if self.ep_size == 4:
            llm_args += " --use_all_gather 1"
        self.llm = MagaServerManager(
            env_args={
                "LOAD_PYTHON_MODEL": "1",
                "FT_SERVER_TEST": "1",
                "FRONTEND_SERVER_COUNT": "1",
                "WORLD_SIZE": "4",
                "TP_SIZE": "4",
                "EP_SIZE": str(self.ep_size),
                "ROLE_TYPE": "PDFUSION",
                "VIT_SEPARATION": "0",
            },
            port=MagaServerManager.get_free_port(),
            device_ids=devices[:4],
            role_name="qwen4_native_fp8_text",
            smoke_args_str=llm_args,
        )
        self.addCleanup(self.llm.stop_server)
        (self.output_dir / "configuration.json").write_text(
            json.dumps(
                dict(
                    checkpoint=self.model_path,
                    devices=devices[:4],
                    llm_args=llm_args,
                    graph=int(graph),
                    requested_capture_batches=list(range(1, 17)),
                    concurrency_limit=16,
                    ep_size=self.ep_size,
                    generation_http_requests=29,
                    expected_generated_responses=44,
                ),
                indent=2,
            )
        )
        self.assertTrue(
            self.llm.start_server(
                model_path=self.model_path, model_type="qwen4_exp", timeout=10800
            )
        )
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        if self.graph_enabled:
            startup = Path(self.llm.log_file_path).read_text(errors="replace")
            captured = sorted(
                set(
                    map(
                        int,
                        re.findall(r"capture success for batch size: (\d+)", startup),
                    )
                )
            )
            (self.output_dir / "capture_coverage.json").write_text(
                json.dumps(
                    dict(
                        captured_batches=captured,
                        scope="startup log; not serving concurrency",
                    ),
                    indent=2,
                )
            )
            self.assertTrue(set(range(1, 17)).issubset(captured), captured)
            level = self.session.post(
                f"http://127.0.0.1:{self.llm.port}/set_log_level",
                json={"log_level": "DEBUG"},
                timeout=10,
            )
            level.raise_for_status()
            self.assertEqual(level.json(), {"status": "ok"})

    def assert_counting_answer(self, answer):
        self.assertTrue(
            all(
                char in "0123456789"
                or char.isspace()
                or unicodedata.category(char).startswith("P")
                for char in answer
            ),
            answer,
        )
        matches = list(re.finditer(r"[0-9]+", answer))
        if matches and matches[-1].end() == len(answer):
            tail, expected = matches[-1].group(), str(len(matches))
            if len(tail) < len(expected) and expected.startswith(tail):
                matches.pop()
        numbers = [int(match.group()) for match in matches]
        self.assertGreaterEqual(len(numbers), 20, answer)
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)), answer)

    def check_atomic_batch16(self):
        # One batch API call makes the engine batch explicit. Unlike the
        # independent client wave, graph mode must actually replay batch16.
        prompt = self.tokenizer.apply_chat_template(
            [
                dict(
                    role="user",
                    content="Write the integers from 1 to 1000 in order, separated by spaces.",
                )
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        query = dict(
            prompt_batch=[prompt] * 16,
            generate_config=dict(
                max_new_tokens=128,
                top_k=1,
                random_seed=42,
                reuse_cache=False,
                aux_info=True,
            ),
        )
        (self.output_dir / "atomic_batch16.request.json").write_text(
            json.dumps(query, indent=2, ensure_ascii=False)
        )
        log_path = Path(self.llm.log_file_path)
        offset = log_path.stat().st_size
        # The inherited post() asserts graph_bs=1, so use this batch-specific
        # request and validate its serving window independently.
        response = self.session.post(
            f"http://127.0.0.1:{self.llm.port}/batch_infer", json=query, timeout=1200
        )
        (self.output_dir / "atomic_batch16.response.json").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        self.assertNotIn("error", actual, actual)
        self.assertNotIn("error_code", actual, actual)
        items = actual["response_batch"]
        self.assertEqual(len(items), 16)
        for item in items:
            self.assertNotIn("error", item, item)
            self.assertNotIn("error_code", item, item)
            self.assertIs(item["finished"], True)
            self.assertEqual(item["aux_info"]["reuse_len"], 0)
            self.assertEqual(item["aux_info"]["output_len"], 128)
            self.assert_counting_answer(item["response"])
        with log_path.open("rb") as log:
            log.seek(offset)
            runtime = log.read().decode(errors="replace")
        (self.output_dir / "atomic_batch16.runtime.log").write_text(runtime)
        batches = list(
            map(
                int,
                re.findall(
                    r"\[PyWrappedModel\] using CUDA graph forward, is_target_verify=0, is_prefill=0, graph_bs=(\d+)",
                    runtime,
                ),
            )
        )
        completed = runtime.count("[PyWrappedModel] CUDA graph forward completed")
        normal = runtime.count(
            "[PyWrappedModel] using normal forward, is_target_verify=0, is_prefill=0"
        )
        (self.output_dir / "atomic_batch16_coverage.json").write_text(
            json.dumps(
                dict(
                    responses_correct=len(items),
                    graph_batches=sorted(set(batches)),
                    batch16_forwards=batches.count(16),
                    completed_graph_forwards=completed,
                    normal_decode_forwards=normal,
                    scope="atomic batch API; not independently arriving clients or a capacity SLA",
                ),
                indent=2,
            )
        )
        if self.graph_enabled:
            self.assertEqual(set(batches), {16}, "atomic batch must replay graph16")
            self.assertGreaterEqual(batches.count(16), 127)
            self.assertEqual(completed, len(batches), "incomplete graph16 replay")
            self.assertEqual(normal, 0, "atomic batch decode fell back to eager")

    def test_combined_generation_and_prefix_reuse(self):
        math_result = self.post(
            "math",
            "/v1/chat/completions",
            dict(
                messages=[
                    dict(
                        role="user",
                        content="What is 4 multiplied by 5? Return only the number.",
                    )
                ],
                max_tokens=16,
                temperature=0,
                seed=42,
                chat_template_kwargs=dict(enable_thinking=False),
            ),
        )
        self.assertEqual(math_result["choices"][0]["message"]["content"].strip(), "20")
        echo = self.post(
            "echo",
            "/v1/chat/completions",
            dict(
                messages=[
                    dict(
                        role="user",
                        content="Repeat exactly this text and nothing else: amber river 4729",
                    )
                ],
                max_tokens=32,
                temperature=0,
                seed=42,
                chat_template_kwargs=dict(enable_thinking=False),
            ),
        )
        self.assertEqual(
            echo["choices"][0]["message"]["content"].strip(), "amber river 4729"
        )
        # Keep an explicit text regression above the SM120 GDN CP-prefill
        # threshold. The existing cache fixture is only about 801 input tokens.
        # Prefix reuse is disabled so all of these tokens reach cold prefill.
        long_records = [
            f"Record {i:04d}: books, maps, and letters are in the library."
            for i in range(160)
        ]
        long_prompt = self.tokenizer.apply_chat_template(
            [
                dict(
                    role="user",
                    content="Here is an archive of independent numbered records.\n"
                    + "\n".join(long_records)
                    + "\nWrite the integers from 1 to 1000 in order, separated by spaces.",
                )
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        expected_input_len = len(self.tokenizer.encode(long_prompt))
        self.assertGreaterEqual(expected_input_len, 2048)
        self.assertLess(expected_input_len + 64, 8192)
        long_prefill = self.text_request(
            "long_text_prefill", long_prompt, False, new_tokens=64
        )
        # Assert the server's actual processed input, not just local estimates.
        actual_input_len = long_prefill["aux_info"]["input_len"]
        self.assertGreaterEqual(actual_input_len, 2048)
        self.assertLess(actual_input_len + 64, 8192)
        self.assertEqual(long_prefill["aux_info"]["reuse_len"], 0)
        self.assertTrue(self.eos_token_ids.isdisjoint(long_prefill["output_ids"][0]))
        (self.output_dir / "long_text_prefill_coverage.json").write_text(
            json.dumps(
                dict(
                    estimated_input_len=expected_input_len,
                    actual_input_len=actual_input_len,
                    output_len=len(long_prefill["output_ids"][0]),
                    reuse_len=0,
                    cache_block_size=128,
                    decode_graph_enabled=self.graph_enabled,
                    scope="cold text prefill >=2048 tokens; graph mode checks subsequent decode replay, not prefill graph capture",
                ),
                indent=2,
            )
        )

        prefix = "Here is an archive of independent numbered records.\n" + "".join(
            f"Record {i:04d}: books, maps, and letters are in the library.\n"
            for i in range(40)
        )
        prompts = [
            self.tokenizer.apply_chat_template(
                [
                    dict(
                        role="user",
                        content=prefix
                        + f"\nWrite the integers from {start} to 1000 in order, separated by spaces.",
                    )
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for start in (1, 2)
        ]
        ids = [self.tokenizer.encode(prompt) for prompt in prompts]
        common_len = next(
            (i for i, (a, b) in enumerate(zip(*ids)) if a != b), min(map(len, ids))
        )
        self.assertGreaterEqual(common_len, 3 * 128)
        references = [
            self.text_request(f"uncached_{i}", prompt, False, start=i + 1)
            for i, prompt in enumerate(prompts)
        ]
        for reference in references:
            self.assertEqual(reference["aux_info"]["reuse_len"], 0)
        for label, index, reuse, expect_hit in (
            ("fill", 0, True, False),
            ("repeat", 0, True, True),
            ("fork", 1, True, True),
            ("fork_repeat", 1, True, True),
            ("bypass", 0, False, False),
        ):
            actual = self.text_request(label, prompts[index], reuse, start=index + 1)
            reused = actual["aux_info"]["reuse_len"]
            if expect_hit:
                self.assertGreaterEqual(reused, 2 * 128, label)
                self.assertEqual(reused % 128, 0)
                self.assertLess(reused, actual["aux_info"]["input_len"])
                if label == "fork":
                    self.assertLessEqual(reused, common_len)
            else:
                self.assertEqual(reused, 0, label)

        # The last sampled token is not forwarded, and checkpoint reclamation
        # drains on the following allocation. Leave enough decode steps for
        # three completed pages and their subsequent allocation, even when the
        # input ends on a page boundary. Keep this inside the intended answer:
        # forced continuation after EOS is not a useful deterministic fixture.
        long_reference = self.text_request("long_uncached", prompts[0], False, 512)
        long_cached = self.text_request("long_cached", prompts[0], True, 512)
        for actual in (long_reference, long_cached):
            self.assertTrue(self.eos_token_ids.isdisjoint(actual["output_ids"][0]))
        self.assertEqual(long_reference["aux_info"]["reuse_len"], 0)
        self.assertGreaterEqual(long_cached["aux_info"]["reuse_len"], 2 * 128)

        # A synchronized wave tests 16 completed clients. The scheduler may
        # admit fewer than 16 simultaneous sequences; record actual graph
        # batches rather than interpreting client count as engine capacity.
        barrier = Barrier(16)
        log_path = Path(self.llm.log_file_path)
        offset = log_path.stat().st_size

        def generate(index):
            query = dict(
                messages=[
                    dict(
                        role="user",
                        content="Write the integers from 1 to 1000 in order, separated by spaces.",
                    )
                ],
                max_tokens=256,
                temperature=0,
                seed=42,
                chat_template_kwargs=dict(enable_thinking=False),
                extra_configs=dict(reuse_cache=False),
            )
            with requests.Session() as session:
                session.trust_env = False
                barrier.wait(timeout=60)
                response = session.post(
                    f"http://127.0.0.1:{self.llm.port}/v1/chat/completions",
                    json=query,
                    timeout=1200,
                )
                (self.output_dir / f"concurrent_{index:02d}.http.txt").write_text(
                    response.text
                )
                response.raise_for_status()
                actual = response.json()
            self.assertNotIn("error", actual, actual)
            self.assertNotIn("error_code", actual, actual)
            answer = actual["choices"][0]["message"]["content"]
            self.assert_counting_answer(answer)
            return actual["usage"]

        with ThreadPoolExecutor(max_workers=16) as pool:
            usages = list(pool.map(generate, range(16)))
        with log_path.open("rb") as log:
            log.seek(offset)
            runtime = log.read().decode(errors="replace")
        batches = list(
            map(
                int,
                re.findall(
                    r"\[PyWrappedModel\] using CUDA graph forward, is_target_verify=0, is_prefill=0, graph_bs=(\d+)",
                    runtime,
                ),
            )
        )
        normal = runtime.count(
            "[PyWrappedModel] using normal forward, is_target_verify=0, is_prefill=0"
        )
        completed = runtime.count("[PyWrappedModel] CUDA graph forward completed")
        (self.output_dir / "concurrent_coverage.json").write_text(
            json.dumps(
                dict(
                    completed_clients=len(usages),
                    usages=usages,
                    observed_graph_batches=sorted(set(batches)),
                    graph_forwards=len(batches),
                    completed_graph_forwards=completed,
                    normal_decode_forwards=normal,
                    batch16_replay_observed=16 in batches,
                    scope="one synchronized client wave; no throughput or memory SLA",
                ),
                indent=2,
            )
        )
        if self.graph_enabled:
            self.assertTrue(batches, "no serving decode replay in concurrent wave")
            self.assertGreater(completed, 0)
            self.assertEqual(normal, 0, "ordinary decode fell back to eager")

        self.check_atomic_batch16()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ep-size", type=int, choices=(1, 4), default=1)
    options, test_args = parser.parse_known_args()
    Qwen4TextContractTest.ep_size = options.ep_size
    unittest.main(argv=[sys.argv[0], *test_args])
