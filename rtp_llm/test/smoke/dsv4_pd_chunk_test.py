"""Real DSV4 P/D transfer with batched chunks and a reused prefix."""

import json
import logging
import os
import re
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import requests
from transformers import AutoTokenizer

# The shared runner's imports use the same smoke package root as entry.py.
sys.path.insert(0, str(Path(__file__).parent.parent))
from smoke.multi_inst_case_runner import PdSeperationCaseRunner  # noqa: E402
from smoke.task_info import TaskInfo, TaskStates  # noqa: E402


class _PDChunkRunner(PdSeperationCaseRunner):
    def curl_server(self, server_manager):
        self.test_case.check_requests(server_manager)
        return TaskStates()


class DSV4PDChunkTest(unittest.TestCase):
    def test_batch_reuses_prefix_and_transfers_chunked_prefill(self):
        model_path = os.environ.get(
            "DSV4_MODEL_PATH", "/mnt/nas1/hf/DeepSeek-V4-Flash-DSpark"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        self.output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # SMOKE_ARGS reserves four GPUs; each independently launched role owns two.
        common_args = os.environ["SMOKE_ARGS"].replace(
            "--world_size 4", "--world_size 2"
        )
        self.chunk_size = PdSeperationCaseRunner._extract_int_arg(
            common_args, "--prefill_chunk_size", 0
        )
        self.assertGreater(self.chunk_size, 0)
        envs = [
            "LOAD_PYTHON_MODEL=1",
            "WORLD_SIZE=2",
            "DG_JIT_CPP_STANDARD=20",
            "RTP_LLM_DEVICE_INPUT=1",
            "ENABLE_LAYER_MICRO_BATCH=0",
            "LINEAR_STEP=1",
            "CP_FORCE_SINGLE_PREFILL=0",
            "FRONTEND_SERVER_COUNT=1",
        ]
        runner = _PDChunkRunner(
            task_info=TaskInfo(
                model_type="deepseek_v4", model_path=model_path, query_result=[]
            ),
            env_args={"prefill": envs, "decode": envs},
            gpu_card="SM100_ARM_CU13",
            smoke_args={
                "prefill": common_args
                + " --role_type PREFILL --tp_size 2 --dp_size 1 --ep_size 2"
                " --cp_rotate_method ALL_GATHER --use_deepep_low_latency 0",
                "decode": common_args
                + " --role_type DECODE --tp_size 1 --dp_size 2 --ep_size 2"
                " --cp_rotate_method PREFILL_CP --prefill_cp_size 2"
                " --use_deepep_low_latency 1",
            },
        )
        runner.test_case = self
        state = runner.run()
        self.assertTrue(state.ret, str(state))

    def request(self, server, label, prompt, reuse, barrier=None):
        query = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": 8,
                "min_new_tokens": 8,
                "top_k": 1,
                "top_p": 1.0,
                "reuse_cache": reuse,
                "can_use_pd_separation": True,
                "return_input_ids": True,
                "return_output_ids": True,
                "aux_info": True,
            },
        }
        with requests.Session() as session:
            session.trust_env = False
            if barrier is not None:
                barrier.wait(timeout=30)
            response = session.post(
                f"http://127.0.0.1:{server.port}/", json=query, timeout=600
            )
        (self.output_dir / f"{label}.http.txt").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        self.assertNotIn("error_code", actual, label)
        self.assertIs(actual.get("finished"), True, label)
        self.assertTrue(actual["aux_info"]["pd_sep"], label)
        self.assertEqual(len(actual["output_ids"][0]), 8, label)
        self.assertEqual(
            len(actual["input_ids"][0]), actual["aux_info"]["input_len"], label
        )
        return actual

    def check_requests(self, server):
        prefix = "Read the archive and answer the final instruction.\n"
        while len(self.tokenizer.encode(prefix)) < self.chunk_size:
            prefix += (
                "The archive contains books, maps, letters, and photographs.\n" * 16
            )
        prompts = []
        prefix_tokens = len(self.tokenizer.encode(prefix))
        for index, tail_tokens in enumerate((128, 1536, 2560, 3584)):
            tail = f"Request {index}:\n"
            while (
                len(self.tokenizer.encode(prefix + tail)) - prefix_tokens < tail_tokens
            ):
                tail += (
                    f"Document {index} describes rivers, mountains, and forests.\n" * 16
                )
            answer = " ".join([f"WORD{index}"] * 8)
            prompts.append(prefix + tail + f"Reply with exactly: {answer}\nAnswer:")

        # No stored golden: obtain serial uncached outputs from this same model.
        references = [
            self.request(server, f"reference_{i}", prompt, False)
            for i, prompt in enumerate(prompts)
        ]
        for reference in references:
            self.assertEqual(reference["aux_info"]["prefill_total_reuse_len"], 0)
        seed = self.request(server, "prefix_seed", prefix + "Reply READY.\n", True)
        self.assertEqual(seed["aux_info"]["prefill_total_reuse_len"], 0)

        log_path = Path(server.log_file_path)
        log_offset = log_path.stat().st_size
        barrier = Barrier(len(prompts))
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = [
                executor.submit(
                    self.request, server, f"batch_{i}", prompt, True, barrier
                )
                for i, prompt in enumerate(prompts)
            ]
            actuals = [future.result() for future in futures]
        for i, (reference, actual) in enumerate(zip(references, actuals)):
            self.assertEqual(actual["input_ids"], reference["input_ids"], f"row {i}")
            self.assertEqual(actual["output_ids"], reference["output_ids"], f"row {i}")
            self.assertIn(f"WORD{i}", actual["response"], f"row {i}")
            aux = actual["aux_info"]
            reused = aux["prefill_total_reuse_len"]
            self.assertGreaterEqual(reused, 256, f"row {i}: prefix must really hit")
            self.assertEqual(reused % 256, 0)
            if i > 0:
                self.assertGreater(aux["input_len"] - reused, self.chunk_size)

        # Concurrency alone does not prove batching or chunking. Inspect actual
        # pre-CP model-input shapes, emitted once per forward by enable_detail_log.
        with log_path.open() as stream:
            stream.seek(log_offset)
            log = stream.read()
        forwards = re.findall(
            r"model_input: GptModelInputs[^\n]*?combo_tokens: "
            r"Tensor\([^)]*?shape=\[(\d+)\]"
            r"[^\n]*?input_lengths: Tensor\([^)]*?shape=\[(\d+)\]",
            log,
        )
        shapes = [(int(tokens), int(rows)) for tokens, rows in forwards]
        (self.output_dir / "batch_shapes.json").write_text(json.dumps(shapes))
        self.assertGreater(len(shapes), len(prompts), shapes)
        self.assertTrue(any(rows > 1 for _, rows in shapes), shapes)
        self.assertTrue(
            all(0 < tokens <= self.chunk_size for tokens, _ in shapes), shapes
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
