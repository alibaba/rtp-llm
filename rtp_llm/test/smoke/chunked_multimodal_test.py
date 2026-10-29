"""Compare real Qwen3-VL inference with images split across prefill chunks."""

import base64
import json
import math
import os
import re
import unittest
from contextlib import ExitStack
from pathlib import Path

import requests

from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.test.utils.port_util import PortManager

CHUNK_SIZE = 256
OUTPUT_TOKENS = 8
DATA_DIR = Path(__file__).parent / "data/model/qwen_vl"
SERVER_ARGS = (
    "--tp_size 1 --dp_size 1 --world_size 1 "
    "--role_type PDFUSION --warm_up 0 --enable_cuda_graph 0 "
    "--max_seq_len 8192 --seq_size_per_block 64 --reuse_cache 1 "
    "--kv_cache_mem_mb 2048 --use_local 1 --use_local_preprocess 1 "
    "--concurrency_limit 1 --max_context_batch_size 1 --enable_detail_log 1"
)


class ChunkedMultimodalTest(unittest.TestCase):
    act_type = "BF16"
    single_image_only = False

    def setUp(self):
        fixture = json.loads((DATA_DIR / "q_r_3.json").read_text())
        self.model_path = os.environ.get("VL_SMOKE_CHECKPOINT", fixture["model_path"])
        config = json.loads((Path(self.model_path) / "config.json").read_text())
        self.assertEqual(config["model_type"], "qwen3_vl")
        self.assertTrue(config["text_config"]["rope_scaling"]["mrope_section"])
        self.assertTrue(config["vision_config"]["deepstack_visual_indexes"])
        self.output_dir = (
            Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / self.act_type.lower()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.port_locks = ExitStack()
        self.addCleanup(self.port_locks.close)
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        encoded_image = base64.b64encode((DATA_DIR / "1.jpeg").read_bytes()).decode()
        self.image = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + encoded_image},
        }
        self.save(
            "model_features.json",
            {
                "checkpoint": self.model_path,
                "act_type": self.act_type,
                "mrope": config["text_config"]["rope_scaling"],
                "deepstack_visual_indexes": config["vision_config"][
                    "deepstack_visual_indexes"
                ],
            },
        )

    def save(self, name, value):
        (self.output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2)
        )

    def start_server(self, chunk_size, stack):
        devices = get_gpu_ids()
        self.assertTrue(devices, "the smoke test needs one reserved GPU")
        ports, locks = PortManager(
            lock_dir=self.output_dir / "port_locks"
        ).get_consecutive_ports(200)
        for lock in locks:
            self.port_locks.callback(lock.__exit__, None, None, None)
        manager = MagaServerManager(
            env_args={
                "LOAD_PYTHON_MODEL": "1",
                "FRONTEND_SERVER_COUNT": "1",
                "WORLD_SIZE": "1",
                "LOG_LEVEL": "DEBUG",
                "FT_SERVER_TEST": "1",
            },
            device_ids=devices[:1],
            port=str(ports[0] + 100),
            role_name=f"multimodal_{self.act_type.lower()}_chunk{chunk_size}",
            smoke_args_str=SERVER_ARGS
            + f" --act_type {self.act_type} --prefill_chunk_size {chunk_size}",
        )
        stack.callback(manager.stop_server)
        self.assertTrue(
            manager.start_server(
                model_type="qwen3_vl", model_path=self.model_path, timeout=600
            ),
            "Qwen3-VL server failed to start",
        )
        self.server_log = Path(manager.log_file_path)
        return f"http://127.0.0.1:{manager.port}"

    def request(self, url, case, content, streaming, image_count):
        # Chunked prefill needs reuse_cache enabled. Distinct early prefixes keep
        # earlier cases from supplying this request's KV cache; reuse_len is checked below.
        content = [{"type": "text", "text": f"Fixture {case}. "}] + content
        query = {
            "model": "qwen3_vl",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": OUTPUT_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": streaming,
            "logprobs": True,
            "top_logprobs": 5,
            "logprobs_mode": "original",
            "extra_configs": {
                "return_all_probs": 2,
                "do_sample": False,
                "min_new_tokens": OUTPUT_TOKENS,
                "reuse_cache": True,
                "aux_info": True,
                "add_vision_id": False,
            },
        }
        if streaming:
            query["stream_options"] = {"include_usage": True}
        log_offset = self.server_log.stat().st_size
        artifact_prefix = f"chunk{self.chunk_size}_{case}"
        with self.session.post(
            url + "/v1/chat/completions", json=query, stream=streaming, timeout=180
        ) as response:
            if not response.ok:
                self.save(
                    artifact_prefix + ".error.json",
                    {"status": response.status_code, "body": response.text},
                )
            response.raise_for_status()
            if streaming:
                events, raw, done = [], [], False
                for line in response.iter_lines(decode_unicode=True):
                    raw.append(line)
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload.lower() == "[done]":
                        done = True
                    else:
                        events.append(json.loads(payload))
                self.save(artifact_prefix + ".sse.json", raw)
                self.assertTrue(events, "missing OpenAI SSE events")
                # The current OpenAI endpoint closes after finish_reason; the
                # native endpoint additionally emits a [done] sentinel.
                self.assertTrue(
                    done
                    or any(
                        choice.get("finish_reason")
                        for event in events
                        for choice in event.get("choices", [])
                    ),
                    "stream ended without a terminal completion",
                )
            else:
                events = [response.json()]
        self.save(artifact_prefix + ".http.json", events)
        rows, text, usages, metadata = [], [], [], []
        for event in events:
            self.assertNotIn("error", event, event)
            if event.get("usage"):
                usages.append(event["usage"])
            if event.get("aux_info"):
                metadata.append(event["aux_info"])
            for choice in event.get("choices", []):
                self.assertEqual(choice["index"], 0)
                rows.extend((choice.get("logprobs") or {}).get("content") or [])
                text.append(
                    (choice.get("delta") or choice.get("message") or {}).get("content")
                    or ""
                )
        self.assertEqual(len(rows), OUTPUT_TOKENS)
        self.assertTrue(usages, "missing token usage")
        usage = usages[-1]
        self.assertEqual(usage["completion_tokens"], OUTPUT_TOKENS)
        image_tokens = usage.get("prompt_tokens_details", {}).get("image_tokens", 0)
        # Each image is longer than two windows, so an intermediate window must
        # start inside its visual tokens regardless of the text-prefix length.
        self.assertGreater(image_tokens, image_count * 2 * CHUNK_SIZE)
        self.assertGreater(usage["prompt_tokens"], image_tokens)
        self.assertEqual("".join(text), "".join(row["token"] for row in rows))
        for row in rows:
            self.assertTrue(math.isfinite(row["logprob"]))
            self.assertLessEqual(row["logprob"], 1e-6)
            self.assertEqual(len(row["top_logprobs"]), 5)
            for candidate in row["top_logprobs"]:
                self.assertTrue(math.isfinite(candidate["logprob"]))
                self.assertLessEqual(candidate["logprob"], 1e-6)
        self.assertTrue(metadata, "missing execution metadata")
        self.assertEqual(metadata[-1]["reuse_len"], 0)
        with self.server_log.open("rb") as log:
            log.seek(log_offset)
            request_log = log.read().decode(errors="replace")
        context_rows = re.findall(r"context stream:[^\n]*", request_log)
        prefixes = sorted(
            {
                int(match.group(1))
                for row in context_rows
                if (match := re.search(r"reuse_length:\s*(\d+)", row))
            }
        )
        self.save(
            artifact_prefix + ".chunk_evidence.json",
            {"context_rows": context_rows, "prefixes": prefixes},
        )
        if self.chunk_size:
            self.assertIn(CHUNK_SIZE, prefixes, "first intermediate chunk missing")
            self.assertIn(2 * CHUNK_SIZE, prefixes, "second intermediate chunk missing")
        else:
            self.assertEqual(prefixes, [0], "reference unexpectedly used chunking")
        return {"rows": rows, "text": "".join(text), "usage": usage}

    def compare(self, reference, actual):
        self.assertEqual(actual["usage"], reference["usage"])
        self.assertEqual(actual["text"], reference["text"])
        for step, (expected, observed) in enumerate(
            zip(reference["rows"], actual["rows"])
        ):
            self.assertEqual(observed["bytes"], expected["bytes"], f"token {step}")
            # BF16 covers image chunking and exact greedy output. FP16 also
            # checks scores; BF16 scores are more sensitive to the prefill shape.
            if self.act_type == "FP16":
                self.assertAlmostEqual(
                    observed["logprob"],
                    expected["logprob"],
                    delta=0.03,
                    msg=f"sampled token {step}",
                )
                expected_top = {
                    tuple(candidate["bytes"]): candidate["logprob"]
                    for candidate in expected["top_logprobs"]
                }
                observed_top = {
                    tuple(candidate["bytes"]): candidate["logprob"]
                    for candidate in observed["top_logprobs"]
                }
                # Candidates near the fifth-place boundary can swap after rounding.
                self.assertGreaterEqual(
                    len(expected_top.keys() & observed_top.keys()), 4
                )
                for token in expected_top.keys() & observed_top.keys():
                    # A log-space delta amplifies tiny tail-probability changes
                    # between FP16 prefill shapes and norm implementations.
                    self.assertTrue(
                        math.isclose(
                            math.exp(observed_top[token]),
                            math.exp(expected_top[token]),
                            rel_tol=0.03,
                            abs_tol=1e-4,
                        ),
                        msg=f"candidate at token {step}",
                    )

    def test_image_crosses_chunks_with_mrope_and_deepstack(self):
        describe = {"type": "text", "text": "Describe this image briefly."}
        cases = [
            ("one_image", [self.image, describe], False, 1),
            (
                "text_then_image_stream",
                [
                    {"type": "text", "text": "Look carefully. " * 40},
                    self.image,
                    describe,
                ],
                True,
                1,
            ),
            (
                "two_images",
                [
                    self.image,
                    {"type": "text", "text": "Here is the same image again."},
                    self.image,
                    {"type": "text", "text": "Describe what both images show."},
                ],
                False,
                2,
            ),
        ]
        if self.single_image_only:
            cases = cases[:1]
        references = {}
        for chunk_size in (0, CHUNK_SIZE):
            self.chunk_size = chunk_size
            with ExitStack() as stack:
                url = self.start_server(chunk_size, stack)
                for case, content, streaming, image_count in cases:
                    with self.subTest(chunk_size=chunk_size, case=case):
                        actual = self.request(
                            url, case, content, streaming, image_count
                        )
                        self.save(f"chunk{chunk_size}_{case}.json", actual)
                        if chunk_size == 0:
                            references[case] = actual
                        else:
                            self.compare(references[case], actual)


class ChunkedMultimodalFP16Test(ChunkedMultimodalTest):
    act_type = "FP16"
    single_image_only = True


if __name__ == "__main__":
    unittest.main()
