"""Real-model chunked logprobs and fused/PD batch behavior."""

import argparse
import copy
import json
import math
import os
import re
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import requests
import torch

from rtp_llm.config.exceptions import ExceptionType
from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.test.utils.port_util import PortManager

FIXTURE = Path(__file__).parent / "data/model/qwen25/q_r_l20_remote_cache_pd_sep.json"
CHUNK_SIZE = 64
OUTPUT_TOKENS = 8


SERVER_ARGS = (
    "--tp_size 1 --dp_size 1 --world_size 1 --act_type FP16 "
    "--warm_up 0 --enable_cuda_graph 0 --max_seq_len 2048 "
    "--seq_size_per_block 64 --reuse_cache 1 --kv_cache_mem_mb 512 "
    "--cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 60000 "
    "--concurrency_limit 4 --max_context_batch_size 1 --enable_detail_log 1"
)


class ChunkedLogprobsTest(unittest.TestCase):
    topology = "fused"

    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text())
        self.model_type = self.fixture["model_type"]
        self.model_path = os.environ.get(
            "CHUNK_LOGPROBS_CHECKPOINT", self.fixture["model_path"]
        )
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.port_locks = ExitStack()
        self.addCleanup(self.port_locks.close)
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)

    def free_port(self):
        ports, locks = PortManager().get_consecutive_ports(200)
        for lock in locks:
            self.port_locks.callback(lock.__exit__, None, None, None)
        return str(ports[0] + 100)

    def start_servers(self, chunk_size, stack):
        devices = get_gpu_ids()
        self.assertGreaterEqual(len(devices), 2 if self.topology == "pd" else 1)
        args = SERVER_ARGS
        env = {
            "LOAD_PYTHON_MODEL": "1",
            "FRONTEND_SERVER_COUNT": "1",
            "WORLD_SIZE": "1",
            "LOG_LEVEL": "DEBUG",
            "FT_SERVER_TEST": "1",
        }
        if self.topology == "fused":
            manager = MagaServerManager(
                env_args=env,
                device_ids=devices[:1],
                port=self.free_port(),
                role_name=f"fused_chunk{chunk_size}",
                smoke_args_str=f"{args} --role_type PDFUSION --prefill_chunk_size {chunk_size}",
            )
            stack.callback(manager.stop_server)
            self.assertTrue(
                manager.start_server(
                    model_type=self.model_type, model_path=self.model_path, timeout=600
                )
            )
            self.prefill_log = Path(manager.log_file_path)
            return f"http://127.0.0.1:{manager.port}", [manager]

        prefill_port, decode_port = self.free_port(), self.free_port()
        route = ServiceRoute(
            service_id="chunk_logprobs",
            use_local=True,
            role_endpoints=[
                GroupEndPoint(
                    group="default",
                    prefill_endpoint=EndPoint(
                        type="Vipserver",
                        address=f"127.0.0.1:{prefill_port}",
                        protocol="http",
                        path="/",
                    ),
                    decode_endpoint=EndPoint(
                        type="Vipserver",
                        address=f"127.0.0.1:{decode_port}",
                        protocol="http",
                        path="/",
                    ),
                )
            ],
        )
        managers = []
        for role, port, peer, gpu in (
            ("PREFILL", prefill_port, decode_port, devices[0]),
            ("DECODE", decode_port, prefill_port, devices[1]),
        ):
            manager = MagaServerManager(
                env_args={
                    **env,
                    "REMOTE_SERVER_PORT": str(peer),
                    "REMOTE_RPC_SERVER_IP": "127.0.0.1",
                    "MODEL_SERVICE_CONFIG": route.model_dump_json(),
                },
                device_ids=[gpu],
                port=port,
                role_name=f"{role.lower()}_chunk{chunk_size}",
                smoke_args_str=f"{args} --role_type {role} --prefill_chunk_size {chunk_size if role == 'PREFILL' else 0}",
            )
            stack.callback(manager.stop_server)
            managers.append(manager)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    m.start_server,
                    model_type=self.model_type,
                    model_path=self.model_path,
                    timeout=600,
                )
                for m in managers
            ]
            for future in futures:
                self.assertTrue(future.result(), "P/D role failed to start")
        self.prefill_log = Path(managers[0].log_file_path)
        return f"http://127.0.0.1:{prefill_port}", managers

    def make_query(
        self, label, streaming, *, output_tokens=OUTPUT_TOKENS, reuse_cache=True
    ):
        # Vary the early prompt prefix across cases to prevent a previous case
        # from satisfying this one entirely through prefix-cache reuse.
        prompt = f"Case {label}. " + "The river flows through the quiet valley. " * 48
        prompt += "Continue this story with one short sentence."
        query = copy.deepcopy(self.fixture["query_result"][0]["query"])
        query.update(
            {
                "model": "qwen_2",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": output_tokens,
                "seed": 123,
                "temperature": 1.0,
                "top_p": 1.0,
                "stream": streaming,
                "logprobs": True,
                "top_logprobs": 5,
                "logprobs_mode": "original",
                "extra_configs": {
                    # Keep generated prefixes identical for probability comparisons.
                    "top_k": 1,
                    "do_sample": True,
                    "min_new_tokens": output_tokens,
                    "reuse_cache": reuse_cache,
                    "aux_info": True,
                },
            }
        )
        return query

    def request(self, url, label, query):
        streaming = query["stream"]
        log_offset = self.prefill_log.stat().st_size
        response = self.session.post(
            url + "/v1/chat/completions", json=query, stream=streaming, timeout=180
        )
        if not response.ok:
            self.save(
                f"chunk{self.chunk_size}_{label}.error.json",
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
                    continue
                events.append(json.loads(payload))
            self.save(f"chunk{self.chunk_size}_{label}.sse.json", raw)
            self.assertTrue(events, "missing OpenAI SSE events")
            self.assertTrue(
                done
                or any(
                    c.get("finish_reason") for e in events for c in e.get("choices", [])
                ),
                "stream ended without a terminal completion",
            )
        else:
            events = [response.json()]
        response.close()
        self.save(
            f"chunk{self.chunk_size}_{label}.http.json",
            {"query": query, "events": events},
        )
        actual = self.check_output(events, query["max_tokens"])
        self.check_chunk_progress(label, log_offset)
        return actual

    def check_output(self, events, output_tokens):
        rows, text, usages, aux = [], [], [], []
        for event in events:
            self.assertNotIn("error", event, event)
            if event.get("usage"):
                usages.append(event["usage"])
            if event.get("aux_info"):
                aux.append(event["aux_info"])
            for choice in event.get("choices", []):
                self.assertEqual(choice["index"], 0)
                rows.extend((choice.get("logprobs") or {}).get("content") or [])
                text.append(
                    (choice.get("delta") or choice.get("message") or {}).get("content")
                    or ""
                )
        self.assertEqual(
            len(rows),
            output_tokens,
            "missing/duplicate probability rows, including first token",
        )
        self.assertTrue(usages, "missing token usage")
        usage = usages[-1]
        self.assertEqual(usage["completion_tokens"], len(rows))
        self.assertGreater(usage["prompt_tokens"], 2 * CHUNK_SIZE)
        for row in rows:
            self.assertTrue(math.isfinite(row["logprob"]))
            self.assertLessEqual(row["logprob"], 1e-6)
            self.assertEqual(len(row["top_logprobs"]), 5)
            for candidate in row["top_logprobs"]:
                self.assertTrue(math.isfinite(candidate["logprob"]))
                self.assertLessEqual(candidate["logprob"], 1e-6)
        self.assertEqual("".join(text), "".join(row["token"] for row in rows))
        self.assertTrue(aux, "missing execution metadata")
        if self.topology == "pd":
            self.assertTrue(aux[-1]["pd_sep"], "request fell back to fused generation")
        reuse = (
            aux[-1].get("prefill_total_reuse_len", 0)
            if self.topology == "pd"
            else aux[-1].get("reuse_len", 0)
        )
        self.assertLess(
            reuse,
            usage["prompt_tokens"] - 2 * CHUNK_SIZE,
            "cache reuse bypassed chunk execution",
        )
        return {"rows": rows, "text": "".join(text), "usage": usage, "aux": aux[-1]}

    def check_chunk_progress(self, label, log_offset):
        if not self.chunk_size:
            return
        with self.prefill_log.open("rb") as log:
            log.seek(log_offset)
            new_log = log.read().decode(errors="replace")
        chunk_rows = re.findall(
            r"context stream:.*reuse_length:(?:64|128|192|256).*", new_log
        )
        self.save(f"chunk{self.chunk_size}_{label}.chunk_evidence.json", chunk_rows)
        self.assertTrue(
            chunk_rows, "this request has no observed intermediate chunk prefix"
        )

    def save(self, name, value):
        (self.output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2)
        )

    def compare(self, reference, actual):
        self.assertEqual(actual["text"], reference["text"])
        self.assertEqual(actual["usage"], reference["usage"])
        expected_scores, observed_scores = [], []
        for expected, observed in zip(reference["rows"], actual["rows"]):
            self.assertEqual(observed["bytes"], expected["bytes"])
            expected_scores.append(expected["logprob"])
            observed_scores.append(observed["logprob"])
            # Preserve duplicate decoded byte strings while ignoring candidate order.
            key = lambda c: (tuple(c["bytes"]), c["token"], c["logprob"])
            expected_top = sorted(expected["top_logprobs"], key=key)
            observed_top = sorted(observed["top_logprobs"], key=key)
            self.assertEqual(
                [(c["bytes"], c["token"]) for c in expected_top],
                [(c["bytes"], c["token"]) for c in observed_top],
            )
            expected_scores.extend(c["logprob"] for c in expected_top)
            observed_scores.extend(c["logprob"] for c in observed_top)
        # Reuse OpenaiComparer's existing FP16 logprob tolerance, including its
        # absolute bound near zero. Chunked attention changes accumulation order.
        torch.testing.assert_close(
            torch.tensor(observed_scores),
            torch.tensor(expected_scores),
            rtol=1e-2,
            atol=1e-2,
        )

    def check_batch(self, url):
        queries = [
            self.make_query(
                f"batch-{output_tokens}",
                streaming=False,
                output_tokens=output_tokens,
                reuse_cache=False,
            )
            for output_tokens in (OUTPUT_TOKENS, 6)
        ]
        references = [
            self.request(url, f"batch-reference-{index}", query)
            for index, query in enumerate(queries)
        ]
        query = {"requests": queries}
        log_offset = self.prefill_log.stat().st_size
        with self.session.post(
            url + "/v1/batch/chat/completions", json=query, timeout=180
        ) as response:
            if not response.ok:
                self.save(
                    f"chunk{self.chunk_size}_batch.error.json",
                    {"status": response.status_code, "body": response.text},
                )
            response.raise_for_status()
            batch = response.json()
        self.save(
            f"chunk{self.chunk_size}_batch.http.json",
            {"query": query, "response": batch},
        )
        self.assertEqual(len(batch["responses"]), len(queries))
        if self.topology == "pd":
            for index, item in enumerate(batch["responses"]):
                with self.subTest(batch_item=index):
                    self.assertIn("error", item, item)
                    error = item["error"]
                    self.assertEqual(error["error_code"], ExceptionType.INVALID_PARAMS)
                    self.assertIn(
                        "PD batch does not support compact logprobs", error["message"]
                    )
            return
        for item, reference, item_query in zip(batch["responses"], references, queries):
            output_tokens = item_query["max_tokens"]
            actual = self.check_output([item], output_tokens)
            # The native batch response exposes the whole terminal output step.
            self.assertEqual(actual["aux"]["step_output_len"], output_tokens)
            self.compare(reference, actual)
        self.check_chunk_progress("batch", log_offset)

    def test_chunk_on_off_probabilities(self):
        references = {}
        # Exercise the PD logprobs rejection without grouped Prefill chunking.
        batch_chunk_size = 0 if self.topology == "pd" else CHUNK_SIZE
        for chunk_size in (0, CHUNK_SIZE):
            self.chunk_size = chunk_size
            with ExitStack() as stack:
                url, managers = self.start_servers(chunk_size, stack)
                for streaming in (False, True):
                    case = "stream" if streaming else "nonstream"
                    actual = self.request(url, case, self.make_query(case, streaming))
                    self.save(f"chunk{chunk_size}_{case}.json", actual)
                    if chunk_size == 0:
                        references[case] = actual
                    else:
                        self.compare(references[case], actual)
                if chunk_size == batch_chunk_size:
                    self.check_batch(url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=("fused", "pd"), default="fused")
    args, rest = parser.parse_known_args()
    ChunkedLogprobsTest.topology = args.topology
    unittest.main(argv=[__file__] + rest)
