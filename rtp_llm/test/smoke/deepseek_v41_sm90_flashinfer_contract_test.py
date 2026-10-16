"""Opt-in Hopper FlashInfer DP8 contract using the unchanged SM100 golden.

Exercises PDFUSION generation, Engram UVA, prefix reuse and decode graphs.
CED bounded decode, PD and DSpark are not covered by this contract.
"""

import argparse
import json
import logging
import shlex
import shutil
import unittest
from pathlib import Path
from urllib.parse import urlsplit

import requests

from rtp_llm.test.smoke.deepseek_v41_contract_test import DeepSeekV41ContractTest
from rtp_llm.test.smoke.deepseek_v41_sm90_contract_test import (
    DeepSeekV41Sm90ContractTest,
)


class _BackendRoutedSession(requests.Session):
    """Pin generation requests while leaving management endpoints unchanged."""

    def __init__(self, role_addr):
        super().__init__()
        self.trust_env = False
        self.role_addr = role_addr

    def request(self, method, url, **kwargs):
        query = kwargs.get("json")
        if (
            method.upper() == "POST"
            and urlsplit(url).path in ("", "/")
            and isinstance(query, dict)
            and isinstance(query.get("generate_config"), dict)
        ):
            config = query["generate_config"]
            route = [dict(self.role_addr)]
            if config.get("role_addrs") not in (None, [], route):
                raise ValueError("Generation request conflicts with the pinned backend")
            # Keep the actual route in the inherited request evidence JSON too.
            config["role_addrs"] = route
        return super().request(method, url, **kwargs)


class DeepSeekV41Sm90FlashInferContractTest(DeepSeekV41Sm90ContractTest):
    moe_strategy = "sm90_flashinfer"
    extra_server_env = {
        **DeepSeekV41Sm90ContractTest.extra_server_env,
        "DSV41_SM90_INDEXED_MOE": "0",
        "DSV41_SM90_MULTIQUERY_INDEXER": "1",
        "DSV41_ENGRAM_REGISTER_CHUNK_MB": "256",
        # A transport failure must not silently retry on a different DP rank.
        "RTP_LLM_PD_ROUTE_RETRY_ON_UNAVAILABLE": "0",
    }

    def setUp(self):
        super().setUp()
        response = self.session.get(
            f"http://127.0.0.1:{self.server.port}/worker_status", timeout=30
        )
        response.raise_for_status()
        status = response.json()
        (self.output_dir / "v41_sm90_flashinfer_worker_status.json").write_text(
            json.dumps(status, indent=2)
        )
        self.assertNotIn("error", status)
        self.assertEqual(int(status["dp_size"]), 8)
        workers = status["results"]
        self.assertEqual(len(workers), 8)
        self.assertEqual({int(worker["dp_rank"]) for worker in workers}, set(range(8)))
        addresses = [worker["address"] for worker in workers]
        self.assertEqual(len(set(addresses)), 8)
        for worker in workers:
            self.assertIs(worker["alive"], True)
            host, port = worker["address"].rsplit(":", 1)
            self.assertTrue(host)
            self.assertTrue(0 < int(port) < 65536)
        selected = next(worker for worker in workers if int(worker["dp_rank"]) == 0)
        host, port = selected["address"].rsplit(":", 1)
        self.backend_route = {
            "role": "PDFUSION",
            "ip": host,
            # Generation transport uses the discovered backend grpc_port.
            "http_port": int(self.server.port),
            "grpc_port": int(port),
        }
        self.backend_route_evidence = {
            "dp_rank": 0,
            "address": selected["address"],
            "role_addr": self.backend_route,
        }
        (self.output_dir / "v41_sm90_flashinfer_backend_route.json").write_text(
            json.dumps(self.backend_route_evidence, indent=2)
        )
        self.session = _BackendRoutedSession(self.backend_route)
        self.addCleanup(self.session.close)

    def start_servers(self):
        import torch

        parser = argparse.ArgumentParser(add_help=False)
        for name in ("tp_size", "dp_size", "ep_size", "world_size"):
            parser.add_argument(f"--{name}", type=int, required=True)
        topology, _ = parser.parse_known_args(shlex.split(self.server_args()))
        self.assertEqual(
            vars(topology),
            {"tp_size": 1, "dp_size": 8, "ep_size": 8, "world_size": 8},
        )
        if not torch.version.cuda or int(torch.version.cuda.split(".")[0]) != 13:
            raise RuntimeError("SM90 FlashInfer smoke requires the CUDA13 runtime")
        if torch.cuda.device_count() != 8 or any(
            torch.cuda.get_device_capability(rank) != (9, 0) for rank in range(8)
        ):
            raise RuntimeError("SM90 DP8/EP8 smoke requires eight locked SM90 GPUs")
        # Both Engram host lookup modes use the same physical tmpfs tables.
        raw = json.loads((Path(self.model_path) / "config.json").read_text())
        config = raw.get("text_config", raw)
        rows = sum(config["engram_num_embeddings"])
        dim = int(config["engram_head_dim"])
        required = rows * (dim + dim // 32)
        capacity = shutil.disk_usage("/dev/shm").total
        if capacity < required:
            raise RuntimeError(
                f"Engram requires {required / 1024**3:.2f} GiB shared table storage, "
                f"but /dev/shm capacity is {capacity / 1024**3:.2f} GiB; "
                "increase container shared memory (256 GiB recommended). "
                "No model forward has run."
            )
        # Skip only the parent backend's Humming-specific startup assertion.
        DeepSeekV41ContractTest.start_servers(self)
        startup = Path(self.graph_server.log_file_path).read_text(errors="replace")
        self.assertIn("explicit SM90 opensource FlashInfer CUTLASS W4A8", startup)
        self.assertNotIn("E8M0 decoded for BF16 MMA", startup)
        self.assertIn("SM90 Engram UVA prefetch enabled", startup)
        self.assertNotIn("Engram shared CPU lookup enabled", startup)

    def assert_decode_graph_replayed_and_record_coverage(self):
        DeepSeekV41ContractTest.assert_decode_graph_replayed_and_record_coverage(self)
        (self.output_dir / "v41_sm90_flashinfer_coverage.json").write_text(
            json.dumps(
                {
                    "requests": self.observed,
                    "backend": self.moe_strategy,
                    "backend_route": self.backend_route_evidence,
                    "topology": {"tp_size": 1, "dp_size": 8, "ep_size": 8},
                    "execution": "decode_cuda_graph",
                    "server_args": self.server_args(),
                    "native_a8_requested": True,
                    "engram": "uva_prefetch",
                    "engram_checkpoint_layers": True,
                    "reference": "committed SM100 golden token prefix",
                    "ced_bounded_decode": "NOT_RUN",
                    "dspark_vision_pd": "NOT_RUN",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        DeepSeekV41Sm90FlashInferContractTest
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
