"""Exercise production eager transport with CPU tensors and mocked collectives."""

from __future__ import annotations

import ast
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "rtp_llm/models_py/modules/dsv4/fp8/attention.py").is_file()
)
SOURCE = (
    ROOT
    / "rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/sm120_decode.py"
)


def load_executor():
    names = {
        "_decode_chunk_tokens",
        "_return_mode",
        "_a2a_payload_bound_exceeded",
        "_validate_a2a_token_counts",
        "Sm120DecodeExecutor",
    }
    selected = [
        node
        for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future] + selected, type_ignores=[])
    )
    namespace = {
        "torch": torch,
        "os": os,
        "_logger": logging.getLogger(__name__),
        "Fp8Fp4ExecutorBase": object,
        "_A2A_PAYLOAD_TOKEN_BOUND": 65536,
        "_DECODE_CHUNK_TOKENS": 4096,
        "is_sm120": lambda device: True,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


class EagerDecodeContractTest(unittest.TestCase):
    def setUp(self):
        self.namespace = load_executor()
        self.executor_type = self.namespace["Sm120DecodeExecutor"]

    def run_transport(self, counts, rank):
        world, hidden, topk = len(counts), 128, 2
        rows = max(counts[rank], 0)
        executor = self.executor_type()
        executor.cfg = types.SimpleNamespace(n_routed_experts=world * 2)
        executor._compute_received = Mock(
            side_effect=lambda payload, total, *args: torch.ones(total, hidden)
        )
        events = []
        quantize = Mock(
            side_effect=lambda x, **kw: (
                x.to(torch.float8_e4m3fn),
                torch.ones(x.shape[0], hidden // 32, dtype=torch.uint8),
            )
        )

        def gather(output, local_count, **kwargs):
            events.append("counts")
            self.assertEqual(local_count.item(), rows)
            output.copy_(torch.tensor(counts).view(world, 1))

        def exchange(output, payload, *, output_split_sizes, input_split_sizes, **kw):
            call = events.count("exchange")
            events.append("exchange")
            self.assertEqual(
                output_split_sizes, counts if call == 0 else [rows] * world
            )
            self.assertEqual(input_split_sizes, [rows] * world if call == 0 else counts)
            output.fill_(1)

        def peer_sum(payload, peers, output):
            events.append("sum")
            output.copy_(payload.view(peers, rows, hidden).sum(dim=0))
            return output

        self.namespace["fp32_peer_sum"] = peer_sum
        dist = types.SimpleNamespace(
            group=types.SimpleNamespace(WORLD=object()),
            get_world_size=lambda group: world,
            get_rank=lambda group: rank,
            all_gather_into_tensor=gather,
            all_to_all_single=exchange,
        )
        with patch.object(torch, "distributed", dist), patch.dict(
            sys.modules, {"flashinfer": types.SimpleNamespace(mxfp8_quantize=quantize)}
        ), patch.dict(os.environ, {"DSV4_MOE_RETURN_DTYPE": "bf16"}):
            result = executor._forward_all_to_all(
                torch.ones(rows, hidden, dtype=torch.bfloat16),
                torch.ones(rows, topk),
                torch.zeros(rows, topk, dtype=torch.int64),
            )
        return result, events, quantize, executor._compute_received

    def test_uneven_and_idle_peers_follow_identical_collective_sequence(self):
        counts = [0, 3, 1, 2, 0, 4, 1, 0]
        for rank, rows in enumerate(counts):
            with self.subTest(rank=rank):
                result, events, quantize, compute = self.run_transport(counts, rank)
                self.assertEqual(
                    [e for e in events if e != "sum"],
                    ["counts", "exchange", "exchange"],
                )
                self.assertEqual(result.shape, (rows, 128))
                self.assertEqual(result.dtype, torch.float32)
                torch.testing.assert_close(result, torch.full_like(result, 8))
                self.assertEqual(quantize.call_count, int(rows > 0))
                compute.assert_called_once()

    def test_all_empty_peers_skip_payload_and_kernel_work(self):
        for rank in range(8):
            result, events, quantize, compute = self.run_transport([0] * 8, rank)
            self.assertEqual(result.shape, (0, 128))
            self.assertEqual(events, ["counts"])
            quantize.assert_not_called()
            compute.assert_not_called()

    def test_oversize_aggregate_is_rejected_consistently(self):
        for rank in range(8):
            with self.subTest(rank=rank), self.assertRaisesRegex(
                RuntimeError, "aggregate EP tokens"
            ):
                self.run_transport([8193] * 8, rank)

    def test_invalid_peer_metadata_is_rejected(self):
        validate = self.namespace["_validate_a2a_token_counts"]
        for counts in ([1, -1], [1, 2**40], [1], [2, 1]):
            with self.subTest(counts=counts), self.assertRaisesRegex(
                RuntimeError, "invalid token counts"
            ):
                validate(counts, world=2, rank=0, local_tokens=1, max_tokens=65536)

    def test_graph_and_replicated_tp_fail_before_transport(self):
        executor = self.executor_type()
        executor._validate_group = Mock()
        executor._forward_all_to_all = Mock()
        for capturing, tp in ((True, 1), (False, 8)):
            executor.cfg = types.SimpleNamespace(attn_tp_size=tp)
            with self.subTest(capturing=capturing, tp=tp), patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=capturing
            ), self.assertRaises(RuntimeError):
                executor.forward(torch.ones(1, 128), None, None)
        executor._forward_all_to_all.assert_not_called()


if __name__ == "__main__":
    unittest.main()
