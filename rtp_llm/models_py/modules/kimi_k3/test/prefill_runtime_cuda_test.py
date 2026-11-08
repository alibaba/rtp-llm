"""Real TP8 CP transport and MTP storage contracts, without checkpoint weights."""

import json
import time
import unittest
from datetime import timedelta
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.models_py.distributed import collective_torch as collective
from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3Model
from rtp_llm.models_py.modules.hybrid.natural_zigzag_plan import NaturalZigzagPlan
from rtp_llm.models_py.modules.kimi_k3.dsa_model_adapter import KimiK31ModelAdapter
from rtp_llm.ops import RoleType
from rtp_llm.test.utils.port_util import PortManager


class CapacityModel(KimiK31ModelAdapter, torch.nn.Module):
    def __init__(self, chunk, rank, size):
        super().__init__()
        self.tp_size, self.tp_rank = size, rank
        self._max_moe_tokens = chunk * 2 * size
        self.prefill_query_cp = self._outer_natural_enabled = True
        self._row_exchange = None
        self.layers = []
        self.config = NS(
            hidden_size=128,
            max_seq_len=1048576,
            moe_prefill_max_tokens_per_rank=2097152,
        )
        self.norm = torch.nn.LayerNorm(128, dtype=torch.bfloat16, device="cuda")
        self.dsa_workspace = NS()
        self.register_buffer("_mtp_prefill_hidden_buffer", None, persistent=False)
        self._mtp_prefill_hidden_valid_rows = 0


def values(indices, hidden_size, epoch):
    columns = torch.arange(hidden_size, device=indices.device)
    return ((indices[:, None] + columns + epoch) % 127).bfloat16()


def zigzag_indices(lengths, rank, size, device):
    # Independent oracle: enumerate front/back intervals, without reading
    # NaturalZigzagPlan's transport segments or the exchanged output.
    indices, offset = [], 0
    for length in lengths:
        padded = length
        while padded % (2 * size):
            padded += 1
        pair = padded // (2 * size)
        indices.extend(range(offset + rank * pair, offset + (rank + 1) * pair))
        indices.extend(
            range(offset + padded - (rank + 1) * pair, offset + padded - rank * pair)
        )
        offset += padded
    return torch.tensor(indices, device=device, dtype=torch.int64)


def check_fp8_projection_staging(checks, device, size):
    # Build the actual FP8 collective at the TP8/64K serving capacity.
    model = KimiK3Model.__new__(KimiK3Model)
    torch.nn.Module.__init__(model)
    model.tp_size = size
    model._fp8_collective = None
    model._max_moe_tokens = 1048576
    model._projection_staging_tokens = 65552
    model.parallelism_config = NS(role_type=RoleType.PREFILL)
    model.config = NS(
        hidden_size=7168,
        attention_projection_quant_config=NS(get_method=lambda: "FP8_PER_BLOCK"),
        k3_runtime_config=NS(attn_res_block_size=0),
    )
    projection = NS(scale_ue8m0=True, weight=NS(device=device))
    model.layers = [NS(attention=NS(input=projection, output=projection), block_size=0)]
    with patch(
        "rtp_llm.models_py.model_desc.kimi_k3._get_group", return_value=dist.group.WORLD
    ):
        model._initialize_fp8_collective()
    fp8 = model._fp8_collective
    checks.assertEqual(fp8.max_m, 65552)
    checks.assertTrue(fp8.can_run_ag(8192))
    checks.assertTrue(fp8.can_run_rs(65536))
    checks.assertFalse(fp8.can_run_ag(8195))
    checks.assertFalse(fp8.can_run_rs(65560))


def worker(rank, size, port):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=size,
        device_id=device,
        timeout=timedelta(seconds=300),
    )
    checks = unittest.TestCase()
    retained = []
    try:
        # Bind RTP's TP lookup to the real NCCL group; all allocations, peer
        # mappings, barriers, copies and asynchronous events remain production.
        with patch.object(
            collective, "get_process_group", return_value=dist.group.WORLD
        ):
            for chunk in (512, 8192, 65552):
                model = CapacityModel(chunk, rank, size)
                retained.append(model)
                resource = NS(is_speculative=True, max_context_batch_size=2)
                model._initialize_mtp_hidden_handoff(resource)
                model._initialize_natural_outer(resource)
                exchange = model._row_exchange
                capacity = model._max_moe_tokens // size
                checks.assertEqual(exchange.source.shape, (capacity, 128))
                checks.assertEqual(exchange.chunk_capacity, min(capacity, 8192))
                checks.assertEqual(
                    model._mtp_prefill_hidden_buffer.shape, (capacity, 128)
                )
                checks.assertGreaterEqual(
                    model.dsa_workspace.route_bytes, exchange.resident_bytes
                )
                pointers = (
                    exchange.source.data_ptr(),
                    model._mtp_prefill_hidden_buffer.data_ptr(),
                )
                shapes = [(1,), (17, 31, 3), (chunk,)]
                if chunk == 512:
                    # The maximum expanded one-token return-sequence layout
                    # fills the physical capacity, beyond the warmup batch.
                    shapes.append((1,) * chunk)
                for epoch, lengths in enumerate(shapes, 1):
                    plan = NaturalZigzagPlan(lengths, size, rank)
                    exchange.prepare_plan(
                        plan, chunk_rows=model.dsa_workspace.route_chunk_rows
                    )
                    local = torch.arange(
                        rank * plan.local_rows,
                        (rank + 1) * plan.local_rows,
                        device=device,
                    )
                    natural = values(local, 128, epoch)
                    expected = values(
                        zigzag_indices(lengths, rank, size, device), 128, epoch
                    )
                    routed = exchange.exchange(natural, plan, "to_zigzag")
                    torch.testing.assert_close(routed, expected, rtol=0, atol=0)
                    restored = exchange.exchange(routed, plan, "to_natural")
                    torch.testing.assert_close(restored, natural, rtol=0, atol=0)
                    consumed = torch.empty_like(routed)

                    def receive(start, block):
                        consumed[start : start + block.shape[0]].copy_(block)

                    exchange.consume_chunks(
                        natural,
                        plan,
                        "to_zigzag",
                        chunk_rows=model.dsa_workspace.route_chunk_rows,
                        consumer=receive,
                    )
                    torch.testing.assert_close(consumed, expected, rtol=0, atol=0)
                    # Publishing strided input must preserve independent BF16
                    # storage, including padding-only local rows.
                    hidden = torch.empty(
                        (plan.local_rows, 256), device=device, dtype=torch.bfloat16
                    )[:, ::2]
                    hidden.copy_(routed)
                    model._query_layout = NS(local_rows=plan.local_rows)
                    published = model._publish_mtp_prefill_hidden(hidden)
                    hidden.zero_()
                    torch.testing.assert_close(published, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        model.get_mtp_target_hidden_states(-1), expected, rtol=0, atol=0
                    )
                model._initialize_mtp_hidden_handoff(resource)
                model._initialize_natural_outer(resource)
                checks.assertIs(model._row_exchange, exchange)
                checks.assertEqual(
                    pointers,
                    (
                        exchange.source.data_ptr(),
                        model._mtp_prefill_hidden_buffer.data_ptr(),
                    ),
                )
                # Overflow is checked before queuing a transport operation or
                # publishing invalid recurrence data.
                invalid = NaturalZigzagPlan((model._max_moe_tokens + 1,), size, rank)
                with checks.assertRaisesRegex(ValueError, "capacity"):
                    exchange.exchange(
                        torch.empty(
                            (invalid.local_rows, 128),
                            device=device,
                            dtype=torch.bfloat16,
                        ),
                        invalid,
                        "to_zigzag",
                    )
                model._query_layout = NS(local_rows=capacity + 1)
                with checks.assertRaisesRegex(ValueError, "rank-local CP storage"):
                    model._publish_mtp_prefill_hidden(
                        torch.empty(
                            (capacity + 1, 128), device=device, dtype=torch.bfloat16
                        )
                    )
                torch.cuda.synchronize()
                dist.barrier()
                print(
                    json.dumps(
                        dict(
                            rank=rank,
                            chunk=chunk,
                            physical_capacity=capacity,
                            slot_rows=exchange.chunk_capacity,
                            route_payload_bytes=exchange.resident_bytes,
                            handoff_bytes=model._mtp_prefill_hidden_buffer.numel() * 2,
                            passed=True,
                        )
                    ),
                    flush=True,
                )
        check_fp8_projection_staging(checks, device, size)
    finally:
        torch.cuda.synchronize()
        dist.destroy_process_group()


class PrefillRuntimeCudaTest(unittest.TestCase):
    def test_tp8_transport_handoff_and_capacity(self):
        self.assertGreaterEqual(
            torch.cuda.device_count(), 8, "requires eight CUDA GPUs"
        )
        manager = PortManager()
        ports, locks = manager.get_consecutive_ports(1)
        processes = []
        try:
            context = mp.get_context("spawn")
            processes = [
                context.Process(target=worker, args=(rank, 8, ports[0]))
                for rank in range(8)
            ]
            for process in processes:
                process.start()
            deadline = time.monotonic() + 900
            for process in processes:
                process.join(timeout=max(0, deadline - time.monotonic()))
            self.assertEqual([process.exitcode for process in processes], [0] * 8)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            for lock in locks:
                lock.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
