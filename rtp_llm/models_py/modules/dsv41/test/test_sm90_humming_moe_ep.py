"""Two-Hopper DeepEP normal + Humming regression without checkpoint loading.

The real Buffer dispatch/combine is observed, not mocked. Only the singleton
binding is injected so this test can size its NVLink allocation from DeepEP's
configuration hints instead of allocating the server's fixed 2 GB buffer.
"""

import multiprocessing as mp
import os
import time
import unittest
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rtp_llm.test.utils.port_util import PortManager

_WORLD = 2
_EXPERTS = 16
_LOCAL_EXPERTS = 8
_DIM = 128


def _fixtures():
    generator = torch.Generator(device="cpu").manual_seed(20261015)
    packed = [
        torch.randint(
            -128,
            128,
            (_EXPERTS, _DIM, _DIM // 2),
            dtype=torch.int8,
            generator=generator,
        )
        for _ in range(3)
    ]
    scales = [
        torch.randint(
            121,
            125,
            (_EXPERTS, _DIM, _DIM // 32),
            dtype=torch.uint8,
            generator=generator,
        )
        for _ in range(3)
    ]
    inputs = (torch.randn((_WORLD, 3, _DIM), generator=generator) * 12).bfloat16()
    # Exact BF16 markers make the source of every received token unambiguous.
    for rank in range(_WORLD):
        inputs[rank, :, 0] = torch.arange(3) + rank * 32 + 1
    weights = torch.tensor(
        [
            [
                [0.05, 0.1, 0.2, 0.25, 0.4, 0.5],
                [0.5, 0.4, 0.25, 0.2, 0.1, 0.05],
                [0.0] * 6,
            ],
            [
                [0.4, 0.25, 0.1, 0.5, 0.05, 0.2],
                [0.2, 0.05, 0.5, 0.1, 0.25, 0.4],
                [0.0] * 6,
            ],
        ],
        dtype=torch.float32,
    )
    mixed = torch.tensor(
        [
            [[0, 1, 2, 8, 9, 10], [4, 5, 6, 12, 13, 14], [-1] * 6],
            [[15, 14, 13, 7, 6, 5], [11, 10, 9, 3, 2, 1], [-1] * 6],
        ],
        dtype=torch.int64,
    )
    only_zero = torch.tensor(
        [
            [[0, 1, 2, 3, 4, 5], [7, 6, 5, 4, 3, 2], [-1] * 6],
            [[5, 4, 3, 2, 1, 0], [2, 3, 4, 5, 6, 7], [-1] * 6],
        ],
        dtype=torch.int64,
    )
    only_one = torch.where(only_zero >= 0, only_zero + _LOCAL_EXPERTS, only_zero)
    return (
        packed,
        scales,
        inputs,
        weights,
        (
            ("cross_rank", mixed),
            ("rank1_empty_recv", only_zero),
            ("rank0_empty_recv", only_one),
        ),
    )


def _decode_fp4(weight, scale):
    # Independent E2M1 nibble decode, not the production weight transformer.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=weight.device,
    )
    raw = weight.to(torch.int16) & 255
    codes = torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)
    return lut[codes.long()] * scale.float().repeat_interleave(32, dim=-1)


def _quantize_reference(x):
    blocks = x.float().reshape(x.shape[0], -1, 128)
    scale = torch.pow(
        2.0, torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    payload = (blocks / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).float()
    return (payload * scale.unsqueeze(-1)).reshape_as(x)


def _reference(x, routes, route_weights, decoded):
    """Explicit quantized math, including BF16 partial sums at the EP boundary."""
    result = torch.zeros_like(x, dtype=torch.float32)
    for row in range(x.shape[0]):
        quantized_x = _quantize_reference(x[row : row + 1])
        rank_outputs = []
        for destination in range(_WORLD):
            slot_outputs = torch.zeros((6, _DIM), dtype=torch.float32, device=x.device)
            for slot, expert in enumerate(routes[row].tolist()):
                if (
                    not destination * _LOCAL_EXPERTS
                    <= expert
                    < (destination + 1) * _LOCAL_EXPERTS
                ):
                    continue
                gate = (
                    (quantized_x @ decoded[0][expert].T)
                    .bfloat16()
                    .float()
                    .clamp(max=10)
                )
                up = (
                    (quantized_x @ decoded[1][expert].T)
                    .bfloat16()
                    .float()
                    .clamp(-10, 10)
                )
                hidden = (
                    torch.nn.functional.silu(gate) * up * route_weights[row, slot]
                ).bfloat16()
                down = (_quantize_reference(hidden) @ decoded[2][expert].T).bfloat16()
                slot_outputs[slot] = down.float().squeeze(0)
            # DeepEPStrategy sends each rank's local FP32 sum as BF16.
            rank_outputs.append(slot_outputs.sum(0).bfloat16().float())
        result[row] = torch.stack(rank_outputs).sum(0).bfloat16().float()
    return result


class _ObservedBuffer:
    def __init__(self, buffer, rank, all_inputs, all_weights):
        self.buffer = buffer
        self.rank = rank
        self.all_inputs = all_inputs
        self.all_weights = all_weights
        self.routes = None
        self.recv_rows = None

    def get_dispatch_layout(self, indices, num_experts):
        assert indices.shape[1] == 8, "top6 must be padded to top8"
        assert bool((indices[:, 6:] == -1).all())
        return self.buffer.get_dispatch_layout(indices, num_experts)

    def dispatch(self, *args, **kwargs):
        result = self.buffer.dispatch(*args, **kwargs)
        recv_x, recv_ids, recv_weights, counts, _, _ = result
        self.recv_rows = recv_x.shape[0]
        lo, hi = self.rank * _LOCAL_EXPERTS, (self.rank + 1) * _LOCAL_EXPERTS
        selected = (self.routes >= lo) & (self.routes < hi)
        expected_rows = int(selected.any(-1).sum())
        assert self.recv_rows == expected_rows, (
            self.rank,
            self.recv_rows,
            expected_rows,
        )
        expected_counts = [
            (self.routes == expert).sum().item() for expert in range(lo, hi)
        ]
        assert list(counts) == expected_counts, (counts, expected_counts)
        received = recv_x.cpu()
        ids = recv_ids.cpu().long()
        weights = recv_weights.cpu()
        for row in range(self.recv_rows):
            marker = int(received[row, 0]) - 1
            source, token = marker // 32, marker % 32
            torch.testing.assert_close(
                received[row], self.all_inputs[source, token], rtol=0, atol=0
            )
            route = self.routes[source, token]
            valid = (route >= lo) & (route < hi)
            expected = torch.cat(
                (torch.where(valid, route - lo, -1), torch.full((2,), -1))
            )
            torch.testing.assert_close(ids[row], expected, rtol=0, atol=0)
            torch.testing.assert_close(
                weights[row, :6][valid],
                self.all_weights[source, token][valid],
                rtol=0,
                atol=0,
            )
        return result

    def combine(self, x, handle, *args, **kwargs):
        assert x.dtype == torch.bfloat16
        assert not args and not kwargs, "combine must not apply router weights again"
        return self.buffer.combine(x, handle)


def _worker(rank, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", rank=rank, world_size=_WORLD, timeout=timedelta(seconds=180)
    )
    # Import after NCCL is loaded so the wrapper can pin EP_NCCL_ROOT_DIR.
    from rtp_llm.models_py.distributed.deepep_wrapper import (
        DeepEPBuffer,
        DeepEPMode,
        DeepEPWrapper,
    )
    from rtp_llm.models_py.modules.dsv41.moe.strategies.base import MoeCfg
    from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import (
        SM90HummingStrategy,
    )
    from rtp_llm.utils.model_weight import W

    hidden_bytes = _DIM * torch.empty((), dtype=torch.bfloat16).element_size()
    nvl_bytes = max(
        config.get_nvl_buffer_size_hint(hidden_bytes, _WORLD)
        for config in (
            DeepEPBuffer.get_dispatch_config(_WORLD),
            DeepEPBuffer.get_combine_config(_WORLD),
        )
    )
    buffer = DeepEPBuffer(
        dist.group.WORLD,
        num_nvl_bytes=nvl_bytes,
        num_rdma_bytes=0,
        low_latency_mode=False,
        explicitly_destroy=True,
    )
    previous_instance = DeepEPWrapper._instance
    try:
        packed, scales, all_inputs, all_weights, cases = _fixtures()
        observed = _ObservedBuffer(buffer, rank, all_inputs, all_weights)
        DeepEPWrapper._instance = SimpleNamespace(
            buffer=observed, mode=DeepEPMode.NORMAL
        )
        cfg = MoeCfg(
            layer_id=0,
            dim=_DIM,
            moe_inter_dim=_DIM,
            n_routed_experts=_EXPERTS,
            n_activated_experts=6,
            swiglu_limit=10.0,
            ep_size=_WORLD,
            ep_rank=rank,
            n_local_experts=_LOCAL_EXPERTS,
            local_expert_start=rank * _LOCAL_EXPERTS,
            local_expert_end=(rank + 1) * _LOCAL_EXPERTS,
            max_tokens_per_rank=32,
            shared_fp8_block_size=32,
        )
        strategy = SM90HummingStrategy(cfg)
        keys = (
            (W.v4_routed_w1_w, W.v4_routed_w1_s),
            (W.v4_routed_w3_w, W.v4_routed_w3_s),
            (W.v4_routed_w2_w, W.v4_routed_w2_s),
        )
        layer_weights = {}
        for (weight_key, scale_key), weight, scale in zip(keys, packed, scales):
            local = slice(cfg.local_expert_start, cfg.local_expert_end)
            layer_weights[weight_key] = weight[local].cuda()
            layer_weights[scale_key] = scale[local].cuda().view(torch.float8_e8m0fnu)
        strategy.setup_weights(layer_weights)
        assert not layer_weights, "expert source tensors must be consumed"
        strategy.setup_runtime()
        decoded = [
            _decode_fp4(w.cuda(), s.cuda().view(torch.float8_e8m0fnu))
            for w, s in zip(packed, scales)
        ]
        x = all_inputs[rank].cuda()
        route_weights = all_weights[rank].cuda()
        numerical_failures = []
        for name, routes in cases:
            observed.routes = routes
            with torch.inference_mode():
                expected = _reference(x, routes[rank], route_weights, decoded)
                actual = strategy(x, route_weights, routes[rank].cuda())
            torch.cuda.synchronize()
            try:
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.05)
            except AssertionError as error:
                # Still exercise both empty-receive rounds before reporting
                # numerical failures; both ranks must reach each collective.
                numerical_failures.append(f"rank={rank} case={name}: {error}")
            torch.testing.assert_close(
                actual[-1], torch.zeros_like(actual[-1]), rtol=0, atol=0
            )
            print(
                f"SM90 EP2 rank={rank} case={name} recv_rows={observed.recv_rows} "
                f"max_abs_diff={(actual - expected).abs().max().item():.6g}",
                flush=True,
            )
            dist.barrier()
        if numerical_failures:
            raise AssertionError("\n".join(numerical_failures))
    finally:
        DeepEPWrapper._instance = previous_instance
        buffer.destroy()
        dist.destroy_process_group()


class SM90HummingEPTest(unittest.TestCase):
    def test_ep2_local_ids_route_weights_and_empty_receive(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD:
            self.skipTest("requires two NVLink-connected SM90 GPUs")
        if any(
            torch.cuda.get_device_capability(rank) != (9, 0) for rank in range(_WORLD)
        ):
            self.skipTest("requires two NVLink-connected SM90 GPUs")
        ports, locks = PortManager().get_consecutive_ports(1)
        processes = []
        exitcodes = []
        try:
            context = mp.get_context("spawn")
            processes = [
                context.Process(target=_worker, args=(rank, ports[0]))
                for rank in range(_WORLD)
            ]
            for process in processes:
                process.start()
            # A common deadline bounds a failed peer's collective/JIT wait.
            deadline = time.monotonic() + 600
            for process in processes:
                process.join(timeout=max(0, deadline - time.monotonic()))
            exitcodes = [process.exitcode for process in processes]
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5)
            for process in processes:
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
            for lock in locks:
                lock.__exit__(None, None, None)
        self.assertEqual(exitcodes, [0, 0])


if __name__ == "__main__":
    unittest.main()
