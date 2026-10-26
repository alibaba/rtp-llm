"""Real eight-GPU streaming decisions and full-window runtime migration."""

import hashlib
import logging
import os
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.eplb import policy
from rtp_llm.eplb.runtime import RuntimeExpertBalancer
from rtp_llm.eplb.test.policy_larges_reference import (
    GROUPS,
    assert_same_outputs,
    history,
)
from rtp_llm.eplb.test.policy_larges_test import checked_plan
from rtp_llm.eplb.test.runtime_test import make_config, make_layer, routing_oracle


def assert_ranks_equal(value, world_size):
    flat = value.contiguous().flatten()
    gathered = torch.empty(
        world_size * flat.numel(), device=value.device, dtype=value.dtype
    )
    dist.all_gather_into_tensor(gathered, flat)
    gathered = gathered.reshape(world_size, -1)
    assert torch.equal(gathered, gathered[0].expand_as(gathered)), "rank disagreement"


def assert_input_equal(source, world_size):
    digest = hashlib.sha256(source.cpu().contiguous().numpy().tobytes()).digest()
    assert_ranks_equal(
        torch.tensor(list(digest), dtype=torch.uint8, device=source.device), world_size
    )


def policy_scenario(rank, world_size):
    stream = torch.cuda.Stream()
    for steps in (10, 100):
        for _, layers, experts, physical, local in GROUPS:
            for kind in ("zipf", "sparse"):
                for seed in (0, 1):
                    source = history(kind, steps, layers, experts, seed).cuda()
                    assert_input_equal(source, world_size)
                    # Each GPU independently checks every chosen decision and
                    # reconstructs all outputs before the cross-rank equality.
                    expected, _ = checked_plan(source, physical, local)
                    for result in expected:
                        assert_ranks_equal(result, world_size)
                    stream.wait_stream(torch.cuda.current_stream())
                    for repeat in range(3):
                        with torch.cuda.stream(stream):
                            # A different input between repetitions detects
                            # stale scratch or cached-output reuse.
                            changed = source.roll(repeat + 1, dims=-1)
                            policy.rebalance_experts_global(changed, physical, local)
                            result = policy.rebalance_experts_global(
                                source, physical, local
                            )
                        torch.cuda.current_stream().wait_stream(stream)
                        assert_same_outputs(result, expected)
                        for value in result:
                            assert_ranks_equal(value, world_size)


def storage_identity(layers):
    return [
        [
            (tensor.data_ptr(), tensor.stride(), tuple(tensor.shape))
            for tensor in (
                *layer.weights,
                layer.physical_to_logical,
                layer.logical_to_physical,
                layer.logical_count,
            )
        ]
        for layer in layers
    ]


def check_published(layers, rank, world_size):
    for layer in layers:
        for value in (
            layer.physical_to_logical,
            layer.logical_to_physical,
            layer.logical_count,
        ):
            assert_ranks_equal(value, world_size)
        local = layer.num_local_experts
        ids = layer.physical_to_logical[rank * local : (rank + 1) * local]
        expected_weights = (
            (
                ids[:, None] * 100
                + torch.arange(layer.weights[0].size(1), device="cuda")
            ).int(),
            (ids[:, None, None] * 10 + torch.arange(6, device="cuda").reshape(2, 3))
            .float()
            .transpose(1, 2),
        )
        for value, expected in zip(layer.weights, expected_weights):
            assert torch.equal(
                value.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )


def runtime_scenario(rank, world_size, window):
    physical = tuple(expert for expert in range(4) for _ in range(4))
    layers = [
        make_layer(physical, layer_id, rank, world_size, "cuda", weight_width=width)
        for layer_id, width in ((2, 6), (5, 6), (7, 8))
    ]
    calls = []

    def planner(source, total, local):
        assert source.size(0) == window and source.dtype == torch.int32
        assert source.is_cuda and source.is_contiguous()
        assert_input_equal(source, world_size)
        actual, _ = checked_plan(source, total, local)
        for result in actual:
            assert_ranks_equal(result, world_size)
        calls.append(tuple(source.shape))
        return actual

    with patch("rtp_llm.eplb.runtime.rebalance_experts_global", side_effect=planner):
        runtime = RuntimeExpertBalancer(
            layers,
            make_config(eplb_stats_window_size=window, eplb_update_time=window),
            rank,
            world_size,
            "large-window-test",
        )
        assert len(runtime.groups) == 2 and len(calls) == 2
        assert all(
            group.updates == 0 and group.window_index == 0 for group in runtime.groups
        )
        identities = storage_identity(layers)
        expected_windows = [torch.zeros_like(group.window) for group in runtime.groups]
        ids = torch.tensor([[0, 1], [2, 3], [-1, 0]], dtype=torch.int32, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                layers[0].map_ids(ids)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = layers[0].map_ids(ids)
        first = window // 4
        update_steps = (first, first + window, first + 2 * window)
        migrations = 0
        for step in range(1, update_steps[-1] + 1):
            hot = 2 if step <= first else (3 if step <= first + window else 0)
            local = layers[0].num_local_experts
            counts = torch.zeros((8, local), dtype=torch.int32, device="cuda")
            for layer in layers:
                local_map = layer.physical_to_logical[rank * local : (rank + 1) * local]
                counts[layer.layer_id] = (local_map == hot).int() * 10000 * (rank + 1)
            original_counts = counts.clone()
            for group, expected in zip(runtime.groups, expected_windows):
                expected[(step - 1) % window].copy_(
                    counts.index_select(0, group.layer_ids)
                )
            changed = runtime.step(counts)
            assert torch.equal(counts, original_counts)
            assert changed == (step in update_steps)
            if step not in update_steps:
                continue
            migrations += 1
            assert all(group.updates == migrations for group in runtime.groups)
            for group, expected in zip(runtime.groups, expected_windows):
                assert group.window_index == step % window
                assert torch.equal(group.window, expected), "physical ring history"
            check_published(layers, rank, world_size)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured, routing_oracle(layers[0], ids))
            assert storage_identity(layers) == identities
        assert migrations == 3
        assert len(calls) == 2 * (
            1 + migrations
        ), "warmup and each real update must plan both groups"


def worker(rank, rendezvous, scenario, world_size):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=300),
    )
    try:
        dist.all_reduce(torch.zeros(1, device="cuda"))
        if scenario == "policy":
            policy_scenario(rank, world_size)
        else:
            runtime_scenario(rank, world_size, int(scenario))
        dist.barrier()
    except Exception:
        logging.exception("large-window EPLB scenario=%s rank=%s", scenario, rank)
        raise
    finally:
        dist.destroy_process_group()


class PolicyLargeDistributedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world_size = 8
        if torch.cuda.device_count() != cls.world_size:
            raise RuntimeError(
                "large-window distributed tests require exactly eight visible GPUs"
            )

    def run_scenario(self, scenario):
        with tempfile.TemporaryDirectory(prefix="eplb-larges-") as directory:
            mp.spawn(
                worker,
                args=(os.path.join(directory, "rendezvous"), scenario, self.world_size),
                nprocs=self.world_size,
                join=True,
            )

    def test_real_eight_gpu_all_three_outputs_and_nondefault_stream(self):
        self.run_scenario("policy")

    def test_runtime_window_10_three_migrations_and_graph_replay(self):
        self.run_scenario("10")

    def test_runtime_window_100_three_migrations_and_graph_replay(self):
        self.run_scenario("100")


if __name__ == "__main__":
    unittest.main()
