import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.eplb.runtime import RuntimeExpertBalancer, _shuffle_weights
from rtp_llm.eplb.test.runtime_test import make_config, make_layer


def _publish_layout(layer, physical):
    inverse = torch.full_like(layer.logical_to_physical, -1, device="cpu")
    counts = torch.zeros_like(layer.logical_count, device="cpu")
    for slot, expert in enumerate(physical.tolist()):
        inverse[expert, counts[expert]] = slot
        counts[expert] += 1
    layer.logical_to_physical.copy_(inverse)
    layer.logical_count.copy_(counts)
    layer.physical_to_logical.copy_(physical)


def _check_weights(layer):
    local = layer.num_local_experts
    ids = layer.physical_to_logical[layer.ep_rank * local : (layer.ep_rank + 1) * local]
    expected_weight = ids[:, None] * 100 + torch.arange(6)
    expected_scale = (
        (ids[:, None, None] * 10 + torch.arange(6).reshape(2, 3))
        .float()
        .transpose(1, 2)
    )
    assert torch.equal(layer.weights[0].cpu(), expected_weight)
    assert torch.equal(layer.weights[1].cpu(), expected_scale)
    if len(layer.weights) > 2:
        expected_fp8 = (ids[:, None] + torch.arange(3)).to(torch.float8_e4m3fn)
        assert torch.equal(
            layer.weights[2].cpu().view(torch.uint8), expected_fp8.view(torch.uint8)
        )


def _run_distributed(rank, path, scenario, world_size):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{path}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(seconds=120),
    )
    try:
        # Initialize WORLD before any point-to-point exchange.
        dist.all_reduce(torch.zeros(1, device="cuda"))
        layer = make_layer(
            (0, 1, 2, 3, 0, 1, 2, 3),
            layer_id=2,
            ep_rank=rank,
            ep_size=world_size,
            device="cuda",
        )
        local = layer.num_local_experts
        local_ids = layer.physical_to_logical[rank * local : (rank + 1) * local].cuda()
        fp8 = (local_ids[:, None] + torch.arange(3, device="cuda")).to(
            torch.float8_e4m3fn
        )
        layer.weights = (*layer.weights, fp8)
        pointers = [
            tensor.data_ptr()
            for tensor in (
                *layer.weights,
                layer.logical_to_physical,
                layer.logical_count,
            )
        ]

        if scenario == "shuffle":
            generator = torch.Generator().manual_seed(735)
            for _ in range(25):
                # Every expert remains represented; replicas and rank ownership
                # change independently, exercising cycles and duplicate targets.
                next_physical = torch.cat(
                    (torch.arange(4), torch.randint(0, 4, (4,), generator=generator))
                )
                next_physical = next_physical[torch.randperm(8, generator=generator)]
                _shuffle_weights(layer, next_physical, world_size)
                _publish_layout(layer, next_physical)
                torch.cuda.synchronize()
                _check_weights(layer)
        else:
            runtime = RuntimeExpertBalancer(
                [layer],
                make_config(eplb_update_time=1, eplb_stats_window_size=8),
                rank,
                world_size,
                1,
            )
            ids = torch.tensor(
                [[0, 1], [2, -1], [3, 2]], dtype=torch.int32, device="cuda"
            )
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    layer.map_ids(ids)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured = layer.map_ids(ids)
            for hot in (2, 3, 0, 1):
                counts = torch.zeros(3, local, dtype=torch.int64, device="cuda")
                current_local = layer.physical_to_logical[
                    rank * local : (rank + 1) * local
                ].cuda()
                counts[2] = (current_local == hot) * (2**33)
                saved = counts.clone()
                runtime.record_step(counts)
                assert runtime.balance()
                assert torch.equal(counts, saved)
                assert int(layer.logical_count[hot]) == 5
                assert runtime._samples == 0
                _check_weights(layer)
                graph.replay()
                torch.cuda.synchronize()
                mapped = captured.cpu().long()
                logical = ids.cpu().long()
                assert mapped[1, 1].item() == -1
                assert torch.equal(
                    layer.physical_to_logical[mapped[logical >= 0]],
                    logical[logical >= 0],
                )
                maps = torch.empty(world_size, 8, dtype=torch.int64, device="cuda")
                dist.all_gather_into_tensor(maps, layer.physical_to_logical.cuda())
                assert torch.equal(maps, maps[0].expand_as(maps))
        assert pointers == [
            tensor.data_ptr()
            for tensor in (
                *layer.weights,
                layer.logical_to_physical,
                layer.logical_count,
            )
        ]
        dist.barrier()
    finally:
        dist.destroy_process_group()


class RuntimeDistributedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world_size = int(os.environ.get("EPLB_TEST_WORLD_SIZE", "2"))
        if cls.world_size not in (2, 8):
            raise ValueError("EPLB_TEST_WORLD_SIZE must be 2 or 8")
        if torch.cuda.device_count() < cls.world_size:
            raise RuntimeError(
                f"runtime_distributed_test requires {cls.world_size} visible CUDA GPUs"
            )

    def run_scenario(self, scenario):
        with tempfile.TemporaryDirectory(prefix="eplb-runtime-") as directory:
            mp.spawn(
                _run_distributed,
                args=(os.path.join(directory, "rendezvous"), scenario, self.world_size),
                nprocs=self.world_size,
                join=True,
            )

    def test_p2p_cycles_replicas_fp8_and_noncontiguous_scales(self):
        self.run_scenario("shuffle")

    def test_runtime_int64_history_and_graph_replay(self):
        self.run_scenario("runtime")


if __name__ == "__main__":
    unittest.main()
