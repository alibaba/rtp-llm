"""Real NCCL and symmetric-memory migration, plus routing graph replay.

The graph here captures map_ids; model/kernel tests cover the MoE computation.
"""

import logging
import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rtp_llm.eplb.runtime import RuntimeExpertBalancer
from rtp_llm.eplb.shuffle import (
    expert_model_parallel_shuffle_inplace,
    expert_model_parallel_shuffle_inplace_intranode,
)
from rtp_llm.eplb.test.runtime_test import make_config, make_layer, routing_oracle


def _publish_layout(layer, physical):
    inverse = torch.full_like(layer.logical_to_physical, -1, device="cpu")
    counts = torch.zeros_like(layer.logical_count, device="cpu")
    for slot, expert in enumerate(physical.cpu().tolist()):
        inverse[expert, counts[expert]] = slot
        counts[expert] += 1
    layer.logical_to_physical.copy_(inverse)
    layer.logical_count.copy_(counts)
    layer.physical_to_logical.copy_(physical)


def _bytes(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8)


def _check_weights(layer):
    local = layer.num_local_experts
    ids = layer.physical_to_logical.cpu()[
        layer.ep_rank * local : (layer.ep_rank + 1) * local
    ]
    expected = (
        (ids[:, None] * 100 + torch.arange(layer.weights[0].size(1))).int(),
        (ids[:, None, None] * 10 + torch.arange(6).reshape(2, 3))
        .float()
        .transpose(1, 2),
        (ids[:, None] + torch.arange(3)).to(torch.float8_e4m3fn),
        (ids[:, None] * 3 + torch.arange(8)).to(torch.int8),
        (ids[:, None] + 127 + torch.arange(3))
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu),
    )
    for index, (actual, reference) in enumerate(zip(layer.weights, expected)):
        assert torch.equal(_bytes(actual.cpu()), _bytes(reference)), (
            layer.ep_rank,
            layer.layer_id,
            index,
        )


def _make_layers(rank, world_size, mixed=False):
    physical = tuple(expert for expert in range(4) for _ in range(4))
    layers = []
    layouts = ((2, 6), (5, 6)) + (((7, 8),) if mixed else ())
    for layer_id, width in layouts:
        layer = make_layer(
            physical, layer_id, rank, world_size, "cuda", weight_width=width
        )
        local = layer.num_local_experts
        ids = layer.physical_to_logical[rank * local : (rank + 1) * local]
        fp8 = (ids[:, None] + torch.arange(3, device="cuda")).to(torch.float8_e4m3fn)
        packed_fp4 = (ids[:, None] * 3 + torch.arange(8, device="cuda")).to(torch.int8)
        ue8m0 = (
            (ids[:, None] + 127 + torch.arange(3, device="cuda"))
            .to(torch.uint8)
            .view(torch.float8_e8m0fnu)
        )
        layer.weights = (*layer.weights, fp8, packed_fp4, ue8m0)
        layers.append(layer)
    return layers


def _identities(layers):
    return [
        [
            (tensor.data_ptr(), tuple(tensor.shape), tensor.stride())
            for tensor in (
                *layer.weights,
                layer.logical_to_physical,
                layer.logical_count,
                layer.physical_to_logical,
            )
        ]
        for layer in layers
    ]


def _check_rank_maps(layers, world_size):
    physical = torch.stack([layer.physical_to_logical for layer in layers])
    gathered = torch.empty(
        world_size * physical.size(0),
        physical.size(1),
        dtype=physical.dtype,
        device="cuda",
    )
    dist.all_gather_into_tensor(gathered, physical)
    gathered = gathered.reshape(world_size, *physical.shape)
    assert torch.equal(gathered, gathered[0].expand_as(gathered))


def _shuffle_scenario(rank, world_size, intranode):
    layers = _make_layers(rank, world_size)
    saved_identity = _identities(layers)
    weights = [layer.weights for layer in layers]
    physical = torch.stack([layer.physical_to_logical for layer in layers])
    if intranode:
        import torch.distributed._symmetric_memory as symm_mem

        buffers = [
            symm_mem.empty(w.shape, dtype=w.dtype, device=w.device) for w in weights[0]
        ]
        handles = [symm_mem.rendezvous(buffer, dist.group.WORLD) for buffer in buffers]
        mirrors = [
            [
                handle.get_buffer(peer, buffer.shape, buffer.dtype)
                for buffer, handle in zip(buffers, handles)
            ]
            for peer in range(world_size)
        ]
    generator = torch.Generator().manual_seed(735)
    # Explicit no-op, full rank swap, then changing replica multiplicities.
    layouts = [physical.cpu().clone(), physical.cpu().roll(8, dims=-1)]
    for _ in range(8):
        rows = []
        for _ in layers:
            row = torch.cat(
                (torch.arange(4), torch.randint(0, 4, (12,), generator=generator))
            )
            rows.append(row[torch.randperm(16, generator=generator)])
        layouts.append(torch.stack(rows))
    for index, target in enumerate(layouts):
        target = target.cuda()
        if index == 1:
            local = layers[0].num_local_experts
            start = rank * local
            old_ids = set(physical[0, start : start + local].cpu().tolist())
            new_ids = set(target[0, start : start + local].cpu().tolist())
            assert new_ids - old_ids, "rank swap must actually require remote experts"
        if intranode:
            expert_model_parallel_shuffle_inplace_intranode(
                mirrors, physical, weights, target, dist.group.WORLD
            )
        else:
            expert_model_parallel_shuffle_inplace(
                physical, weights, target, dist.group.WORLD, p2p_groups=3
            )
        for layer, row in zip(layers, target):
            _publish_layout(layer, row)
        physical.copy_(target)
        torch.cuda.synchronize()
        for layer in layers:
            _check_weights(layer)
        _check_rank_maps(layers, world_size)
        assert _identities(layers) == saved_identity


def _runtime_scenario(rank, world_size, force_p2p):
    layers = _make_layers(rank, world_size, mixed=True)
    runtime = RuntimeExpertBalancer(layers, make_config(), rank, world_size, "target")
    assert [len(group.layers) for group in runtime.groups] == [2, 1]
    if force_p2p:
        # Exercise the real cross-domain transport on one host. This does not
        # simulate or claim to validate a multi-node network/topology probe.
        for group in runtime.groups:
            group.shuffle_initialized = True
            group.intranode_shuffle = False
    saved_identity = _identities(layers)
    ids = torch.tensor([[0, 1], [2, -1], [3, 2]], dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layers[0].map_ids(ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = layers[0].map_ids(ids)
    graph.replay()
    torch.cuda.synchronize()
    original_mapped = captured.clone()
    graph_output_changed = False
    expected_windows = [torch.zeros_like(group.window) for group in runtime.groups]
    step = 0
    for update, hot in enumerate((2, 3, 0, 1), start=1):
        for _ in range(1 if update == 1 else 4):
            step += 1
            local = layers[0].num_local_experts
            counts = torch.zeros(8, local, dtype=torch.int32, device="cuda")
            for layer in layers:
                local_map = layer.physical_to_logical[rank * local : (rank + 1) * local]
                counts[layer.layer_id] = (local_map == hot).int() * 10000
            saved_counts = counts.clone()
            before = [layer.physical_to_logical.clone() for layer in layers]
            for group, expected in zip(runtime.groups, expected_windows):
                expected[(step - 1) % 4].copy_(counts.index_select(0, group.layer_ids))
            changed = runtime.step(counts)
            assert torch.equal(counts, saved_counts)
            # A rank with no hot expert has a valid all-zero local history.
            for group, expected in zip(runtime.groups, expected_windows):
                assert torch.equal(group.window, expected)
            expected_update = step == 1 or (step - 1) % 4 == 0
            assert changed == expected_update
            if expected_update:
                assert all(
                    not torch.equal(old, layer.physical_to_logical)
                    for old, layer in zip(before, layers)
                )
                assert all(group.updates == update for group in runtime.groups)
                assert all(group.window_index == step % 4 for group in runtime.groups)
                assert all(int(layer.logical_count[hot]) == 13 for layer in layers)
                if not force_p2p:
                    assert all(group.intranode_shuffle for group in runtime.groups)
            for layer in layers:
                _check_weights(layer)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured, routing_oracle(layers[0], ids))
            graph_output_changed |= not torch.equal(captured, original_mapped)
            _check_rank_maps(layers, world_size)
            assert _identities(layers) == saved_identity
    assert (
        graph_output_changed
    ), "captured routing must observe an actually changed mapping"


def _mixed_prompt_scenario(rank, world_size, after_prefill):
    layers = _make_layers(rank, world_size, mixed=True)
    runtime = RuntimeExpertBalancer(
        layers,
        make_config(eplb_rearrange_after_prefill=after_prefill),
        rank,
        world_size,
        "target-dp-normal",
        is_normal_engine=True,
        sync_prompt_phase=True,
    )
    flag = runtime.prompt_flag
    assert flag.device.type == "cuda" and flag.dtype == torch.int32
    assert flag.numel() == 1
    flag_pointer = flag.data_ptr()
    saved_identity = _identities(layers)
    # Different DP inputs take turns entering prefill. All other ranks decode.
    # Every rank must nevertheless follow one globally agreed EPLB phase.
    prompt_ranks = (0, 1, None, 1, 0, None, None, None, None)
    if after_prefill:
        expected_updates = (0, 0, 1, 1, 1, 2, 2, 2, 2)
        expected_indices = (1, 2, 3, 0, 1, 2, 3, 0, 1)
        expected_steps = (3, 3, 0, 3, 3, 0, 1, 2, 3)
    else:
        expected_updates = (0, 0, 1, 1, 1, 1, 1, 1, 2)
        expected_indices = (0, 0, 1, 1, 1, 2, 3, 0, 1)
        expected_steps = (3, 3, 0, 0, 0, 1, 2, 3, 0)
    local = layers[0].num_local_experts
    for index, prompt_rank in enumerate(prompt_ranks):
        hot = 2 if index < 3 else 3
        counts = torch.zeros(8, local, dtype=torch.int32, device="cuda")
        for layer in layers:
            local_map = layer.physical_to_logical[rank * local : (rank + 1) * local]
            counts[layer.layer_id] = (local_map == hot).int() * 10000 * (rank + 1)
        original_counts = counts.clone()
        old_windows = [group.window.clone() for group in runtime.groups]
        old_maps = [layer.physical_to_logical.clone() for layer in layers]
        changed = runtime.step(counts, is_prompt=rank == prompt_rank)
        assert torch.equal(counts, original_counts)
        assert runtime.prompt_flag.data_ptr() == flag_pointer
        assert flag.item() == int(prompt_rank is not None)
        for group, old_window in zip(runtime.groups, old_windows):
            assert group.updates == expected_updates[index]
            assert group.window_index == expected_indices[index]
            assert group.steps == expected_steps[index]
            if prompt_rank is not None and not after_prefill:
                assert torch.equal(group.window, old_window)
            else:
                written = (group.window_index - 1) % group.window.size(0)
                assert torch.equal(
                    group.window[written], counts.index_select(0, group.layer_ids)
                )
        if index == 2:
            assert changed
            assert all(
                not torch.equal(old, layer.physical_to_logical)
                for old, layer in zip(old_maps, layers)
            )
        if index == 0 or expected_updates[index] == expected_updates[index - 1]:
            assert not changed
        for layer in layers:
            _check_weights(layer)
        # Real NCCL checks that scheduling state and maps agree on every rank.
        state = torch.tensor(
            [
                flag.item(),
                int(changed),
                *(
                    value
                    for group in runtime.groups
                    for value in (group.updates, group.window_index, group.steps)
                ),
            ],
            dtype=torch.int32,
            device="cuda",
        )
        states = torch.empty(
            world_size, state.numel(), dtype=state.dtype, device="cuda"
        )
        dist.all_gather_into_tensor(states, state)
        assert torch.equal(states, states[0].expand_as(states))
        _check_rank_maps(layers, world_size)
        assert _identities(layers) == saved_identity


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
        # Initialize WORLD before P2P, symmetric-memory, or DeepEP topology query.
        dist.all_reduce(torch.zeros(1, device="cuda"))
        if scenario in ("shuffle_p2p", "shuffle_symm"):
            _shuffle_scenario(rank, world_size, scenario == "shuffle_symm")
        elif scenario in ("mixed_prompt_skip", "mixed_prompt_defer"):
            _mixed_prompt_scenario(rank, world_size, scenario == "mixed_prompt_defer")
        else:
            _runtime_scenario(rank, world_size, scenario == "runtime_p2p")
        dist.barrier()
    except Exception:
        # Teardown can wait for a peer collective and hide the first failure.
        logging.exception("EPLB scenario=%s rank=%s failed", scenario, rank)
        raise
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
        self.run_scenario("shuffle_p2p")

    def test_symm_mem_cycles_replicas_and_shared_mirror_reuse(self):
        self.run_scenario("shuffle_symm")

    def test_runtime_all_layout_groups_and_routing_graph_with_symm_mem(self):
        self.run_scenario("runtime_symm")

    def test_runtime_all_layout_groups_and_routing_graph_with_p2p(self):
        self.run_scenario("runtime_p2p")

    def test_normal_dp_mixed_prefill_globally_skips_sampling(self):
        self.run_scenario("mixed_prompt_skip")

    def test_normal_dp_mixed_prefill_globally_defers_rearrange_to_decode(self):
        self.run_scenario("mixed_prompt_defer")


if __name__ == "__main__":
    unittest.main()
