"""EPLB equivalence for the private gate-pack kernels used by V4.1."""

import importlib
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

# Load the actual kernel modules without importing the engine's native bindings.
_PACKAGE = types.ModuleType("_dsv41_private_gate_pack_test")
_PACKAGE.__path__ = [str(Path(__file__).resolve().parent.parent / "moe")]
sys.modules[_PACKAGE.__name__] = _PACKAGE
_PACK = importlib.import_module(f"{_PACKAGE.__name__}._mega_gate_pack_triton")
_SE_PACK = importlib.import_module(f"{_PACKAGE.__name__}._mega_se_gate_pack_triton")

if _PACK.triton is not None:
    import triton
    import triton.language as tl

    _map_eplb_ids = _PACK._map_eplb_ids

    @triton.jit
    def _map_test_kernel(
        ids_ptr,
        token_indices_ptr,
        map_ptr,
        count_ptr,
        out_ptr,
        ep_rank,
        size: tl.constexpr,
        map_stride: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK)
        valid = offsets < size
        ids = tl.load(ids_ptr + offsets, mask=valid, other=-1)
        rows = tl.load(token_indices_ptr + offsets, mask=valid, other=0)
        physical = _map_eplb_ids(
            ids, rows, valid, map_ptr, count_ptr, ep_rank, map_stride
        )
        tl.store(out_ptr + offsets, physical, mask=valid)


def _layout(experts, copies=1, shift=0, device="cuda"):
    logical = torch.cat((torch.arange(experts), torch.arange(16) // copies)).roll(shift)
    mapping = torch.full((experts, 17), -1, dtype=torch.int32)
    counts = torch.zeros(experts, dtype=torch.int32)
    for physical, expert in enumerate(logical.tolist()):
        mapping[expert, counts[expert]] = physical
        counts[expert] += 1
    return mapping.to(device), counts.to(device)


def _torch_map(ids, mapping, counts, rank, rows=None):
    if rows is None:
        rows = torch.arange(ids.shape[0], device=ids.device).unsqueeze(1)
    logical = ids.clamp_min(0).long()
    rotation = (rank + rows * 23333) & 0xFFFFFFFF
    replica = rotation % counts[logical]
    return torch.where(ids >= 0, mapping[logical, replica], -1).to(ids.dtype)


def _inputs(tokens, experts=384, topk=6, dim=256, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(20261028)
    bias = torch.arange(experts, device=device, dtype=torch.float32) * 0.003
    # Make the duplicated experts participate in normal routing as well as hash.
    bias[:16] += 8
    return SimpleNamespace(
        x=torch.randn(
            (tokens, dim), generator=generator, device=device, dtype=torch.bfloat16
        ),
        scores=torch.randn(
            (tokens, experts), generator=generator, device=device, dtype=torch.bfloat16
        ),
        bias=bias,
        input_ids=torch.arange(tokens, device=device, dtype=torch.int64) % 19,
        tid2eid=(
            torch.arange(19, device=device).unsqueeze(1)
            + torch.arange(topk, device=device)
        )
        % experts,
        topk=topk,
    )


def _outputs(inputs, shared=False, block_m=32):
    tokens, dim = inputs.x.shape
    kwargs = dict(device=inputs.x.device)
    out = SimpleNamespace(
        x=torch.full((tokens, dim), 0, dtype=torch.float8_e4m3fn, **kwargs),
        sf=torch.full((tokens, dim // 128), -123, dtype=torch.int32, **kwargs),
        indices=torch.full((tokens, inputs.topk), -123, dtype=torch.int64, **kwargs),
        weights=torch.full((tokens, inputs.topk), -123, dtype=torch.float32, **kwargs),
    )
    if shared:
        active_rows = ((tokens + block_m - 1) // block_m) * (
            (block_m + 127) // 128 * 128
        )
        out.shared_sf = torch.full(
            (dim // 128, active_rows), -123, dtype=torch.int32, **kwargs
        ).T
    return out


def _run(inputs, out, hash_routing, shared=False, mapping=None, counts=None, rank=0):
    args = [inputs.x, inputs.scores]
    args += [inputs.input_ids, inputs.tid2eid] if hash_routing else [inputs.bias]
    args += [out.x, out.sf]
    if shared:
        args += [out.shared_sf]
    args += [out.indices, out.weights]
    kwargs = dict(
        route_scale=2.5,
        to_physical_map=mapping,
        logical_count=counts,
        ep_rank=rank,
    )
    suffix = "hash" if hash_routing else "nonhash"
    if shared:
        kwargs["block_m"] = 32
        function = getattr(_SE_PACK, f"fused_mega_moe_se_gate_pack_{suffix}")
    else:
        function = getattr(_PACK, f"fused_mega_moe_gate_pack_{suffix}")
    function(*args, **kwargs)


class PrivateGatePackEplbHostTest(unittest.TestCase):
    def test_empty_inputs_do_not_launch_or_touch_outputs(self):
        inputs = _inputs(0, device="cpu")
        mapping, counts = _layout(384, device="cpu")
        for shared in (False, True):
            for hash_routing in (False, True):
                with self.subTest(shared=shared, hash=hash_routing):
                    out = _outputs(inputs, shared)
                    with patch.object(_PACK, "_block_m") as block_m:
                        _run(inputs, out, hash_routing, shared, mapping, counts, 31)
                    block_m.assert_not_called()
                    self.assertEqual(out.indices.numel(), 0)

    def test_se_forwards_borrowed_maps_and_stages_scales(self):
        inputs = _inputs(3, device="cpu")
        mapping, counts = _layout(384, device="cpu")
        for hash_routing in (False, True):
            with self.subTest(hash=hash_routing):
                out = _outputs(inputs, shared=True)
                suffix = "hash" if hash_routing else "nonhash"
                with patch.object(
                    _SE_PACK, f"fused_mega_moe_gate_pack_{suffix}"
                ) as pack, patch.object(
                    _SE_PACK, "stage_mega_moe_se_shared_l1_scales"
                ) as stage:
                    _run(inputs, out, hash_routing, True, mapping, counts, 7)
                self.assertIs(pack.call_args.kwargs["to_physical_map"], mapping)
                self.assertIs(pack.call_args.kwargs["logical_count"], counts)
                self.assertEqual(pack.call_args.kwargs["ep_rank"], 7)
                stage.assert_called_once_with(out.sf, out.shared_sf, 3, 32)


@unittest.skipUnless(
    _PACK.triton is not None and torch.cuda.is_available(), "CUDA and Triton required"
)
class PrivateGatePackEplbGpuTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("DSV4_MEGA_MOE_GATE_PACK_BLOCK_M", None)

    def _assert_equivalent(self, ref, got, mapping, counts, rank, shared=False):
        torch.testing.assert_close(
            got.indices, _torch_map(ref.indices, mapping, counts, rank), rtol=0, atol=0
        )
        for actual, expected in (
            (got.weights, ref.weights),
            (got.x.view(torch.uint8), ref.x.view(torch.uint8)),
            (got.sf, ref.sf),
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if shared:
            torch.testing.assert_close(got.shared_sf, ref.shared_sf, rtol=0, atol=0)

    def test_hash_and_nonhash_match_torch_mapping_for_prefill_shapes(self):
        cases = ((1, 1, 0), (17, 6, 3), (257, 32, 31), (1025, 6, 3), (2049, 6, 0))
        for hash_routing in (False, True):
            for tokens, topk, rank in cases:
                with self.subTest(
                    hash=hash_routing, tokens=tokens, topk=topk, rank=rank
                ):
                    inputs = _inputs(tokens, topk=topk)
                    mapping, counts = _layout(384, copies=4, shift=11)
                    ref, got = _outputs(inputs), _outputs(inputs)
                    _run(inputs, ref, hash_routing)
                    _run(
                        inputs,
                        got,
                        hash_routing,
                        mapping=mapping,
                        counts=counts,
                        rank=rank,
                    )
                    self._assert_equivalent(ref, got, mapping, counts, rank)

    def test_nonfinite_fallback_is_mapped_without_changing_packed_values(self):
        mapping, counts = _layout(384, copies=2, shift=7)
        for hash_routing in (False, True):
            with self.subTest(hash=hash_routing):
                inputs = _inputs(17)
                inputs.x[0, 0] = float("nan")
                inputs.x[1, 128] = float("inf")
                inputs.scores[2, 2] = float("nan")
                inputs.scores[3, 3] = -float("inf")
                ref, got = _outputs(inputs), _outputs(inputs)
                _run(inputs, ref, hash_routing)
                _run(inputs, got, hash_routing, mapping=mapping, counts=counts, rank=3)
                self._assert_equivalent(ref, got, mapping, counts, 3)
                self.assertTrue(torch.isfinite(got.weights).all().item())

    def test_eplb_reuses_jit_across_token_counts_ranks_and_map_updates(self):
        token_counts = (
            1,
            2,
            15,
            16,
            17,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            257,
            1023,
            1024,
            1025,
            2047,
            2048,
            2049,
            8192,
        )
        layouts = [_layout(384, copies=n, shift=11 + n) for n in (1, 2, 4)]
        for hash_routing in (False, True):
            suffix = "hash" if hash_routing else "nonhash"
            kernel = getattr(_PACK, f"_mega_moe_gate_pack_{suffix}_kernel")
            mapping, counts = (tensor.clone() for tensor in layouts[0])
            # Only the existing BLOCK_M=2/4/8 buckets need separate kernels.
            for tokens in (1, 1024, 2048):
                inputs = _inputs(tokens, dim=5120)
                _run(inputs, _outputs(inputs), hash_routing, False, mapping, counts, 0)
            torch.cuda.synchronize()
            # A new specialization reaches _do_compile even on a disk-cache hit.
            # Fail there, while still launching the actual cached GPU kernels.
            with patch.object(
                kernel,
                "_do_compile",
                side_effect=AssertionError("Unexpected EPLB gate-pack JIT miss"),
            ):
                for tokens in token_counts:
                    inputs = _inputs(tokens, dim=5120)
                    for shared in (False, True):
                        out = _outputs(inputs, shared)
                        for index, rank in enumerate((0, 1, 3, 16, 31, 64, 71)):
                            with self.subTest(
                                hash=hash_routing,
                                tokens=tokens,
                                shared=shared,
                                rank=rank,
                            ):
                                source_map, source_count = layouts[index % len(layouts)]
                                if index % 2:
                                    mapping.copy_(source_map)
                                    counts.copy_(source_count)
                                else:
                                    mapping, counts = (
                                        source_map.clone(),
                                        source_count.clone(),
                                    )
                                _run(
                                    inputs,
                                    out,
                                    hash_routing,
                                    shared,
                                    mapping,
                                    counts,
                                    rank,
                                )
            torch.cuda.synchronize()

    def test_graph_replay_reads_inplace_map_updates_including_se(self):
        for shared in (False, True):
            for hash_routing in (False, True):
                with self.subTest(shared=shared, hash=hash_routing):
                    inputs = _inputs(65)
                    mapping, counts = _layout(384, copies=1, shift=7)
                    layouts = [_layout(384, copies=n, shift=11 + n) for n in (2, 4, 1)]
                    ref, got = _outputs(inputs, shared), _outputs(inputs, shared)
                    _run(inputs, ref, hash_routing, shared)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(2):
                            _run(inputs, got, hash_routing, shared, mapping, counts, 31)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        _run(inputs, got, hash_routing, shared, mapping, counts, 31)
                    tracked = (mapping, counts, got.indices, got.weights, got.x, got.sf)
                    if shared:
                        tracked += (got.shared_sf,)
                    addresses = tuple(t.data_ptr() for t in tracked)
                    previous = got.indices.clone()
                    for new_map, new_count in layouts:
                        mapping.copy_(new_map)
                        counts.copy_(new_count)
                        graph.replay()
                        self._assert_equivalent(ref, got, mapping, counts, 31, shared)
                        self.assertFalse(torch.equal(got.indices, previous))
                        previous = got.indices.clone()
                        self.assertEqual(
                            addresses, tuple(t.data_ptr() for t in tracked)
                        )

    def test_uint32_wrap_invalid_ids_and_padding(self):
        mapping, counts = _layout(384, copies=4, shift=19)
        rows = torch.tensor(
            [0, 1, 184071, 184072, 184073, 184074, 1 << 20, 1 << 32, 17],
            device="cuda",
            dtype=torch.int64,
        )
        ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, -1], device="cuda")
        for rank in (0, 3, 31):
            with self.subTest(rank=rank):
                out = torch.full((16,), -123, device="cuda", dtype=torch.int64)
                _map_test_kernel[(1,)](
                    ids,
                    rows,
                    mapping,
                    counts,
                    out,
                    rank,
                    size=ids.numel(),
                    map_stride=mapping.stride(0),
                    BLOCK=16,
                )
                expected = _torch_map(ids, mapping, counts, rank, rows)
                torch.testing.assert_close(out[: ids.numel()], expected, rtol=0, atol=0)
                self.assertTrue((out[ids.numel() :] == -123).all().item())


if __name__ == "__main__":
    unittest.main()
