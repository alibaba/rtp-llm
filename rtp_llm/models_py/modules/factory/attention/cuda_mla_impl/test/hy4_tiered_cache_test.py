"""HY4 packed KV equivalence across HBM, pinned backing and graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.pinned_mla_cache import (
    PinnedMlaWorkingSet,
)


def packed_cache(tokens, seed=731):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.empty((tokens, 656), device="cuda", dtype=torch.uint8)
    values = torch.randn((tokens, 512), generator=generator, device="cuda")
    packed[:, :512] = values.to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.exp2(
        torch.randint(-3, 4, (tokens, 4), generator=generator, device="cuda").float()
    )
    packed[:, 512:528] = scales.view(torch.uint8)
    rope = torch.randn((tokens, 64), generator=generator, device="cuda").bfloat16()
    packed[:, 528:] = rope.view(torch.uint8)
    return packed.view(-1, 64, 656)


def dequantize(packed):
    rows = packed.reshape(-1, 656)
    values = rows[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = (
        rows[:, 512:528].contiguous().view(torch.float32).repeat_interleave(128, dim=1)
    )
    rope = rows[:, 528:].contiguous().view(torch.bfloat16)
    return torch.cat(((values * scales).bfloat16(), rope), dim=1)


class Hy4TieredCacheTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.fail("CUDA is required for tiered cache validation")

    def make_working_set(self, references, capacity=2048, hbm_tokens=256):
        host = [ref[hbm_tokens // 64 :].cpu().pin_memory() for ref in references]
        resident = [
            torch.empty(
                ((hbm_tokens + capacity) // 64, 64, 656),
                device=ref.device,
                dtype=ref.dtype,
            )
            for ref in references
        ]
        for target, source in zip(resident, references):
            target[: hbm_tokens // 64].copy_(source[: hbm_tokens // 64])
        return PinnedMlaWorkingSet(
            host,
            capacity,
            64,
            references[0].device,
            hbm_tokens=hbm_tokens,
            hbm_cache=resident,
        )

    def test_long_and_cp_local_prefill_exact_with_permuted_blocks(self):
        for global_length in (10000, 32768, 65536, 100000):
            for cp in (1, 8):
                length = (global_length + cp - 1) // cp
                with self.subTest(global_length=global_length, cp=cp):
                    tokens = ((length + 63) // 64 + 4) * 64
                    reference = packed_cache(tokens)
                    working = self.make_working_set([reference])
                    pages = (length + 63) // 64
                    # Include both tiers and a zero-length middle request. Bounds
                    # are ragged and request block tables deliberately differ.
                    table = (
                        torch.stack(
                            (
                                torch.arange(pages),
                                torch.arange(pages),
                                torch.arange(pages).flip(0),
                            )
                        )
                        .int()
                        .cuda()
                    )
                    lengths = torch.tensor(
                        [length, 0, max(1, length // 3)],
                        device="cuda",
                        dtype=torch.int32,
                    )
                    starts = torch.tensor(
                        [0, length, length], device="cuda", dtype=torch.int32
                    )
                    out = torch.empty(
                        (length + max(1, length // 3), 576),
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    working.gather_bf16(0, out, table, lengths, starts)
                    logical = []
                    for req, size in ((0, length), (2, max(1, length // 3))):
                        positions = torch.arange(size, device="cuda")
                        logical.append(
                            table[req, positions // 64].long() * 64 + positions % 64
                        )
                    expected = dequantize(reference)[torch.cat(logical)]
                    torch.testing.assert_close(out, expected, atol=0, rtol=0)
                    self.assertFalse(working.started)

    def test_cp_empty_query_rank_still_publishes_kv_before_history_gather(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
            flashmla_sparse_cp_impl as cp,
        )

        reference = packed_cache(4096)
        working = self.make_working_set([reference])
        slots = torch.tensor([3, 270, -1, 1000], device="cuda", dtype=torch.int64)
        updates = reference.flatten(0, 1)[[7, 8, 9, 10]].clone()
        expected = reference.clone()
        expected.flatten(0, 1)[slots[[0, 1, 3]]] = updates[[0, 1, 3]]
        cache = SimpleNamespace(kv_cache_base=working.backing[0])
        op = cp.SparseMlaFp8CPOp.__new__(cp.SparseMlaFp8CPOp)
        op.kv_restore_unpad_indices = torch.arange(4, device="cuda")
        op.kv_cache_sharded = True
        op.sharded_slot_mapping = slots
        op.mla_params = SimpleNamespace(slot_mapping=slots)
        op.attn_inputs = SimpleNamespace()
        op.write_cache_store_impl = None
        op.total_local_ids = torch.empty(0, device="cuda", dtype=torch.int64)
        op._gather = object()
        op._allocate_fused_kv = Mock(return_value=torch.empty(0, device="cuda"))
        events = []

        def write(ckv, rope, target, params, slot_mapping_override):
            torch.testing.assert_close(
                slot_mapping_override, torch.arange(4, device="cuda"), atol=0, rtol=0
            )
            target.kv_cache_base.flatten(0, 1).copy_(updates)

        def published(*args):
            actual = torch.cat((working.resident[0][:4], working.backing[0].cuda()))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            events.append("published")

        def gather(cache_arg, out, pinned_cache):
            self.assertIs(pinned_cache[0], working)
            self.assertEqual(events, ["published"])
            events.append("gathered")

        op.kv_cache_write_op = SimpleNamespace(forward=write)
        op._gather_sharded_kv_cache = gather
        with patch.object(
            cp, "all_gather", side_effect=lambda x, **kwargs: x
        ), patch.object(cp.common, "apply_write_cache_store", side_effect=published):
            result = op.forward(
                None,
                torch.zeros(4, 512, device="cuda"),
                torch.zeros(4, 64, device="cuda"),
                topk=None,
                batch_indice_d=None,
                kv_cache=cache,
                pinned_cache=(working, 0),
            )
        self.assertIsNone(result)
        self.assertEqual(events, ["published", "gathered"])

    def test_decode_prefill_transition_and_graph_write_through(self):
        references = [packed_cache(4096, seed=seed) for seed in (731, 732)]
        working = self.make_working_set(references)
        ids = torch.tensor([[3, 270, 1000, 270, -1]], device="cuda", dtype=torch.int32)
        slots = torch.tensor([3, 270, 1000, -1], device="cuda", dtype=torch.int64)
        updates = [ref.flatten(0, 1)[[7, 8, 9, 10]].clone() for ref in references]
        table = torch.arange(64, device="cuda", dtype=torch.int32).view(1, -1)
        lengths = torch.tensor([4096], device="cuda", dtype=torch.int32)
        starts = torch.zeros(1, device="cuda", dtype=torch.int32)
        gathered = [
            torch.empty((4096, 576), device="cuda", dtype=torch.bfloat16)
            for _ in references
        ]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        def execute():
            working.begin(ids)
            for layer in range(2):
                working.write(layer, slots, updates[layer])
                working.gather_bf16(layer, gathered[layer], table, lengths, starts)

        with torch.cuda.stream(stream):
            execute()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            execute()
        for replay in range(4):
            for layer in range(2):
                updates[layer][:, 528:].fill_(replay)
                references[layer].flatten(0, 1)[slots[:3].long()] = updates[layer][:3]
            graph.replay()
            torch.cuda.synchronize()
            for layer in range(2):
                torch.testing.assert_close(
                    gathered[layer], dequantize(references[layer]), atol=0, rtol=0
                )
                physical = working.physical_indices
                actual = working.resident[layer].flatten(0, 1)[physical[0, :4].long()]
                expected = references[layer].flatten(0, 1)[ids[0, :4].long()]
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
