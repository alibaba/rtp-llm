"""CPU forward contracts using the production class and explicit attention reference.

Only backend kernels, collectives and cache writers are doubled. These tests do
not claim CUDA stream safety or backend numerical equivalence.
"""

import itertools
import unittest
import weakref
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha import (
    allgather_cp_impl as impl,
)


def attention(q, k, v, qptr, kptr, causal):
    outputs, lses = [], []
    for row in range(len(qptr) - 1):
        qi = q[int(qptr[row]) : int(qptr[row + 1])].float()
        ki = k[int(kptr[row]) : int(kptr[row + 1])].float()
        vi = v[int(kptr[row]) : int(kptr[row + 1])].float()
        ki = ki.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
        vi = vi.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
        if not ki.shape[0]:
            outputs.append(torch.zeros_like(qi))
            lses.append(torch.full(qi.shape[:2], -torch.inf))
            continue
        scores = torch.einsum("thd,shd->hts", qi, ki) * q.shape[-1] ** -0.5
        if causal:
            mask = torch.arange(ki.shape[0])[None, :] > (
                torch.arange(qi.shape[0])[:, None] + ki.shape[0] - qi.shape[0]
            )
            scores.masked_fill_(mask[None, :, :], -torch.inf)
        outputs.append(torch.einsum("hts,shd->thd", scores.softmax(-1), vi))
        lses.append(torch.logsumexp(scores, -1).T)
    return torch.cat(outputs).to(torch.bfloat16), torch.cat(lses)


def merge_state(v_a, s_a, v_b, s_b):
    s = torch.logaddexp(s_a, s_b)
    v = (
        v_a.float() * (s_a - s).exp()[..., None]
        + v_b.float() * (s_b - s).exp()[..., None]
    )
    return v.to(v_a.dtype), s


class Harness:
    def __init__(self, prefix, fa4, sharded, cache_dtype, chunks, cp_rank, padded):
        self.events = []
        self.gather_refs = []
        self.restore_refs = []
        self.pool_ref = None
        self.prefix_storage = None
        self.prefix_live = None
        self.first_ragged_pool_live = None
        self.prefix_variant = "normal"
        self.part_refs = []
        self.q_layouts = []
        self.part_q_source_ptrs = []
        self.op = o = impl.PCPAllGatherAttnOp.__new__(impl.PCPAllGatherAttnOp)
        o.__dict__.update(
            _use_forward_opt=False,
            head_dim=8,
            num_qo_heads=4,
            num_kv_heads=2,
            prefill_cp_size=4,
            _kv_sharded=sharded,
            _cp_size=4 if sharded else 1,
            _cp_rank=cp_rank if sharded else 0,
            seq_size_per_block=4,
            has_prefix=prefix,
            _fa4_prefix=fa4,
            _fa4_no_prefix=fa4,
            attn_configs=NS(kv_cache_dtype=cache_dtype),
        )
        t = sum(chunks)
        o.cu_seqlens = torch.tensor([0] + list(itertools.accumulate(chunks)))
        o.qo_indptr = o.cu_seqlens // 2
        o.kv_indptr_part0 = o.qo_indptr * (cp_rank + 1)
        o.kv_indptr_part1 = o.qo_indptr * (8 - cp_rank)
        q0, q1, kv0, kv1, restore = [], [], [], [], []
        offset = 0
        global_offset = 0
        for chunk in chunks:
            q0.extend(range(offset, offset + chunk // 2))
            q1.extend(range(offset + chunk // 2, offset + chunk))
            kv0.extend(range(global_offset, global_offset + chunk // 2 * (cp_rank + 1)))
            kv1.extend(range(global_offset, global_offset + chunk // 2 * (8 - cp_rank)))
            restore.extend(
                range(global_offset, global_offset + chunk * 4 - (1 if padded else 0))
            )
            offset += chunk
            global_offset += chunk * 4
        o.q0_idx, o.q1_idx, o.kv0_idx, o.kv1_idx, o.kv_restore_unpad_indices = [
            torch.tensor(x, dtype=torch.long) for x in [q0, q1, kv0, kv1, restore]
        ]
        # Mixed zero/nonzero prefixes exercises an empty prefix member in a hit batch.
        lengths = [0 if i % 2 else 3 for i in range(len(chunks))]
        o.attn_inputs = NS(prefix_lengths=torch.tensor(lengths))
        self.prefix_ptr = torch.tensor([0] + list(itertools.accumulate(lengths)))
        gen = torch.Generator().manual_seed(431)
        self.prefix_k = torch.randn(sum(lengths), 2, 8, generator=gen).bfloat16()
        self.prefix_v = torch.randn(sum(lengths), 2, 8, generator=gen).bfloat16()
        self.qkv = torch.randn(t, 64, generator=gen).bfloat16()
        self.cache = NS(kv_cache_base=torch.zeros(t * 4, 2, 2, 4, 8).to(cache_dtype))
        self.params = NS(
            batch_indice_d=torch.zeros(len(restore), dtype=torch.int32),
            positions_d=torch.arange(len(restore), dtype=torch.int32),
            page_indice_d=torch.arange(t * 4),
            decode_page_indptr_d=torch.tensor([0, t * 4]),
            paged_kv_last_page_len_d=torch.tensor([4]),
        )
        o._physical_block_table = lambda: torch.arange(t * 4).view(1, -1)
        o._run_fa4_paged_prefix = lambda q, pool, ptr: self.prefix(q, pool, fa4=True)
        prefix_runner = lambda q, pool, return_lse: self.prefix(q, pool, fa4=False)
        o.prefill_wrappers = {"paged": {"prefix": NS(run=prefix_runner)}}
        o._run_ragged_part = lambda part, q, k, v, return_lse=False: self.ragged(
            q,
            k,
            v,
            o.qo_indptr,
            o.kv_indptr_part0 if part == "part0" else o.kv_indptr_part1,
            return_lse,
            False,
        )
        o._run_fa4_ragged = lambda q, k, v, qp, kp, return_lse=False: self.ragged(
            q, k, v, qp, kp, return_lse, True
        )

    def all_gather(self, value, group):
        result = value.repeat(4, 1)
        self.gather_refs.append(weakref.ref(result))
        return result

    def write(self, k, v, pool, slots):
        self.events.append((k.clone(), v.clone(), slots.clone()))
        self.restore_refs.extend([weakref.ref(k), weakref.ref(v)])
        valid = slots >= 0
        # Mutate the shared backing cache, making cache parity a value assertion.
        flat = pool.view(-1)
        values = torch.cat([k[valid].flatten(), v[valid].flatten()]).to(pool.dtype)
        flat[: values.numel()] = values

    def append(self, **kw):
        self.write(
            kw["append_key"],
            kw["append_value"],
            kw["paged_kv_cache"],
            kw["positions"].long(),
        )

    def gather_pool(self, pool, *args, **kwargs):
        out = pool.clone()
        self.pool_ref = weakref.ref(out)
        return out

    def prefix(self, q, pool, fa4):
        self.prefix_live = (
            [r() is not None for r in self.gather_refs],
            [r() is not None for r in self.restore_refs],
        )
        self.q_layouts.append((q.is_contiguous(), q.stride(-1), q.data_ptr()))
        dtype = self.cache.kv_cache_base.dtype if fa4 else q.dtype
        out, lse = attention(
            q.to(dtype),
            self.prefix_k.to(dtype),
            self.prefix_v.to(dtype),
            self.op.cu_seqlens,
            self.prefix_ptr,
            False,
        )
        if self.prefix_variant == "noncontiguous":
            out = out.transpose(0, 1).contiguous().transpose(0, 1)
        elif self.prefix_variant == "extra_row":
            out = torch.cat([out, torch.zeros_like(out[:1])])
        elif self.prefix_variant == "float32":
            out = out.float()
        self.prefix_storage = out.data_ptr()
        return out, lse

    def ragged(self, q, k, v, qp, kp, return_lse, fa4):
        if self.part_refs:
            assert all(
                ref() is None for ref in self.part_refs
            ), "previous part tensors still alive"
        self.part_refs = [weakref.ref(tensor) for tensor in (q, k, v)]
        assert q.is_contiguous(), "ragged Q must be materialized by index_select"
        if self.pool_ref is not None and self.first_ragged_pool_live is None:
            self.first_ragged_pool_live = self.pool_ref() is not None
        if fa4:
            q = q.to(k.dtype)
        out, lse = attention(q, k, v, qp, kp, True)
        return (out, lse) if return_lse else out

    def reference(self):
        """Compute both parts without mutation or the production scheduling loop."""
        o = self.op
        q, k, v = self.qkv.split([32, 16, 16], dim=-1)
        q = q.reshape(-1, 4, 8)
        k, v = (x.repeat(4, 1).reshape(-1, 2, 8) for x in (k, v))
        result = torch.empty_like(q)
        if o.has_prefix:
            dtype = self.cache.kv_cache_base.dtype if o._fa4_prefix else q.dtype
            prefix, prefix_lse = attention(
                q.to(dtype),
                self.prefix_k.to(dtype),
                self.prefix_v.to(dtype),
                o.cu_seqlens,
                self.prefix_ptr,
                False,
            )
        for qi, ki, kp in [
            (o.q0_idx, o.kv0_idx, o.kv_indptr_part0),
            (o.q1_idx, o.kv1_idx, o.kv_indptr_part1),
        ]:
            dtype = self.cache.kv_cache_base.dtype if o._fa4_no_prefix else q.dtype
            value, lse = attention(
                q[qi].to(dtype), k[ki].to(dtype), v[ki].to(dtype), o.qo_indptr, kp, True
            )
            if o.has_prefix:
                value, _ = merge_state(prefix[qi], prefix_lse[qi], value, lse)
            result[qi] = value
        indices = o.kv_restore_unpad_indices
        slots = self.params.positions_d.long()
        if o._kv_sharded:
            slots = torch.where((slots // 4) % 4 == o._cp_rank, slots, -1)
        dtype = self.qkv.dtype if o._kv_sharded else self.cache.kv_cache_base.dtype
        return result, (k[indices].to(dtype), v[indices].to(dtype), slots)

    def run(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _cp_slot_mapping
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        def slots(positions, bt, batch, page, eb, ratio, cp, rank, **kw):
            return torch.where((positions // page) % cp == rank, positions, -1)

        def merge(v_a, s_a, v_b, s_b):
            value, lse = merge_state(v_a, s_a, v_b, s_b)
            # The synthetic float32 prefix tests only storage selection. Keep
            # merged values in BF16 so indexed assignment obeys the caller API.
            return value.to(v_b.dtype), lse

        index_select = torch.index_select

        def select(tensor, dim, index):
            if tensor.ndim == 3 and tensor.shape[1] == self.op.num_qo_heads:
                self.part_q_source_ptrs.append(tensor.data_ptr())
            return index_select(tensor, dim, index)

        with ExitStack() as stack:
            stack.enter_context(patch.object(torch, "index_select", select))
            for name, replacement in dict(
                all_gather=self.all_gather,
                cast_kv_for_cache_append=lambda k, v, cache, dtype: (
                    k.to(dtype),
                    v.to(dtype),
                ),
                append_paged_kv_cache=self.append,
                fill_fp8_kv_cache_scale=lambda *a, **kw: None,
                gather_cp_sharded_prefix_pool=self.gather_pool,
                merge_state=merge,
            ).items():
                stack.enter_context(patch.object(impl, name, replacement))
            stack.enter_context(
                patch.object(_cp_slot_mapping, "cp_kv_slot_mapping", slots)
            )
            stack.enter_context(
                patch.object(rtp_llm_ops, "mha_kv_write_cache", self.write, create=True)
            )
            return self.op.forward(self.qkv, self.cache, self.params)


class TestDenseForwardCpu(unittest.TestCase):
    def test_forward_values_cache_and_lifetimes(self):
        for case in itertools.product(
            [False, True],
            [False, True],
            [False, True],
            [torch.bfloat16, torch.float8_e4m3fn],
            [[4], [4, 2, 6]],
            [0, 3],
            [False, True],
        ):
            with self.subTest(case=case):
                h = Harness(*case)
                expected, writes = h.reference()
                original = h.qkv.clone()
                actual = h.run()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(h.qkv, original, rtol=0, atol=0)
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(len(h.events), 1)
                for observed, wanted in zip(h.events[0], writes):
                    torch.testing.assert_close(
                        observed.float(), wanted.float(), rtol=0, atol=0
                    )
                # Assert cache contents as well as the write arguments.
                key, value, slots = writes
                live = slots >= 0
                packed = torch.cat([key[live].flatten(), value[live].flatten()])
                cached = h.cache.kv_cache_base.flatten()
                torch.testing.assert_close(
                    cached[: packed.numel()].float(),
                    packed.to(cached.dtype).float(),
                    rtol=0,
                    atol=0,
                )
                self.assertEqual(cached[packed.numel() :].float().count_nonzero(), 0)
                self.assertTrue(all(ref() is None for ref in h.gather_refs))
                self.assertTrue(all(ref() is None for ref in h.part_refs))
                self.assertEqual(len(h.part_q_source_ptrs), 2)
                if not h.op.has_prefix or h.op._fa4_prefix:
                    self.assertEqual(h.part_q_source_ptrs, [h.qkv.data_ptr()] * 2)
                if h.op.has_prefix:
                    self.assertEqual(actual.data_ptr(), h.prefix_storage)
                    self.assertFalse(
                        any(h.prefix_live[1]), "cache append buffers still alive"
                    )
                    contiguous, last_stride, ptr = h.q_layouts[0]
                    self.assertEqual(last_stride, 1)
                    if h.op._fa4_prefix:
                        self.assertFalse(contiguous)
                        self.assertEqual(ptr, h.qkv.data_ptr())
                    else:
                        self.assertTrue(contiguous)
                        self.assertNotEqual(ptr, h.qkv.data_ptr())
                    if h.op._kv_sharded:
                        self.assertFalse(h.first_ragged_pool_live)
                        self.assertIsNone(h.pool_ref())

    def test_prefix_storage_fallback(self):
        for variant, fa4 in itertools.product(
            ["noncontiguous", "extra_row", "float32"], [False, True]
        ):
            with self.subTest(variant=variant, fa4=fa4):
                h = Harness(True, fa4, True, torch.bfloat16, [4, 2, 6], 3, True)
                h.prefix_variant = variant
                expected, _ = h.reference()
                actual = h.run()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, torch.bfloat16)
                self.assertTrue(actual.is_contiguous())
                self.assertNotEqual(actual.data_ptr(), h.prefix_storage)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
