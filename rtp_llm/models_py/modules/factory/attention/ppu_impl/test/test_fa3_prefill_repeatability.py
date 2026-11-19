"""Real PPU paged-prefill repeats over strided HND caches."""

import math
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.modules.factory.attention.ppu_impl.fa3_mha import (
    FA3PrefillPagedAttnOp,
)
from rtp_llm.ops import AttentionConfigs


@unittest.skipUnless(get_device_type() == DeviceType.Ppu, "requires PPU")
class PpuFa3PrefillRepeatabilityTest(unittest.TestCase):
    HEADS, KV_HEADS, DIM, PAGE = 16, 8, 128, 64

    def _case(self, length, batch, **mask):
        torch.manual_seed(20261102 + length + batch)
        pages = math.ceil(length / self.PAGE)
        cache = torch.randn(
            1 + batch * pages,
            2,
            self.KV_HEADS,
            self.PAGE,
            self.DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q = torch.randn(
            batch * length,
            self.HEADS,
            self.DIM,
            dtype=cache.dtype,
            device="cuda",
        )
        table = torch.arange(1, 1 + batch * pages, dtype=torch.int32).reshape(
            batch, pages
        )
        config = AttentionConfigs()
        config.head_num, config.kv_head_num = self.HEADS, self.KV_HEADS
        config.size_per_head, config.kernel_tokens_per_block = self.DIM, self.PAGE
        config.max_seq_len = 2048
        op = FA3PrefillPagedAttnOp(config, SimpleNamespace(is_cuda_graph=False), **mask)
        lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
        cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * length
        op.fa3_params = SimpleNamespace(
            cache_seqlens=lengths,
            block_table=table.cuda(),
            cu_seqlens_q=cu,
            max_seqlen_q=length,
            max_seqlen_k=length,
            scheduler_metadata=op._get_scheduler_metadata(
                batch, length, length, lengths, cu, q.dtype
            ),
        )
        return op, q, cache, table

    def _reference(self, q, cache, table, length):
        expected, positions = [], []
        for batch, block_ids in enumerate(table):
            selected = cache[block_ids.long().cuda()].cpu().float()
            k, v = [
                selected[:, plane]
                .permute(1, 0, 2, 3)
                .reshape(self.KV_HEADS, -1, self.DIM)[:, :length]
                .repeat_interleave(self.HEADS // self.KV_HEADS, dim=0)
                for plane in (0, 1)
            ]
            for offset in (0, length // 2, length - 1):
                position = batch * length + offset
                query = q[position].cpu().float()
                scores = torch.einsum("hd,hld->hl", query, k[:, : offset + 1])
                scores *= self.DIM**-0.5
                expected.append(
                    torch.einsum("hl,hld->hd", scores.softmax(-1), v[:, : offset + 1])
                )
                positions.append(position)
        return torch.stack(expected), positions

    def test_repeats_ignore_unused_pages_and_partial_page_padding(self):
        for length, batch in ((500, 1), (499, 4), (512, 1), (500, 4)):
            with self.subTest(length=length, batch=batch):
                op, q, cache, table = self._case(length, batch)
                expected, positions = self._reference(q, cache, table, length)
                first = None
                for poison in (0.0, 17.0, -29.0, 1000.0, 0.0, 17.0):
                    cache[0].fill_(poison)
                    if length % self.PAGE:
                        cache[
                            table[:, -1].long().cuda(), :, :, length % self.PAGE :, :
                        ] = poison
                    output = op.forward(q, SimpleNamespace(kv_cache_base=cache))
                    torch.cuda.synchronize()
                    actual = output.cpu()
                    if first is None:
                        first = actual
                        torch.testing.assert_close(
                            actual[positions].float(), expected, atol=1e-2, rtol=1e-2
                        )
                    else:
                        self.assertTrue(
                            torch.equal(actual, first), "prefill output changed"
                        )

    def test_draft_masks_over_paged_prefix_and_query_suffix(self):
        # The 8 proposal queries follow a 67-token prefix, across a page boundary.
        for causal, window_size in ((False, (-1, -1)), (True, (15, 0))):
            klen, qlen = 75, 8
            op, q, cache, table = self._case(
                klen, 1, causal=causal, window_size=window_size
            )
            q = q[-qlen:]
            params = op.fa3_params
            params.max_seqlen_q = qlen
            params.cu_seqlens_q = torch.tensor(
                [0, qlen], device=q.device, dtype=torch.int32
            )
            params.scheduler_metadata = op._get_scheduler_metadata(
                1, qlen, klen, params.cache_seqlens, params.cu_seqlens_q, q.dtype
            )
            actual = op.forward(q, SimpleNamespace(kv_cache_base=cache))
            pages = cache[table[0].long().to(q.device)].float()

            def dense(plane):
                return (
                    pages[:, plane]
                    .permute(0, 2, 1, 3)
                    .reshape(-1, self.KV_HEADS, self.DIM)[:klen]
                    .repeat_interleave(self.HEADS // self.KV_HEADS, dim=1)
                )

            k, v = dense(0), dense(1)
            for j in range(qlen):
                pos = klen - qlen + j
                lo, hi = (
                    (max(0, pos - window_size[0]), pos + 1) if causal else (0, klen)
                )
                score = (
                    torch.einsum("hd,thd->ht", q[j].float(), k[lo:hi]) * self.DIM**-0.5
                )
                expected = torch.einsum("ht,thd->hd", score.softmax(-1), v[lo:hi])
                torch.testing.assert_close(
                    actual[j].float(), expected, atol=1e-2, rtol=1e-2
                )

    def test_draft_graph_refreshes_both_masks_after_request_pages_change(self):
        from rtp_llm.models_py.model_desc.qwen3_dspark_model import _PpuDraftAttention
        from rtp_llm.ops import RopeStyle
        from rtp_llm.ops.compute_ops import PyAttentionInputs, get_typemeta

        # Two requests keep the same graph bucket but exchange physical pages.
        # The dense oracle uses only the current request's valid K/V (including
        # the 8 query slots); neither the FA3 scheduler nor its page map is reused.
        qlen, batch = 8, 2
        config = AttentionConfigs()
        config.head_num, config.kv_head_num = self.HEADS, self.KV_HEADS
        config.size_per_head, config.kernel_tokens_per_block = self.DIM, self.PAGE
        config.max_seq_len, config.need_rope_kv_cache = 256, False
        config.rope_config.style = RopeStyle.No
        a = PyAttentionInputs()
        a.is_prefill = a.is_target_verify = a.is_cuda_graph = True
        a.dtype = get_typemeta(torch.empty(1, dtype=torch.bfloat16))
        a.input_lengths = torch.full((batch,), qlen, dtype=torch.int32).pin_memory()
        a.prefix_lengths = torch.tensor([65, 90], dtype=torch.int32).pin_memory()
        a.sequence_lengths = torch.empty(0, dtype=torch.int32).pin_memory()
        a.cu_seqlens = torch.tensor(
            [0, qlen, batch * qlen], dtype=torch.int32
        ).pin_memory()
        a.cu_seqlens_device = a.cu_seqlens.cuda()
        a.kv_cache_kernel_block_id = torch.tensor(
            [[1, 2], [3, 4]], dtype=torch.int32
        ).pin_memory()
        a.kv_cache_kernel_block_id_device = a.kv_cache_kernel_block_id.cuda()
        torch.manual_seed(53)
        q = torch.randn(
            batch * qlen, self.HEADS, self.DIM, device="cuda", dtype=torch.bfloat16
        )
        cache = torch.randn(
            5, 2, self.KV_HEADS, self.PAGE, self.DIM, device="cuda", dtype=q.dtype
        )
        wrapped = SimpleNamespace(kv_cache_base=cache)
        plans = _PpuDraftAttention(config, a, (16, None))
        for idx in range(2):
            plans.forward(q, wrapped, idx)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = [plans.forward(q, wrapped, idx) for idx in range(2)]
        for prefixes, pages in [
            ([94, 57], [[3, 4], [1, 2]]),
            ([65, 87], [[2, 1], [4, 3]]),
        ]:
            a.prefix_lengths.copy_(torch.tensor(prefixes))
            a.kv_cache_kernel_block_id.copy_(torch.tensor(pages))
            a.kv_cache_kernel_block_id_device.copy_(a.kv_cache_kernel_block_id)
            cache.normal_()
            q.normal_()
            plans.prepare_cuda_graph(a)
            graph.replay()
            torch.cuda.synchronize()
            for b, prefix in enumerate(prefixes):
                length = prefix + qlen
                own = cache[pages[b]].float().cpu()
                k, v = [
                    own[:, plane]
                    .permute(0, 2, 1, 3)
                    .reshape(-1, self.KV_HEADS, self.DIM)[:length]
                    .repeat_interleave(self.HEADS // self.KV_HEADS, dim=1)
                    for plane in (0, 1)
                ]
                for j in range(qlen):
                    query = q[b * qlen + j].float().cpu()
                    for mask, (lo, hi) in enumerate(
                        ((max(0, prefix + j - 15), prefix + j + 1), (0, length))
                    ):
                        scores = (
                            torch.einsum("hd,thd->ht", query, k[lo:hi]) * self.DIM**-0.5
                        )
                        expected = torch.einsum(
                            "ht,thd->hd", scores.softmax(-1), v[lo:hi]
                        )
                        torch.testing.assert_close(
                            captured[mask][b * qlen + j].float().cpu(),
                            expected,
                            atol=1e-2,
                            rtol=1e-2,
                        )

    def test_graph_replay_matches_eager_after_query_updates(self):
        op, q, cache, _ = self._case(500, 1)
        wrapped = SimpleNamespace(kv_cache_base=cache)
        for _ in range(3):
            op.forward(q, wrapped)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op.forward(q, wrapped)
        for repeat in range(4):
            fresh = torch.randn_like(q)
            eager = op.forward(fresh, wrapped)
            torch.cuda.synchronize()
            expected = eager.cpu()
            q.copy_(fresh)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(captured.cpu(), expected), f"replay {repeat}")


if __name__ == "__main__":
    unittest.main()
