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

    def _case(self, length, batch):
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
        op = FA3PrefillPagedAttnOp(config, SimpleNamespace(is_cuda_graph=False))
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
