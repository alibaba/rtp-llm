import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class KVCacheWriteCudaGraphTest(unittest.TestCase):
    def _check_padded_replay(self, dtype, pages_per_request=1, dim=64):
        batch_size, tokens, heads, page_size = 4, 5, 2, 16
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        block_ids = torch.arange(
            1, batch_size * pages_per_request + 1, dtype=torch.int32
        ).view(batch_size, pages_per_request)

        def plan(active, prefixes, forbid_realloc):
            params.fill_params(
                torch.tensor(prefixes, dtype=torch.int32),
                torch.empty(0, dtype=torch.int32),
                torch.tensor(
                    [tokens] * active + [0] * (batch_size - active), dtype=torch.int32
                ),
                block_ids,
                page_size,
                forbid_realloc,
            )

        cache = torch.full(
            (batch_size * pages_per_request + 2, 2, heads, page_size, dim),
            7.0,
            device="cuda",
            dtype=dtype,
        )
        # Non-contiguous QKV slices are also used by the attention implementation.
        kv = torch.arange(batch_size * tokens * heads * dim * 2, device="cuda").reshape(
            batch_size * tokens, heads, dim * 2
        )
        kv = (kv % 11).to(dtype)
        key, value = kv[..., :dim], kv[..., dim:]
        writer = KVCacheWriteOp(heads, dim, physical_page_size=page_size)
        writer.set_params(params)
        layer_cache = SimpleNamespace(kv_cache_base=cache)
        plan(batch_size, [pages_per_request * page_size - tokens] * batch_size, False)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            writer.forward(key, value, layer_cache, token_indptr=params.qo_indptr_d)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            writer.forward(key, value, layer_cache, token_indptr=params.qo_indptr_d)

        cases = [
            (3, [0, 3, 7, 0]),
            (2, [6, 10, 0, 0]),
            (0, [0] * 4),
            (4, [1, 2, 3, 4]),
        ]
        if pages_per_request > 1:
            cases += [(3, [14, 20, 27, 0]), (1, [15, 0, 0, 0])]
        for active, prefixes in cases:
            with self.subTest(dtype=dtype, active=active):
                cache.fill_(7)
                expected = cache.float().clone()
                for row in range(active):
                    start = row * tokens
                    for token in range(tokens):
                        pos = prefixes[row] + token
                        page_id = int(block_ids[row, pos // page_size])
                        expected[page_id, 0, :, pos % page_size] = key[
                            start + token
                        ].float()
                        expected[page_id, 1, :, pos % page_size] = value[
                            start + token
                        ].float()
                plan(active, prefixes, True)
                graph.replay()
                # Includes pages belonging to inactive requests and all prefix rows.
                torch.testing.assert_close(cache.float(), expected, rtol=0, atol=0)

    def test_bfloat16_padding_does_not_write_cached_pages(self):
        self._check_padded_replay(torch.bfloat16)

    def test_fp8_padding_does_not_write_cached_pages(self):
        self._check_padded_replay(torch.float8_e4m3fn)

    def test_bfloat16_replay_across_page_boundary(self):
        self._check_padded_replay(torch.bfloat16, pages_per_request=2, dim=256)

    def test_fp8_replay_across_page_boundary(self):
        self._check_padded_replay(torch.float8_e4m3fn, pages_per_request=2, dim=256)


if __name__ == "__main__":
    unittest.main()
