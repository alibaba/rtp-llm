"""BF16 Indexer score contracts for prefill tiles and unchanged fallbacks."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score


@unittest.skipUnless(torch.cuda.is_available(), "CUDA component test")
class TestIndexerScoreTriton(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3101)
        self.tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.tf32

    def _inputs(self, batch, queries, keys, heads=32):
        q = (torch.randn(batch, queries, heads, 128, device="cuda") * 0.05).bfloat16()
        k = (torch.randn(batch, keys, 128, device="cuda") * 0.05).bfloat16()
        weights = torch.randn(batch, queries, heads * 2, device="cuda")
        weights.div_(64)
        weights = weights[..., ::2]
        qp = (
            (torch.arange(queries, device="cuda").flip(0) * 7 - 5)
            .expand(batch, -1)
            .clone()
        )
        kp = (torch.randperm(keys, device="cuda") * 3).expand(batch, -1).contiguous()
        qr = ((torch.arange(queries, device="cuda") // 32) % 2).expand(batch, -1)
        kr = ((torch.arange(keys, device="cuda") // 128) % 2).expand(batch, -1)
        return q, k, weights, qp.long(), kp.int(), qr.int(), kr.int()

    def _reference(self, q, k, weights, qp=None, kp=None, qr=None, kr=None, ratio=1):
        expected = torch.zeros(
            q.shape[:2] + (k.shape[1],), dtype=torch.float32, device="cuda"
        )
        # Independent FP32 matmuls, with no tiled masks or Triton scheduling.
        for head in range(q.shape[2]):
            dot = q[:, :, head].float() @ k.float().transpose(1, 2)
            expected += dot.relu() * weights[:, :, head, None].float()
        valid = torch.ones_like(expected, dtype=torch.bool)
        if qp is not None:
            if kp is None:
                kp = torch.arange(k.shape[1], device="cuda")[None]
            valid &= kp[:, None] < (qp[:, :, None] + 1) // ratio
        if qr is not None:
            valid &= qr[:, :, None] == kr[:, None]
        return expected.masked_fill(~valid, -float("inf"))

    def _assert_score(self, actual, expected):
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_prefill_global_positions_request_masks_and_tail_tiles(self):
        for batch, queries, keys, ratio, bf16_weights in [
            (1, 32, 128, 1, False),
            (2, 33, 129, 1, False),
            (2, 65, 257, 3, True),
        ]:
            with self.subTest(batch=batch, queries=queries, keys=keys, ratio=ratio):
                q, k, w, qp, kp, qr, kr = self._inputs(batch, queries, keys)
                if bf16_weights:
                    w = w.bfloat16()
                # The first full query tile is causal-empty; another tile
                # has live keys. Include request-empty and partial tail tiles.
                qp[:, :32] = -1
                qp[:, 32:] = 4096
                if queries > 33:
                    qr[:, -1] = 99
                storage = torch.full((batch, queries, keys + 7), 123.0, device="cuda")
                out = storage[..., :keys]
                actual = v4_indexer_score(
                    q,
                    k,
                    w,
                    qp,
                    ratio,
                    kv_positions=kp,
                    query_requests=qr,
                    key_requests=kr,
                    out=out,
                )
                self.assertEqual(actual.data_ptr(), out.data_ptr())
                self._assert_score(
                    actual, self._reference(q, k, w, qp, kp, qr, kr, ratio)
                )
                self.assertTrue(torch.all(storage[..., keys:] == 123.0))
                self.assertTrue(torch.isneginf(actual[:, :32]).all())

    def test_prefill_without_global_keys_or_request_mask(self):
        q, k, w, qp, *_ = self._inputs(1, 65, 257)
        actual = v4_indexer_score(q, k, w, qp, compress_ratio=2)
        self._assert_score(actual, self._reference(q, k, w, qp, ratio=2))

    def test_all_masked_tiles_write_over_existing_output(self):
        q, k, w, qp, kp, qr, kr = self._inputs(1, 65, 257)
        out = torch.full((1, 65, 257), 123.0, device="cuda")
        qp.fill_(-1)
        actual = v4_indexer_score(q, k, w, qp, kv_positions=kp, out=out)
        self.assertTrue(torch.isneginf(actual).all())
        qp.fill_(4096)
        qr.fill_(99)
        actual = v4_indexer_score(
            q,
            k,
            w,
            qp,
            kv_positions=kp,
            query_requests=qr,
            key_requests=kr,
            out=out,
        )
        self.assertTrue(torch.isneginf(actual).all())

    def test_short_rows_other_heads_and_decode_fallbacks(self):
        for queries, keys, heads, masked in [
            (7, 13, 32, True),
            (33, 129, 64, True),
            (1, 257, 32, False),
            (65, 257, 32, False),
        ]:
            with self.subTest(queries=queries, keys=keys, heads=heads, masked=masked):
                q, k, w, qp, kp, qr, kr = self._inputs(1, queries, keys, heads)
                if masked:
                    actual = v4_indexer_score(
                        q,
                        k,
                        w,
                        qp,
                        kv_positions=kp,
                        query_requests=qr,
                        key_requests=kr,
                    )
                    expected = self._reference(q, k, w, qp, kp, qr, kr)
                else:
                    actual = v4_indexer_score(
                        q, k, w, query_requests=qr, key_requests=kr
                    )
                    expected = self._reference(q, k, w, qr=qr, kr=kr)
                self._assert_score(actual, expected)

    def test_empty_inputs_preserve_output_shape(self):
        for queries, keys in [(0, 129), (65, 0)]:
            with self.subTest(queries=queries, keys=keys):
                q, k, w, qp, kp, qr, kr = self._inputs(1, queries, keys)
                out = torch.empty(1, queries, keys, device="cuda")
                actual = v4_indexer_score(
                    q,
                    k,
                    w,
                    qp,
                    kv_positions=kp,
                    query_requests=qr,
                    key_requests=kr,
                    out=out,
                )
                self.assertEqual(actual.shape, (1, queries, keys))
                self.assertEqual(actual.data_ptr(), out.data_ptr())

    def test_graph_replay_rechecks_positions_and_requests(self):
        q, k, w, qp, kp, qr, kr = self._inputs(1, 65, 257)
        qp = qp.int()
        qr.zero_()
        kr.zero_()
        qp.fill_(-1)
        out = torch.empty(1, 65, 257, device="cuda")

        def run():
            return v4_indexer_score(
                q,
                k,
                w,
                qp,
                kv_positions=kp,
                query_requests=qr,
                key_requests=kr,
                out=out,
            )

        for _ in range(10):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for position, request in [(4096, 0), (-1, 0), (4096, 99), (4096, 0)]:
            qp.fill_(position)
            qr.fill_(request)
            graph.replay()
            self._assert_score(out, self._reference(q, k, w, qp, kp, qr, kr))


if __name__ == "__main__":
    unittest.main()
