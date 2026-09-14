"""DFlash mask/layout parity, paged writes and changing-length graph replay."""

import math
import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.dflash_attention import (
    DFlashCacheLayout,
    dflash_paged_attention,
    dflash_write_paged_kv,
    is_supported,
)


def _decode_cache(cache, layout):
    """Independent physical-page decoding to [pages,P,Hkv,D]."""
    pages, _, heads, width, dim = cache.shape
    key, value = cache[:, 0].contiguous(), cache[:, 1].contiguous()
    if layout != DFlashCacheLayout.CUDA:
        key = key.view(pages, heads, dim // 8, width, 8)
        key = key.permute(0, 1, 3, 2, 4).reshape(pages, heads, width, dim)
        if layout == DFlashCacheLayout.AITER_VECTOR:
            value = value.view(pages, heads, width // 8, dim, 8)
            value = value.permute(0, 1, 2, 4, 3).reshape(pages, heads, width, dim)
        else:
            value = value.view(pages, heads, dim, width).transpose(-1, -2)
    return key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)


def _mask(q_width, length, causal, window_size, device="cpu"):
    query_positions = torch.arange(q_width, device=device) + length - q_width
    key_positions = torch.arange(length, device=device)
    visible = query_positions[:, None] >= 0
    visible = visible.expand(q_width, length).clone()
    if causal:
        visible &= key_positions[None, :] <= query_positions[:, None]
    if window_size:
        visible &= key_positions[None, :] > query_positions[:, None] - window_size
        if not causal:
            visible &= key_positions[None, :] < query_positions[:, None] + window_size
    return visible


def _reference(query, cache, table, lengths, q_width, causal, window_size, layout):
    key_pages, value_pages = _decode_cache(cache, layout)
    heads, dim = query.shape[1:]
    repeat = heads // cache.shape[2]
    out = torch.zeros_like(query, dtype=torch.float32)
    for request, length in enumerate(lengths.cpu().tolist()):
        if not 0 < length <= table.shape[1] * cache.shape[3]:
            continue
        pages = table[request, : math.ceil(length / cache.shape[3])].long()
        key = key_pages.index_select(0, pages).flatten(0, 1)[:length]
        value = value_pages.index_select(0, pages).flatten(0, 1)[:length]
        key = key.float().transpose(0, 1).repeat_interleave(repeat, dim=0)
        value = value.float().transpose(0, 1).repeat_interleave(repeat, dim=0)
        q = query[request * q_width : (request + 1) * q_width].float().transpose(0, 1)
        scores = q @ key.transpose(-1, -2) / math.sqrt(dim)
        visible = _mask(q_width, length, causal, window_size, query.device)
        scores.masked_fill_(~visible, -float("inf"))
        # Rows preceding position zero can occur in padded/tiny-length fixtures.
        probs = scores.softmax(-1).nan_to_num_(0.0)
        out[request * q_width : (request + 1) * q_width] = (probs @ value).transpose(
            0, 1
        )
    return out


class DFlashAttentionContractTest(unittest.TestCase):
    def test_swa_has_exact_window_and_excludes_future_noise(self):
        for window in (2048, 4096):
            with self.subTest(window=window):
                length = window * 2
                position = length - 8
                mask = _mask(8, length, True, window)
                self.assertEqual(mask.sum(1).tolist(), [window] * 8)
                self.assertFalse(mask[0, position - window].item())
                self.assertTrue(mask[0, position - window + 1].item())
                self.assertTrue(mask[0, position].item())
                self.assertFalse(mask[0, position + 1].item())
                self.assertTrue(_mask(8, length, False, 0)[0, -1].item())

    def test_device_gate_rejects_cpu_without_device_reads(self):
        query = torch.empty(8, 16, 128, dtype=torch.bfloat16)
        cache = torch.empty(1, 2, 4, 16, 128, dtype=torch.bfloat16)
        self.assertFalse(is_supported(query, cache, 8))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA or ROCm GPU")
class DFlashAttentionGpuTest(unittest.TestCase):
    def _fixture(
        self, batch, q_width, capacity, heads=16, kv_heads=4, dim=128, page=16
    ):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(7103 + batch + q_width)
        query = torch.randn(
            batch * q_width,
            heads,
            dim,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        count = batch * math.ceil(capacity / page)
        cache = torch.randn(
            count,
            2,
            kv_heads,
            page,
            dim,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        table = torch.randperm(count, device=device, generator=generator).int()
        table = table.view(batch, -1)
        return query, cache, table

    def _assert_parity(
        self, query, cache, table, lengths, q_width, causal, window, layout, out=None
    ):
        actual = dflash_paged_attention(
            query,
            cache,
            table,
            lengths,
            q_width,
            causal=causal,
            window_size=window,
            cache_layout=layout,
            out=out,
        )
        expected = _reference(
            query, cache, table, lengths, q_width, causal, window, layout
        )
        torch.testing.assert_close(actual.float(), expected, atol=2e-3, rtol=2e-2)
        self.assertTrue(torch.isfinite(actual).all().item())
        return actual

    def test_layouts_and_mixed_lengths(self):
        for q_width in (8, 16):
            query, cache, table = self._fixture(4, q_width, 4096)
            lengths = torch.tensor(
                [2047, 2048, 2049, 4096], device="cuda", dtype=torch.int32
            )
            for layout in DFlashCacheLayout:
                for causal, window in ((False, 0), (True, 2048)):
                    with self.subTest(q_width=q_width, layout=layout, causal=causal):
                        self._assert_parity(
                            query,
                            cache,
                            table,
                            lengths,
                            q_width,
                            causal,
                            window,
                            layout,
                        )

    def test_batch_query_and_head_geometry(self):
        for batch, q_width in ((1, 5), (2, 6), (4, 7), (8, 8), (16, 16), (32, 8)):
            with self.subTest(batch=batch, q_width=q_width):
                query, cache, table = self._fixture(
                    batch, q_width, 256, heads=32, kv_heads=8
                )
                lengths = torch.full((batch,), 129, device="cuda", dtype=torch.int32)
                self._assert_parity(
                    query,
                    cache,
                    table,
                    lengths,
                    q_width,
                    True,
                    2048,
                    DFlashCacheLayout.CUDA,
                )

    def test_supported_head_dims_and_page_sizes(self):
        for dim, page in ((64, 8), (128, 32), (256, 16)):
            query, cache, table = self._fixture(
                1, 8, 64, heads=6, kv_heads=2, dim=dim, page=page
            )
            lengths = torch.tensor([33], device="cuda", dtype=torch.int32)
            for layout in DFlashCacheLayout:
                with self.subTest(dim=dim, page=page, layout=layout):
                    self._assert_parity(
                        query, cache, table, lengths, 8, True, 16, layout
                    )

    def test_noncontiguous_query_and_tiny_lengths(self):
        query, cache, table = self._fixture(4, 8, 32)
        packed = torch.randn(32, 24, 128, device="cuda", dtype=torch.bfloat16)
        query = packed[:, :16]
        lengths = torch.tensor([0, 1, 8, 17], device="cuda", dtype=torch.int32)
        for layout in DFlashCacheLayout:
            self._assert_parity(query, cache, table, lengths, 8, True, 2048, layout)

    def test_paged_write_all_layouts_and_padding(self):
        _, cache, table = self._fixture(2, 8, 64)
        requests = torch.tensor([0, 0, 0, 1, 1, -1], device="cuda", dtype=torch.int32)
        positions = torch.tensor(
            [15, 16, 17, 0, 63, -1], device="cuda", dtype=torch.int32
        )
        packed = torch.randn(6, 8, 128, device="cuda", dtype=torch.bfloat16)
        key, value = packed[:, :4], packed[:, 4:]
        for layout in DFlashCacheLayout:
            cache.fill_(3.0)
            dflash_write_paged_kv(
                key, value, cache, table, requests, positions, cache_layout=layout
            )
            key_pages, value_pages = _decode_cache(cache, layout)
            expected_k = torch.full_like(key_pages, 3.0)
            expected_v = torch.full_like(value_pages, 3.0)
            for row, (request, position) in enumerate(
                zip(requests.cpu().tolist(), positions.cpu().tolist())
            ):
                if request >= 0:
                    page = table[request, position // 16].item()
                    expected_k[page, position % 16] = key[row]
                    expected_v[page, position % 16] = value[row]
            torch.testing.assert_close(key_pages, expected_k, atol=0, rtol=0)
            torch.testing.assert_close(value_pages, expected_v, atol=0, rtol=0)

    def test_graph_growth_shrink_and_page_remapping(self):
        q_width = 8
        query, cache, table = self._fixture(4, q_width, 8192)
        lengths = torch.full((4,), 4096, device="cuda", dtype=torch.int32)
        out = torch.empty_like(query)
        for layout in DFlashCacheLayout:
            for causal, window in ((False, 0), (True, 2048), (True, 4096)):
                for _ in range(3):
                    dflash_paged_attention(
                        query,
                        cache,
                        table,
                        lengths,
                        q_width,
                        causal=causal,
                        window_size=window,
                        cache_layout=layout,
                        out=out,
                    )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    dflash_paged_attention(
                        query,
                        cache,
                        table,
                        lengths,
                        q_width,
                        causal=causal,
                        window_size=window,
                        cache_layout=layout,
                        out=out,
                    )
                pointer = out.data_ptr()
                for sizes in (
                    [0, 8, 2048, 4096],
                    [8192, 2049, 17, 0],
                    [1, 4096, 8191, 8],
                ):
                    lengths.copy_(torch.tensor(sizes, device="cuda", dtype=torch.int32))
                    table.copy_(table.flip(0))
                    query.normal_()
                    graph.replay()
                    torch.cuda.synchronize()
                    expected = _reference(
                        query, cache, table, lengths, q_width, causal, window, layout
                    )
                    torch.testing.assert_close(
                        out.float(), expected, atol=2e-3, rtol=2e-2
                    )
                    self.assertEqual(out.data_ptr(), pointer)

    def test_query_write_graph_after_partial_acceptance(self):
        # Rejected query rows remain physically present. A shorter accepted
        # prefix followed by query replacement must produce the cropped-cache
        # result without ever making old future scratch visible.
        query, cache, table = self._fixture(2, 8, 64)
        key = torch.randn(16, 4, 128, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        requests = torch.arange(2, device="cuda", dtype=torch.int32).repeat_interleave(
            8
        )
        positions = torch.empty(16, device="cuda", dtype=torch.int32)
        lengths = torch.empty(2, device="cuda", dtype=torch.int32)
        out = torch.empty_like(query)
        for layout in DFlashCacheLayout:
            prefixes = torch.tensor([16, 16], device="cuda", dtype=torch.int32)
            positions.copy_(
                (prefixes[:, None] + torch.arange(8, device="cuda")).flatten()
            )
            lengths.copy_(prefixes + 8)

            def forward():
                dflash_write_paged_kv(
                    key, value, cache, table, requests, positions, cache_layout=layout
                )
                dflash_paged_attention(
                    query,
                    cache,
                    table,
                    lengths,
                    8,
                    causal=False,
                    cache_layout=layout,
                    out=out,
                )

            for _ in range(3):
                forward()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                forward()
            for accepted in ([0, 7], [3, 0], [7, 3]):
                key.normal_()
                value.normal_()
                positions.copy_(
                    (prefixes[:, None] + torch.arange(8, device="cuda")).flatten()
                )
                lengths.copy_(prefixes + 8)
                graph.replay()
                expected = _reference(query, cache, table, lengths, 8, False, 0, layout)
                torch.testing.assert_close(out.float(), expected, atol=2e-3, rtol=2e-2)
                # Commit target features in the same dense positions, including
                # rejected rows, just as the fixed-width engine commit does.
                key.normal_()
                value.normal_()
                dflash_write_paged_kv(
                    key, value, cache, table, requests, positions, cache_layout=layout
                )
                prefixes += torch.tensor(accepted, device="cuda", dtype=torch.int32) + 1

    def test_unsupported_layout_and_dtype(self):
        query, cache, table = self._fixture(1, 8, 32)
        self.assertFalse(is_supported(query.float(), cache, 8))
        self.assertFalse(is_supported(query, cache, 9))
        self.assertFalse(is_supported(query, cache, 8, cache_layout=7))
        self.assertFalse(is_supported(query[:, :15], cache, 8))


if __name__ == "__main__":
    unittest.main()
