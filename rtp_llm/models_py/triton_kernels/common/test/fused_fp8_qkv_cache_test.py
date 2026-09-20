import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.fused_fp8_qkv_cache import (
    fused_fp8_qkv_cache,
    is_query_supported,
    is_supported,
    quantize_fp8_query,
)


def _quantize_reference(tensor, scale):
    return (tensor.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)


class FusedFP8QKVCacheTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        if torch.version.hip is not None or torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("Native NVIDIA E4M3FN conversion requires SM89 or newer")
        self.device = torch.device("cuda")
        torch.manual_seed(1769)

    def _case(
        self,
        lengths,
        prefixes,
        *,
        padded_rows=0,
        q_heads=8,
        kv_heads=2,
        page_size=64,
        strided=False,
        scales=(1.0, 1.0, 1.0),
    ):
        head_dim = 256
        rows = sum(lengths) + padded_rows
        width = (q_heads + 2 * kv_heads) * head_dim
        qkv = torch.randn(
            (rows, width + (128 if strided else 0)),
            dtype=torch.bfloat16,
            device=self.device,
        )[:, :width]
        pages = len(lengths) * 4
        page_span = 2 * kv_heads * page_size * head_dim
        physical_span = page_span + (1536 if strided else 0)
        storage = torch.full(
            (pages, physical_span), 13, device=self.device, dtype=torch.float32
        ).to(torch.float8_e4m3fn)
        cache = storage.as_strided(
            (pages, 2, kv_heads, page_size, head_dim),
            (
                physical_span,
                kv_heads * page_size * head_dim,
                page_size * head_dim,
                head_dim,
                1,
            ),
        )
        table_storage = torch.full(
            (len(lengths), 8), -1, device=self.device, dtype=torch.int32
        )
        table = table_storage[:, ::2]
        table.copy_(
            torch.arange(pages - 1, -1, -1, device=self.device, dtype=torch.int32).view(
                len(lengths), 4
            )
        )
        cu = torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            device=self.device,
            dtype=torch.int32,
        )
        prefix = torch.tensor(prefixes, device=self.device, dtype=torch.int32)
        out = torch.full(
            (rows, q_heads, head_dim), 19, device=self.device, dtype=torch.float32
        ).to(torch.float8_e4m3fn)
        kwargs = dict(
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            max_query_len=max(lengths),
            q_scale=scales[0],
            k_scale=scales[1],
            v_scale=scales[2],
            out=out,
        )
        return dict(
            qkv=qkv,
            cache=cache,
            storage=storage,
            table=table,
            cu=cu,
            prefix=prefix,
            kwargs=kwargs,
        )

    def _reference(self, case):
        qkv = case["qkv"]
        kwargs = case["kwargs"]
        q_dim = kwargs["num_q_heads"] * kwargs["head_dim"]
        kv_dim = kwargs["num_kv_heads"] * kwargs["head_dim"]
        ref_q = torch.zeros_like(kwargs["out"])
        ref_storage = case["storage"].clone()
        ref_cache = ref_storage.as_strided(case["cache"].shape, case["cache"].stride())
        cu = case["cu"].tolist()
        prefixes = case["prefix"].tolist()
        table = case["table"].tolist()
        for batch, prefix in enumerate(prefixes):
            for row in range(cu[batch], cu[batch + 1]):
                ref_q[row].copy_(
                    _quantize_reference(qkv[row, :q_dim], kwargs["q_scale"]).view_as(
                        ref_q[row]
                    )
                )
                position = prefix + row - cu[batch]
                page_col, offset = divmod(position, kwargs["page_size"])
                if position < 0 or page_col >= len(table[batch]):
                    continue
                page = table[batch][page_col]
                if page < 0 or page >= ref_cache.shape[0]:
                    continue
                for component, scale in enumerate(
                    (kwargs["k_scale"], kwargs["v_scale"])
                ):
                    begin = q_dim + component * kv_dim
                    ref_cache[page, component, :, offset, :].copy_(
                        _quantize_reference(
                            qkv[row, begin : begin + kv_dim], scale
                        ).view(kwargs["num_kv_heads"], kwargs["head_dim"])
                    )
            length = cu[batch + 1] - cu[batch]
            end_pos = prefix + length
            if length > 0 and end_pos > 0 and end_pos % kwargs["page_size"]:
                col, offset = divmod(end_pos, kwargs["page_size"])
                if col < len(table[batch]):
                    page = table[batch][col]
                    if 0 <= page < ref_cache.shape[0]:
                        ref_cache[page, 1, :, offset:, :].zero_()
        return ref_q, ref_storage

    def _run(self, case):
        return fused_fp8_qkv_cache(
            case["qkv"],
            case["cache"],
            case["table"],
            case["cu"],
            case["prefix"],
            **case["kwargs"],
        )

    def _assert_result(self, case, reference):
        ref_q, ref_storage = reference
        # Byte equality also catches overflow-to-NaN and negative-zero errors.
        self.assertTrue(
            torch.equal(
                case["kwargs"]["out"].view(torch.uint8), ref_q.view(torch.uint8)
            ),
            "Quantized Q differs from the saturating FP32 reference",
        )
        self.assertTrue(
            torch.equal(
                case["storage"].view(torch.uint8), ref_storage.view(torch.uint8)
            ),
            "Cache contents, unused pages or physical page padding were corrupted",
        )

    def test_prefill_page64_head256_crosses_pages_with_prefix(self):
        case = self._case([129, 7, 65], [60, 127, 2])
        reference = self._reference(case)
        self.assertIs(self._run(case), case["kwargs"]["out"])
        self._assert_result(case, reference)

    def test_invalid_activation_nan_is_not_hidden_by_saturation(self):
        case = self._case([1], [0])
        q_dim = 8 * 256
        case["qkv"][0, 0] = float("nan")
        case["qkv"][0, 1] = float("inf")
        case["qkv"][0, 2] = -float("inf")
        case["qkv"][0, q_dim] = float("nan")
        case["qkv"][0, q_dim + 2 * 256] = float("nan")
        output = self._run(case).float().flatten()
        page = int(case["table"][0, 0].item())
        self.assertTrue(torch.isnan(output[0]).item())
        self.assertEqual(output[1].item(), 448.0)
        self.assertEqual(output[2].item(), -448.0)
        self.assertTrue(torch.isnan(case["cache"][page, 0, 0, 0, 0].float()).item())
        self.assertTrue(torch.isnan(case["cache"][page, 1, 0, 0, 0].float()).item())
        decoded = quantize_fp8_query(case["qkv"][:, :q_dim]).float().flatten()
        self.assertTrue(torch.isnan(decoded[0]).item())
        torch.testing.assert_close(decoded[1:3], output[1:3], rtol=0, atol=0)

    def test_hybrid_physical_page_and_packed_qkv_strides(self):
        case = self._case([67, 3], [63, 126], strided=True)
        reference = self._reference(case)
        self._run(case)
        self._assert_result(case, reference)

    def test_decode_query_quantization_clears_only_unused_value_tail(self):
        for ndim in (2, 3):
            with self.subTest(ndim=ndim):
                case = self._case(
                    [1, 1, 1, 0], [64, 127, 66, 0], padded_rows=1, strided=True
                )
                case["storage"].view(torch.uint8).fill_(127)  # E4M3 NaN
                query = case["qkv"][:, :2048]
                if ndim == 3:
                    query = query.view(4, 8, 256)
                seq_lens = torch.tensor(
                    [65, 128, 67, 0], device=self.device, dtype=torch.int32
                )
                expected = case["storage"].clone()
                expected_cache = expected.as_strided(
                    case["cache"].shape, case["cache"].stride()
                )
                table = case["table"].tolist()
                for batch, length in enumerate(seq_lens.tolist()):
                    if length and length % 64:
                        expected_cache[
                            table[batch][length // 64], 1, :, length % 64 :, :
                        ].zero_()
                output = quantize_fp8_query(
                    query,
                    scale=0.5,
                    kv_cache=case["cache"],
                    block_table=case["table"],
                    seq_lens=seq_lens,
                )
                self.assertTrue(
                    torch.equal(
                        output.view(torch.uint8),
                        _quantize_reference(query, 0.5).view(torch.uint8),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        case["storage"].view(torch.uint8), expected.view(torch.uint8)
                    )
                )

    def test_qwen35_tensor_parallel_head_counts(self):
        for q_heads, kv_heads in ((8, 1), (16, 2), (32, 4)):
            with self.subTest(q_heads=q_heads, kv_heads=kv_heads):
                case = self._case([5], [62], q_heads=q_heads, kv_heads=kv_heads)
                reference = self._reference(case)
                self._run(case)
                self._assert_result(case, reference)

    def test_mtp_five_tokens_zero_length_entries_and_padded_tail(self):
        case = self._case([5, 0, 1, 5, 0], [62, 0, 127, 125, 0], padded_rows=14)
        reference = self._reference(case)
        self._run(case)
        self._assert_result(case, reference)

    def test_all_empty_queries_zero_entire_padded_output(self):
        case = self._case([0, 0, 0], [0, 0, 0], padded_rows=17)
        reference = self._reference(case)
        self._run(case)
        self._assert_result(case, reference)

    def test_nonunit_scales_and_saturation(self):
        case = self._case([5, 1], [63, 129], scales=(0.5, 0.3, 2.0))
        extremes = torch.tensor(
            [-8192, -449, -448, -0.0, 0.0, 448, 449, 8192],
            device=self.device,
            dtype=torch.bfloat16,
        )
        case["qkv"][:, :8].copy_(extremes)
        for start in (2048, 2560):
            case["qkv"][:, start : start + 8].copy_(extremes)
        reference = self._reference(case)
        self._run(case)
        self._assert_result(case, reference)
        self.assertTrue(torch.isfinite(case["kwargs"]["out"].float()).all().item())

    def test_invalid_page_ids_and_out_of_range_position_do_not_write(self):
        case = self._case([5, 5, 5], [63, 63, 256])
        case["table"][0].fill_(-1)
        case["table"][1].fill_(case["cache"].shape[0])
        reference = self._reference(case)
        self._run(case)
        self._assert_result(case, reference)

    def test_cuda_graph_replay_uses_updated_metadata(self):
        case = self._case([5, 0, 3], [63, 0, 126], padded_rows=7, strided=True)
        # Warm the exact specialization before capture, including wrapper gate.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._run(case)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._run(case)
        case["qkv"].normal_()
        case["cu"].copy_(
            torch.tensor([0, 1, 6, 6], device=self.device, dtype=torch.int32)
        )
        case["prefix"].copy_(
            torch.tensor([63, 125, 0], device=self.device, dtype=torch.int32)
        )
        case["table"].copy_(case["table"].flip(0))
        case["storage"].zero_()
        reference = self._reference(case)
        graph.replay()
        self._assert_result(case, reference)

    def test_strided_query_conversion_2d_and_3d(self):
        for shape in ((7, 2048), (7, 8, 256)):
            with self.subTest(shape=shape):
                if len(shape) == 2:
                    query = torch.randn(
                        (7, 2560), device=self.device, dtype=torch.bfloat16
                    )[:, :2048]
                else:
                    query = torch.randn(
                        (7, 16, 256), device=self.device, dtype=torch.bfloat16
                    )[:, ::2]
                query[0].fill_(8192)
                output = quantize_fp8_query(query, scale=0.25)
                expected = _quantize_reference(query, 0.25)
                self.assertEqual(tuple(output.shape), shape)
                self.assertTrue(output.is_contiguous())
                self.assertTrue(
                    torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
                )

    def test_support_gate_rejects_invalid_layout_or_scale(self):
        case = self._case([1], [0])
        args = (
            case["qkv"],
            case["cache"],
            case["table"],
            case["cu"],
            case["prefix"],
        )
        self.assertTrue(is_supported(*args, **case["kwargs"]))
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            kwargs = dict(case["kwargs"], q_scale=scale)
            self.assertFalse(is_supported(*args, **kwargs))
            with self.assertRaises(ValueError):
                fused_fp8_qkv_cache(*args, **kwargs)
        self.assertFalse(is_query_supported(case["qkv"].float()))
        self.assertFalse(is_query_supported(case["qkv"][:, ::2]))


if __name__ == "__main__":
    unittest.main()
