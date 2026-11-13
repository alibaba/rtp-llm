import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models_py.modules.qwen4_exp.gdn_decode_triton import (
    _run_packed_decode,
    try_packed_decode,
)
from rtp_llm.models_py.triton_kernels.fla.block import store_ssm_state_to_block_map
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule_fwd
from rtp_llm.models_py.triton_kernels.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_fwd,
)
from rtp_llm.models_py.triton_kernels.fla.gdn_gating import fused_gdn_gating
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd


class GdnSm90DispatchTest(unittest.TestCase):
    def test_cp_prefill_is_enabled_by_default(self):
        from rtp_llm.models_py.model_desc.qwen3_next import (
            Qwen3NextGatedDeltaNetPrefill,
        )
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
            Qwen4Sm120GdnPrefill,
        )

        with mock.patch.object(
            Qwen3NextGatedDeltaNetPrefill, "__init__", return_value=None
        ):
            candidate = Qwen4Sm120GdnPrefill()
        self.assertTrue(candidate.sm90_cp_prefill)
        self.assertTrue(candidate.sm90_grouped_prepare)
        output = object()
        helper = mock.Mock(return_value=output)
        packed = SimpleNamespace(is_cuda=True, device="cuda:0")
        package = "rtp_llm.models_py.modules.qwen4_exp"
        with (
            mock.patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            mock.patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=78),
            ),
            mock.patch(
                f"{package}.gdn_prefill_adapter._sm90_cp_available", return_value=True
            ),
            mock.patch.dict(
                "sys.modules",
                {
                    f"{package}.gdn_prefill_cp_sm90": SimpleNamespace(
                        try_sm90_prefill=helper
                    )
                },
            ),
        ):
            actual = candidate._fla(packed, None, None, None, 64, None, None)
        self.assertIs(actual, output)
        helper.assert_called_once_with(
            candidate, packed, None, None, None, 64, None, None
        )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "requires SM90",
)
class GdnSm90CandidateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261112)

    def assertBytesEqual(self, value, expected):
        self.assertTrue(
            torch.equal(
                value.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def test_inference_output_projection_reads_current_weight(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
            CudaF16Linear,
        )
        from rtp_llm.models_py.modules.qwen4_exp.gdn_projection import Sm90GdnLinear

        with torch.inference_mode():
            weight = (
                torch.randint(-3, 4, (3072, 2560), device="cuda").to(torch.bfloat16)
                * 0.125
            )
        native = CudaF16Linear(weight)
        candidate = Sm90GdnLinear(native)
        value = (
            torch.randint(-3, 4, (8192, 3072), device="cuda").to(torch.bfloat16) * 0.125
        )
        self.assertTrue(candidate.weight.is_inference())
        self.assertEqual(candidate.weight.data_ptr(), native.weight.data_ptr())
        self.assertBytesEqual(candidate(value), native(value))
        self.assertIsNone(candidate._packed)
        self.assertEqual(candidate._plans, {})
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = candidate(value)
        for _ in range(3):
            with torch.inference_mode():
                native.weight.add_(0.125)
            graph.replay()
            self.assertBytesEqual(result, native(value))
            self.assertBytesEqual(candidate(value), native(value))
        graph.reset()
        with torch.inference_mode():
            candidate.weight = candidate.weight.clone()
        self.assertBytesEqual(candidate(value), native(value))

    def test_grouped_value_preparation_preserves_rounding_and_tail_guards(self):
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_adapter import (
            _sm90_cp_available,
        )

        if not _sm90_cp_available():
            self.skipTest("requires the qualified FlashInfer CP dependencies")
        from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_cp_sm90 import (
            _copy_value_alpha_beta,
            _prepare_cp,
        )

        for rows in (1, 3, 17, 2049):
            with self.subTest(rows=rows):
                packed = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
                projected = torch.randn(rows, 8240, device="cuda", dtype=torch.bfloat16)
                a, b = projected[:, -48:].split(24, dim=-1)
                a[0].fill_(30)
                b[0, ::2].fill_(-30)
                alog = torch.randn(24, device="cuda", dtype=torch.float32)
                bias = torch.randn(24, device="cuda", dtype=torch.bfloat16)
                saved = (packed.clone(), projected.clone())
                outputs = []
                for tokens, heads in ((16, 1), (4, 8)):
                    value = torch.full(
                        (rows + 2, 24, 128),
                        7,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    alpha = torch.full(
                        (rows + 2, 24), 7, device="cuda", dtype=torch.float32
                    )
                    beta = torch.full_like(alpha, 7)
                    _copy_value_alpha_beta[
                        ((rows + tokens - 1) // tokens, 24 // heads)
                    ](
                        packed,
                        a,
                        b,
                        alog,
                        bias,
                        value[1:-1],
                        alpha[1:-1],
                        beta[1:-1],
                        rows,
                        a.stride(0),
                        8,
                        24,
                        TOKENS=tokens,
                        HEADS=heads,
                        num_warps=4,
                    )
                    for tensor in (value, alpha, beta):
                        self.assertTrue(torch.all(tensor[0] == 7).item())
                        self.assertTrue(torch.all(tensor[-1] == 7).item())
                    outputs.append((value, alpha, beta))
                for actual, expected in zip(outputs[1], outputs[0]):
                    self.assertBytesEqual(actual, expected)
                self.assertBytesEqual(packed, saved[0])
                self.assertBytesEqual(projected, saved[1])

        rows = 32768
        packed = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
        projected = torch.randn(rows, 8240, device="cuda", dtype=torch.bfloat16)
        a, b = projected[:, -48:].split(24, dim=-1)
        expected = _prepare_cp(packed, a, b, alog, bias, 8, 24)
        actual = _prepare_cp(packed, a, b, alog, bias, 8, 24, group_heads=True)
        for value, reference in zip(actual, expected):
            self.assertBytesEqual(value, reference)

    def test_decode_pages_null_padding_and_strided_pool(self):
        batch = 8
        for page in (64, 128):
            with self.subTest(page=page):
                qkv = torch.randn(batch, 5120, device="cuda", dtype=torch.bfloat16)
                projection = torch.randn(
                    batch, 8240, device="cuda", dtype=torch.bfloat16
                )
                b, a = projection[:, -48:].split(24, dim=-1)
                alog = torch.randn(24, device="cuda", dtype=torch.float32)
                bias = torch.randn(24, device="cuda", dtype=torch.bfloat16)
                lengths = torch.tensor(
                    [0, 2, page, page + 1, page + 2, 2 * page, 2 * page + 1, 2],
                    device="cuda",
                    dtype=torch.int32,
                )
                mapping = torch.arange(
                    1, batch * 3 + 1, device="cuda", dtype=torch.int32
                ).view(batch, 3)
                mapping[-1, 0] = 0
                raw = (
                    torch.randn(
                        batch * 3 + 1,
                        24 * 128 * 128 + 512,
                        device="cuda",
                        dtype=torch.float32,
                    )
                    * 0.1
                )
                original = raw.clone()
                state = raw[:, : 24 * 128 * 128].view(batch * 3 + 1, 24, 128, 128)
                q, k, v = qkv.view(batch, 1, 40, 128).split((8, 8, 24), dim=2)
                g, beta = fused_gdn_gating(alog, a, b, bias)
                expected, _ = fused_recurrent_gated_delta_rule_fwd(
                    q,
                    k,
                    v,
                    g.view(batch, 1, 24),
                    beta.view(batch, 1, 24),
                    128**-0.5,
                    state,
                    block_map=mapping,
                    seq_size_per_block=page,
                    sequence_lengths=lengths,
                    use_qk_l2norm_in_kernel=True,
                )
                expected_state = raw.clone()
                raw.copy_(original)
                result = try_packed_decode(
                    qkv, a, b, alog, bias, state, mapping, lengths, page
                )
                self.assertIsNotNone(result)
                self.assertBytesEqual(result, expected)
                self.assertBytesEqual(raw, expected_state)
                padded = torch.empty(batch, 5128, device="cuda", dtype=torch.bfloat16)[
                    :, :5120
                ]
                self.assertIsNone(
                    try_packed_decode(
                        padded, a, b, alog, bias, state, mapping, lengths, page
                    )
                )
                self.assertIsNone(
                    try_packed_decode(
                        qkv.half(), a, b, alog, bias, state, mapping, lengths, page
                    )
                )
                self.assertIsNone(
                    try_packed_decode(
                        qkv, a, b, alog, bias, state, mapping, lengths, 32
                    )
                )

    def test_decode_small_batch_value_tile(self):
        for batch in (16, 32):
            with self.subTest(batch=batch):
                qkv = torch.randn(batch, 5120, device="cuda", dtype=torch.bfloat16)
                a = torch.randn(batch, 24, device="cuda", dtype=torch.bfloat16)
                b = torch.randn_like(a)
                alog = torch.randn(24, device="cuda", dtype=torch.float32)
                bias = torch.randn(24, device="cuda", dtype=torch.bfloat16)
                lengths = torch.full((batch,), 65, device="cuda", dtype=torch.int32)
                mapping = torch.arange(
                    1, 2 * batch + 1, device="cuda", dtype=torch.int32
                ).view(batch, 2)
                states = (
                    torch.randn(
                        2 * batch + 1, 24, 128, 128, device="cuda", dtype=torch.float32
                    )
                    * 0.05
                )
                initial = states.clone()
                expected = _run_packed_decode(
                    qkv, a, b, alog, bias, states, mapping, lengths, 64, value_tile=8
                )
                expected_state = states.clone()
                states.copy_(initial)
                output = try_packed_decode(
                    qkv, a, b, alog, bias, states, mapping, lengths, 64
                )
                self.assertBytesEqual(output, expected)
                self.assertBytesEqual(states, expected_state)

    def test_prefill_checkpoint_tail_and_prefix(self):
        for page, rows, prefix in (
            (64, 1, 0),
            (64, 64, 0),
            (64, 65, 0),
            (128, 63, 0),
            (128, 128, 0),
            (128, 129, 0),
            (64, 2049, 128),
            (128, 2049, 256),
        ):
            with self.subTest(page=page, rows=rows, prefix=prefix):
                q = l2norm_fwd(
                    torch.randn(1, rows, 8, 128, device="cuda", dtype=torch.bfloat16)
                )
                k = l2norm_fwd(torch.randn_like(q))
                v = (
                    torch.randn(1, rows, 24, 128, device="cuda", dtype=torch.bfloat16)
                    * 0.1
                )
                g = -torch.rand(1, rows, 24, device="cuda", dtype=torch.float32) * 0.1
                beta = torch.rand(1, rows, 24, device="cuda", dtype=torch.bfloat16)
                initial = (
                    torch.randn(1, 24, 128, 128, device="cuda", dtype=torch.float32)
                    * 0.1
                )
                cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
                prefixes = torch.tensor([prefix], device="cuda", dtype=torch.int32)
                count = (prefix + rows + page - 1) // page + 1
                mapping = torch.arange(
                    count, 0, -1, device="cuda", dtype=torch.int32
                ).view(1, count)
                states = (
                    torch.randn(
                        count + 1, 24, 128, 128, device="cuda", dtype=torch.float32
                    )
                    * 0.1
                )
                original = states.clone()
                _, expected, _, final, _, h, _ = chunk_gated_delta_rule_fwd(
                    q, k, v, g, beta, 128**-0.5, initial, True, cu
                )
                store_ssm_state_to_block_map(
                    h, final, prefixes, cu, mapping, states, page, 64
                )
                if page == 64 and rows > 64:
                    first_page = mapping[0, prefix // page]
                    self.assertBytesEqual(states[first_page], h[0, 1])
                expected_state, expected_final = states.clone(), final.clone()
                states.copy_(original)
                _, result, _, final, _, _, _ = chunk_gated_delta_rule_fwd(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    128**-0.5,
                    initial,
                    True,
                    cu,
                    page_state_store=(states, mapping, prefixes, page),
                )
                self.assertBytesEqual(result, expected)
                self.assertBytesEqual(final, expected_final)
                self.assertBytesEqual(states, expected_state)

    def test_projection_graph_streams_and_weight_refresh(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
            CudaF16Linear,
        )
        from rtp_llm.models_py.modules.qwen4_exp.gdn_projection import (
            Sm90GdnLinear,
            _library,
            _prefill_projection,
        )

        for columns, inner in ((8240, 2560), (2560, 3072)):
            weight = (
                torch.randint(-3, 4, (inner, columns), device="cuda").to(torch.bfloat16)
                * 0.125
            )
            native = CudaF16Linear(weight)
            candidate = Sm90GdnLinear(native)
            for rows in (16, 32, 64, 8192, 32768):
                value = (
                    torch.randint(-3, 4, (rows, inner), device="cuda").to(
                        torch.bfloat16
                    )
                    * 0.125
                )
                expected = native(value)
                self.assertBytesEqual(candidate(value), expected)
                h20 = torch.cuda.get_device_properties().multi_processor_count == 78
                if rows == 8192 and h20:
                    storage = torch.empty(
                        rows * inner + 1, device="cuda", dtype=torch.bfloat16
                    )
                    unaligned = storage[1:].view(rows, inner)
                    unaligned.copy_(value)
                    selected = (
                        candidate._packed if columns == 2560 else candidate.weight
                    )
                    uncached = torch.empty_like(expected)
                    _prefill_projection[(rows // 64 * ((columns + 127) // 128),)](
                        unaligned,
                        selected,
                        uncached,
                        columns,
                        inner,
                        *selected.stride(),
                        num_warps=4,
                        num_stages=3,
                    )
                    self.assertBytesEqual(candidate(unaligned), uncached)
                    self.assertBytesEqual(candidate(unaligned), uncached)
                if (
                    h20
                    and _library() is not None
                    and rows != 8192
                    and not (rows == 32 and columns == 2560)
                ):
                    packed = (
                        columns == 8240
                        and rows <= 64
                        or columns == 2560
                        and rows >= 8192
                    )
                    self.assertTrue(candidate._plans[(rows, packed)].valid)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = candidate(value)
                graph.replay()
                self.assertBytesEqual(actual, expected)
                graph.reset()
                streams = [torch.cuda.Stream(), torch.cuda.Stream()]
                ready = torch.cuda.Event()
                ready.record()
                outputs = []
                for stream in streams:
                    stream.wait_event(ready)
                    with torch.cuda.stream(stream):
                        outputs.append(candidate(value))
                for stream in streams:
                    stream.synchronize()
                for result in outputs:
                    self.assertBytesEqual(result, expected)
            native.weight.add_(0.125)
            value = torch.ones(16, inner, device="cuda", dtype=torch.bfloat16)
            self.assertBytesEqual(candidate(value), native(value))
            candidate.weight = native.weight.clone()
            self.assertBytesEqual(candidate(value), native(value))
            tail = value[:3]
            self.assertBytesEqual(candidate(tail), native(tail))
            with torch.inference_mode():
                candidate.weight = native.weight.clone()
            self.assertBytesEqual(candidate(value), native(value))


if __name__ == "__main__":
    unittest.main(verbosity=2)
