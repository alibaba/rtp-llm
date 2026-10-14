"""Numerical comparisons using the real FlashMLA and RTP projection kernels."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8._v41_mega_attention import MegaAttention
from rtp_llm.models_py.modules.dsv4.fp8._v41_output_projection import (
    grouped_output_projection,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention import _prepare_wo_a_stacked
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8, rope_only
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_decode_attention_fusion import (
    V41Fp4DualDecodeAttentionCUDA,
)
from rtp_llm.models_py.modules.dsv4.rope import precompute_freqs_cis
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear


class V41MegaAttentionRequirementsCPU(unittest.TestCase):
    def test_projection_consumes_fused_pair_and_preserves_fallback(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_wo_a_quant

        linear = V41MXFP8Linear.__new__(V41MXFP8Linear)
        torch.nn.Module.__init__(linear)
        linear.forward = Mock()
        linear.forward_quantized = Mock()
        mega = MegaAttention.__new__(MegaAttention)
        mega.groups, mega.rank = 8, 1024
        mega.weight, mega.weight_scales = object(), object()
        pair = SimpleNamespace(data=torch.empty(2, 8, 4096))
        output, quantized = object(), (object(), object())
        for result in (quantized, None):
            with self.subTest(fused=result is not None), patch.object(
                _v41_wo_a_quant, "try_grouped_quant", return_value=result
            ) as fused, patch("deep_gemm.fp8_einsum") as gemm:
                linear.forward.reset_mock()
                linear.forward_quantized.reset_mock()
                mega.project(SimpleNamespace(wo_b=linear), pair, out=output)
                fused.assert_called_once_with(pair, (mega.weight, mega.weight_scales))
                if result is not None:
                    linear.forward_quantized.assert_called_once_with(
                        *result, out=output
                    )
                    linear.forward.assert_not_called()
                    gemm.assert_not_called()
                else:
                    gemm.assert_called_once()
                    linear.forward.assert_called_once()
                    linear.forward_quantized.assert_not_called()
                    self.assertIs(linear.forward.call_args.kwargs["out"], output)

    @staticmethod
    def _attention(weight):
        linear = V41MXFP8Linear.__new__(V41MXFP8Linear)
        torch.nn.Module.__init__(linear)
        linear.weight = weight
        return SimpleNamespace(
            wq_b=linear, head_dim=512, rope_head_dim=64, n_heads=64, n_groups=8
        )

    def test_unsupported_device_raises_instead_of_selecting_another_backend(self):
        attn = self._attention(torch.empty(1, 1))
        with self.assertRaisesRegex(RuntimeError, "mega attention requires SM10x"):
            MegaAttention.create(attn)

    def test_missing_native_api_raises_instead_of_selecting_another_backend(self):
        attn = self._attention(
            SimpleNamespace(is_cuda=True, device=torch.device("cuda"))
        )
        native = SimpleNamespace()
        package = SimpleNamespace(fused_norm_rope_attn_rope_cast=object(), cuda=native)
        with patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ), patch.dict(
            "sys.modules", {"flash_mla": package, "flash_mla.cuda": native}
        ), self.assertRaisesRegex(
            ImportError, "without mega attention"
        ):
            MegaAttention.create(attn)

    def test_tp_heads_reach_native_backend(self):
        native = SimpleNamespace(
            **{
                name: Mock()
                for name in (
                    "fused_norm_rope_attn_rope_cast_fwd",
                    "fused_norm_rope_attn_rope_cast_decode",
                    "permute_q_b_proj",
                    "permute_wv_proj",
                )
            }
        )
        kernels = object()
        package = SimpleNamespace(fused_norm_rope_attn_rope_cast=kernels, cuda=native)
        for heads in (8, 16, 32, 64, 128):
            with self.subTest(heads=heads):
                attn = self._attention(
                    SimpleNamespace(is_cuda=True, device=torch.device("cuda"))
                )
                attn.n_heads, attn.n_groups, attn.layer_id = heads, heads // 8, 0
                with patch.object(
                    torch.cuda, "get_device_capability", return_value=(10, 3)
                ), patch.dict(
                    "sys.modules", {"flash_mla": package, "flash_mla.cuda": native}
                ), patch.object(MegaAttention, "__init__", return_value=None) as init:
                    self.assertIsInstance(MegaAttention.create(attn), MegaAttention)
                    init.assert_called_once_with(attn, kernels)

    def test_old_flashmla_wrapper_has_actionable_error(self):
        attn = self._attention(
            SimpleNamespace(is_cuda=True, device=torch.device("cuda"))
        )
        package = SimpleNamespace(cuda=SimpleNamespace())
        with patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ), patch.dict(
            "sys.modules", {"flash_mla": package, "flash_mla.cuda": package.cuda}
        ), self.assertRaisesRegex(ImportError, "open_source/deps requirements"):
            MegaAttention.create(attn)

    def test_tp_padding_preserves_chunk_layout_and_current_sink(self):
        for heads in (8, 16, 32, 64, 128):
            with self.subTest(heads=heads):
                mega = MegaAttention.__new__(MegaAttention)
                mega.kernel_heads = max(64, heads)
                mega.groups = heads // 8
                mega.cos_sin_cache = object()
                # Independent, head-major oracle: the Q linear outputs
                # channel chunks interleaved across heads, not head-major Q.
                q = torch.arange(2 * heads * 512).reshape(2, heads, 512)
                interleaved = q.reshape(2, heads, 32, 16).permute(0, 2, 1, 3)
                interleaved = interleaved.contiguous().view(1, 2, heads, 512)
                sink = torch.arange(heads, dtype=torch.float32)
                attn = SimpleNamespace(
                    n_heads=heads, attn_sink=sink, softmax_scale=512**-0.5
                )
                positions = torch.tensor([3, 41], dtype=torch.int64)
                args = mega._arguments(attn, interleaved, positions)
                recovered = args["q"].view(2, 32, mega.kernel_heads, 16)
                recovered = recovered.permute(0, 2, 1, 3).reshape(
                    2, mega.kernel_heads, 512
                )
                torch.testing.assert_close(recovered[:, :heads], q)
                self.assertEqual(torch.count_nonzero(recovered[:, heads:]).item(), 0)
                torch.testing.assert_close(args["attn_sink"][:heads], sink)
                self.assertTrue(torch.isposinf(args["attn_sink"][heads:]).all())
                self.assertEqual(args["n_wv_group"], mega.kernel_heads // 8)
                self.assertEqual(args["token_positions"].dtype, torch.int32)
                torch.testing.assert_close(args["token_positions"], positions.int())
                attn.attn_sink = sink + 7
                torch.testing.assert_close(
                    mega._arguments(attn, interleaved, positions)["attn_sink"][:heads],
                    attn.attn_sink,
                )

    def test_tp_output_crops_groups_without_repacking_scales(self):
        for heads in (8, 16, 32, 64, 128):
            with self.subTest(heads=heads):
                mega = MegaAttention.__new__(MegaAttention)
                mega.groups = heads // 8
                groups = max(8, mega.groups)
                data = torch.arange(3 * groups * 4096).view(3, groups, 4096)
                # Native packed UE8M0 scales: token dimension has stride 1,
                # rounded to a multiple of four for TMA.
                scales = torch.arange(groups * 32 * 4, dtype=torch.int32)
                scales = scales.view(groups, 32, 4).permute(2, 0, 1)[:3]
                pair = mega._output(data, scales)
                self.assertEqual(pair.data.shape, (3, mega.groups, 4096))
                torch.testing.assert_close(pair.data, data[:, : mega.groups])
                torch.testing.assert_close(pair.scales, scales[:, : mega.groups])
                self.assertEqual(pair.scales.stride(), scales.stride())
                self.assertEqual(pair.scales.data_ptr(), scales.data_ptr())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class V41MegaAttentionCUDA(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("mega attention requires SM10x")
        torch.manual_seed(8217)
        self.context = torch.inference_mode()
        self.context.__enter__()
        self.addCleanup(self.context.__exit__, None, None, None)

    @staticmethod
    def _attention(rank=64, heads=64):
        groups, dimension, q_rank = heads // 8, 512, 128
        q_weight = torch.randn(heads * dimension, q_rank, device="cuda").to(
            torch.float8_e4m3fn
        )
        q_scales = torch.full(
            (heads * dimension // 32, q_rank // 32), 0.125, device="cuda"
        ).to(torch.float8_e8m0fnu)
        wo_weight = torch.randn(groups * rank, 8 * dimension, device="cuda").to(
            torch.float8_e4m3fn
        )
        wo_scales = torch.full(
            (groups * rank // 32, 8 * dimension // 32), 0.125, device="cuda"
        ).to(torch.float8_e8m0fnu)
        weight, scales = _prepare_wo_a_stacked(wo_weight, wo_scales, groups, rank, 4096)

        def wo_b(value, out=None):
            return value if out is None else out.copy_(value)

        return SimpleNamespace(
            layer_id=0,
            n_heads=heads,
            n_groups=groups,
            head_dim=dimension,
            rope_head_dim=64,
            o_lora_rank=rank,
            softmax_scale=dimension**-0.5,
            wq_b=V41MXFP8Linear(q_weight, q_scales),
            _wo_a_stk_w=weight,
            _wo_a_stk_s=scales,
            wo_b=wo_b,
            freqs_cis=precompute_freqs_cis(
                64, 2048, 0, 160000, 16, 32, 1, device=torch.device("cuda")
            ),
            attn_sink=torch.linspace(-4, 4, heads, device="cuda"),
            _shared_attention={},
        )

    def _check_output(self, actual, expected):
        self.assertTrue(torch.isfinite(actual).all().item())
        actual, expected = actual.float().flatten(), expected.float().flatten()
        relative_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(actual, expected, dim=0)
        print(
            f"mega projection relative_l2={relative_l2.item():.6f} cosine={cosine.item():.6f}",
            flush=True,
        )
        self.assertLess(relative_l2.item(), 0.05)
        self.assertGreater(cosine.item(), 0.999)

    def test_tp_prefill_and_decode_match_unfused_attention(self):
        from flash_mla import flash_mla_sparse_fwd, get_mla_metadata

        for heads in (8, 16, 32):
            with self.subTest(heads=heads):
                attn = self._attention(heads=heads)
                original_q = attn.wq_b
                mega = MegaAttention.create(attn)
                qr = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
                positions = torch.tensor([127, 233], device="cuda", dtype=torch.int32)
                freqs = attn.freqs_cis[positions.long()]
                q = original_q(qr).view(2, heads, 512)
                rope_only(q, freqs, 64)
                # The reference sparse kernel also requires >=64 heads.
                # Its Q is head-major, unlike Mega's chunk-interleaved Q.
                reference_q = torch.nn.functional.pad(q, (0, 0, 0, 64 - heads))
                sink = torch.nn.functional.pad(
                    attn.attn_sink, (0, 64 - heads), value=float("inf")
                )
                kv = torch.randn(257, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.3
                indices = torch.randint(
                    0, 257, (2, 1, 128), device="cuda", dtype=torch.int32
                )
                indices[:, :, ::17] = -1
                lengths = torch.tensor([37, 128], device="cuda", dtype=torch.int32)
                output, _, _ = flash_mla_sparse_fwd(
                    q=reference_q,
                    kv=kv,
                    indices=indices,
                    sm_scale=attn.softmax_scale,
                    attn_sink=sink,
                    topk_length=lengths,
                )
                expected = grouped_output_projection(
                    output[:, :heads].contiguous(),
                    freqs,
                    attn._wo_a_stk_w,
                    attn._wo_a_stk_s,
                )
                pair = mega.prefill(
                    attn,
                    attn.wq_b(qr).view(2, heads, 512),
                    positions,
                    kv,
                    indices,
                    lengths,
                )
                self._check_output(mega.project(attn, pair), expected)

                case = V41Fp4DualDecodeAttentionCUDA._case(2, 1, 128)
                reference = case.op.forward(
                    reference_q.view(2, 1, 64, 512),
                    case.swa_pool,
                    sink,
                    case.swa_topk,
                    get_mla_metadata()[0],
                    extra_k_cache=case.global_pool,
                    extra_topk_idxs=case.global_topk,
                )
                expected = grouped_output_projection(
                    reference[..., :heads, :].contiguous(),
                    freqs,
                    attn._wo_a_stk_w,
                    attn._wo_a_stk_s,
                )

                def run():
                    pair = mega.decode(
                        attn,
                        attn.wq_b(qr).view(2, 1, heads, 512),
                        positions,
                        case.swa_pool,
                        case.swa_topk,
                        case.global_pool,
                        case.global_topk.view(2, -1),
                    )
                    return mega.project(attn, pair)

                self._check_output(run(), expected)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                # Padding must be replayed from live queries, positions,
                # sink weights and slots, rather than capture-time values.
                qr.copy_(torch.randn_like(qr))
                positions.add_(17)
                attn.attn_sink.add_(1)
                V41Fp4DualDecodeAttentionCUDA._set_slots(case, 1)
                graph.replay()
                torch.testing.assert_close(captured, run(), rtol=0, atol=0)
                def resident_addresses():
                    return tuple(
                        t.data_ptr()
                        for t in (
                            attn.wq_b.weight,
                            attn.wq_b.weight_scales,
                            mega.weight,
                            mega.weight_scales,
                            mega.cos_sin_cache,
                        )
                    )

                addresses = resident_addresses()
                before_reload = run().clone()
                # Simulate parent reload restoring checkpoint-layout Q into
                # resident storage, then replay the already captured graph.
                attn.wq_b.weight.copy_(original_q.weight)
                attn.wq_b.weight_scales.copy_(original_q.weight_scales)
                mega.reload(attn)
                self.assertEqual(resident_addresses(), addresses)
                torch.testing.assert_close(run(), before_reload, rtol=0, atol=0)
                graph.replay()
                torch.testing.assert_close(captured, run(), rtol=0, atol=0)

    def test_mega_pair_fused_projection_matches_unfused_and_replays(self):
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_wo_a_quant

        attn = self._attention(rank=1024)
        attn.wo_b = V41MXFP8Linear(
            (torch.randn(5120, 8192, device="cuda") * 0.02).to(torch.float8_e4m3fn),
            torch.ones(160, 256, device="cuda").to(torch.float8_e8m0fnu),
        )
        mega = MegaAttention.create(attn)
        self.assertTrue(_v41_wo_a_quant.warmup(mega.weight, mega.weight_scales))
        for rows in (1, 64, 384, 512):
            with self.subTest(rows=rows):
                q = torch.randn(rows, 64, 512, device="cuda", dtype=torch.bfloat16)
                kv = torch.randn(257, 1, 512, device="cuda", dtype=torch.bfloat16)
                indices = torch.randint(
                    0, 257, (rows, 1, 128), device="cuda", dtype=torch.int32
                )
                indices[:, :, ::17] = -1
                positions = torch.arange(rows, device="cuda", dtype=torch.int32)
                lengths = torch.full((rows,), 128, device="cuda", dtype=torch.int32)
                pair = mega.prefill(attn, q, positions, kv, indices, lengths)
                self.assertTrue(
                    _v41_wo_a_quant.is_supported(
                        pair, (mega.weight, mega.weight_scales)
                    )
                )
                # Compare only this change: identical real FlashMLA outputs
                # and permuted weights, with/without the restored epilogue.
                with patch.object(
                    _v41_wo_a_quant, "try_grouped_quant", return_value=None
                ):
                    expected = mega.project(attn, pair)
                out = torch.empty_like(expected)
                self.assertIs(mega.project(attn, pair, out=out), out)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = mega.project(attn, pair)
                # The capture must consume current inputs, not the warmup's.
                pair.data.zero_()
                graph.replay()
                torch.testing.assert_close(
                    captured, mega.project(attn, pair), rtol=0, atol=0
                )

    def test_prefill_projection_matches_original_with_masks_and_positions(self):
        from flash_mla import flash_mla_sparse_fwd

        for rows in (1, 11):
            with self.subTest(rows=rows):
                attn = self._attention()
                original_q = attn.wq_b
                original_weight = original_q.weight.view(torch.uint8).clone()
                mega = attn._mega_attention = MegaAttention.create(attn)
                torch.testing.assert_close(
                    original_q.weight.view(torch.uint8), original_weight
                )
                qr = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
                positions = (
                    torch.arange(rows, device="cuda", dtype=torch.int32) * 37 + 3
                )
                freqs = attn.freqs_cis[positions.long()]
                q = original_q(qr).view(rows, 64, 512)
                rope_only(q, freqs, 64)
                kv = torch.randn(257, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.3
                indices = torch.randint(
                    0, 257, (rows, 1, 128), device="cuda", dtype=torch.int32
                )
                indices[:, :, ::17] = -1
                lengths = torch.linspace(37, 128, rows, device="cuda").int()
                output, _, _ = flash_mla_sparse_fwd(
                    q=q,
                    kv=kv,
                    indices=indices,
                    sm_scale=attn.softmax_scale,
                    attn_sink=attn.attn_sink,
                    topk_length=lengths,
                )
                expected = grouped_output_projection(
                    output, freqs, attn._wo_a_stk_w, attn._wo_a_stk_s
                )
                pair = mega.prefill(
                    attn,
                    attn.wq_b(qr).view(rows, 64, 512),
                    positions,
                    kv,
                    indices,
                    lengths,
                )
                actual = torch.empty_like(expected)
                result = AttentionV41FP8._project_output(attn, pair, out=actual)
                self.assertIs(result, actual)
                self._check_output(actual, expected)

    def test_decode_projection_matches_original_over_fp8_fp4_planar_pages(self):
        from flash_mla import get_mla_metadata

        for batch, span, entries, dual in ((1, 1, 64, False), (3, 2, 128, True)):
            with self.subTest(batch=batch, span=span, dual=dual):
                attn = self._attention()
                original_q = attn.wq_b
                mega = attn._mega_attention = MegaAttention.create(attn)
                case = V41Fp4DualDecodeAttentionCUDA._case(batch, span, entries)
                attn.attn_sink = case.sink
                qr = torch.randn(batch, span, 128, device="cuda", dtype=torch.bfloat16)
                positions = (
                    torch.arange(batch * span, device="cuda", dtype=torch.int32) * 43
                    + 127
                )
                freqs = attn.freqs_cis[positions.long()]
                case.q = original_q(qr).view(batch, span, 64, 512)
                rope_only(case.q.view(-1, 64, 512), freqs, 64)
                output = case.op.forward(
                    case.q,
                    case.swa_pool,
                    case.sink,
                    case.swa_topk,
                    get_mla_metadata()[0],
                    extra_k_cache=case.global_pool if dual else None,
                    extra_topk_idxs=case.global_topk if dual else None,
                )
                expected = grouped_output_projection(
                    output, freqs, attn._wo_a_stk_w, attn._wo_a_stk_s
                )
                pair = mega.decode(
                    attn,
                    attn.wq_b(qr).view(batch, span, 64, 512),
                    positions,
                    case.swa_pool,
                    case.swa_topk,
                    case.global_pool if dual else None,
                    case.global_topk.view(batch * span, -1) if dual else None,
                )
                self._check_output(
                    AttentionV41FP8._project_output(attn, pair), expected
                )

    def test_mega_decode_graph_replay_updates_queries_positions_and_slots(self):
        attn = self._attention()
        original_q = attn.wq_b
        mega = attn._mega_attention = MegaAttention.create(attn)
        case = V41Fp4DualDecodeAttentionCUDA._case(2, 1, 128)
        attn.attn_sink = case.sink
        qr = torch.randn(2, 1, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.tensor([127, 233], device="cuda", dtype=torch.int32)

        def run():
            pair = mega.decode(
                attn,
                attn.wq_b(qr).view(2, 1, 64, 512),
                positions,
                case.swa_pool,
                case.swa_topk,
                case.global_pool,
                case.global_topk.view(2, -1),
            )
            return AttentionV41FP8._project_output(attn, pair)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for phase in (0, 1):
            qr.copy_(torch.randn_like(qr))
            positions.add_(17)
            V41Fp4DualDecodeAttentionCUDA._set_slots(case, phase + 1)
            graph.replay()
            torch.testing.assert_close(captured, run(), rtol=0, atol=0)

        # Simulate the parent reload restoring the standard checkpoint layout,
        # then verify that our derived weights retain graph-baked addresses.
        def resident_pointers():
            tensors = (
                attn.wq_b.weight,
                attn.wq_b.weight_scales,
                mega.weight,
                mega.weight_scales,
                mega.cos_sin_cache,
            )
            return tuple(t.data_ptr() for t in tensors)

        pointers = resident_pointers()
        packed_q = attn.wq_b.weight.view(torch.uint8).clone()
        attn.wq_b.weight.copy_(original_q.weight)
        attn.wq_b.weight_scales.copy_(original_q.weight_scales)
        mega.reload(attn)
        self.assertEqual(pointers, resident_pointers())
        torch.testing.assert_close(attn.wq_b.weight.view(torch.uint8), packed_q)
        graph.replay()
        torch.testing.assert_close(captured, run(), rtol=0, atol=0)

    def test_dspark_five_queries_use_mega_with_noncausal_slots(self):
        from flash_mla import get_mla_metadata

        from rtp_llm.models_py.model_desc import deepseek_v4_dspark_model as dspark

        attn = self._attention()
        original_q = attn.wq_b
        attn._mega_attention = MegaAttention.create(attn)
        attn._project_output = lambda pair: AttentionV41FP8._project_output(attn, pair)
        attn._prefill_output_all_reduce = lambda output: None
        batch, gamma = 2, 5
        case = V41Fp4DualDecodeAttentionCUDA._case(batch, gamma, 128)
        attn.attn_sink = case.sink
        # Every query sees the entire five-token proposal block, including
        # future proposal slots. Invalid padding must contribute no attention.
        case.swa_topk[:, :, :5] = torch.arange(5, device="cuda", dtype=torch.int32)
        case.swa_topk[:, :, 80:] = -1
        positions = torch.tensor([123, 251], device="cuda", dtype=torch.int32)[:, None]
        positions = positions + torch.arange(gamma, device="cuda", dtype=torch.int32)
        qr = torch.randn(batch, gamma, 128, device="cuda", dtype=torch.bfloat16)
        freqs = attn.freqs_cis[positions.reshape(-1).long()]
        original = original_q(qr).view(batch, gamma, 64, 512)
        rope_only(original.view(-1, 64, 512), freqs, 64)
        expected = case.op.forward(
            original, case.swa_pool, case.sink, case.swa_topk, get_mla_metadata()[0]
        )
        expected = grouped_output_projection(
            expected, freqs, attn._wo_a_stk_w, attn._wo_a_stk_s
        )
        model = SimpleNamespace(
            _swa_cache_bound=lambda *args: nullcontext((attn, 128, case.swa_pool)),
            _proposal_metadata=lambda *args: (
                positions.reshape(-1),
                case.swa_topk.reshape(batch * gamma, -1),
                None,
            ),
        )

        def qkv(attention, value, query_positions, *, apply_q_rope):
            self.assertFalse(apply_q_rope)
            torch.testing.assert_close(query_positions, positions.reshape(-1))
            return SimpleNamespace(
                q=attention.wq_b(value).view(batch, gamma, 64, 512), kv=None
            )

        with patch.object(dspark, "decode_compute_qkv", side_effect=qkv), patch.object(
            dspark, "_write_dspark_swa"
        ) as write, patch.object(
            dspark, "get_or_build_sched_meta", side_effect=AssertionError("old backend")
        ):
            actual = dspark.DeepSeekV4DSparkModel._forward_dspark_attention(
                model, 0, qr, positions, None, None, None, 128, None, {}
            )
        write.assert_called_once()
        self.assertEqual(actual.shape, (batch, gamma, 512))
        self._check_output(actual, expected)

    def test_kv_write_is_identical_when_q_rope_moves_into_attention(self):
        from rtp_llm.models_py.modules.dsv4.fp8._v41_qkv_rope_cache_triton import (
            try_fused_qkv_rope_cache,
        )

        q = torch.randn(2, 2, 64, 512, device="cuda", dtype=torch.bfloat16)
        raw_kv = torch.randn(2, 2, 512, device="cuda", dtype=torch.bfloat16)
        norm = torch.ones(512, device="cuda", dtype=torch.bfloat16)
        positions = torch.tensor([0, 3, 127, 131], device="cuda", dtype=torch.int32)
        freqs = self._attention().freqs_cis
        slots = torch.arange(4, device="cuda", dtype=torch.int32)
        pool = torch.zeros(1, 128, 528, device="cuda", dtype=torch.uint8)
        new_pool = torch.zeros_like(pool)
        unchanged_q = q.clone()
        original = try_fused_qkv_rope_cache(
            q.clone(), raw_kv, norm, positions, freqs, pool, slots
        )
        moved = try_fused_qkv_rope_cache(
            unchanged_q,
            raw_kv,
            norm,
            positions,
            freqs,
            new_pool,
            slots,
            apply_q_rope=False,
        )
        self.assertIsNotNone(original)
        self.assertIsNotNone(moved)
        torch.testing.assert_close(moved.q, q, rtol=0, atol=0)
        torch.testing.assert_close(moved.kv, original.kv, rtol=0, atol=0)
        torch.testing.assert_close(new_pool, pool, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
