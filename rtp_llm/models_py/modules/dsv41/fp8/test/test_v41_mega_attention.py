"""Native FlashMLA mega attention layout, sparse semantics and graph checks."""

import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_mega_attention as mega


def _q_fused(q):
    tokens, heads, _ = q.shape
    return q.view(tokens, heads, 32, 16).transpose(1, 2).contiguous().view_as(q)


def _unpack_scale(scale):
    shifts = torch.arange(4, device=scale.device) * 8
    return ((scale.to(torch.int64).unsqueeze(-1) >> shifts) & 255).flatten(-2)


def _dequant_output(pair, heads):
    data, scale = pair
    factor = torch.exp2(_unpack_scale(scale).float() - 127).repeat_interleave(32, -1)
    data = data.float() * factor
    tokens, groups, _ = data.shape
    return (
        data.view(tokens, groups, 16, 8, 32)
        .transpose(2, 3)
        .contiguous()
        .view(tokens, groups * 8, 512)[:, :heads]
    )


def _state(device="cpu", heads=16, backend=None):
    groups = heads // 8
    return mega.MegaAttention(
        backend or Mock(),
        torch.zeros(heads * 512, 128, dtype=torch.float8_e4m3fn, device=device),
        torch.zeros(1, heads * 512, dtype=torch.int32, device=device).T,
        torch.zeros(groups, 128, 4096, dtype=torch.float8_e4m3fn, device=device),
        torch.zeros(groups, 32, 128, dtype=torch.int32, device=device).transpose(1, 2),
        torch.zeros(256, 64, device=device),
        torch.zeros(max(64, heads), device=device),
        512**-0.5,
        heads,
        groups,
    )


class MegaAttentionCPU(unittest.TestCase):
    def test_disabled_configuration_skips_backend_loading(self):
        with patch.dict(os.environ, {"DSV41_MEGA_ATTENTION": "0"}), patch.object(
            mega, "_backend", side_effect=AssertionError("backend imported")
        ):
            self.assertIsNone(mega.prepare(SimpleNamespace()))

    def _supported_cpu_fixture(self):
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        # Supply the packed-weight contract without executing a CUDA packer.
        state = _state()
        linear = V41MXFP8Linear.__new__(V41MXFP8Linear)
        torch.nn.Module.__init__(linear)
        linear.weight = state.query_weight
        linear.weight_scales = state.query_scale
        linear.N, linear.K = linear.weight.shape
        return SimpleNamespace(
            wq_b=linear,
            _wo_a_stk_w=state.output_weight,
            _wo_a_stk_s=state.output_scale,
            freqs_cis=torch.zeros(256, 32, dtype=torch.complex64),
            attn_sink=torch.zeros(state.heads, dtype=torch.float32),
            skip_post_q_norm=True,
            n_heads=state.heads,
            n_groups=state.groups,
            head_dim=512,
            rope_head_dim=64,
            o_lora_rank=128,
        )

    def test_enabled_by_default_and_switch_disables_mega_attention(self):
        attn = self._supported_cpu_fixture()
        with patch.dict(os.environ, {}, clear=True), patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ), patch.object(mega, "_device_capability", return_value=(10, 3)), patch.object(
            mega, "_backend", return_value=object()
        ) as backend, patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")
        ):
            self.assertTrue(mega.is_supported(attn))
            backend.assert_called_once_with()
            backend.reset_mock()
            with patch.dict(os.environ, {"DSV41_MEGA_ATTENTION": "0"}):
                self.assertFalse(mega.is_supported(attn))
                self.assertIsNone(mega.prepare(attn))
            backend.assert_not_called()

    def test_cpu_falls_back_before_hardware_probe(self):
        attn = self._supported_cpu_fixture()
        with patch.dict(os.environ, {}, clear=True), patch.object(
            mega, "_device_capability", side_effect=AssertionError("CUDA queried")
        ), patch.object(
            mega, "_backend", side_effect=AssertionError("backend imported")
        ):
            self.assertIsNone(mega.prepare(attn))

    def test_default_requires_sm10x_and_native_backend(self):
        attn = self._supported_cpu_fixture()
        for capability, backend in (((9, 0), object()), ((10, 3), None)):
            with self.subTest(capability=capability, backend=backend), patch.dict(
                os.environ, {}, clear=True
            ), patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            ), patch.object(
                mega, "_device_capability", return_value=capability
            ), patch.object(
                mega, "_backend", return_value=backend
            ):
                self.assertIsNone(mega.prepare(attn))

    def test_partial_wheel_and_import_failures_reject_backend(self):
        required = (
            "fused_norm_rope_attn_rope_cast_fwd",
            "fused_norm_rope_attn_rope_cast_decode",
            "permute_q_b_proj",
            "permute_wv_proj",
        )
        self.addCleanup(mega._backend.cache_clear)
        for missing in required:
            backend = SimpleNamespace(
                **{name: Mock() for name in required if name != missing}
            )
            mega._backend.cache_clear()
            with self.subTest(missing=missing), patch.object(
                mega.importlib, "import_module", return_value=backend
            ):
                self.assertIsNone(mega._backend())
        for exception in (ImportError("old wheel"), OSError("missing library")):
            mega._backend.cache_clear()
            with self.subTest(exception=exception), patch.object(
                mega.importlib, "import_module", side_effect=exception
            ):
                self.assertIsNone(mega._backend())

    def test_preparation_errors_are_not_hidden(self):
        with patch.object(mega, "is_supported", return_value=True), patch.object(
            mega, "_backend", return_value=object()
        ), patch.object(
            mega, "_permuted_weights", side_effect=RuntimeError("kernel failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                mega.prepare(SimpleNamespace())

    def test_q_permutation_keeps_scales_and_avoids_native_tp_restrictions(self):
        attn = self._supported_cpu_fixture()
        rows, width = attn.wq_b.weight.shape
        weight = (torch.arange(rows * width).reshape(rows, width) % 31 - 15).float()
        attn.wq_b.weight.copy_(weight.to(torch.float8_e4m3fn))
        attn.wq_b.weight_scales.copy_(torch.arange(rows).int()[:, None])
        backend = SimpleNamespace(
            permute_q_b_proj=Mock(
                side_effect=AssertionError("old native head restriction")
            ),
            permute_wv_proj=Mock(return_value=(attn._wo_a_stk_w, attn._wo_a_stk_s)),
        )
        qw, qs, _, _ = mega._permuted_weights(attn, backend)
        x = torch.arange(width).float()[None] / width
        original_q = (x @ attn.wq_b.weight.float().T).view(1, attn.n_heads, 512)
        actual_q = (x @ qw.float().T).view_as(original_q)
        torch.testing.assert_close(actual_q, _q_fused(original_q), rtol=0, atol=0)
        expected_scales = _q_fused(torch.arange(rows).view(1, attn.n_heads, 512))
        torch.testing.assert_close(
            qs[:, 0], expected_scales.flatten().int(), rtol=0, atol=0
        )
        self.assertEqual(qs.stride(0), 1)
        backend.permute_q_b_proj.assert_not_called()

    def test_tp4_padding_preserves_chunk_layout(self):
        state = _state()
        standard = torch.arange(3 * 16 * 512).view(3, 16, 512).to(torch.bfloat16)
        actual = state.pad_q(_q_fused(standard))
        expected = _q_fused(torch.nn.functional.pad(standard, (0, 0, 0, 48)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(state.query_weight.shape[0], 16 * 512)
        self.assertEqual(state.padded_groups, 8)

    def test_prefill_pybind_contract_keeps_positions_and_ragged_lengths(self):
        state = _state()
        q = torch.ones(3, 16, 512, dtype=torch.bfloat16)
        kv = torch.ones(257, 1, 512, dtype=torch.bfloat16)
        indices = torch.full((3, 128), -1, dtype=torch.int32)
        lengths = torch.tensor([0, 17, 127], dtype=torch.int32)
        positions = torch.tensor([28672, 17, 32766], dtype=torch.int32)
        output = (object(), object())
        state.backend.fused_norm_rope_attn_rope_cast_fwd.return_value = (
            *output,
            None,
            None,
        )
        self.assertTrue(state.supports_prefill(kv, indices, lengths, positions))
        result = state.prefill(q, kv, indices, positions, lengths)
        self.assertEqual(result, output)
        args = state.backend.fused_norm_rope_attn_rope_cast_fwd.call_args.args
        self.assertEqual(len(args), 18)
        self.assertEqual(args[0].shape, (3, 64, 512))
        self.assertEqual(args[2].shape, (3, 1, 128))
        self.assertIs(args[6], lengths)
        self.assertIs(args[9], positions)
        self.assertEqual(args[13:], (8, 32, True, True, True))
        self.assertFalse(state.supports_prefill(kv.float(), indices, lengths))
        self.assertFalse(state.supports_prefill(kv, indices[:, :17], lengths))

    def test_decode_retains_padded_page_strides_and_joint_softmax_pools(self):
        state = _state()
        swa = torch.empty(4, 33792, dtype=torch.uint8).as_strided(
            (4, 32, 528), (33792, 528, 1)
        )
        extra = torch.empty(3, 73728, dtype=torch.uint8).as_strided(
            (3, 128, 288), (73728, 288, 1)
        )
        ids = torch.full((6, 128), -1, dtype=torch.int32)
        positions = torch.tensor([7, 8, 9, 23, 24, 25], dtype=torch.int32)
        state.backend.fused_norm_rope_attn_rope_cast_decode.return_value = (
            None,
            None,
            None,
        )
        self.assertTrue(state.supports_decode(swa, ids, extra, ids))
        state.decode(
            torch.zeros(2, 3, 64, 512),
            swa,
            ids,
            positions,
            extra_pool=extra,
            extra_indices=ids,
        )
        args = state.backend.fused_norm_rope_attn_rope_cast_decode.call_args.args
        self.assertEqual(len(args), 21)
        self.assertEqual(args[1].stride(0), 33792)
        self.assertEqual(args[7].stride(0), 73728)
        self.assertIs(args[12], positions)
        self.assertEqual(args[16:], (8, 32, True, True, True))
        self.assertFalse(state.supports_decode(swa, ids.long(), extra, ids))
        # FlashMLA TMA addresses whole token rows, including padded pages.
        invalid = torch.empty(4, 17408, dtype=torch.uint8).as_strided(
            (4, 32, 528), (17408, 528, 1)
        )
        self.assertFalse(state.supports_decode(invalid, ids))
        invalid_extra = torch.empty(3, 37376, dtype=torch.uint8).as_strided(
            (3, 128, 288), (37376, 288, 1)
        )
        self.assertFalse(state.supports_decode(swa, ids, invalid_extra, ids))

    def test_prepared_tensors_are_nonpersistent_module_buffers(self):
        state = _state()
        self.assertEqual(
            set(dict(state.named_buffers())),
            {
                "query_weight",
                "query_scale",
                "output_weight",
                "output_scale",
                "cos_sin",
                "sink",
            },
        )
        self.assertEqual(dict(state.state_dict()), {})

    def test_explicit_reload_preserves_captured_buffer_addresses(self):
        state = _state()
        tensors = dict(state.named_buffers())
        pointers = {name: value.data_ptr() for name, value in tensors.items()}
        originals = tuple(
            torch.full_like(tensors[name].float(), 2).to(tensors[name].dtype)
            for name in ("query_weight", "query_scale", "output_weight", "output_scale")
        )
        table = torch.ones(256, 32, dtype=torch.complex64) * (2 + 3j)
        attn = SimpleNamespace(freqs_cis=table, attn_sink=torch.arange(16).float())
        with patch.dict(os.environ, {"DSV41_MEGA_ATTENTION": "0"}), patch.object(
            mega, "_permuted_weights", return_value=originals
        ):
            state.reload_weights(attn)
        self.assertEqual(
            pointers, {name: value.data_ptr() for name, value in state.named_buffers()}
        )
        torch.testing.assert_close(
            state.cos_sin, torch.cat((table.real, table.imag), -1)
        )
        torch.testing.assert_close(state.query_weight.float(), originals[0].float())
        torch.testing.assert_close(state.sink[:16], attn.attn_sink)
        self.assertTrue(torch.isposinf(state.sink[16:]).all().item())


def _rotate(x, freqs, inverse=False):
    result = x.clone()
    tail = torch.view_as_complex(x[..., -64:].float().reshape(*x.shape[:-1], 32, 2))
    table = freqs.conj() if inverse else freqs
    result[..., -64:] = (
        torch.view_as_real(tail * table[:, None]).flatten(-2).to(x.dtype)
    )
    return result


def _attention_reference(q, kv_rows, indices, positions, table, sink, lengths=None):
    q = _rotate(q, table[positions.long()])
    ids = indices.long()
    valid = (ids >= 0) & (ids < kv_rows.shape[0])
    if lengths is not None:
        valid &= torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]
    kv = kv_rows[ids.clamp(0, kv_rows.shape[0] - 1)].float()
    scores = torch.einsum("mhd,mkd->mhk", q.float(), kv) * (512**-0.5)
    scores.masked_fill_(~valid[:, None], -torch.inf)
    # Explicit sink in the softmax denominator; +inf sinks force zero output.
    positive_sink = torch.isposinf(sink)
    safe_sink = torch.where(positive_sink, torch.full_like(sink, 10000), sink)
    scores = torch.cat((scores, safe_sink[None, :, None].expand(q.shape[0], -1, 1)), -1)
    probability = torch.nan_to_num(scores.softmax(-1))[..., :-1]
    output = torch.einsum("mhk,mkd->mhd", probability, kv).to(torch.bfloat16)
    return _rotate(output, table[positions.long()], inverse=True).float()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class MegaAttentionCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10 or mega._backend() is None:
            raise unittest.SkipTest("requires SM10x and native mega FlashMLA")

    def _fixture(self, heads, rows=6):
        torch.manual_seed(1701 + heads)
        state = _state("cuda", heads, mega._backend())
        angles = torch.randn(256, 32, device="cuda") * 0.5
        table = torch.polar(torch.ones_like(angles), angles)
        state.cos_sin.copy_(torch.cat((table.real, table.imag), -1))
        state.sink[:heads].normal_()
        state.sink[0] = -torch.inf
        state.sink[1] = torch.inf
        state.sink[heads:] = torch.inf
        q = (torch.randn(rows, heads, 512, device="cuda") * 0.5).to(torch.bfloat16)
        pos = torch.tensor(
            [31, 99, 3, 201, 107, 255], device="cuda", dtype=torch.int32
        )[:rows]
        return state, table, q, pos

    def _assert_quantized_close(self, actual, expected):
        self.assertTrue(torch.isfinite(actual).all().item())
        difference = actual - expected
        relative_l2 = difference.norm() / expected.norm().clamp_min(1e-7)
        print(f"mega attention relative_l2={relative_l2.item():.6f}", flush=True)
        self.assertLess(relative_l2.item(), 0.05)
        self.assertLess(
            difference.abs().max().item(), 0.125 * expected.abs().max().item() + 0.003
        )
        torch.testing.assert_close(
            actual[:, 1], torch.zeros_like(actual[:, 1]), rtol=0, atol=0
        )

    def test_prefill_ragged_cp_positions_tp4_and_full_heads(self):
        for heads in (8, 16, 32, 64, 128):
            with self.subTest(heads=heads):
                state, table, q, pos = self._fixture(heads)
                kv = (torch.randn(384, 512, device="cuda") * 0.5).to(torch.bfloat16)
                indices = torch.randint(
                    0, 384, (6, 128), device="cuda", dtype=torch.int32
                )
                indices[:, -7:] = -1
                lengths = torch.tensor(
                    [0, 17, 64, 127, 91, 128], device="cuda", dtype=torch.int32
                )
                pair = state.prefill(
                    _q_fused(q), kv.unsqueeze(1), indices, pos, lengths
                )
                reference = _attention_reference(
                    q, kv, indices, pos, table, state.sink[:heads], lengths
                )
                self._assert_quantized_close(_dequant_output(pair, heads), reference)
                torch.testing.assert_close(
                    _dequant_output(pair, heads)[0],
                    torch.zeros_like(reference[0]),
                    rtol=0,
                    atol=0,
                )

    def test_decode_fp8_swa_fp4_extra_and_graph_replay(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_fp4_triton as fp4
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_swa_triton as swa

        # DP4 decode retains all 64 local heads; 16 also checks TP4 support.
        for heads in (64, 16):
            with self.subTest(heads=heads):
                state, table, q, pos = self._fixture(heads)
                swa_pool = torch.zeros(
                    4, 33792, dtype=torch.uint8, device="cuda"
                ).as_strided((4, 32, 528), (33792, 528, 1))
                fp4_pool = torch.zeros(
                    2, 73728, dtype=torch.uint8, device="cuda"
                ).as_strided((2, 128, 288), (73728, 288, 1))
                swa_slots = torch.arange(128, device="cuda")
                fp4_slots = torch.arange(256, device="cuda")
                swa.quantize_and_insert_swa_k_cache(
                    torch.randn(128, 512, device="cuda", dtype=torch.bfloat16),
                    swa_pool,
                    swa_slots,
                )
                fp4.quantize_and_insert_k_cache_fp4(
                    torch.randn(256, 512, device="cuda", dtype=torch.bfloat16),
                    fp4_pool,
                    fp4_slots,
                )
                swa_rows = swa.dequantize_swa_k_cache(swa_pool, swa_slots)
                fp4_rows = fp4.dequantize_k_cache_slots_fp4(fp4_pool, fp4_slots)
                si = torch.randint(0, 128, (6, 128), device="cuda", dtype=torch.int32)
                ei = torch.randint(0, 256, (6, 128), device="cuda", dtype=torch.int32)
                si[:, -9:] = -1
                ei[0] = -1
                query = _q_fused(q)

                def run():
                    return state.decode(
                        query, swa_pool, si, pos, extra_pool=fp4_pool, extra_indices=ei
                    )

                for _ in range(3):
                    run()
                pair = run()
                combined = torch.cat((si, torch.where(ei >= 0, ei + 128, ei)), -1)
                reference = _attention_reference(
                    q,
                    torch.cat((swa_rows, fp4_rows)),
                    combined,
                    pos,
                    table,
                    state.sink[:heads],
                )
                self._assert_quantized_close(_dequant_output(pair, heads), reference)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run()
                query.mul_(0.5)
                pos.add_(1).remainder_(256)
                graph.replay()
                expected = run()
                torch.testing.assert_close(
                    captured[0].view(torch.uint8),
                    expected[0].view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(captured[1], expected[1], rtol=0, atol=0)

    def test_decode_swa_only_small_batches_and_all_head_counts(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_swa_triton as swa

        for heads in (8, 32, 64, 128):
            for rows in (1, 3):
                with self.subTest(heads=heads, rows=rows):
                    state, table, q, pos = self._fixture(heads, rows=rows)
                    # Preserve the token-interleaved ABI and exercise a nontrivial
                    # physical page stride, as in the dual-pool decode test.
                    pool = torch.zeros(
                        7, 33792, dtype=torch.uint8, device="cuda"
                    ).as_strided((7, 32, 528), (33792, 528, 1))
                    slots = torch.arange(224, device="cuda")
                    swa.quantize_and_insert_swa_k_cache(
                        torch.randn(224, 512, device="cuda", dtype=torch.bfloat16),
                        pool,
                        slots,
                    )
                    keys = swa.dequantize_swa_k_cache(pool, slots)
                    indices = torch.randint(
                        0, 224, (rows, 128), device="cuda", dtype=torch.int32
                    )
                    indices[:, -9:] = -1
                    lengths = torch.tensor(
                        [17, 0, 128], device="cuda", dtype=torch.int32
                    )[:rows]
                    self.assertTrue(state.supports_decode(pool, indices))
                    pair = state.decode(
                        _q_fused(q), pool, indices, pos, swa_lengths=lengths
                    )
                    reference = _attention_reference(
                        q, keys, indices, pos, table, state.sink[:heads], lengths
                    )
                    actual = _dequant_output(pair, heads)
                    self._assert_quantized_close(actual, reference)
                    if rows == 3:
                        torch.testing.assert_close(
                            actual[1], torch.zeros_like(actual[1]), rtol=0, atol=0
                        )

    def test_decode_kv_insert_fuses_query_padding_without_rotating_query(self):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_qkv_rope_cache_triton import (
            try_fused_qkv_rope_cache,
        )

        state, table, q, pos = self._fixture(16)
        q = _q_fused(q).view(2, 3, 16, 512)
        original = q.clone()
        raw = torch.randn(2, 3, 768, device="cuda", dtype=torch.bfloat16)[..., :512]
        norm = torch.ones(512, device="cuda", dtype=torch.bfloat16)
        slots = torch.tensor([1, 3, -1, 7, 9, 13], device="cuda")
        pool = torch.zeros(1, 32, 528, device="cuda", dtype=torch.uint8)
        baseline_pool = torch.zeros_like(pool)
        baseline = try_fused_qkv_rope_cache(
            q.clone(), raw, norm, pos, table, baseline_pool, slots
        )
        actual = try_fused_qkv_rope_cache(
            q,
            raw,
            norm,
            pos,
            table,
            pool,
            slots,
            apply_q_rope=False,
            pad_query_heads=64,
        )
        self.assertIsNotNone(actual)
        torch.testing.assert_close(q, original, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.q.view(6, 64, 512), state.pad_q(original), rtol=0, atol=0
        )
        torch.testing.assert_close(actual.kv, baseline.kv, rtol=0, atol=0)
        torch.testing.assert_close(pool, baseline_pool, rtol=0, atol=0)

    def test_startup_weight_permutations_and_query_gemm(self):
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        heads, groups, rank = 16, 2, 128
        weight = (torch.randn(heads * 512, 128, device="cuda") * 0.1).to(
            torch.float8_e4m3fn
        )
        scales = torch.ones(heads * 16, 4, device="cuda").to(torch.float8_e8m0fnu)
        linear = V41MXFP8Linear(weight, scales)
        output_weight = (torch.randn(groups, rank, 4096, device="cuda") * 0.1).to(
            torch.float8_e4m3fn
        )
        exponents = torch.randint(
            125, 129, (groups, rank, 128), device="cuda", dtype=torch.int64
        )
        packed = (
            (
                exponents.view(groups, rank, 32, 4)
                << (torch.arange(4, device="cuda") * 8)
            )
            .sum(-1)
            .int()
            .transpose(1, 2)
            .contiguous()
            .transpose(1, 2)
        )
        angles = torch.randn(256, 32, device="cuda")
        attn = SimpleNamespace(
            wq_b=linear,
            _wo_a_stk_w=output_weight,
            _wo_a_stk_s=packed,
            freqs_cis=torch.polar(torch.ones_like(angles), angles),
            attn_sink=torch.zeros(heads, device="cuda"),
            skip_post_q_norm=True,
            n_heads=heads,
            n_groups=groups,
            head_dim=512,
            rope_head_dim=64,
            o_lora_rank=rank,
            softmax_scale=512**-0.5,
        )
        with patch.dict(os.environ, {"DSV41_MEGA_ATTENTION": "1"}):
            state = mega.prepare(attn)
        self.assertIsNotNone(state)
        q = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(
            state.project_q(linear, q),
            _q_fused(linear(q).view(6, heads, 512)),
            rtol=0,
            atol=0,
        )
        permutation = torch.arange(128, device="cuda").view(8, 16).T.reshape(-1)
        expected_weight = (
            output_weight.view(groups, rank, 8, 16, 32)
            .transpose(2, 3)
            .contiguous()
            .view_as(output_weight)
        )
        torch.testing.assert_close(
            state.output_weight.view(torch.uint8),
            expected_weight.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            _unpack_scale(state.output_scale),
            exponents.index_select(-1, permutation),
            rtol=0,
            atol=0,
        )

    def _projection_fixture(self, heads=64, rank=128):
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        groups = heads // 8
        torch.manual_seed(8217 + heads)
        linear = V41MXFP8Linear(
            (torch.randn(heads * 512, 128, device="cuda") * 0.1).to(
                torch.float8_e4m3fn
            ),
            torch.ones(heads * 16, 4, device="cuda").to(torch.float8_e8m0fnu),
        )
        angles = torch.randn(256, 32, device="cuda") * 0.5
        attn = SimpleNamespace(
            wq_b=linear,
            _wo_a_stk_w=(torch.randn(groups, rank, 4096, device="cuda") * 0.02).to(
                torch.float8_e4m3fn
            ),
            _wo_a_stk_s=torch.full(
                (groups, 32, rank), 0x7F7F7F7F, device="cuda", dtype=torch.int32
            ).transpose(1, 2),
            freqs_cis=torch.polar(torch.ones_like(angles), angles),
            attn_sink=torch.randn(heads, device="cuda"),
            skip_post_q_norm=True,
            n_heads=heads,
            n_groups=groups,
            head_dim=512,
            rope_head_dim=64,
            o_lora_rank=rank,
            softmax_scale=512**-0.5,
            wo_b=lambda value, out=None: value if out is None else out.copy_(value),
            _prefill_output_all_reduce=lambda value: None,
        )
        attn._mega_attention = mega.prepare(attn)
        return attn

    def _assert_projection_close(self, actual, reference):
        actual, reference = actual.float().flatten(), reference.float().flatten()
        self.assertTrue(torch.isfinite(actual).all().item())
        relative_l2 = (actual - reference).norm() / reference.norm().clamp_min(1e-7)
        cosine = torch.nn.functional.cosine_similarity(actual, reference, dim=0)
        print(
            f"mega projection relative_l2={relative_l2.item():.6f} "
            f"cosine={cosine.item():.6f}",
            flush=True,
        )
        self.assertLess(relative_l2.item(), 0.05)
        self.assertGreater(cosine.item(), 0.999)

    @torch.inference_mode()
    def test_tp_projection_matches_unfused_and_reload_preserves_graph(self):
        for heads in (8, 16, 32, 64):
            with self.subTest(heads=heads):
                attn = self._projection_fixture(heads)
                state = attn._mega_attention
                qr = torch.randn(6, 128, dtype=torch.bfloat16, device="cuda")
                kv = torch.randn(384, 512, dtype=torch.bfloat16, device="cuda")
                indices = torch.randint(
                    0, 384, (6, 128), device="cuda", dtype=torch.int32
                )
                indices[:, -9:] = -1
                positions = torch.tensor(
                    [31, 99, 3, 201, 107, 255], device="cuda"
                ).int()

                def call():
                    q = state.project_q(attn.wq_b, qr)
                    return state.project_output(
                        state.prefill(q, kv.unsqueeze(1), indices, positions), attn.wo_b
                    )

                def reference():
                    q = attn.wq_b(qr).view(6, heads, 512)
                    output = _attention_reference(
                        q, kv, indices, positions, attn.freqs_cis, attn.attn_sink
                    )
                    return (
                        torch.einsum(
                            "mgk,grk->mgr",
                            output.view(6, heads // 8, 4096),
                            attn._wo_a_stk_w.float(),
                        )
                        .to(torch.bfloat16)
                        .flatten(1)
                    )

                for _ in range(3):
                    call()
                self._assert_projection_close(call(), reference())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = call()
                pointers = [value.data_ptr() for value in state.buffers()]
                # Reload changes derived Q/Wo weights, RoPE and sinks in place.
                attn.wq_b.weight.copy_(
                    (-attn.wq_b.weight.float()).to(attn.wq_b.weight.dtype)
                )
                attn._wo_a_stk_w.copy_(
                    (-attn._wo_a_stk_w.float()).to(attn._wo_a_stk_w.dtype)
                )
                attn.freqs_cis.mul_(complex(0, 1))
                attn.attn_sink.add_(1)
                state.reload_weights(attn)
                self.assertEqual(
                    pointers, [value.data_ptr() for value in state.buffers()]
                )
                qr.mul_(0.5)
                graph.replay()
                torch.testing.assert_close(captured, call(), atol=0, rtol=0)
                self._assert_projection_close(captured, reference())

    @torch.inference_mode()
    def test_dspark_five_queries_keep_noncausal_slots_and_lengths(self):
        from rtp_llm.models_py.model_desc import (
            deepseek_v41_base_dspark_model as dspark,
        )
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_swa_triton as swa

        attn = self._projection_fixture()
        state = attn._mega_attention
        batch, gamma, rows = 2, 5, 10
        qr = torch.randn(batch, gamma, 128, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(rows, 512, dtype=torch.bfloat16, device="cuda")
        pool = torch.zeros(1, 32, 528, dtype=torch.uint8, device="cuda")
        slots = torch.arange(rows, device="cuda")
        positions = torch.arange(31, 31 + rows, device="cuda").view(batch, gamma)
        # All five proposal KVs are visible even to the first query; positive
        # slots after the per-row length deliberately test length masking.
        indices = torch.zeros(rows, 128, dtype=torch.int32, device="cuda")
        indices[:, :gamma] = slots.view(batch, gamma).repeat_interleave(gamma, 0).int()
        lengths = torch.tensor([gamma, 0], dtype=torch.int32, device="cuda")
        model = SimpleNamespace(
            _swa_cache_bound=lambda *args: nullcontext((attn, 32, pool)),
            _proposal_metadata=lambda *args: (slots, indices, lengths),
        )

        def project(attention, value, query_positions, *, mega_attention):
            self.assertIs(mega_attention, state)
            torch.testing.assert_close(query_positions, positions.reshape(-1))
            return SimpleNamespace(q=state.project_q(attn.wq_b, value), kv=kv)

        with patch.object(dspark, "decode_compute_qkv", side_effect=project):
            actual = dspark.DeepSeekV4DSparkModel._forward_dspark_attention(
                model, 0, qr, positions, None, None, None, 32, None, {}
            )
        keys = swa.dequantize_swa_k_cache(pool, slots)
        q = attn.wq_b(qr).view(rows, 64, 512)
        reference = _attention_reference(
            q,
            keys,
            indices,
            positions.reshape(-1),
            attn.freqs_cis,
            attn.attn_sink,
            lengths.repeat_interleave(gamma),
        )
        reference = (
            torch.einsum(
                "mgk,grk->mgr", reference.view(rows, 8, 4096), attn._wo_a_stk_w.float()
            )
            .to(torch.bfloat16)
            .reshape(batch, gamma, -1)
        )
        self._assert_projection_close(actual, reference)

    @torch.inference_mode()
    def test_mega_fp8_pair_keeps_fused_wo_a_projection_and_graph(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_wo_a_quant
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        attn = self._projection_fixture(rank=1024)
        attn.wo_b = V41MXFP8Linear(
            (torch.randn(5120, 8192, device="cuda") * 0.02).to(torch.float8_e4m3fn),
            torch.ones(160, 256, device="cuda").to(torch.float8_e8m0fnu),
        )
        state = attn._mega_attention
        self.assertTrue(_v41_wo_a_quant.warmup(state.output_weight, state.output_scale))
        for rows in (1, 64, 512):
            with self.subTest(rows=rows):
                q = _q_fused(torch.randn(rows, 64, 512, device="cuda").bfloat16())
                kv = torch.randn(257, 1, 512, device="cuda").bfloat16()
                indices = torch.randint(0, 257, (rows, 128), device="cuda").int()
                positions = torch.arange(rows, device="cuda").int().remainder_(256)
                pair = state.prefill(q, kv, indices, positions)
                self.assertTrue(
                    _v41_wo_a_quant.is_supported(
                        pair, (state.output_weight, state.output_scale)
                    )
                )
                with patch.object(
                    _v41_wo_a_quant, "try_grouped_quant", return_value=None
                ):
                    expected = state.project_output(pair, attn.wo_b)
                out = torch.empty_like(expected)
                self.assertIs(state.project_output(pair, attn.wo_b, out=out), out)
                torch.testing.assert_close(out, expected, atol=0, rtol=0)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = state.project_output(pair, attn.wo_b)
                pair[0].zero_()
                graph.replay()
                torch.testing.assert_close(
                    captured, state.project_output(pair, attn.wo_b), atol=0, rtol=0
                )


if __name__ == "__main__":
    unittest.main()
