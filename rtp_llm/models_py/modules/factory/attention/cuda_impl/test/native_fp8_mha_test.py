"""SM100 integration coverage for Qwen3.5 FP8 Q and paged FP8 KV.

The reference starts with the same quantized inputs as the attention kernel.
FP8 attention's internal arithmetic is still lower precision than FP32 attention,
so quantization byte equality alone cannot establish attention accuracy. Check
both relative error and an absolute bound, and compare graph replay separately.
"""

import gc
import itertools
import math
import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_impl import trtllm_gen
from rtp_llm.models_py.modules.factory.attention.cuda_impl.trtllm_gen import (
    FlashInferTRTLLMDecodeImpl,
    FlashInferTRTLLMDecodeOp,
    FlashInferTRTLLMParams,
    FlashInferTRTLLMPrefillImpl,
    FlashInferTRTLLMPrefillOp,
    FlashInferTRTLLMSpecDecodeImpl,
)
from rtp_llm.models_py.triton_kernels.common.fused_fp8_qkv_cache import (
    fused_fp8_qkv_cache,
)
from rtp_llm.ops import AttentionConfigs, KvCacheDataType, RopeStyle
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCachePrefillOpQKVOut,
    FusedRopeKVCachePrefillOpQOut,
    LayerKVCache,
    PyAttentionInputs,
    get_typemeta,
)

HEAD_DIM = 256
QUERY_HEADS = 8
KV_HEADS = 1
PAGE_SIZE = 64
Q_DIM = QUERY_HEADS * HEAD_DIM
PACKED_DIM = (QUERY_HEADS + 2 * KV_HEADS) * HEAD_DIM


def _config():
    config = AttentionConfigs()
    config.dtype = torch.bfloat16
    config.kv_cache_dtype = KvCacheDataType.FP8
    config.need_rope_kv_cache = True
    config.rope_config.style = RopeStyle.Mrope
    config.size_per_head = HEAD_DIM
    config.head_num = QUERY_HEADS
    config.kv_head_num = KV_HEADS
    config.kernel_tokens_per_block = PAGE_SIZE
    return config


def _quantize(value, scale):
    return (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)


def _cumulative(lengths, device="cuda"):
    return torch.tensor(
        [0] + list(itertools.accumulate(lengths)), device=device, dtype=torch.int32
    )


class _AttentionInputsWithDeviceMirrors:
    """Supply mirrors normally initialized by the C++ model/graph runner.

    PyAttentionInputs exposes these device mirrors as read-only and its Python
    constructor leaves them undefined. Keep every other field on the actual
    binding, while exercising the real Python Impl and CUDA kernels below.
    """

    def __init__(self, inputs):
        self._inputs = inputs
        self.input_lengths_device = inputs.input_lengths.cuda()
        self.prefix_lengths_device = inputs.prefix_lengths.cuda()
        self.sequence_lengths_plus_1_device = inputs.sequence_lengths.cuda() + 1

    def __getattr__(self, name):
        return getattr(self._inputs, name)

    def __setattr__(self, name, value):
        if name in (
            "_inputs",
            "input_lengths_device",
            "prefix_lengths_device",
            "sequence_lengths_plus_1_device",
        ):
            object.__setattr__(self, name, value)
        else:
            setattr(self._inputs, name, value)


class NativeFP8GateTest(unittest.TestCase):
    def test_workspace_lease_uses_real_graph_flag_and_native_base_gate(self):
        config = _config()
        config.rope_config.index_factor = 1
        inputs = PyAttentionInputs()
        inputs.kv_cache_kernel_block_id_device = torch.zeros(4, 8, dtype=torch.int32)
        with (
            patch.object(
                trtllm_gen, "_g_trt_graph_workspaces", weakref.WeakValueDictionary()
            ),
            patch.object(trtllm_gen, "is_sm10x", return_value=True),
            patch.object(
                trtllm_gen, "get_trt_workspace_buffer", return_value=torch.empty(32)
            ) as allocate,
            patch.object(trtllm_gen, "release_trt_workspace_buffer") as release,
        ):
            for enabled, graph, style, expected in (
                ("1", False, RopeStyle.Base, False),
                ("0", True, RopeStyle.Base, False),
                ("1", True, RopeStyle.Mrope, False),
                ("1", True, RopeStyle.Base, True),
            ):
                with (
                    self.subTest(enabled=enabled, graph=graph, style=style),
                    patch.dict(os.environ, {"RTP_QWEN35_NATIVE_FP8_ATTN": enabled}),
                ):
                    inputs.is_cuda_graph = graph
                    config.rope_config.style = style
                    lease = trtllm_gen._native_base_graph_workspace(config, inputs)
                    self.assertEqual(lease is not None, expected)
                    del lease
            allocate.assert_called_once_with(device="cpu")
            release.assert_called_once()

    def test_workspace_lease_shares_storage_views_and_releases_once(self):
        config = _config()
        config.rope_config.style = RopeStyle.Base
        config.rope_config.index_factor = 1
        tables = torch.zeros(4, 8, dtype=torch.int32)
        inputs = PyAttentionInputs()
        inputs.is_cuda_graph = True
        inputs.kv_cache_kernel_block_id_device = tables[:2]
        with (
            patch.dict(os.environ, {"RTP_QWEN35_NATIVE_FP8_ATTN": "1"}),
            patch.object(
                trtllm_gen, "_g_trt_graph_workspaces", weakref.WeakValueDictionary()
            ) as leases,
            patch.object(trtllm_gen, "is_sm10x", return_value=True),
            patch.object(
                trtllm_gen,
                "get_trt_workspace_buffer",
                side_effect=lambda **_: torch.empty(32),
            ) as allocate,
            patch.object(trtllm_gen, "release_trt_workspace_buffer") as release,
        ):
            first = trtllm_gen._native_base_graph_workspace(config, inputs)
            inputs.kv_cache_kernel_block_id_device = tables[1:]
            second = trtllm_gen._native_base_graph_workspace(config, inputs)
            self.assertIs(first, second)
            inputs.kv_cache_kernel_block_id_device = tables.clone()
            other = trtllm_gen._native_base_graph_workspace(config, inputs)
            self.assertIsNot(first, other)
            self.assertNotEqual(first.buffer.data_ptr(), other.buffer.data_ptr())
            self.assertEqual(allocate.call_count, 2)
            owner = weakref.ref(first)
            prefill = FlashInferTRTLLMPrefillOp(config, first)
            verify = FlashInferTRTLLMDecodeOp(config, second)
            shared_buffer = first.buffer
            del first, second, tables
            gc.collect()
            self.assertIsNotNone(owner())
            self.assertGreater(owner().storage.nbytes(), 0)
            del prefill
            gc.collect()
            release.assert_not_called()
            self.assertIs(verify.workspace_buffer, shared_buffer)
            del verify
            gc.collect()
            self.assertIsNone(owner())
            release.assert_called_once_with(shared_buffer)
            del other
            gc.collect()
            self.assertEqual(release.call_count, 2)
            self.assertEqual(len(leases), 0)

    def test_gate_includes_only_single_axis_base_rope(self):
        with (
            patch.dict(os.environ, {"RTP_QWEN35_NATIVE_FP8_ATTN": "1"}),
            patch.object(trtllm_gen, "is_sm10x", return_value=True),
        ):
            for style, axes, expected in (
                (RopeStyle.Mrope, 3, True),
                (RopeStyle.Base, 1, True),
                (RopeStyle.Base, 3, False),
                (RopeStyle.No, 1, False),
            ):
                with self.subTest(style=style, axes=axes):
                    config = _config()
                    config.rope_config.style = style
                    config.rope_config.index_factor = axes
                    self.assertEqual(
                        trtllm_gen.use_native_fp8_attention(config), expected
                    )
            config.rope_config.style = RopeStyle.Base
            with patch.dict(os.environ, {"RTP_QWEN35_NATIVE_FP8_ATTN": "0"}):
                self.assertFalse(trtllm_gen.use_native_fp8_attention(config))

    def test_base_position_buffer_is_prepared_once_and_preserves_explicit_ids(self):
        config = _config()
        config.rope_config.style = RopeStyle.Base
        config.rope_config.index_factor = 1
        explicit = torch.arange(15, dtype=torch.int32)
        for positions in (None, torch.empty(0, dtype=torch.int32), explicit):
            with self.subTest(positions=positions):
                calls = []
                inputs = SimpleNamespace(
                    cu_seqlens_device=torch.tensor([0, 5, 10, 15], dtype=torch.int32)
                )
                impl = SimpleNamespace(
                    native_fp8=True,
                    attn_configs=config,
                    attn_inputs=inputs,
                    rope_params=SimpleNamespace(position_ids=positions, max_seq_len=5),
                    _cg=trtllm_gen._init_decode_cg_params(
                        3, torch.empty(3, 8), torch.empty(3), torch.empty(3)
                    ),
                    prepare_cuda_graph=calls.append,
                )
                trtllm_gen._init_native_base_positions(impl)
                if positions is explicit:
                    self.assertIs(impl.rope_params.position_ids, explicit)
                    self.assertEqual(impl._base_position_max_q, 0)
                    self.assertEqual(calls, [])
                else:
                    self.assertEqual(impl.rope_params.position_ids.shape, (15,))
                    self.assertEqual(impl.rope_params.position_ids.dtype, torch.int32)
                    self.assertEqual(impl._base_position_max_q, 5)
                    self.assertEqual(calls, [inputs])

    def test_disabled_gate_preserves_bf16_query_and_metadata(self):
        query = torch.randn(5, QUERY_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        cache = LayerKVCache()
        cache.kv_cache_base = torch.empty(
            1, 2, KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=torch.float8_e4m3fn
        )
        params = FlashInferTRTLLMParams(
            batch_size=1,
            max_q_len=5,
            max_kv_len=10,
            max_seq_len=10,
            seq_lens=torch.tensor([10], dtype=torch.int32),
            block_tables=torch.tensor([[0]], dtype=torch.int32),
            cu_seqlens=_cumulative([5], "cpu"),
            cu_kv_seqlens=_cumulative([10], "cpu"),
        )
        for op_class, namespace, kernel_name in (
            (
                FlashInferTRTLLMPrefillOp,
                trtllm_gen.flashinfer.prefill,
                "trtllm_batch_context_with_kv_cache",
            ),
            (
                FlashInferTRTLLMDecodeOp,
                trtllm_gen.flashinfer.decode,
                "trtllm_batch_decode_with_kv_cache",
            ),
        ):
            with (
                self.subTest(op=op_class.__name__),
                patch.dict(os.environ, {"RTP_QWEN35_NATIVE_FP8_ATTN": "0"}),
                patch.object(trtllm_gen, "is_sm10x", return_value=True),
                patch.object(
                    trtllm_gen, "get_trt_workspace_buffer", return_value=torch.empty(0)
                ),
                patch.object(trtllm_gen, "release_trt_workspace_buffer"),
                patch.object(
                    trtllm_gen,
                    "quantize_fp8_query",
                    side_effect=AssertionError("Disabled path must retain BF16 Q"),
                ),
                patch.object(
                    namespace, kernel_name, return_value=query.clone()
                ) as call,
            ):
                op = op_class(_config())
                self.assertFalse(op.native_fp8)
                output = op.forward(query, cache, params)
                args = call.call_args.kwargs
                self.assertEqual(args["query"].dtype, torch.bfloat16)
                self.assertEqual(args["query"].data_ptr(), query.data_ptr())
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertIs(args["seq_lens"], params.seq_lens)
                self.assertIs(args["block_tables"], params.block_tables)
                self.assertEqual(args["bmm1_scale"], HEAD_DIM**-0.5)
                self.assertEqual(args["bmm2_scale"], 1.0)
                if op_class is FlashInferTRTLLMPrefillOp:
                    self.assertIs(args["cum_seq_lens_q"], params.cu_seqlens)
                    self.assertIs(args["cum_seq_lens_kv"], params.cu_kv_seqlens)
                else:
                    self.assertEqual(args["q_len_per_req"], 5)
                del op


class NativeFP8MHAIntegrationTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not trtllm_gen.is_sm10x():
            self.skipTest("TRTLLM Gen FP8 Q/KV integration requires SM100")
        torch.manual_seed(20260920)
        env = patch.dict(os.environ, {"RTP_QWEN35_NATIVE_FP8_ATTN": "1"})
        env.start()
        self.addCleanup(env.stop)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", old_tf32)

    def _case(self, lengths, prefixes, scales=(1.0, 1.0, 1.0), page_columns=None):
        kv_lengths = [q + p for q, p in zip(lengths, prefixes)]
        columns = page_columns or math.ceil(max(kv_lengths) / PAGE_SIZE)
        batch = len(lengths)
        table = torch.randperm(batch * columns, device="cuda").to(torch.int32)
        table = table.reshape(batch, columns)
        qkv = torch.randn(
            sum(lengths), PACKED_DIM + 128, device="cuda", dtype=torch.bfloat16
        )[:, :PACKED_DIM]
        cache = torch.zeros(
            batch * columns,
            2,
            KV_HEADS,
            PAGE_SIZE,
            HEAD_DIM,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        dense_kv = []
        offset = 0
        for b, (length, prefix) in enumerate(zip(lengths, prefixes)):
            k = torch.randn(columns * PAGE_SIZE, 1, HEAD_DIM, device="cuda").bfloat16()
            v = torch.randn_like(k)
            tail = qkv[offset : offset + length, Q_DIM:]
            k[prefix : prefix + length] = tail[:, :HEAD_DIM].reshape(
                length, 1, HEAD_DIM
            )
            v[prefix : prefix + length] = tail[:, HEAD_DIM:].reshape(
                length, 1, HEAD_DIM
            )
            dense_kv.append((k[: prefix + length], v[: prefix + length]))
            for component, (tensor, scale) in enumerate(zip((k, v), scales[1:])):
                cached = _quantize(tensor, scale)
                # The fused helper must replace these poison values with the new KV.
                cached[prefix : prefix + length] = 29.0
                cache[table[b].long(), component] = cached.reshape(
                    columns, PAGE_SIZE, KV_HEADS, HEAD_DIM
                ).permute(0, 2, 1, 3)
            offset += length
        layer_cache = LayerKVCache()
        layer_cache.kv_cache_base = cache
        return SimpleNamespace(
            lengths=lengths,
            prefixes=prefixes,
            kv_lengths=kv_lengths,
            scales=scales,
            qkv=qkv,
            cache=cache,
            layer_cache=layer_cache,
            table=table,
            cu=_cumulative(lengths),
            prefix=torch.tensor(prefixes, device="cuda", dtype=torch.int32),
            dense_kv=dense_kv,
        )

    def _prepare(self, case, decode):
        inputs = SimpleNamespace(
            is_prefill=not decode or case.lengths[0] > 1,
            input_lengths=torch.tensor(case.lengths, dtype=torch.int32),
            prefix_lengths=torch.tensor(case.prefixes, dtype=torch.int32),
            sequence_lengths=torch.tensor(case.prefixes, dtype=torch.int32),
            cu_seqlens_device=case.cu,
            kv_cache_kernel_block_id_device=case.table,
        )
        op_class = FlashInferTRTLLMDecodeOp if decode else FlashInferTRTLLMPrefillOp
        op = op_class(_config())
        self.assertTrue(op.native_fp8)
        params = op.prepare(inputs)
        torch.testing.assert_close(
            params.seq_lens,
            torch.tensor(case.kv_lengths, device="cuda", dtype=torch.int32),
        )
        return op, params

    def _forward(self, case, op, params, fused=True):
        q_scale, k_scale, v_scale = case.scales
        query = fused_fp8_qkv_cache(
            case.qkv,
            case.cache,
            case.table,
            case.cu,
            case.prefix,
            num_q_heads=QUERY_HEADS,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            page_size=PAGE_SIZE,
            max_query_len=max(case.lengths),
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )
        if not fused:
            # Also cover DecodeOp's independent quantizer for its native RoPE path.
            query = case.qkv[:, :Q_DIM].reshape(-1, QUERY_HEADS, HEAD_DIM)
        return op.forward(
            query,
            case.layer_cache,
            params,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        ).reshape(-1, QUERY_HEADS, HEAD_DIM)

    def _reference(self, case):
        refs, row_ids = [], []
        offset = 0
        for length, prefix, (k, v) in zip(case.lengths, case.prefixes, case.dense_kv):
            # Full 16661-token prefill is executed, but sample FP32 reference rows
            # to avoid creating an O(N^2) attention matrix in the test.
            indices = sorted(
                {
                    0,
                    min(1, length - 1),
                    min(63, length - 1),
                    min(64, length - 1),
                    min(511, length - 1),
                    length - 1,
                }
            )
            rows = torch.tensor([offset + i for i in indices], device="cuda")
            q = case.qkv[rows, :Q_DIM].reshape(-1, QUERY_HEADS, HEAD_DIM)
            q = _quantize(q, case.scales[0]).float() * case.scales[0]
            k = _quantize(k, case.scales[1]).float() * case.scales[1]
            v = _quantize(v, case.scales[2]).float() * case.scales[2]
            logits = torch.einsum("qhd,khd->hqk", q, k.expand(-1, QUERY_HEADS, -1))
            logits *= HEAD_DIM**-0.5
            causal = (
                torch.arange(prefix + length, device="cuda")[None, :]
                <= (prefix + torch.tensor(indices, device="cuda"))[:, None]
            )
            weights = torch.softmax(logits.masked_fill(~causal[None], -torch.inf), -1)
            refs.append(
                torch.einsum("hqk,khd->qhd", weights, v.expand(-1, QUERY_HEADS, -1))
            )
            row_ids.extend(offset + i for i in indices)
            offset += length
        return torch.tensor(row_ids, device="cuda"), torch.cat(refs)

    def _assert_accuracy(self, output, case):
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all().item())
        rows, reference = self._reference(case)
        actual = output[rows].float()
        error = actual - reference
        # Native FP8-Q/KV vs FP32 on the same quantized inputs measured up to
        # 2.6% relative L2 on SM100. This is kernel arithmetic error, in addition
        # to input quantization; it is not permission for cache/metadata errors.
        # Check each request independently: an exact short request must not hide
        # corrupt pages or stale metadata in the much longer requests.
        offset = 0
        sample_ids = rows.tolist()
        for length in case.lengths:
            mask = [offset <= row < offset + length for row in sample_ids]
            relative_l2 = (
                error[mask].norm() / reference[mask].norm().clamp_min(1e-20)
            ).item()
            self.assertLess(relative_l2, 0.04)
            offset += length
        self.assertLess(error.abs().max().item(), 0.12)
        self.assertGreater(
            torch.nn.functional.cosine_similarity(
                actual.flatten(), reference.flatten(), dim=0
            ).item(),
            0.999,
        )

    def _run_case(self, lengths, prefixes, *, decode=False, scales=(1.0, 1.0, 1.0)):
        case = self._case(lengths, prefixes, scales)
        op, params = self._prepare(case, decode)
        namespace = (
            trtllm_gen.flashinfer.decode if decode else trtllm_gen.flashinfer.prefill
        )
        kernel = (
            "trtllm_batch_decode_with_kv_cache"
            if decode
            else "trtllm_batch_context_with_kv_cache"
        )
        with patch.object(namespace, kernel, wraps=getattr(namespace, kernel)) as call:
            output = self._forward(case, op, params)
        self.assertEqual(call.call_args.kwargs["query"].dtype, torch.float8_e4m3fn)
        self.assertEqual(call.call_args.kwargs["out_dtype"], torch.bfloat16)
        self.assertAlmostEqual(
            call.call_args.kwargs["bmm1_scale"],
            scales[0] * scales[1] / math.sqrt(HEAD_DIM),
        )
        self.assertEqual(call.call_args.kwargs["bmm2_scale"], scales[2])
        self._assert_accuracy(output, case)
        direct_query_output = self._forward(case, op, params, fused=False)
        torch.testing.assert_close(direct_query_output, output, rtol=0, atol=0)

    def test_prefill_without_prefix(self):
        self._run_case([65, 129], [0, 0])

    def test_prefill_with_long_prefix(self):
        self._run_case([5, 129], [16656, 16532])

    def test_full_video_length_prefill(self):
        self._run_case([16661], [0])

    def test_decode_one_token(self):
        self._run_case([1, 1, 1], [0, 128, 16660], decode=True)

    def test_mtp_five_token_decode(self):
        self._run_case([5, 5, 5], [0, 124, 16656], decode=True)

    def test_nonunit_scales_prefill_and_mtp(self):
        for decode, lengths, prefixes in (
            (False, [65, 129], [64, 16532]),
            (True, [5, 5], [252, 16656]),
        ):
            with self.subTest(decode=decode):
                self._run_case(lengths, prefixes, decode=decode, scales=(0.5, 0.3, 2.0))

    def test_cuda_graph_replay_updates_lengths_and_page_tables(self):
        for decode, lengths in ((False, [65, 129]), (True, [1, 1]), (True, [5, 5])):
            with self.subTest(decode=decode, lengths=lengths):
                case = self._case(lengths, [0, 64], page_columns=5)
                changed = self._case(lengths, [127, 128], page_columns=5)
                op, params = self._prepare(case, decode)
                params.max_seq_len = params.max_kv_len = 5 * PAGE_SIZE
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._forward(case, op, params)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = self._forward(case, op, params)
                graph.replay()
                self._assert_accuracy(captured, case)
                # Keep graph addresses fixed while changing both the logical
                # prefix lengths and physical placement of every sequence.
                case.qkv.copy_(changed.qkv)
                case.cache.copy_(changed.cache)
                case.table.copy_(changed.table)
                case.prefix.copy_(changed.prefix)
                params.seq_lens.copy_(
                    torch.tensor(changed.kv_lengths, device="cuda", dtype=torch.int32)
                )
                if params.cu_kv_seqlens is not None:
                    params.cu_kv_seqlens.copy_(_cumulative(changed.kv_lengths))
                case.prefixes = changed.prefixes
                case.kv_lengths = changed.kv_lengths
                case.dense_kv = changed.dense_kv
                for _ in range(3):
                    graph.replay()
                self._assert_accuracy(captured, case)
                eager = self._forward(case, op, params)
                torch.testing.assert_close(captured, eager, rtol=0, atol=0)
                del graph, captured, op

    def _real_mtp_inputs(self, lengths, prefixes, table, positions):
        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        inputs.is_target_verify = True
        inputs.is_s_padded = True
        inputs.is_cuda_graph = True
        inputs.dtype = get_typemeta(torch.empty(0, dtype=torch.bfloat16))
        inputs.input_lengths = torch.tensor(lengths, dtype=torch.int32).pin_memory()
        inputs.prefix_lengths = torch.tensor(prefixes, dtype=torch.int32).pin_memory()
        inputs.sequence_lengths = inputs.prefix_lengths.clone().pin_memory()
        inputs.cu_seqlens = _cumulative(lengths, "cpu").pin_memory()
        inputs.cu_seqlens_device = inputs.cu_seqlens.cuda()
        inputs.cu_kv_seqlens_device = _cumulative(
            [prefix + length for prefix, length in zip(prefixes, lengths)]
        )
        inputs.context_total_kv_length = sum(prefixes) + sum(lengths)
        inputs.total_tokens = sum(lengths)
        inputs.padding_offset = torch.zeros(
            len(lengths) * 5, dtype=torch.int32
        ).pin_memory()
        inputs.combo_position_ids = positions.clone()
        inputs.kv_cache_kernel_block_id = table.cpu().pin_memory()
        inputs.kv_cache_kernel_block_id_device = table.clone()
        return _AttentionInputsWithDeviceMirrors(inputs)

    @staticmethod
    def _real_mtp_config():
        config = _config()
        config.max_seq_len = 2048
        config.tokens_per_block = PAGE_SIZE
        config.rope_config.dim = 64
        config.rope_config.base = 10000000
        config.rope_config.index_factor = 3
        config.rope_config.mrope_dim1 = 11
        config.rope_config.mrope_dim2 = 11
        config.rope_config.mrope_dim3 = 10
        return config

    @staticmethod
    def _mtp_positions(prefixes):
        # Distinct, positive axes exercise actual mRoPE rather than identity
        # rotation. Target verification follows an existing nonempty prefix.
        return torch.tensor(
            [
                [prefix + token + 1, prefix + token + 7, prefix + token + 19]
                for prefix in prefixes
                for token in range(5)
            ],
            device="cuda",
            dtype=torch.int32,
        ).flatten()

    def _real_mtp_reference(self, config, source, cache, table, prefixes):
        live_rows = len(prefixes) * 5
        inputs = self._real_mtp_inputs(
            [5] * len(prefixes),
            prefixes,
            table[: len(prefixes)],
            self._mtp_positions(prefixes),
        )
        reference_cache = LayerKVCache()
        reference_cache.kv_cache_base = cache.clone()
        reference_cache.kv_scale_base = torch.ones(
            cache.shape[0], 2 * KV_HEADS * PAGE_SIZE, device="cuda"
        )
        # Use the actual legacy c308 QOut writer on unpadded requests. Its
        # padding-row writes to reserved page zero are not part of the oracle.
        rope = FusedRopeKVCachePrefillOpQOut(config)
        query = rope.forward(
            source[:live_rows].clone(), reference_cache, rope.prepare(inputs)
        )
        dense_kv = []
        for row, prefix in enumerate(prefixes):
            pages = reference_cache.kv_cache_base[table[row].long()]
            dense_kv.append(
                tuple(
                    pages[:, component]
                    .permute(0, 2, 1, 3)
                    .reshape(-1, KV_HEADS, HEAD_DIM)[: prefix + 5]
                    for component in (0, 1)
                )
            )
        reference = SimpleNamespace(
            lengths=[5] * len(prefixes),
            prefixes=prefixes,
            qkv=query.reshape(live_rows, Q_DIM),
            dense_kv=dense_kv,
            scales=(1.0, 1.0, 1.0),
        )
        return query, reference_cache.kv_cache_base, reference

    def _run_real_mtp_graph(self, *, poison_unused_v=False):
        batch, query_len, page_columns = 4, 5, 8
        config = self._real_mtp_config()
        # Page zero is reserved, exactly as in the serving graph runner.
        tables = torch.arange(
            1, 1 + batch * page_columns, device="cuda", dtype=torch.int32
        ).reshape(batch, page_columns)
        prefixes = [63, 124, 190, 253]
        inputs = self._real_mtp_inputs(
            [query_len] * batch, prefixes, tables, self._mtp_positions(prefixes)
        )
        source = torch.randn(
            batch * query_len, PACKED_DIM, device="cuda", dtype=torch.bfloat16
        )
        qkv = torch.empty_like(source)
        layer_cache = LayerKVCache()
        layer_cache.kv_cache_base = torch.randn(
            1 + batch * page_columns,
            2,
            KV_HEADS,
            PAGE_SIZE,
            HEAD_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        ).to(torch.float8_e4m3fn)
        layer_cache.kv_scale_base = torch.ones(
            layer_cache.kv_cache_base.shape[0],
            2 * KV_HEADS * PAGE_SIZE,
            device="cuda",
        )
        impl = FlashInferTRTLLMSpecDecodeImpl(config, inputs)
        self.assertTrue(impl.native_fp8)
        self.assertIsInstance(impl.rope_kvcache_impl, FusedRopeKVCachePrefillOpQKVOut)
        impl.fmha_params.max_seq_len = page_columns * PAGE_SIZE
        queries = []
        original_forward = impl.fmha_impl.forward

        def observe_query(query, *args, **kwargs):
            queries.append(query)
            return original_forward(query, *args, **kwargs)

        def forward():
            # QKVOut rotates in place. Every model step produces fresh QKV;
            # copying inside the graph reproduces that contract on each replay.
            qkv.copy_(source)
            return impl.forward(qkv, layer_cache, 0)

        with patch.object(impl.fmha_impl, "forward", new=observe_query):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                impl.prepare_cuda_graph(inputs)
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = forward()
            captured_query = queries[-1]
            metadata = (
                inputs.input_lengths_device,
                inputs.prefix_lengths_device,
                inputs.cu_seqlens_device,
                inputs.kv_cache_kernel_block_id_device,
                inputs.combo_position_ids,
            )
            pointers = tuple(tensor.data_ptr() for tensor in metadata)
            # Retain the cache across stages. Roll page ownership so a new
            # request uses storage previously written by a different request.
            for stage, live_prefixes in enumerate(
                ([63, 124, 190], [125, 191], [60, 128, 254])
            ):
                with self.subTest(stage=stage, poison_unused_v=poison_unused_v):
                    live = len(live_prefixes)
                    lengths = [query_len] * live + [0] * (batch - live)
                    padded_prefixes = list(live_prefixes) + [0] * (batch - live)
                    table = tables.roll(stage, 0).roll(stage, 1)
                    table[live:].zero_()
                    updated = self._real_mtp_inputs(
                        lengths,
                        padded_prefixes,
                        table,
                        self._mtp_positions(padded_prefixes),
                    )
                    for field in (
                        "input_lengths",
                        "prefix_lengths",
                        "sequence_lengths",
                        "input_lengths_device",
                        "prefix_lengths_device",
                        "cu_seqlens",
                        "cu_seqlens_device",
                        "cu_kv_seqlens_device",
                        "kv_cache_kernel_block_id",
                        "kv_cache_kernel_block_id_device",
                        "combo_position_ids",
                    ):
                        getattr(inputs, field).copy_(getattr(updated, field))
                    inputs.context_total_kv_length = updated.context_total_kv_length
                    inputs.total_tokens = updated.total_tokens
                    source.normal_()
                    # Padded input poison must never reach valid Q or any KV page.
                    source[live * query_len :].fill_(float("nan"))
                    if poison_unused_v:
                        for row, prefix in enumerate(live_prefixes):
                            column, offset = divmod(prefix + query_len, PAGE_SIZE)
                            if offset:
                                page = int(table[row, column].item())
                                layer_cache.kv_cache_base[page, 1, :, offset:].fill_(
                                    float("nan")
                                )
                    query_ref, cache_ref, reference = self._real_mtp_reference(
                        config, source, layer_cache.kv_cache_base, table, live_prefixes
                    )
                    # The only intentional difference from c308 QOut is clearing
                    # the unused V suffix of the final page. Keep K tails, other
                    # pages and reserved page zero byte-exact in this oracle.
                    for row, prefix in enumerate(live_prefixes):
                        column, offset = divmod(prefix + query_len, PAGE_SIZE)
                        if offset:
                            page = int(table[row, column].item())
                            cache_ref[page, 1, :, offset:].zero_()
                    impl.prepare_cuda_graph(inputs)
                    for _ in range(2):
                        graph.replay()
                    valid = live * query_len
                    self.assertEqual(
                        tuple(tensor.data_ptr() for tensor in metadata), pointers
                    )
                    self.assertTrue(
                        torch.equal(
                            captured_query[:valid].view(torch.uint8),
                            _quantize(query_ref, 1.0).view(torch.uint8),
                        ),
                        "Fused query differs from the actual c308 QOut query",
                    )
                    self.assertEqual(
                        torch.count_nonzero(captured_query[valid:].float()).item(), 0
                    )
                    self.assertTrue(
                        torch.equal(
                            layer_cache.kv_cache_base.view(torch.uint8),
                            cache_ref.view(torch.uint8),
                        ),
                        "Cache must match c308 valid writes and zero only unused V tails",
                    )
                    self._assert_accuracy(
                        captured[:valid].reshape(-1, QUERY_HEADS, HEAD_DIM), reference
                    )
                    eager = forward()
                    torch.testing.assert_close(
                        captured[:valid], eager[:valid], rtol=0, atol=0
                    )

    def test_real_mtp_impl_graph_padding_and_reused_pages(self):
        self._run_real_mtp_graph()

    def test_real_mtp_impl_graph_does_not_read_nan_v_tail(self):
        self._run_real_mtp_graph(poison_unused_v=True)

    def _base_inputs(
        self,
        lengths,
        prefixes,
        table,
        *,
        capacity,
        decode=False,
        implicit_positions=False,
    ):
        # Pack valid tokens; the graph's remaining allocation is padding. Use
        # positions different from cache offsets to test the explicit-ID path.
        positions, padding_offsets = [], []
        for batch, (length, prefix) in enumerate(zip(lengths, prefixes)):
            for token in range(length):
                padding_offsets.append(batch * capacity + token - len(positions))
                positions.append(prefix + token + 11)
        padding = len(lengths) * capacity - len(positions)
        positions += [0] * padding
        padding_offsets += [0] * padding
        inputs = self._real_mtp_inputs(
            lengths,
            prefixes,
            table,
            torch.tensor(positions, device="cuda", dtype=torch.int32),
        )
        inputs.is_prefill = not decode
        inputs.is_target_verify = False
        inputs.padding_offset = torch.tensor(
            padding_offsets, device="cuda", dtype=torch.int32
        )
        inputs.sequence_lengths_plus_1_device.copy_(
            torch.tensor(
                [prefix + length for prefix, length in zip(prefixes, lengths)],
                device="cuda",
                dtype=torch.int32,
            )
        )
        if implicit_positions:
            inputs.combo_position_ids = torch.empty(0, dtype=torch.int32, device="cuda")
        if decode:
            # PyWrappedModel's pure decode has empty prefill metadata. Its
            # separate decode_cu_seqlens is not the prefill cu_seqlens pointer.
            inputs.cu_seqlens.zero_()
            inputs.cu_seqlens_device.zero_()
            inputs.cu_kv_seqlens_device.zero_()
            inputs.total_tokens = 0
        return inputs

    def _base_reference(
        self,
        config,
        source,
        cache,
        table,
        lengths,
        prefixes,
        *,
        decode=False,
        implicit_positions=False,
    ):
        valid = sum(lengths)
        inputs = self._base_inputs(
            lengths,
            prefixes,
            table[: len(lengths)],
            capacity=max(lengths),
            implicit_positions=implicit_positions,
        )
        if decode and not implicit_positions:
            # c308's cached Base decode rotates at sequence_lengths and ignores
            # combo_position_ids. QOut uses explicit IDs, so align its oracle
            # positions with that unchanged decode contract, not prefill's +11.
            self.assertTrue(all(length == 1 for length in lengths))
            inputs.combo_position_ids.copy_(
                torch.tensor(prefixes, device="cuda", dtype=torch.int32)
            )
        reference_cache = LayerKVCache()
        reference_cache.kv_cache_base = cache.clone()
        reference_cache.kv_scale_base = torch.ones(
            cache.shape[0], 2 * KV_HEADS * PAGE_SIZE, device="cuda"
        )
        # QOut is the actual c308 RoPE/cache writer, independent of the new
        # QKVOut + Triton route and also a q=1 oracle for ordinary draft decode.
        rope = FusedRopeKVCachePrefillOpQOut(config)
        query = rope.forward(
            source[:valid].clone(), reference_cache, rope.prepare(inputs)
        )
        dense_kv = []
        for row, (length, prefix) in enumerate(zip(lengths, prefixes)):
            pages = reference_cache.kv_cache_base[table[row].long()]
            dense_kv.append(
                tuple(
                    pages[:, component]
                    .permute(0, 2, 1, 3)
                    .reshape(-1, KV_HEADS, HEAD_DIM)[: prefix + length]
                    for component in (0, 1)
                )
            )
            column, offset = divmod(prefix + length, PAGE_SIZE)
            if offset:
                page = int(table[row, column].item())
                reference_cache.kv_cache_base[page, 1, :, offset:].zero_()
        reference = SimpleNamespace(
            lengths=lengths,
            prefixes=prefixes,
            qkv=query.reshape(valid, Q_DIM),
            dense_kv=dense_kv,
            scales=(1.0, 1.0, 1.0),
        )
        return query, reference_cache.kv_cache_base, reference

    def _run_base_draft_graph(
        self, *, decode, implicit_positions=False, speculative=False
    ):
        batch, columns, capacity = 4, 8, 1 if decode else 5
        config = self._real_mtp_config()
        config.max_seq_len = columns * PAGE_SIZE
        config.rope_config.style = RopeStyle.Base
        config.rope_config.index_factor = 1
        tables = torch.arange(
            1, 1 + batch * columns, device="cuda", dtype=torch.int32
        ).reshape(batch, columns)
        inputs = self._base_inputs(
            [capacity] * batch,
            [config.max_seq_len - capacity] * batch,
            tables,
            capacity=capacity,
            decode=decode,
            implicit_positions=implicit_positions,
        )
        # Match CudaGraphRunner.capturePrefill: initialize the largest legal
        # prefix before constructing the attention implementation. Production
        # also uses dummy position IDs during capture; replay supplies real IDs.
        inputs.combo_position_ids.zero_()
        source = torch.randn(
            batch * capacity, PACKED_DIM, device="cuda", dtype=torch.bfloat16
        )
        qkv = torch.empty_like(source)
        cache = LayerKVCache()
        cache.kv_cache_base = torch.randn(
            1 + batch * columns, 2, KV_HEADS, PAGE_SIZE, HEAD_DIM, device="cuda"
        ).to(torch.float8_e4m3fn)
        cache.kv_scale_base = torch.ones(
            cache.kv_cache_base.shape[0], 2 * KV_HEADS * PAGE_SIZE, device="cuda"
        )
        impl_class = FlashInferTRTLLMPrefillImpl
        if decode:
            impl_class = FlashInferTRTLLMDecodeImpl
        elif speculative:
            impl_class = FlashInferTRTLLMSpecDecodeImpl
        if implicit_positions and not decode:
            # Production first performs a separate datatype warmup with zero
            # prefixes, then constructs each actual graph at maximum prefix.
            # A c308 PREFIX_PROMPT=False warmup specialization must not be
            # mistaken for the implementation captured for later nonzero KV.
            warmup_inputs = self._base_inputs(
                [capacity] * batch,
                [0] * batch,
                tables,
                capacity=capacity,
                implicit_positions=True,
            )
            warmup_impl = impl_class(config, warmup_inputs)
            _, warmup_cache_ref, warmup_reference = self._base_reference(
                config,
                source,
                cache.kv_cache_base,
                tables,
                [capacity] * batch,
                [0] * batch,
                implicit_positions=True,
            )
            warmup_output = warmup_impl.forward(source.clone(), cache, 0)
            self._assert_accuracy(
                warmup_output.reshape(-1, QUERY_HEADS, HEAD_DIM), warmup_reference
            )
            self.assertTrue(
                torch.equal(
                    cache.kv_cache_base.view(torch.uint8),
                    warmup_cache_ref.view(torch.uint8),
                )
            )
            del warmup_impl
        impl = impl_class(config, inputs)
        self.assertTrue(impl.fmha_impl.native_fp8)
        if decode:
            self.assertIsNone(impl.fmha_params.cu_seqlens)
        if implicit_positions and not decode:
            self.assertEqual(impl.rope_params.position_ids.numel(), batch * capacity)
            torch.testing.assert_close(
                impl.rope_params.position_ids,
                torch.tensor(
                    list(range(config.max_seq_len - capacity, config.max_seq_len))
                    * batch,
                    device="cuda",
                    dtype=torch.int32,
                ),
                rtol=0,
                atol=0,
            )
        if not decode:
            self.assertIsInstance(
                impl.rope_kvcache_impl, FusedRopeKVCachePrefillOpQKVOut
            )
        self.assertEqual(
            (
                impl.fmha_params.max_seq_len
                if decode or speculative
                else impl.fmha_params.max_kv_len
            ),
            config.max_seq_len,
        )
        namespace = (
            trtllm_gen.flashinfer.decode
            if decode or speculative
            else trtllm_gen.flashinfer.prefill
        )
        kernel_name = (
            "trtllm_batch_decode_with_kv_cache"
            if decode or speculative
            else "trtllm_batch_context_with_kv_cache"
        )
        kernel = getattr(namespace, kernel_name)
        queries = []

        def observe_query(*args, **kwargs):
            queries.append(kwargs["query"])
            self.assertEqual(kwargs["query"].dtype, torch.float8_e4m3fn)
            self.assertEqual(kwargs["out_dtype"], torch.bfloat16)
            return kernel(*args, **kwargs)

        def forward():
            qkv.copy_(source)
            return impl.forward(qkv, cache, 0)

        with patch.object(namespace, kernel_name, new=observe_query):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                impl.prepare_cuda_graph(inputs)
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = forward()
            captured_query = queries[-1]
            fields = (
                "input_lengths",
                "prefix_lengths",
                "sequence_lengths",
                "input_lengths_device",
                "prefix_lengths_device",
                "sequence_lengths_plus_1_device",
                "cu_seqlens",
                "cu_seqlens_device",
                "cu_kv_seqlens_device",
                "kv_cache_kernel_block_id",
                "kv_cache_kernel_block_id_device",
                "combo_position_ids",
                "padding_offset",
            )
            pointers = tuple(getattr(inputs, field).data_ptr() for field in fields)
            position_pointer = impl.rope_params.position_ids.data_ptr()
            for stage, (lengths, prefixes) in enumerate(
                (
                    ([1, 3, 5], [63, 126, 190]),
                    ([2, 1], [127, 255]),
                    ([5, 2, 4], [60, 191, 254]),
                )
            ):
                with self.subTest(decode=decode, stage=stage):
                    if decode:
                        lengths = [1] * len(lengths)
                    elif speculative:
                        lengths = [capacity] * len(lengths)
                        if stage == 0:
                            # Keep this page full so its old poisoned tail
                            # cannot become another request's valid prefix.
                            prefixes[0] = PAGE_SIZE - capacity
                    if implicit_positions and stage == 2:
                        prefixes[0] = 0
                    live, valid = len(lengths), sum(lengths)
                    table = tables.roll(stage, 0).roll(stage, 1)
                    table[live:].zero_()
                    updated = self._base_inputs(
                        lengths + [0] * (batch - live),
                        prefixes + [0] * (batch - live),
                        table,
                        capacity=capacity,
                        decode=decode,
                        implicit_positions=implicit_positions,
                    )
                    for field in fields:
                        getattr(inputs, field).copy_(getattr(updated, field))
                    source.normal_()
                    source[valid:].fill_(float("nan"))
                    for row, (length, prefix) in enumerate(zip(lengths, prefixes)):
                        column, offset = divmod(prefix + length, PAGE_SIZE)
                        if offset:
                            page = int(table[row, column].item())
                            cache.kv_cache_base[page, 1, :, offset:].fill_(float("nan"))
                    query_ref, cache_ref, reference = self._base_reference(
                        config,
                        source,
                        cache.kv_cache_base,
                        table,
                        lengths,
                        prefixes,
                        decode=decode,
                        implicit_positions=implicit_positions,
                    )
                    impl.prepare_cuda_graph(inputs)
                    if implicit_positions and not decode:
                        expected_positions = [
                            prefix + local
                            for prefix, length in zip(prefixes, lengths)
                            for local in range(length)
                        ]
                        expected_positions += [0] * (batch * capacity - valid)
                        torch.testing.assert_close(
                            impl.rope_params.position_ids,
                            torch.tensor(
                                expected_positions, device="cuda", dtype=torch.int32
                            ),
                            rtol=0,
                            atol=0,
                        )
                        self.assertEqual(
                            impl.rope_params.position_ids.data_ptr(), position_pointer
                        )
                    for _ in range(2):
                        graph.replay()
                    self.assertEqual(
                        tuple(getattr(inputs, field).data_ptr() for field in fields),
                        pointers,
                    )
                    self.assertTrue(
                        torch.equal(
                            captured_query[:valid]
                            .reshape(valid, QUERY_HEADS, HEAD_DIM)
                            .view(torch.uint8),
                            _quantize(query_ref, 1.0)
                            .reshape(valid, QUERY_HEADS, HEAD_DIM)
                            .view(torch.uint8),
                        )
                    )
                    if not decode:
                        self.assertEqual(
                            torch.count_nonzero(captured_query[valid:].float()).item(),
                            0,
                        )
                    # Legacy decode writes padded QKV to reserved page zero.
                    # Every allocated page must still match exactly; prefill
                    # must leave even reserved page zero untouched.
                    start = 1 if decode else 0
                    self.assertTrue(
                        torch.equal(
                            cache.kv_cache_base[start:].view(torch.uint8),
                            cache_ref[start:].view(torch.uint8),
                        )
                    )
                    self._assert_accuracy(
                        captured[:valid].reshape(-1, QUERY_HEADS, HEAD_DIM), reference
                    )
                    eager = forward()
                    torch.testing.assert_close(
                        captured[:valid], eager[:valid], rtol=0, atol=0
                    )

    def test_real_base_draft_prefill_graph_with_variable_lengths(self):
        self._run_base_draft_graph(decode=False)

    def test_real_base_draft_decode_graph_with_reused_pages(self):
        self._run_base_draft_graph(decode=True)

    def test_real_base_draft_prefill_graph_without_positions(self):
        self._run_base_draft_graph(decode=False, implicit_positions=True)

    def test_real_base_draft_spec_decode_graph_without_positions(self):
        self._run_base_draft_graph(
            decode=False, implicit_positions=True, speculative=True
        )

    def test_real_base_draft_decode_graph_without_positions(self):
        self._run_base_draft_graph(decode=True, implicit_positions=True)

    def test_base_graph_buckets_share_workspace_with_ordered_replay(self):
        batch, columns, capacity = 4, 8, 5
        config = self._real_mtp_config()
        config.max_seq_len = columns * PAGE_SIZE
        config.rope_config.style = RopeStyle.Base
        config.rope_config.index_factor = 1
        table = torch.arange(
            1, 1 + batch * columns, device="cuda", dtype=torch.int32
        ).reshape(batch, columns)
        cache = LayerKVCache()
        cache.kv_cache_base = torch.randn(
            1 + batch * columns, 2, KV_HEADS, PAGE_SIZE, HEAD_DIM, device="cuda"
        ).to(torch.float8_e4m3fn)
        cache.kv_scale_base = torch.ones(
            cache.kv_cache_base.shape[0], 2 * KV_HEADS * PAGE_SIZE, device="cuda"
        )
        buckets = []
        for impl_class, lengths, prefixes in (
            (FlashInferTRTLLMSpecDecodeImpl, [5, 5, 5, 5], [59, 123, 187, 251]),
            (FlashInferTRTLLMPrefillImpl, [5, 3, 0, 0], [127, 190, 0, 0]),
        ):
            inputs = self._base_inputs(
                lengths,
                [config.max_seq_len - capacity] * batch,
                table,
                capacity=capacity,
                implicit_positions=True,
            )
            # Production capture buckets slice the same runner-owned table.
            inputs.kv_cache_kernel_block_id_device = table[:batch]
            impl = impl_class(config, inputs)
            source = torch.randn(
                batch * capacity, PACKED_DIM, device="cuda", dtype=torch.bfloat16
            )
            qkv = torch.empty_like(source)

            def forward():
                qkv.copy_(source)
                return impl.forward(qkv, cache, 0)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = forward()
            graph.replay()
            torch.cuda.synchronize()
            buckets.append(
                (impl, inputs, source, qkv, graph, output, lengths, prefixes)
            )
        self.assertIs(
            buckets[0][0].fmha_impl._workspace_lease,
            buckets[1][0].fmha_impl._workspace_lease,
        )
        self.assertIsNotNone(buckets[0][0].fmha_impl._workspace_lease)
        fields = (
            "prefix_lengths",
            "prefix_lengths_device",
            "sequence_lengths",
            "sequence_lengths_plus_1_device",
            "cu_kv_seqlens_device",
        )
        # Context and multi-token decode use the same scratch in alternating
        # graphs. As in CudaGraphRunner, each replay completes before the next.
        for index in (0, 1, 0, 1):
            impl, inputs, source, qkv, graph, output, lengths, prefixes = buckets[index]
            updated = self._base_inputs(
                lengths, prefixes, table, capacity=capacity, implicit_positions=True
            )
            for field in fields:
                getattr(inputs, field).copy_(getattr(updated, field))
            source.normal_()
            live = sum(length > 0 for length in lengths)
            valid = sum(lengths)
            source[valid:].fill_(float("nan"))
            _, expected_cache, reference = self._base_reference(
                config,
                source,
                cache.kv_cache_base,
                table,
                lengths[:live],
                prefixes[:live],
                implicit_positions=True,
            )
            impl.prepare_cuda_graph(inputs)
            graph.replay()
            torch.cuda.synchronize()
            self._assert_accuracy(
                output[:valid].reshape(-1, QUERY_HEADS, HEAD_DIM), reference
            )
            self.assertTrue(
                torch.equal(
                    cache.kv_cache_base.view(torch.uint8),
                    expected_cache.view(torch.uint8),
                )
            )
            qkv.copy_(source)
            eager = impl.forward(qkv, cache, 0)
            torch.testing.assert_close(output[:valid], eager[:valid], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
