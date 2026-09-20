"""GPU tests for the sparse GQA FMHA impl and the factory routing key.

The impl's forward is a thin adapter (split qkv -> reshape -> kernel ->
reshape back); it is validated against the torch reference on the same
split/reshape geometry.  The routing key is pure logic.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    _select_attention_impl_key,
)
from rtp_llm.models_py.modules.qwen4_exp import sparse_gqa_impl
from rtp_llm.models_py.modules.qwen4_exp.sparse_fmha import (
    sparse_prefill_attn_torch_reference,
)
from rtp_llm.models_py.modules.qwen4_exp.sparse_gqa_impl import SparseGqaFmhaImpl
from rtp_llm.models_py.modules.qwen4_exp.sparse_paged_fmha import sparse_paged_gqa_attn

_H_Q, _H_KV, _D = 24, 2, 256
_DEV = "cuda"


def _write_packed_kv_to_cache(qkv, cache, block_table, prefixes, input_lengths):
    q_width = _H_Q * _D
    kv_width = _H_KV * _D
    _, packed_k, packed_v = torch.split(qkv, [q_width, kv_width, kv_width], dim=-1)
    offset = 0
    page_size = int(cache.shape[3])
    for request_idx, (prefix, query_len) in enumerate(
        zip(prefixes.tolist(), input_lengths.tolist())
    ):
        for query_idx in range(query_len):
            position = prefix + query_idx
            physical = int(block_table[request_idx, position // page_size].item())
            token_offset = position % page_size
            cache[physical, 0, :, token_offset].copy_(
                packed_k[offset + query_idx].view(_H_KV, _D)
            )
            cache[physical, 1, :, token_offset].copy_(
                packed_v[offset + query_idx].view(_H_KV, _D)
            )
        offset += query_len


def _ragged_paged_reference(qkv, cache, block_table, prefixes, input_lengths, selected):
    q_width = _H_Q * _D
    packed_q = qkv[:, :q_width].view(-1, _H_Q, _D)
    output = torch.zeros_like(packed_q)
    page_size = int(cache.shape[3])
    offset = 0
    for request_idx, (prefix, query_len) in enumerate(
        zip(prefixes.tolist(), input_lengths.tolist())
    ):
        for query_idx in range(query_len):
            visible = prefix + query_idx + 1
            indices = selected[offset + query_idx]
            indices = indices[(indices >= 0) & (indices < visible)]
            for q_head in range(_H_Q):
                kv_head = q_head // (_H_Q // _H_KV)
                keys = []
                values = []
                for token_idx in indices.tolist():
                    physical = int(
                        block_table[request_idx, token_idx // page_size].item()
                    )
                    token_offset = token_idx % page_size
                    keys.append(cache[physical, 0, kv_head, token_offset])
                    values.append(cache[physical, 1, kv_head, token_offset])
                if keys:
                    keys_tensor = torch.stack(keys).float()
                    values_tensor = torch.stack(values).float()
                    scores = (
                        packed_q[offset + query_idx, q_head].float() @ keys_tensor.T
                    ) / _D**0.5
                    output[offset + query_idx, q_head] = (
                        scores.softmax(dim=-1) @ values_tensor
                    ).to(output.dtype)
        offset += query_len
    return output.reshape(-1, q_width)


def _configs(is_sparse=True, use_mla=False, is_prefill=True, opt_in=True):
    """``opt_in`` mirrors the formal sparse-GQA routing config field."""
    attn_configs = SimpleNamespace(
        head_num=_H_Q,
        kv_head_num=_H_KV,
        size_per_head=_D,
        kernel_tokens_per_block=4,
        is_sparse=is_sparse,
        use_mla=use_mla,
        use_sparse_gqa_fmha=opt_in,
    )
    attn_inputs = SimpleNamespace(
        is_prefill=is_prefill,
        input_lengths=torch.tensor([0], dtype=torch.int32),
        sequence_lengths=torch.zeros(1, dtype=torch.int32),
        prefix_lengths=torch.zeros(1, dtype=torch.int32),
        kv_cache_kernel_block_id_device=None,
        is_target_verify=False,
        is_cuda_graph=False,
        is_s_padded=False,
        context_parallel_info=None,
        cache_store_inputs=None,
    )
    return attn_configs, attn_inputs


class SparseGqaRoutingTest(unittest.TestCase):
    def test_sparse_qwen4_prefill_routes_to_sparse_gqa(self):
        attn_configs, attn_inputs = _configs()
        self.assertEqual(
            _select_attention_impl_key(attn_configs, attn_inputs), "sparse_gqa"
        )

    def test_unwired_model_keeps_the_dense_fallback(self):
        """Indexer on but the sparse impl not opted in -> dense (mha).

        This is the state until the attention module runs the indexer and
        hands over its selection.
        """
        attn_configs, attn_inputs = _configs(opt_in=False)
        self.assertEqual(_select_attention_impl_key(attn_configs, attn_inputs), "mha")

    def test_decode_routes_to_sparse_gqa(self):
        attn_configs, attn_inputs = _configs(is_prefill=False)
        self.assertEqual(
            _select_attention_impl_key(attn_configs, attn_inputs), "sparse_gqa"
        )

    def test_dense_model_routes_to_mha(self):
        attn_configs, attn_inputs = _configs(is_sparse=False)
        self.assertEqual(_select_attention_impl_key(attn_configs, attn_inputs), "mha")

    def test_mla_takes_priority(self):
        attn_configs, attn_inputs = _configs(use_mla=True)
        self.assertEqual(_select_attention_impl_key(attn_configs, attn_inputs), "mla")

    def test_impl_support_matches_the_key_logic(self):
        attn_configs, attn_inputs = _configs()
        self.assertTrue(SparseGqaFmhaImpl.support(attn_configs, attn_inputs))
        attn_configs, attn_inputs = _configs(is_prefill=False)
        self.assertTrue(SparseGqaFmhaImpl.support(attn_configs, attn_inputs))
        attn_configs, attn_inputs = _configs(is_sparse=False)
        self.assertFalse(SparseGqaFmhaImpl.support(attn_configs, attn_inputs))


class SparseGqaImplForwardTest(unittest.TestCase):
    class _IdentityRopeWriter:
        def __init__(self, attn_configs):
            pass

        def prepare(self, attn_inputs):
            return object()

        def forward(self, qkv, kv_cache, params):
            return qkv

    class _IdentityDecodeWriter:
        def __init__(self, attn_configs):
            self.q_width = attn_configs.head_num * attn_configs.size_per_head

        def prepare(self, attn_inputs):
            return object()

        def forward(self, qkv, kv_cache, params):
            return qkv[:, : self.q_width]

    class _FailingRopeWriter:
        def forward(self, qkv, kv_cache, params):
            raise RuntimeError("injected fused writer failure")

    def _impl(self, batch=1, seq_len=8):
        attn_configs, attn_inputs = _configs()
        attn_inputs.input_lengths = torch.full((batch,), seq_len, dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.zeros(batch, dtype=torch.int32)
        with patch.object(
            sparse_gqa_impl,
            "FusedRopeKVCachePrefillOpQKVOut",
            self._IdentityRopeWriter,
        ):
            return SparseGqaFmhaImpl(attn_configs, attn_inputs)

    def _decode_impl(self, batch=2):
        attn_configs, attn_inputs = _configs(is_prefill=False)
        attn_inputs.sequence_lengths = torch.tensor(
            [4 + 2 * index for index in range(batch)], dtype=torch.int32
        )
        attn_inputs.kv_cache_kernel_block_id_device = torch.arange(
            1, batch * 3 + 1, dtype=torch.int32, device=_DEV
        ).reshape(batch, 3)
        with patch.object(
            sparse_gqa_impl,
            "FusedRopeKVCacheDecodeOp",
            self._IdentityDecodeWriter,
        ):
            return SparseGqaFmhaImpl(attn_configs, attn_inputs)

    def test_main_cache_phase_is_marked_before_fused_writer(self):
        impl = self._impl(batch=1, seq_len=1)
        impl.rope_kvcache_impl = self._FailingRopeWriter()
        impl.begin_qsa_cache_transaction()

        with self.assertRaisesRegex(RuntimeError, "fused writer failure"):
            impl.forward(
                torch.randn(1, (_H_Q + 2 * _H_KV) * _D),
                selected_indices=torch.zeros(1, 1, dtype=torch.int32),
            )

        self.assertTrue(impl.qsa_main_cache_mutation_started())

    def test_forward_matches_the_torch_reference(self):
        torch.manual_seed(0)
        batch, seq_len, k_sel = 1, 8, 16
        impl = self._impl(batch, seq_len)

        tokens = batch * seq_len
        qkv = (torch.randn(tokens, (_H_Q + 2 * _H_KV) * _D, device=_DEV) * 0.3).to(
            torch.bfloat16
        )
        sel = -torch.ones(batch, seq_len, k_sel, dtype=torch.int32, device=_DEV)
        for s in range(seq_len):
            n = min(k_sel, s + 1)
            sel[0, s, :n] = torch.arange(n, device=_DEV, dtype=torch.int32)

        out = impl.forward(qkv, kv_cache=None, layer_idx=0, selected_indices=sel)

        # Independent path: split/reshape here, call the reference directly.
        q_width = _H_Q * _D
        kv_width = _H_KV * _D
        q, k, v = torch.split(qkv, [q_width, kv_width, kv_width], dim=-1)
        q = q.reshape(batch, seq_len, _H_Q, _D).transpose(1, 2).contiguous()
        k = k.reshape(batch, seq_len, _H_KV, _D).transpose(1, 2).contiguous()
        v = v.reshape(batch, seq_len, _H_KV, _D).transpose(1, 2).contiguous()
        ref = sparse_prefill_attn_torch_reference(q, k, v, sel)
        ref = ref.transpose(1, 2).reshape(tokens, q_width)

        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)

    def test_missing_selection_is_rejected(self):
        impl = self._impl()
        qkv = torch.zeros(8, (_H_Q + 2 * _H_KV) * _D, device=_DEV).to(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "selected_indices"):
            impl.forward(qkv, kv_cache=None, layer_idx=0)

    def test_target_verify_must_use_context_style_prefill(self):
        attn_configs, attn_inputs = _configs(is_prefill=False)
        attn_inputs.is_target_verify = True
        attn_inputs.input_lengths = torch.tensor([4], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.tensor([7], dtype=torch.int32)
        attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)

        with patch.object(
            sparse_gqa_impl, "FusedRopeKVCachePrefillOpQKVOut"
        ) as rope_writer:
            with self.assertRaisesRegex(RuntimeError, "context-style prefill"):
                SparseGqaFmhaImpl(attn_configs, attn_inputs)

        rope_writer.assert_not_called()

    def test_target_verify_writes_main_cache_then_uses_paged_prefix(self):
        batch, query_len, selected_width = 1, 4, 7
        attn_configs, attn_inputs = _configs(is_prefill=True)
        attn_inputs.is_target_verify = True
        attn_inputs.input_lengths = torch.tensor([query_len], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.tensor([7], dtype=torch.int32)
        attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id_device = torch.tensor(
            [[1, 2, 3]], dtype=torch.int32, device=_DEV
        )
        events = []

        class _RecordingWriter(self._IdentityRopeWriter):
            def forward(inner_self, qkv, kv_cache, params):
                events.append("main_rope_kv_write")
                return qkv

        observed = {}

        def _paged(q, cache, block_table, kv_lens, indices, **kwargs):
            events.append("paged_read")
            observed.update(
                q_shape=tuple(q.shape),
                kv_lens=kv_lens.cpu(),
                indices_shape=tuple(indices.shape),
            )
            return q

        with (
            patch.object(
                sparse_gqa_impl,
                "FusedRopeKVCachePrefillOpQKVOut",
                _RecordingWriter,
            ),
            patch.object(
                sparse_gqa_impl.common,
                "apply_write_cache_store",
                side_effect=lambda *args: events.append("cache_store"),
            ),
            patch(
                "rtp_llm.models_py.modules.qwen4_exp.sparse_paged_fmha."
                "sparse_paged_gqa_attn",
                side_effect=_paged,
            ),
        ):
            impl = SparseGqaFmhaImpl(attn_configs, attn_inputs)
            qkv = torch.randn(
                batch * query_len,
                (_H_Q + 2 * _H_KV) * _D,
                dtype=torch.bfloat16,
                device=_DEV,
            )
            selected = torch.zeros(
                batch * query_len,
                selected_width,
                dtype=torch.int32,
                device=_DEV,
            )
            cache = SimpleNamespace(
                kv_cache_base=torch.zeros(
                    4, 2, _H_KV, 4, _D, dtype=torch.bfloat16, device=_DEV
                )
            )
            output = impl.forward(qkv, cache, selected_indices=selected)

        self.assertEqual(output.shape, (batch * query_len, _H_Q * _D))
        self.assertEqual(events, ["main_rope_kv_write", "cache_store", "paged_read"])
        self.assertEqual(observed["q_shape"], (batch, _H_Q, query_len, _D))
        self.assertEqual(observed["indices_shape"], (batch, query_len, selected_width))
        torch.testing.assert_close(
            observed["kv_lens"],
            torch.tensor([[8, 9, 10, 11]], dtype=torch.int32),
        )

    def test_target_verify_rejects_bad_stored_selection_before_main_write(self):
        batch, query_len = 1, 4
        attn_configs, attn_inputs = _configs(is_prefill=True)
        attn_inputs.is_target_verify = True
        attn_inputs.input_lengths = torch.tensor([query_len], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.tensor([7], dtype=torch.int32)
        attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id_device = torch.tensor(
            [[1, 2, 3]], dtype=torch.int32, device=_DEV
        )
        events = []

        class _RecordingWriter(self._IdentityRopeWriter):
            def forward(inner_self, qkv, kv_cache, params):
                events.append("main_rope_kv_write")
                return qkv

        with (
            patch.object(
                sparse_gqa_impl,
                "FusedRopeKVCachePrefillOpQKVOut",
                _RecordingWriter,
            ),
            patch.object(
                sparse_gqa_impl.common,
                "apply_write_cache_store",
                side_effect=lambda *args: events.append("cache_store"),
            ),
        ):
            impl = SparseGqaFmhaImpl(attn_configs, attn_inputs)
            qkv = torch.randn(
                batch * query_len,
                (_H_Q + 2 * _H_KV) * _D,
                dtype=torch.bfloat16,
                device=_DEV,
            )
            malformed = torch.zeros(
                batch * query_len - 1, 7, dtype=torch.int32, device=_DEV
            )
            impl.set_selected_indices(malformed)
            cache = SimpleNamespace(
                kv_cache_base=torch.zeros(
                    4, 2, _H_KV, 4, _D, dtype=torch.bfloat16, device=_DEV
                )
            )

            with self.assertRaisesRegex(ValueError, "one row per token"):
                impl.forward(qkv, cache)

        self.assertEqual(events, [])
        self.assertIs(impl._selected_indices, malformed)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_target_verify_prevalidates_packed_and_hnd_main_cache_layouts(self):
        batch, query_len = 1, 4
        attn_configs, attn_inputs = _configs(is_prefill=True)
        attn_inputs.is_target_verify = True
        attn_inputs.input_lengths = torch.tensor([query_len], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.tensor([7], dtype=torch.int32, device=_DEV)
        attn_inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        attn_inputs.kv_cache_kernel_block_id_device = torch.tensor(
            [[1, 2, 3]], dtype=torch.int32, device=_DEV
        )
        with patch.object(
            sparse_gqa_impl,
            "FusedRopeKVCachePrefillOpQKVOut",
            self._IdentityRopeWriter,
        ):
            impl = SparseGqaFmhaImpl(attn_configs, attn_inputs)

        required_width = 2 * _H_KV * 4 * _D
        backing = torch.zeros(4, required_width + 8, dtype=torch.bfloat16, device=_DEV)
        padded_rows = backing[:, :required_width]
        self.assertFalse(padded_rows.is_contiguous())
        events = []
        runtime = SimpleNamespace(
            main_cache=SimpleNamespace(kv_cache_base=padded_rows),
            main_inputs=attn_inputs,
            validate_before_projection=lambda **kwargs: events.append("side"),
        )
        hidden = torch.zeros(batch * query_len, 16, dtype=torch.bfloat16, device=_DEV)

        impl.validate_qsa_before_side_write(runtime, object(), hidden)
        self.assertEqual(events, ["side"])

        malformed_hnd = torch.zeros(
            4, 2, _H_KV, 4, _D, dtype=torch.bfloat16, device=_DEV
        ).transpose(1, 2)
        runtime.main_cache = SimpleNamespace(kv_cache_base=malformed_hnd)
        with self.assertRaisesRegex(RuntimeError, "HND inner layout"):
            impl.validate_qsa_before_side_write(runtime, object(), hidden)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_decode_does_not_skip_main_cache_preflight(self):
        impl = self._decode_impl(batch=1)
        events = []
        runtime = SimpleNamespace(
            main_cache=SimpleNamespace(
                kv_cache_base=torch.zeros(1, dtype=torch.bfloat16, device=_DEV)
            ),
            main_inputs=impl.attn_inputs,
            validate_before_projection=lambda **kwargs: events.append("side"),
        )
        hidden = torch.zeros(1, 16, dtype=torch.bfloat16, device=_DEV)

        with self.assertRaisesRegex(RuntimeError, "packed 2-D or HND 5-D"):
            impl.validate_qsa_before_side_write(runtime, object(), hidden)

        self.assertEqual(events, ["side"])

    def test_ragged_packed_selection_matches_per_request_reference(self):
        impl = self._impl(batch=2, seq_len=4)
        impl.attn_inputs.input_lengths = torch.tensor([3, 5], dtype=torch.int32)
        qkv = torch.randn(8, (_H_Q + 2 * _H_KV) * _D, device=_DEV, dtype=torch.bfloat16)
        selected = torch.full((8, 5), -1, device=_DEV, dtype=torch.int32)
        offset = 0
        for seq_len in (3, 5):
            for row in range(seq_len):
                selected[offset + row, : row + 1] = torch.arange(
                    row + 1, device=_DEV, dtype=torch.int32
                )
            offset += seq_len

        got = impl.forward(qkv, selected_indices=selected)

        q_width = _H_Q * _D
        kv_width = _H_KV * _D
        q, k, v = torch.split(qkv, [q_width, kv_width, kv_width], dim=-1)
        expected = []
        offset = 0
        for seq_len in (3, 5):
            end = offset + seq_len
            seq_q = q[offset:end].reshape(1, seq_len, _H_Q, _D).transpose(1, 2)
            seq_k = k[offset:end].reshape(1, seq_len, _H_KV, _D).transpose(1, 2)
            seq_v = v[offset:end].reshape(1, seq_len, _H_KV, _D).transpose(1, 2)
            seq_out = sparse_prefill_attn_torch_reference(
                seq_q.contiguous(),
                seq_k.contiguous(),
                seq_v.contiguous(),
                selected[offset:end].unsqueeze(0),
            )
            expected.append(seq_out.transpose(1, 2).reshape(seq_len, q_width))
            offset = end
        torch.testing.assert_close(got, torch.cat(expected), atol=1e-2, rtol=1e-2)

    def test_mtp_incremental_ragged_prefill_reads_paged_prefix(self):
        torch.manual_seed(31)
        input_lengths = torch.tensor([3, 2], dtype=torch.int32)
        prefixes = torch.tensor([3, 7], dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32, device=_DEV)
        block_table = torch.tensor(
            [[1, 2, 3], [4, 5, 6]], dtype=torch.int32, device=_DEV
        )
        attn_configs, attn_inputs = _configs(is_prefill=True)
        attn_inputs.input_lengths = input_lengths
        attn_inputs.prefix_lengths = prefixes
        attn_inputs.cu_seqlens_device = cu_seqlens
        attn_inputs.kv_cache_kernel_block_id_device = block_table
        with patch.object(
            sparse_gqa_impl,
            "FusedRopeKVCachePrefillOpQKVOut",
            self._IdentityRopeWriter,
        ):
            impl = SparseGqaFmhaImpl(attn_configs, attn_inputs)
        impl.set_mtp_draft_mode(True)

        qkv = torch.randn(
            5,
            (_H_Q + 2 * _H_KV) * _D,
            dtype=torch.bfloat16,
            device=_DEV,
        )
        cache = torch.randn(7, 2, _H_KV, 4, _D, dtype=torch.bfloat16, device=_DEV)
        _write_packed_kv_to_cache(qkv, cache, block_table, prefixes, input_lengths)
        selected = torch.full((5, 6), -1, dtype=torch.int32, device=_DEV)
        selected[0, :4] = torch.tensor([0, 1, 2, 3], device=_DEV)
        selected[1, :5] = torch.tensor([0, 1, 2, 3, 4], device=_DEV)
        selected[2, :5] = torch.tensor([0, 2, 3, 4, 5], device=_DEV)
        selected[3, :5] = torch.tensor([0, 2, 4, 6, 7], device=_DEV)
        selected[4, :5] = torch.tensor([1, 3, 5, 7, 8], device=_DEV)
        expected = _ragged_paged_reference(
            qkv, cache, block_table, prefixes, input_lengths, selected
        )
        calls = []

        def _record_paged(q, paged_cache, table, kv_lens, indices, **kwargs):
            calls.append((tuple(q.shape), kv_lens.cpu().tolist(), tuple(indices.shape)))
            return sparse_paged_gqa_attn(
                q, paged_cache, table, kv_lens, indices, **kwargs
            )

        with (
            patch(
                "rtp_llm.models_py.modules.qwen4_exp.sparse_fmha."
                "sparse_prefill_attn",
                side_effect=AssertionError("local sparse prefill must not run"),
            ),
            patch(
                "rtp_llm.models_py.modules.qwen4_exp.sparse_paged_fmha."
                "sparse_paged_gqa_attn",
                side_effect=_record_paged,
            ),
        ):
            got = impl.forward(
                qkv,
                kv_cache=SimpleNamespace(kv_cache_base=cache),
                selected_indices=selected,
            )

        torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)
        self.assertEqual(
            calls,
            [
                ((1, _H_Q, 3, _D), [[4, 5, 6]], (1, 3, 6)),
                ((1, _H_Q, 2, _D), [[8, 9]], (1, 2, 6)),
            ],
        )

    def test_nonzero_prefix_requires_the_paged_bridge_not_local_prefill(self):
        events = []

        class _RecordingWriter(self._IdentityRopeWriter):
            def forward(inner_self, qkv, kv_cache, params):
                events.append("main_rope_kv_write")
                return qkv

        attn_configs, attn_inputs = _configs(is_prefill=True)
        attn_inputs.input_lengths = torch.tensor([2], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.tensor([3], dtype=torch.int32)
        with patch.object(
            sparse_gqa_impl,
            "FusedRopeKVCachePrefillOpQKVOut",
            _RecordingWriter,
        ):
            impl = SparseGqaFmhaImpl(attn_configs, attn_inputs)
        qkv = torch.zeros(
            2,
            (_H_Q + 2 * _H_KV) * _D,
            dtype=torch.bfloat16,
            device=_DEV,
        )
        selected = torch.zeros(2, 1, dtype=torch.int32, device=_DEV)

        with (
            patch.object(
                sparse_gqa_impl.common,
                "apply_write_cache_store",
                side_effect=lambda *args: events.append("cache_store"),
            ),
            patch(
                "rtp_llm.models_py.modules.qwen4_exp.sparse_fmha."
                "sparse_prefill_attn",
                side_effect=AssertionError("local sparse prefill must not run"),
            ),
        ):
            # The prefix case must route to the ragged paged bridge; without a
            # main KV cache the bridge refuses before any local prefill or
            # cache write can run.
            with self.assertRaisesRegex(RuntimeError, "requires the main KV cache"):
                impl.forward(qkv, selected_indices=selected)

        self.assertEqual(events, [])

    def test_mtp_draft_prefix_free_prefill_keeps_local_path(self):
        impl = self._impl(batch=2, seq_len=2)
        impl.set_mtp_draft_mode(True)
        qkv = torch.zeros(
            4,
            (_H_Q + 2 * _H_KV) * _D,
            dtype=torch.bfloat16,
            device=_DEV,
        )
        selected = torch.tensor(
            [[0, -1], [0, 1], [0, -1], [0, 1]],
            dtype=torch.int32,
            device=_DEV,
        )

        with (
            patch(
                "rtp_llm.models_py.modules.qwen4_exp.sparse_fmha."
                "sparse_prefill_attn",
                side_effect=lambda q, k, v, indices: torch.zeros_like(q),
            ) as local_prefill,
            patch(
                "rtp_llm.models_py.modules.qwen4_exp.sparse_paged_fmha."
                "sparse_paged_gqa_attn",
                side_effect=AssertionError("paged prefill must not run"),
            ),
        ):
            output = impl.forward(qkv, selected_indices=selected)

        self.assertEqual(output.shape, (4, _H_Q * _D))
        self.assertEqual(local_prefill.call_count, 2)

    def test_incremental_prefill_missing_block_has_no_main_write_side_effect(self):
        events = []

        class _RecordingWriter(self._IdentityRopeWriter):
            def forward(inner_self, qkv, kv_cache, params):
                events.append("main_rope_kv_write")
                kv_cache.kv_cache_base.fill_(17)
                return qkv

        attn_configs, attn_inputs = _configs(is_prefill=True)
        attn_inputs.input_lengths = torch.tensor([3, 2], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.tensor([3, 7], dtype=torch.int32)
        attn_inputs.cu_seqlens_device = torch.tensor(
            [0, 3, 5], dtype=torch.int32, device=_DEV
        )
        attn_inputs.kv_cache_kernel_block_id_device = torch.tensor(
            [[1, 2, 3], [4, 5, 0]], dtype=torch.int32, device=_DEV
        )
        with patch.object(
            sparse_gqa_impl,
            "FusedRopeKVCachePrefillOpQKVOut",
            _RecordingWriter,
        ):
            impl = SparseGqaFmhaImpl(attn_configs, attn_inputs)
        impl.set_mtp_draft_mode(True)
        qkv = torch.zeros(
            5,
            (_H_Q + 2 * _H_KV) * _D,
            dtype=torch.bfloat16,
            device=_DEV,
        )
        selected = torch.zeros(5, 1, dtype=torch.int32, device=_DEV)
        cache = torch.randn(7, 2, _H_KV, 4, _D, dtype=torch.bfloat16, device=_DEV)
        original = cache.clone()

        with patch.object(
            sparse_gqa_impl.common,
            "apply_write_cache_store",
            side_effect=lambda *args: events.append("cache_store"),
        ):
            with self.assertRaisesRegex(RuntimeError, "unallocated"):
                impl.forward(
                    qkv,
                    kv_cache=SimpleNamespace(kv_cache_base=cache),
                    selected_indices=selected,
                )

        self.assertEqual(events, [])
        torch.testing.assert_close(cache, original)

    def test_main_rope_writer_and_cache_store_run_before_sparse_kernel(self):
        events = []

        class _RecordingRopeWriter(self._IdentityRopeWriter):
            def forward(inner_self, qkv, kv_cache, params):
                events.append("rope_kv_write")
                return qkv

        attn_configs, attn_inputs = _configs()
        attn_inputs.input_lengths = torch.tensor([1], dtype=torch.int32)
        attn_inputs.prefix_lengths = torch.zeros(1, dtype=torch.int32)
        qkv = torch.zeros(1, (_H_Q + 2 * _H_KV) * _D, device=_DEV, dtype=torch.bfloat16)
        selected = torch.zeros(1, 1, device=_DEV, dtype=torch.int32)

        def _record_sparse(q, k, v, indices):
            events.append("sparse_attention")
            return torch.zeros_like(q)

        with (
            patch.object(
                sparse_gqa_impl,
                "FusedRopeKVCachePrefillOpQKVOut",
                _RecordingRopeWriter,
            ),
            patch.object(
                sparse_gqa_impl.common,
                "apply_write_cache_store",
                side_effect=lambda *args: events.append("cache_store"),
            ),
            patch(
                "rtp_llm.models_py.modules.qwen4_exp.sparse_fmha.sparse_prefill_attn",
                side_effect=_record_sparse,
            ),
        ):
            impl = SparseGqaFmhaImpl(attn_configs, attn_inputs)
            impl.forward(qkv, selected_indices=selected)

        self.assertEqual(events, ["rope_kv_write", "cache_store", "sparse_attention"])

    def test_multi_token_decode_uses_paged_sparse_cache(self):
        batch, query_len, selected_width = 2, 3, 5
        impl = self._decode_impl(batch=batch)
        qkv = torch.randn(
            batch * query_len,
            (_H_Q + 2 * _H_KV) * _D,
            device=_DEV,
            dtype=torch.bfloat16,
        )
        selected = torch.zeros(
            batch * query_len,
            selected_width,
            dtype=torch.int32,
            device=_DEV,
        )
        kv_cache = SimpleNamespace(
            kv_cache_base=torch.zeros(
                7, 2, _H_KV, 4, _D, device=_DEV, dtype=torch.bfloat16
            )
        )
        observed = {}

        def _record_paged(
            q,
            cache,
            block_table,
            kv_lens,
            indices,
            *,
            page_size,
            kv_head_num,
        ):
            observed.update(
                q_shape=tuple(q.shape),
                cache=cache,
                block_table=block_table,
                kv_lens=kv_lens.cpu(),
                indices_shape=tuple(indices.shape),
                page_size=page_size,
                kv_head_num=kv_head_num,
            )
            return q

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.sparse_paged_fmha."
            "sparse_paged_gqa_attn",
            side_effect=_record_paged,
        ):
            output = impl.forward(qkv, kv_cache=kv_cache, selected_indices=selected)

        self.assertEqual(output.shape, (batch * query_len, _H_Q * _D))
        self.assertEqual(observed["q_shape"], (batch, _H_Q, query_len, _D))
        self.assertIs(observed["cache"], kv_cache.kv_cache_base)
        self.assertIs(
            observed["block_table"],
            impl.attn_inputs.kv_cache_kernel_block_id_device,
        )
        torch.testing.assert_close(
            observed["kv_lens"],
            torch.tensor([[5, 6, 7], [7, 8, 9]], dtype=torch.int32),
        )
        self.assertEqual(observed["indices_shape"], (batch, query_len, selected_width))
        self.assertEqual(observed["page_size"], 4)
        self.assertEqual(observed["kv_head_num"], _H_KV)


if __name__ == "__main__":
    unittest.main()
