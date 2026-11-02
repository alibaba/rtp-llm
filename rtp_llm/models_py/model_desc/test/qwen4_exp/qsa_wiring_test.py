from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc import qwen3_next, qwen4_exp
from rtp_llm.models_py.model_desc.test.qwen4_exp.fixtures import (
    _EPS,
    _QsaMainCachePhase,
)
from rtp_llm.utils.model_weight import W


class Qwen4ExpQSAConstructionTest(TestCase):
    @staticmethod
    def _fake_base_init(module, *args, **kwargs):
        nn.Module.__init__(module)

    def _config(self, *, is_mtp=False):
        return SimpleNamespace(
            enable_qwen4_qsa=True,
            is_mtp=is_mtp,
            _qwen4_indexer_head_dim=8,
            _qwen4_indexer_compress_ratio=4,
            _qwen4_indexer_budget=8,
            _qwen4_indexer_kv_heads=1,
            attn_config=SimpleNamespace(
                indexer_head_num=4,
                rope_config=SimpleNamespace(
                    index_factor=3,
                    dim=8,
                    mrope_dim1=2,
                    mrope_dim2=1,
                    mrope_dim3=1,
                    mrope_interleaved=True,
                    base=10_000,
                ),
            ),
        )

    def _weights(self):
        return {
            W.qwen4_indexer_qk_proj_w: torch.randn(40, 16),
            W.qwen4_indexer_q_ln_gamma: torch.randn(8),
            W.qwen4_indexer_k_ln_gamma: torch.randn(8),
        }

    def test_constructs_checkpoint_backed_indexer(self):
        weights = self._weights()
        with patch.object(
            qwen3_next.Qwen3NextAttention,
            "__init__",
            self._fake_base_init,
        ):
            attention = qwen4_exp.Qwen4ExpAttention(
                SimpleNamespace(),
                SimpleNamespace(),
                weights,
                _EPS,
                self._config(),
                layer_idx=3,
            )

        self.assertIs(attention.qsa_indexer.qk_proj, weights[W.qwen4_indexer_qk_proj_w])
        self.assertEqual(attention.qsa_indexer.max_selected, 11)
        self.assertEqual(attention.layer_idx, 3)

    @staticmethod
    def _prefill_inputs():
        positions = torch.tensor(
            [
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [0, 0, 0],
                [1, 3, 5],
            ],
            dtype=torch.int32,
        )
        return SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            input_lengths=torch.tensor([3, 2], dtype=torch.int32),
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            combo_position_ids=positions.reshape(-1),
        )

    @staticmethod
    def _decode_inputs():
        return SimpleNamespace(
            is_prefill=False,
            is_target_verify=False,
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            input_lengths=torch.tensor([5, 7], dtype=torch.int32),
            prefix_lengths=torch.empty(0, dtype=torch.int32),
            sequence_lengths=torch.tensor([5, 7], dtype=torch.int32),
            combo_position_ids=torch.tensor(
                [[5, 5, 5], [7, 7, 7]], dtype=torch.int32
            ).reshape(-1),
        )

    def _attention(self, *, is_mtp=False):
        with patch.object(
            qwen3_next.Qwen3NextAttention,
            "__init__",
            self._fake_base_init,
        ):
            attention = qwen4_exp.Qwen4ExpAttention(
                SimpleNamespace(),
                SimpleNamespace(),
                self._weights(),
                _EPS,
                self._config(is_mtp=is_mtp),
                layer_idx=3,
            )
        attention.qkv_proj = nn.Identity()
        attention.qk_fuse_norm = None
        attention.gate = nn.Linear(16, 16, bias=False)
        attention.gate.weight.data.zero_()
        attention.o_proj = nn.Identity()
        attention.tp_size = 1
        return attention

    def _runtime(self, attention, inputs):
        class _Runtime:
            main_cache = None
            main_inputs = inputs
            is_mtp_draft = attention.is_mtp_draft
            rollback_count = 0
            finalize_count = 0

            def rollback_side_cache(runtime_self):
                runtime_self.rollback_count += 1

            def finalize_side_cache(runtime_self):
                runtime_self.finalize_count += 1

            def validate_before_projection(runtime_self, **kwargs):
                runtime_self.validated_before_projection = True

            def write_prefill_indexer_cache(runtime_self, raw_keys, **kwargs):
                runtime_self.raw_keys = raw_keys
                lengths = [int(value) for value in inputs.input_lengths.tolist()]
                max_len = max(lengths)
                rope_dim = int(attention.qsa_rope_config.dim)
                cos = torch.ones(len(lengths), max_len, rope_dim)
                sin = torch.zeros_like(cos)
                return lengths, cos, sin, {}

            def select_decode_tokens(runtime_self, q, raw_keys, **kwargs):
                runtime_self.q = q
                runtime_self.raw_keys = raw_keys
                return torch.zeros(
                    q.shape[0],
                    attention.qsa_indexer.max_selected,
                    dtype=torch.int32,
                    device=q.device,
                )

            def select_target_verify_tokens(runtime_self, q, raw_keys, **kwargs):
                runtime_self.q = q
                runtime_self.raw_keys = raw_keys
                return torch.zeros(
                    q.shape[0],
                    attention.qsa_indexer.max_selected,
                    dtype=torch.int32,
                    device=q.device,
                )

            def select_draft_incremental_prefill_tokens(
                runtime_self, q, raw_keys, **kwargs
            ):
                runtime_self.draft_q = q
                runtime_self.raw_keys = raw_keys
                return torch.zeros(
                    q.shape[0],
                    attention.qsa_indexer.max_selected,
                    dtype=torch.int32,
                    device=q.device,
                )

            def select_draft_prefix_reuse_prefill_tokens(
                runtime_self, q, raw_keys, **kwargs
            ):
                runtime_self.draft_prefix_reuse_q = q
                runtime_self.raw_keys = raw_keys
                return torch.zeros(
                    q.shape[0],
                    attention.qsa_indexer.max_selected,
                    dtype=torch.int32,
                    device=q.device,
                )

        return _Runtime()

    def test_ragged_prefill_hands_packed_indexer_selection_to_sparse_fmha(self):
        class _SparseFmha(_QsaMainCachePhase):
            def __init__(self):
                self.selected = None

            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                return qkv

        attention = self._attention()
        fmha = _SparseFmha()
        hidden = torch.randn(5, 16)
        inputs = self._prefill_inputs()
        runtime = self._runtime(attention, inputs)

        with patch.object(
            attention.qsa_indexer,
            "project",
            wraps=attention.qsa_indexer.project,
        ) as project:
            output = attention(hidden, fmha, None, inputs, qsa_runtime=runtime)

        self.assertEqual(output.shape, hidden.shape)
        project.assert_called_once_with(hidden)
        self.assertEqual(runtime.raw_keys.shape, (5, 8))
        self.assertEqual(fmha.selected.shape, (5, 11))
        self.assertEqual(fmha.selected.dtype, torch.int32)
        self.assertEqual(runtime.rollback_count, 0)
        self.assertEqual(runtime.finalize_count, 1)
        # Each request owns a local token-index space; the second request must
        # restart at zero instead of indexing into the first request's K/V.
        self.assertLessEqual(int(fmha.selected[:3].max().item()), 2)
        self.assertLessEqual(int(fmha.selected[3:].max().item()), 1)

    def test_main_attention_failure_rolls_back_qsa_side_transaction(self):
        class _FailingSparseFmha(_QsaMainCachePhase):
            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                raise RuntimeError("injected main attention failure")

        attention = self._attention()
        inputs = self._prefill_inputs()
        runtime = self._runtime(attention, inputs)

        with self.assertRaisesRegex(RuntimeError, "injected main attention failure"):
            attention(
                torch.randn(5, 16),
                _FailingSparseFmha(),
                None,
                inputs,
                qsa_runtime=runtime,
            )

        self.assertEqual(runtime.rollback_count, 1)
        self.assertEqual(runtime.finalize_count, 0)

    def test_failure_after_main_cache_write_preserves_side_cache(self):
        class _FailingSparseFmha(_QsaMainCachePhase):
            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                self.main_cache_mutation_started = True
                raise RuntimeError("injected failure after main cache write")

        attention = self._attention()
        inputs = self._prefill_inputs()
        runtime = self._runtime(attention, inputs)

        with self.assertRaisesRegex(RuntimeError, "after main cache write"):
            attention(
                torch.randn(5, 16),
                _FailingSparseFmha(),
                None,
                inputs,
                qsa_runtime=runtime,
            )

        self.assertEqual(runtime.rollback_count, 0)
        self.assertEqual(runtime.finalize_count, 1)

    def test_decode_projects_once_and_hands_paged_selection_to_sparse_fmha(self):
        class _SparseFmha(_QsaMainCachePhase):
            def __init__(self):
                self.selected = None

            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                return qkv

        attention = self._attention()
        fmha = _SparseFmha()
        hidden = torch.randn(2, 16)
        inputs = self._decode_inputs()
        runtime = self._runtime(attention, inputs)

        with patch.object(
            attention.qsa_indexer,
            "project",
            wraps=attention.qsa_indexer.project,
        ) as project:
            output = attention(hidden, fmha, None, inputs, qsa_runtime=runtime)

        self.assertEqual(output.shape, hidden.shape)
        project.assert_called_once_with(hidden)
        self.assertEqual(runtime.q.shape, (2, 4, 8))
        self.assertEqual(runtime.raw_keys.shape, (2, 8))
        self.assertEqual(fmha.selected.shape, (2, 11))
        self.assertEqual(fmha.selected.dtype, torch.int32)

    def test_target_verify_projects_once_and_hands_paged_selection_to_sparse_fmha(self):
        class _SparseFmha(_QsaMainCachePhase):
            def __init__(self):
                self.selected = None

            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                return qkv

        attention = self._attention()
        fmha = _SparseFmha()
        inputs = self._prefill_inputs()
        inputs.is_target_verify = True
        inputs.input_lengths = torch.tensor([4], dtype=torch.int32)
        inputs.prefix_lengths = torch.tensor([7], dtype=torch.int32)
        inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        runtime = self._runtime(attention, inputs)

        with patch.object(
            attention.qsa_indexer,
            "project",
            wraps=attention.qsa_indexer.project,
        ) as project:
            output = attention(
                torch.randn(4, 16), fmha, None, inputs, qsa_runtime=runtime
            )

        self.assertEqual(output.shape, (4, 16))
        project.assert_called_once()
        self.assertTrue(runtime.validated_before_projection)
        self.assertEqual(runtime.raw_keys.shape, (4, 8))
        self.assertEqual(fmha.selected.shape, (4, 11))

    def test_mtp_draft_nonzero_prefix_sets_explicit_gate_and_uses_incremental_path(
        self,
    ):
        class _SparseFmha(_QsaMainCachePhase):
            def __init__(self):
                self.is_mtp_draft = False
                self.selected = None

            def set_mtp_draft_mode(self, enabled):
                self.is_mtp_draft = enabled

            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                return qkv

        attention = self._attention(is_mtp=True)
        fmha = _SparseFmha()
        inputs = self._prefill_inputs()
        inputs.prefix_lengths = torch.tensor([7, 11], dtype=torch.int32)
        inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        runtime = self._runtime(attention, inputs)

        output = attention(torch.randn(5, 16), fmha, None, inputs, qsa_runtime=runtime)

        self.assertEqual(output.shape, (5, 16))
        self.assertTrue(fmha.is_mtp_draft)
        self.assertEqual(runtime.draft_q.shape, (5, 4, 8))
        self.assertEqual(runtime.raw_keys.shape, (5, 8))
        self.assertEqual(fmha.selected.shape, (5, 11))

    def test_mtp_draft_long_prefix_reuse_uses_the_warm_prefill_path(self):
        class _SparseFmha(_QsaMainCachePhase):
            def __init__(self):
                self.is_mtp_draft = False
                self.selected = None

            def set_mtp_draft_mode(self, enabled):
                self.is_mtp_draft = enabled

            def set_selected_indices(self, selected):
                self.selected = selected

            def forward(self, qkv, kv_cache, layer_idx):
                return qkv

        attention = self._attention(is_mtp=True)
        fmha = _SparseFmha()
        inputs = self._prefill_inputs()
        inputs.input_lengths = torch.tensor([5], dtype=torch.int32)
        inputs.prefix_lengths = torch.tensor([8], dtype=torch.int32)
        inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        inputs.combo_position_ids = torch.tensor(
            [[position, position, position] for position in range(8, 13)],
            dtype=torch.int32,
        ).reshape(-1)
        runtime = self._runtime(attention, inputs)

        output = attention(torch.randn(5, 16), fmha, None, inputs, qsa_runtime=runtime)

        self.assertEqual(output.shape, (5, 16))
        self.assertTrue(fmha.is_mtp_draft)
        self.assertEqual(runtime.draft_prefix_reuse_q.shape, (5, 4, 8))
        self.assertFalse(hasattr(runtime, "draft_q"))
        self.assertEqual(fmha.selected.shape, (5, 11))

    def test_mtp_short_mixed_prefix_batch_uses_prefill_not_rejection_commit(self):
        attention = self._attention(is_mtp=True)
        for prefixes in ([0, 128], [128, 0]):
            for length in (1, 2, 4):
                with self.subTest(prefixes=prefixes, length=length):
                    inputs = self._prefill_inputs()
                    inputs.input_lengths = torch.tensor(
                        [length, length], dtype=torch.int32
                    )
                    inputs.prefix_lengths = torch.tensor(prefixes, dtype=torch.int32)
                    inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
                    runtime = self._runtime(attention, inputs)
                    selected = attention._qsa_selected_indices(
                        torch.randn(2 * length, 16), runtime
                    )
                    self.assertEqual(selected.shape, (2 * length, 11))
                    self.assertEqual(
                        runtime.draft_prefix_reuse_q.shape, (2 * length, 4, 8)
                    )
                    self.assertFalse(hasattr(runtime, "draft_q"))

    def test_mtp_pd_short_prefix_hit_uses_prefill_not_rejection_commit(self):
        attention = self._attention(is_mtp=True)
        for length in (1, 3, 4):
            with self.subTest(length=length):
                inputs = self._prefill_inputs()
                inputs.input_lengths = torch.tensor([length], dtype=torch.int32)
                inputs.prefix_lengths = torch.tensor([128], dtype=torch.int32)
                inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
                inputs.cache_store_inputs = SimpleNamespace()
                runtime = self._runtime(attention, inputs)
                selected = attention._qsa_selected_indices(
                    torch.randn(length, 16), runtime
                )
                self.assertEqual(selected.shape, (length, 11))
                self.assertEqual(runtime.draft_prefix_reuse_q.shape, (length, 4, 8))
                self.assertFalse(hasattr(runtime, "draft_q"))
                inputs.cache_store_inputs = None
                commit_runtime = self._runtime(attention, inputs)
                attention._qsa_selected_indices(torch.randn(length, 16), commit_runtime)
                self.assertEqual(commit_runtime.draft_q.shape, (length, 4, 8))
                self.assertFalse(hasattr(commit_runtime, "draft_prefix_reuse_q"))

    def test_target_verify_validation_fails_before_projection_and_main_writer(self):
        events = []

        class _SparseFmha(_QsaMainCachePhase):
            def set_selected_indices(self, selected):
                events.append("selection")

            def validate_qsa_before_side_write(self, runtime, indexer, hidden):
                events.append("validation")
                raise RuntimeError("invalid target geometry")

            def forward(self, qkv, kv_cache, layer_idx):
                events.append("main_writer")
                return qkv

        attention = self._attention()
        inputs = self._prefill_inputs()
        inputs.is_target_verify = True
        inputs.input_lengths = torch.tensor([4], dtype=torch.int32)
        inputs.prefix_lengths = torch.tensor([7], dtype=torch.int32)
        inputs.sequence_lengths = torch.empty(0, dtype=torch.int32)
        runtime = self._runtime(attention, inputs)

        with patch.object(
            attention.qsa_indexer,
            "project",
            wraps=attention.qsa_indexer.project,
        ) as project:
            with self.assertRaisesRegex(RuntimeError, "invalid target geometry"):
                attention(
                    torch.randn(4, 16),
                    _SparseFmha(),
                    None,
                    inputs,
                    qsa_runtime=runtime,
                )

        project.assert_not_called()
        self.assertEqual(events, ["validation"])

    def test_enabled_indexer_rejects_dense_fmha(self):
        attention = self._attention()

        with self.assertRaisesRegex(RuntimeError, "dense fallback is forbidden"):
            attention(torch.randn(5, 16), object(), None, self._prefill_inputs())

    def test_enabled_indexer_requires_side_cache_context(self):
        class _SparseFmha:
            def set_selected_indices(self, selected):
                pass

        attention = self._attention()
        with self.assertRaisesRegex(RuntimeError, "side-cache regions"):
            attention(torch.randn(5, 16), _SparseFmha(), None, self._prefill_inputs())


if __name__ == "__main__":
    main()
