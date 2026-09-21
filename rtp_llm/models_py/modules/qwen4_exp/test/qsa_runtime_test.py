from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.modules.qwen4_exp.indexer import (
    apply_partial_rope,
    build_interleaved_mrope,
    build_qsa_rope,
)
from rtp_llm.models_py.modules.qwen4_exp.norm import exact_head_rms_norm
from rtp_llm.models_py.modules.qwen4_exp.qsa_runtime import (
    Qwen4ExpQSARuntimeContext,
    select_qsa_paged_tokens,
)
from rtp_llm.ops import RopeStyle


class Qwen4ExpQSARuntimeTest(TestCase):
    D = 4
    RATIO = 4
    KV_TOKENS_PER_BLOCK = 8
    STATE_TOKENS_PER_BLOCK = 8

    def setUp(self):
        torch.manual_seed(17)
        self.lengths = [5, 8]
        self.total_tokens = sum(self.lengths)
        positions = []
        for seq_len in self.lengths:
            positions.extend([[pos, pos, pos] for pos in range(seq_len)])
        self.position_ids = torch.tensor(positions, dtype=torch.int32).reshape(-1)
        self.cu_seqlens = torch.tensor([0, 5, 13], dtype=torch.int32)
        self.input_lengths = torch.tensor(self.lengths, dtype=torch.int32)
        self.prefix_lengths = torch.zeros(2, dtype=torch.int32)
        self.sequence_lengths = torch.zeros(2, dtype=torch.int32)
        self.kv_base = torch.zeros(7, 2 * self.D * 2, dtype=torch.uint8)
        self.state_base = torch.zeros(8, 2 * self.RATIO * self.D, dtype=torch.float32)
        self.kv_table = torch.tensor([[3], [1]], dtype=torch.int32)
        self.state_table = torch.tensor([[6], [2]], dtype=torch.int32)
        self.rope_config = SimpleNamespace(
            style=RopeStyle.Mrope,
            index_factor=3,
            dim=4,
            mrope_dim1=1,
            mrope_dim2=1,
            mrope_dim3=0,
            mrope_interleaved=True,
            base=10_000,
        )
        self.indexer = SimpleNamespace(
            head_dim=self.D,
            compress_ratio=self.RATIO,
            k_norm_gamma=torch.rand(self.D),
            norm_eps=1e-6,
        )

    def _inputs(self, *, kv_table=None, state_table=None, **overrides):
        values = dict(
            is_prefill=True,
            is_target_verify=False,
            is_cuda_graph=False,
            is_s_padded=False,
            context_parallel_info=None,
            cache_store_inputs=None,
            input_lengths=self.input_lengths,
            prefix_lengths=self.prefix_lengths,
            sequence_lengths=self.sequence_lengths,
            cu_seqlens_device=self.cu_seqlens,
            cu_kv_seqlens_device=self.cu_seqlens,
            combo_position_ids=self.position_ids,
            kv_cache_kernel_block_id_device=kv_table,
            kv_cache_block_id_device=state_table,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def _context(self, *, kv_base=None, state_base=None, **state_overrides):
        main_inputs = self._inputs()
        kv_inputs = self._inputs(kv_table=self.kv_table)
        state_inputs = self._inputs(state_table=self.state_table, **state_overrides)
        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=main_inputs,
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=self.kv_base if kv_base is None else kv_base,
                seq_size_per_block=self.KV_TOKENS_PER_BLOCK,
            ),
            indexer_kv_inputs=kv_inputs,
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=self.state_base if state_base is None else state_base,
                seq_size_per_block=self.STATE_TOKENS_PER_BLOCK,
            ),
            indexer_state_inputs=state_inputs,
        )

    def _expected(self, raw_keys, cos, sin, start):
        pooled = raw_keys.float().mean(dim=0).to(raw_keys.dtype)
        pooled = exact_head_rms_norm(
            pooled, self.indexer.k_norm_gamma, self.indexer.norm_eps
        )
        return apply_partial_rope(pooled, cos, sin).to(torch.bfloat16)

    def _single_request_context(
        self,
        kv_base,
        state_base,
        *,
        is_prefill,
        input_length,
        sequence_length,
        position_ids,
        target_verify=False,
        prefix_length=0,
    ):
        device = kv_base.device
        input_lengths = torch.tensor([input_length], dtype=torch.int32, device=device)
        sequence_lengths = torch.tensor(
            [sequence_length], dtype=torch.int32, device=device
        )
        prefix_lengths = torch.tensor([prefix_length], dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor(
            [0, input_length if is_prefill else 1],
            dtype=torch.int32,
            device=device,
        )
        position_ids = torch.as_tensor(
            position_ids, dtype=torch.int32, device=device
        ).reshape(-1)
        kv_table = torch.tensor([[3]], dtype=torch.int32, device=device)
        state_table = torch.tensor([[6]], dtype=torch.int32, device=device)

        def _inputs(*, kv_table=None, state_table=None):
            return SimpleNamespace(
                is_prefill=is_prefill,
                is_target_verify=target_verify,
                is_cuda_graph=False,
                is_s_padded=False,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sequence_lengths=sequence_lengths,
                cu_seqlens_device=cu_seqlens,
                cu_kv_seqlens_device=cu_seqlens,
                combo_position_ids=position_ids,
                kv_cache_kernel_block_id_device=kv_table,
                kv_cache_block_id_device=state_table,
            )

        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=_inputs(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=kv_base,
                seq_size_per_block=self.KV_TOKENS_PER_BLOCK,
            ),
            indexer_kv_inputs=_inputs(kv_table=kv_table),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=state_base,
                seq_size_per_block=self.STATE_TOKENS_PER_BLOCK,
            ),
            indexer_state_inputs=_inputs(state_table=state_table),
        )

    def _decode_indexer(self, heads=2):
        return SimpleNamespace(
            head_dim=self.D,
            n_heads=heads,
            compress_ratio=self.RATIO,
            token_budget=8,
            k_norm_gamma=self.indexer.k_norm_gamma,
            norm_eps=self.indexer.norm_eps,
        )

    @staticmethod
    def _base_rope_config(
        index_factor, *, style=RopeStyle.Base, indexer_is_neox_style=True
    ):
        return SimpleNamespace(
            style=style,
            index_factor=index_factor,
            dim=4,
            base=10_000,
            scale=1.0,
            indexer_is_neox_style=indexer_is_neox_style,
        )

    @staticmethod
    def _set_position_ids(context, position_ids):
        position_ids = torch.as_tensor(
            position_ids,
            dtype=torch.int32,
            device=context.main_inputs.input_lengths.device,
        ).reshape(-1)
        for inputs in (
            context.main_inputs,
            context.indexer_kv_inputs,
            context.indexer_state_inputs,
        ):
            inputs.combo_position_ids = position_ids

    def _target_verify_context(self, *, q_len=4, committed_length=7):
        """Mirror the target-verify ABI produced by MtpExecutor.

        Target verification is represented as a context-style invocation:
        ``input_lengths=[gamma + 1]``, ``prefix_lengths=[committed]`` and an
        empty ``sequence_lengths`` tensor.
        """
        device = self.kv_base.device
        input_lengths = torch.tensor([q_len], dtype=torch.int32, device=device)
        prefix_lengths = torch.tensor(
            [committed_length], dtype=torch.int32, device=device
        )
        sequence_lengths = torch.empty(0, dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor([0, q_len], dtype=torch.int32, device=device)
        cu_kv_seqlens = torch.tensor(
            [0, committed_length + q_len], dtype=torch.int32, device=device
        )
        position_ids = torch.arange(
            committed_length,
            committed_length + q_len,
            dtype=torch.int32,
            device=device,
        )
        position_ids = position_ids.unsqueeze(1).expand(-1, 3).reshape(-1)
        kv_table = torch.tensor([[3, 4]], dtype=torch.int32, device=device)
        state_table = torch.tensor([[6, 7]], dtype=torch.int32, device=device)

        def _inputs(*, kv_table=None, state_table=None):
            return SimpleNamespace(
                is_prefill=True,
                is_target_verify=True,
                is_cuda_graph=False,
                is_s_padded=False,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sequence_lengths=sequence_lengths,
                cu_seqlens_device=cu_seqlens,
                cu_kv_seqlens_device=cu_kv_seqlens,
                combo_position_ids=position_ids,
                kv_cache_kernel_block_id_device=kv_table,
                kv_cache_block_id_device=state_table,
            )

        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=_inputs(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=self.kv_base,
                seq_size_per_block=self.KV_TOKENS_PER_BLOCK,
            ),
            indexer_kv_inputs=_inputs(kv_table=kv_table),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=self.state_base,
                seq_size_per_block=self.STATE_TOKENS_PER_BLOCK,
            ),
            indexer_state_inputs=_inputs(state_table=state_table),
        )

    def _draft_incremental_context(
        self,
        kv_base,
        state_base,
        *,
        prefixes,
        lengths,
        kv_table=None,
        state_table=None,
        position_factor=3,
        is_mtp_draft=True,
        position_overrides=None,
    ):
        device = kv_base.device
        batch_size = len(lengths)
        input_lengths = torch.tensor(lengths, dtype=torch.int32, device=device)
        prefix_lengths = torch.tensor(prefixes, dtype=torch.int32, device=device)
        sequence_lengths = torch.empty(0, dtype=torch.int32, device=device)
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = input_lengths.cumsum(0)
        cu_kv_seqlens = torch.zeros_like(cu_seqlens)
        cu_kv_seqlens[1:] = (prefix_lengths + input_lengths).cumsum(0)
        positions = []
        for prefix, length in zip(prefixes, lengths):
            positions.extend(range(prefix, prefix + length))
        if position_overrides is not None:
            positions = position_overrides
        position_ids = torch.tensor(positions, dtype=torch.int32, device=device)
        if position_factor == 3:
            position_ids = position_ids.unsqueeze(1).expand(-1, 3)
        position_ids = position_ids.reshape(-1)
        if kv_table is None:
            kv_table = torch.arange(
                1,
                1 + batch_size * 3,
                dtype=torch.int32,
                device=device,
            ).view(batch_size, 3)
        if state_table is None:
            state_table = torch.arange(
                1 + batch_size * 3,
                1 + batch_size * 6,
                dtype=torch.int32,
                device=device,
            ).view(batch_size, 3)

        def _inputs(*, kv_table=None, state_table=None):
            return SimpleNamespace(
                is_prefill=True,
                is_target_verify=False,
                is_cuda_graph=False,
                is_s_padded=False,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sequence_lengths=sequence_lengths,
                cu_seqlens_device=cu_seqlens,
                cu_kv_seqlens_device=cu_kv_seqlens,
                combo_position_ids=position_ids,
                kv_cache_kernel_block_id_device=kv_table,
                kv_cache_block_id_device=state_table,
            )

        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=_inputs(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=kv_base,
                seq_size_per_block=self.KV_TOKENS_PER_BLOCK,
            ),
            indexer_kv_inputs=_inputs(kv_table=kv_table),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=state_base,
                seq_size_per_block=self.STATE_TOKENS_PER_BLOCK,
            ),
            indexer_state_inputs=_inputs(state_table=state_table),
            is_mtp_draft=is_mtp_draft,
        )

    def test_ragged_prefill_writes_each_tag_local_pool(self):
        first = torch.randn(5, self.D, dtype=torch.bfloat16)
        second = torch.randn(8, self.D, dtype=torch.bfloat16)
        raw_keys = torch.cat([first, second])

        lengths, cos, sin, result = self._context().write_prefill_indexer_cache(
            raw_keys, indexer=self.indexer, rope_config=self.rope_config
        )

        self.assertEqual(lengths, self.lengths)
        self.assertEqual(result["num_state_writes"], 13)
        self.assertEqual(result["num_kv_writes"], 3)
        kv_pool = self.kv_base.view(torch.bfloat16).view(7, 2, self.D)
        state_pool = self.state_base.view(8, 2 * self.RATIO, self.D)
        torch.testing.assert_close(
            kv_pool[3, 0], self._expected(first[:4], cos[0, 0], sin[0, 0], 0)
        )
        torch.testing.assert_close(
            kv_pool[1, 0], self._expected(second[:4], cos[1, 0], sin[1, 0], 0)
        )
        torch.testing.assert_close(
            kv_pool[1, 1], self._expected(second[4:8], cos[1, 4], sin[1, 4], 4)
        )
        torch.testing.assert_close(state_pool[6, 4], first[4].float())
        torch.testing.assert_close(state_pool[2, 7], second[7].float())

    def test_base_rope_prefill_factor_one_and_three_match(self):
        raw_keys = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)
        scalar_positions = []
        for seq_len in self.lengths:
            scalar_positions.extend(range(seq_len))

        factor_one = self._context(
            kv_base=torch.zeros_like(self.kv_base),
            state_base=torch.zeros_like(self.state_base),
        )
        self._set_position_ids(factor_one, scalar_positions)
        _, cos_one, sin_one, _ = factor_one.write_prefill_indexer_cache(
            raw_keys,
            indexer=self.indexer,
            rope_config=self._base_rope_config(1),
        )

        factor_three = self._context(
            kv_base=torch.zeros_like(self.kv_base),
            state_base=torch.zeros_like(self.state_base),
        )
        positions_three = (
            torch.tensor(scalar_positions, dtype=torch.int32).unsqueeze(1).expand(-1, 3)
        )
        self._set_position_ids(factor_three, positions_three)
        _, cos_three, sin_three, _ = factor_three.write_prefill_indexer_cache(
            raw_keys,
            indexer=self.indexer,
            rope_config=self._base_rope_config(3),
        )

        torch.testing.assert_close(cos_one, cos_three)
        torch.testing.assert_close(sin_one, sin_three)

    def test_base_rope_prefill_rejects_non_logical_position_before_write(self):
        raw_keys = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)
        scalar_positions = []
        for seq_len in self.lengths:
            scalar_positions.extend(range(seq_len))
        scalar_positions[4] += 1
        for style in (RopeStyle.Base, "Base", 1):
            with self.subTest(style=style):
                context = self._context()
                self._set_position_ids(context, scalar_positions)
                with patch(
                    "rtp_llm.models_py.modules.qwen4_exp.qsa_runtime."
                    "write_indexer_cache"
                ) as writer:
                    with self.assertRaisesRegex(
                        RuntimeError, "logical cache positions"
                    ):
                        context.write_prefill_indexer_cache(
                            raw_keys,
                            indexer=self.indexer,
                            rope_config=self._base_rope_config(1, style=style),
                        )

                writer.assert_not_called()

    def test_non_neox_indexer_rope_is_rejected_before_prefill_write(self):
        raw_keys = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)
        scalar_positions = []
        for seq_len in self.lengths:
            scalar_positions.extend(range(seq_len))
        context = self._context()
        self._set_position_ids(context, scalar_positions)

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa_runtime.write_indexer_cache"
        ) as writer:
            with self.assertRaisesRegex(RuntimeError, "indexer_is_neox_style=true"):
                context.write_prefill_indexer_cache(
                    raw_keys,
                    indexer=self.indexer,
                    rope_config=self._base_rope_config(1, indexer_is_neox_style=False),
                )

        writer.assert_not_called()

    def test_bad_2d_pool_row_is_rejected_before_mutation(self):
        bad_kv = torch.zeros(7, self.kv_base.shape[1] - 1, dtype=torch.uint8)
        raw_keys = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)

        with self.assertRaisesRegex(RuntimeError, "row has"):
            self._context(kv_base=bad_kv).write_prefill_indexer_cache(
                raw_keys, indexer=self.indexer, rope_config=self.rope_config
            )

        self.assertEqual(int(torch.count_nonzero(bad_kv)), 0)
        self.assertEqual(int(torch.count_nonzero(self.state_base)), 0)

    def test_unsupported_or_inconsistent_side_metadata_is_rejected(self):
        raw_keys = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)
        cases = (
            ("prefill", {"is_prefill": False}),
            ("prefix", {"prefix_lengths": torch.tensor([1, 0], dtype=torch.int32)}),
            ("target-verify mode", {"is_target_verify": True}),
            ("CUDA Graph", {"is_cuda_graph": True}),
            ("PD", {"cache_store_inputs": object()}),
        )
        for expected, overrides in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    self._context(**overrides).write_prefill_indexer_cache(
                        raw_keys, indexer=self.indexer, rope_config=self.rope_config
                    )
        self.assertEqual(int(torch.count_nonzero(self.kv_base)), 0)
        self.assertEqual(int(torch.count_nonzero(self.state_base)), 0)

    def test_target_verify_wider_than_one_compression_group_is_rejected(self):
        q_len = self.RATIO + 1
        raw_keys = torch.randn(q_len, self.D, dtype=torch.bfloat16)
        q = torch.randn(q_len, 2, self.D, dtype=torch.bfloat16)
        self.kv_base.random_(0, 256)
        self.state_base.normal_()
        kv_before = self.kv_base.clone()
        state_before = self.state_base.clone()
        context = self._target_verify_context(q_len=q_len)
        indexer = self._decode_indexer()

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa_runtime.write_indexer_cache"
        ) as writer:
            with self.assertRaisesRegex(RuntimeError, "gamma\\+1=.*<= compress_ratio"):
                context.select_target_verify_tokens(
                    q,
                    raw_keys,
                    indexer=indexer,
                    rope_config=self.rope_config,
                )

        writer.assert_not_called()
        torch.testing.assert_close(self.kv_base, kv_before)
        torch.testing.assert_close(self.state_base, state_before)

    @skipUnless(torch.cuda.is_available(), "CUDA is required for target preflight")
    def test_target_verify_state_table_does_not_wrap_past_its_absolute_width(self):
        self.kv_base = self.kv_base.cuda()
        self.state_base = self.state_base.cuda()
        self.indexer.k_norm_gamma = self.indexer.k_norm_gamma.cuda()
        self.kv_base.random_(0, 256)
        self.state_base.normal_()
        kv_before = self.kv_base.clone()
        state_before = self.state_base.clone()
        context = self._target_verify_context(q_len=1, committed_length=8)
        context.indexer_state_inputs.kv_cache_block_id_device = torch.tensor(
            [[6]], dtype=torch.int32, device=self.state_base.device
        )
        raw_keys = torch.randn(
            1, self.D, dtype=torch.bfloat16, device=self.kv_base.device
        )
        q = torch.randn(1, 2, self.D, dtype=torch.bfloat16, device=self.kv_base.device)

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa_runtime.write_indexer_cache"
        ) as writer:
            with self.assertRaisesRegex(RuntimeError, "does not cover"):
                context.select_target_verify_tokens(
                    q,
                    raw_keys,
                    indexer=self._decode_indexer(),
                    rope_config=self.rope_config,
                )

        writer.assert_not_called()
        self.assertTrue(torch.equal(self.kv_base, kv_before))
        self.assertTrue(torch.equal(self.state_base, state_before))

    @skipUnless(torch.cuda.is_available(), "CUDA is required for paged scoring")
    def test_target_verify_uses_prefix_for_every_query_row(self):
        self.kv_base = self.kv_base.cuda()
        self.state_base = self.state_base.cuda()
        self.indexer.k_norm_gamma = self.indexer.k_norm_gamma.cuda()
        q_len = self.RATIO
        committed = 7
        context = self._target_verify_context(q_len=q_len, committed_length=committed)
        indexer = self._decode_indexer()
        raw_keys = torch.randn(
            q_len, self.D, dtype=torch.bfloat16, device=self.kv_base.device
        )
        q = torch.randn(
            q_len, 2, self.D, dtype=torch.bfloat16, device=self.kv_base.device
        )
        observed = {}

        def _score(q, weight, pool, table, lengths, *, block_size, max_ctx_len):
            observed.update(q=q, lengths=lengths, max_ctx_len=max_ctx_len)
            return torch.tensor(
                [[0.25, 3.0]] * q_len,
                dtype=torch.float32,
                device=q.device,
            )

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score."
            "qsa_paged_indexer_score",
            side_effect=_score,
        ):
            selected = context.select_target_verify_tokens(
                q,
                raw_keys,
                indexer=indexer,
                rope_config=self.rope_config,
            )

        self.assertEqual(observed["q"].shape, (1, q_len, 2, self.D))
        torch.testing.assert_close(
            observed["lengths"],
            torch.full(
                (1, q_len),
                2,
                dtype=torch.int32,
                device=observed["lengths"].device,
            ),
        )
        self.assertEqual(observed["max_ctx_len"], 2)
        for row in range(q_len):
            visible = committed + row + 1
            valid = selected[row][selected[row] >= 0]
            self.assertTrue(bool(torch.all(valid < visible)))
            tail = torch.arange(8, visible, dtype=torch.int32, device=selected.device)
            if tail.numel():
                self.assertTrue(
                    set(tail.tolist()).issubset(set(valid.tolist())),
                    msg=f"row={row}, selected={valid.tolist()}",
                )

        kv_pool = self.kv_base.view(torch.bfloat16).view(7, 2, self.D)
        state_pool = self.state_base.view(8, 2 * self.RATIO, self.D)
        self.assertGreater(int(torch.count_nonzero(kv_pool[3, 1])), 0)
        torch.testing.assert_close(state_pool[6, 7], raw_keys[0].float())
        torch.testing.assert_close(state_pool[7, 0], raw_keys[1].float())
        torch.testing.assert_close(state_pool[7, 2], raw_keys[3].float())

    @skipUnless(torch.cuda.is_available(), "CUDA is required for paged scoring")
    def test_target_verify_never_overwrites_the_committed_partial_group(self):
        self.indexer.k_norm_gamma = self.indexer.k_norm_gamma.cuda()
        device = self.indexer.k_norm_gamma.device
        q_len = self.RATIO

        def _score(q, weight, pool, table, lengths, *, block_size, max_ctx_len):
            return torch.zeros(
                q.shape[0] * q.shape[1],
                max_ctx_len,
                dtype=torch.float32,
                device=q.device,
            )

        for residue in range(self.RATIO):
            with self.subTest(prefix_residue=residue):
                committed = 2 * self.RATIO + residue
                self.kv_base = torch.zeros(
                    7,
                    2 * self.D * 2,
                    dtype=torch.uint8,
                    device=device,
                )
                self.state_base = torch.zeros(
                    8,
                    2 * self.RATIO * self.D,
                    dtype=torch.float32,
                    device=device,
                )
                state_pool = self.state_base.view(8, 2 * self.RATIO, self.D)
                committed_slots = []
                for position in range(committed - residue, committed):
                    logical_block = position // self.STATE_TOKENS_PER_BLOCK
                    self.assertLess(logical_block, 2)
                    physical_block = (6, 7)[logical_block]
                    offset = position % (2 * self.RATIO)
                    value = torch.full(
                        (self.D,),
                        float(position + 1),
                        dtype=torch.float32,
                        device=device,
                    )
                    state_pool[physical_block, offset].copy_(value)
                    committed_slots.append((physical_block, offset, value.clone()))

                context = self._target_verify_context(
                    q_len=q_len, committed_length=committed
                )
                raw_keys = torch.randn(
                    q_len, self.D, dtype=torch.bfloat16, device=device
                )
                q = torch.randn(q_len, 2, self.D, dtype=torch.bfloat16, device=device)
                with patch(
                    "rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score."
                    "qsa_paged_indexer_score",
                    side_effect=_score,
                ):
                    context.select_target_verify_tokens(
                        q,
                        raw_keys,
                        indexer=self._decode_indexer(),
                        rope_config=self.rope_config,
                    )

                for physical_block, offset, expected in committed_slots:
                    torch.testing.assert_close(
                        state_pool[physical_block, offset], expected
                    )

    @skipUnless(torch.cuda.is_available(), "CUDA is required for paged scoring")
    def test_draft_incremental_prefill_is_ragged_and_visibility_bounded(self):
        device = torch.device("cuda")
        indexer = self._decode_indexer()
        indexer.k_norm_gamma = indexer.k_norm_gamma.to(device)
        kv_base = torch.zeros(12, 2 * self.D * 2, dtype=torch.uint8, device=device)
        state_base = torch.zeros(
            12, 2 * self.RATIO * self.D, dtype=torch.float32, device=device
        )
        kv_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device=device)
        state_table = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32, device=device)
        lengths = [1, self.RATIO]
        prefixes = [self.RATIO * 2 - 1, self.RATIO * 2 - 1]
        context = self._draft_incremental_context(
            kv_base,
            state_base,
            prefixes=prefixes,
            lengths=lengths,
            kv_table=kv_table,
            state_table=state_table,
        )
        token_count = sum(lengths)
        q = torch.randn(token_count, 2, self.D, dtype=torch.bfloat16, device=device)
        raw = torch.randn(token_count, self.D, dtype=torch.bfloat16, device=device)
        observed = []

        def _score(q, weight, pool, table, lengths, *, block_size, max_ctx_len):
            observed.append((q.shape, lengths.clone(), table.clone(), max_ctx_len))
            rows = q.shape[1]
            return torch.arange(
                rows * max_ctx_len, dtype=torch.float32, device=device
            ).view(rows, max_ctx_len)

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score."
            "qsa_paged_indexer_score",
            side_effect=_score,
        ):
            selected = context.select_draft_incremental_prefill_tokens(
                q,
                raw,
                indexer=indexer,
                rope_config=self._base_rope_config(3),
            )

        self.assertEqual(selected.shape, (token_count, 8 + self.RATIO - 1))
        self.assertEqual(len(observed), 2)
        self.assertEqual(observed[0][0], (1, 1, 2, self.D))
        self.assertEqual(observed[1][0], (1, self.RATIO, 2, self.D))
        torch.testing.assert_close(
            observed[0][1], torch.tensor([[2]], dtype=torch.int32, device=device)
        )
        torch.testing.assert_close(
            observed[1][1],
            torch.full((1, self.RATIO), 2, dtype=torch.int32, device=device),
        )
        visible_lengths = [8, 8, 9, 10, 11]
        for row, visible in enumerate(visible_lengths):
            valid = selected[row][selected[row] >= 0]
            self.assertTrue(bool(torch.all(valid < visible)))

        kv_pool = kv_base.view(torch.bfloat16).view(12, 2, self.D)
        state_pool = state_base.view(12, 2 * self.RATIO, self.D)
        self.assertGreater(int(torch.count_nonzero(kv_pool[1, 1])), 0)
        self.assertGreater(int(torch.count_nonzero(kv_pool[3, 1])), 0)
        torch.testing.assert_close(state_pool[5, 7], raw[0].float())
        torch.testing.assert_close(state_pool[7, 7], raw[1].float())
        torch.testing.assert_close(state_pool[8, 0], raw[2].float())
        torch.testing.assert_close(state_pool[8, 2], raw[4].float())

    def test_draft_incremental_prefill_rejects_before_side_writer(self):
        q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        cases = (
            (
                "explicit MTP draft",
                self._draft_incremental_context(
                    self.kv_base,
                    self.state_base,
                    prefixes=[7],
                    lengths=[1],
                    is_mtp_draft=False,
                ),
                q,
                raw,
            ),
            (
                r"max\(input_lengths\).*<= compress_ratio",
                self._draft_incremental_context(
                    self.kv_base,
                    self.state_base,
                    prefixes=[7],
                    lengths=[self.RATIO + 1],
                ),
                torch.randn(self.RATIO + 1, 2, self.D, dtype=torch.bfloat16),
                torch.randn(self.RATIO + 1, self.D, dtype=torch.bfloat16),
            ),
            (
                "logical cache positions",
                self._draft_incremental_context(
                    self.kv_base,
                    self.state_base,
                    prefixes=[7],
                    lengths=[1],
                    position_overrides=[8],
                ),
                q,
                raw,
            ),
            (
                "unallocated",
                self._draft_incremental_context(
                    self.kv_base,
                    self.state_base,
                    prefixes=[7],
                    lengths=[1],
                    state_table=torch.tensor([[0]], dtype=torch.int32),
                ),
                q,
                raw,
            ),
        )
        for expected, context, case_q, case_raw in cases:
            with self.subTest(expected=expected), patch(
                "rtp_llm.models_py.modules.qwen4_exp.qsa_runtime." "write_indexer_cache"
            ) as writer:
                kv_before = context.indexer_kv_cache.kv_cache_base.clone()
                state_before = context.indexer_state_cache.kv_cache_base.clone()
                with self.assertRaisesRegex(RuntimeError, expected):
                    context.select_draft_incremental_prefill_tokens(
                        case_q,
                        case_raw,
                        indexer=self._decode_indexer(),
                        rope_config=self._base_rope_config(3),
                    )
                writer.assert_not_called()
                self.assertTrue(
                    torch.equal(context.indexer_kv_cache.kv_cache_base, kv_before)
                )
                self.assertTrue(
                    torch.equal(context.indexer_state_cache.kv_cache_base, state_before)
                )

    def test_ordinary_prefill_still_rejects_nonzero_prefix_reuse(self):
        context = self._draft_incremental_context(
            self.kv_base,
            self.state_base,
            prefixes=[7],
            lengths=[1],
            is_mtp_draft=False,
        )
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa_runtime.write_indexer_cache"
        ) as writer:
            with self.assertRaisesRegex(RuntimeError, "prefix reuse"):
                context.write_prefill_indexer_cache(
                    torch.randn(1, self.D, dtype=torch.bfloat16),
                    indexer=self.indexer,
                    rope_config=self._base_rope_config(3),
                )
        writer.assert_not_called()

    @skipUnless(torch.cuda.is_available(), "CUDA is required for paged scoring")
    def test_draft_incremental_prefill_accepted_continuation_completes_groups(self):
        device = torch.device("cuda")
        self.indexer.k_norm_gamma = self.indexer.k_norm_gamma.to(device)
        indexer = self._decode_indexer()
        kv_base = torch.zeros(8, 2 * self.D * 2, dtype=torch.uint8, device=device)
        state_base = torch.zeros(
            8, 2 * self.RATIO * self.D, dtype=torch.float32, device=device
        )
        kv_table = torch.tensor([[1, 2, 3]], dtype=torch.int32, device=device)
        state_table = torch.tensor([[4, 5, 6]], dtype=torch.int32, device=device)
        raw = torch.randn(8, self.D, dtype=torch.bfloat16, device=device)
        state_pool = state_base.view(8, 2 * self.RATIO, self.D)
        state_pool[4, :3].copy_(raw[:3].float())

        def _score(q, weight, pool, table, lengths, *, block_size, max_ctx_len):
            return torch.zeros(
                q.shape[1], max_ctx_len, dtype=torch.float32, device=device
            )

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score."
            "qsa_paged_indexer_score",
            side_effect=_score,
        ):
            first = self._draft_incremental_context(
                kv_base,
                state_base,
                prefixes=[3],
                lengths=[2],
                kv_table=kv_table,
                state_table=state_table,
            ).select_draft_incremental_prefill_tokens(
                torch.randn(2, 2, self.D, dtype=torch.bfloat16, device=device),
                raw[3:5],
                indexer=indexer,
                rope_config=self._base_rope_config(3),
            )
            second = self._draft_incremental_context(
                kv_base,
                state_base,
                prefixes=[5],
                lengths=[3],
                kv_table=kv_table,
                state_table=state_table,
            ).select_draft_incremental_prefill_tokens(
                torch.randn(3, 2, self.D, dtype=torch.bfloat16, device=device),
                raw[5:8],
                indexer=indexer,
                rope_config=self._base_rope_config(3),
            )

        self.assertTrue(bool(torch.all(first[first >= 0] < 5)))
        per_row_visible = (6, 7, 8)
        for row, visible in enumerate(per_row_visible):
            valid = second[row][second[row] >= 0]
            self.assertTrue(bool(torch.all(valid < visible)))

        positions = torch.arange(8, dtype=torch.int32, device=device)
        transported = positions.unsqueeze(1).expand(-1, 3).reshape(-1)
        cos, sin = build_qsa_rope(
            transported,
            self._base_rope_config(3),
            token_count=8,
            dtype=torch.bfloat16,
            device=device,
            logical_positions=positions,
        )
        kv_pool = kv_base.view(torch.bfloat16).view(8, 2, self.D)
        torch.testing.assert_close(
            kv_pool[1, 0], self._expected(raw[:4], cos[0], sin[0], 0)
        )
        torch.testing.assert_close(
            kv_pool[1, 1], self._expected(raw[4:8], cos[4], sin[4], 4)
        )

    @skipUnless(torch.cuda.is_available(), "CUDA is required for paged scoring")
    def test_target_prefix_reuse_accepts_mixed_cold_and_hot_rows(self):
        device = torch.device("cuda")
        indexer = self._decode_indexer()
        indexer.k_norm_gamma = indexer.k_norm_gamma.to(device)
        kv_base = torch.zeros(16, 2 * self.D * 2, dtype=torch.uint8, device=device)
        state_base = torch.zeros(
            16, 2 * self.RATIO * self.D, dtype=torch.float32, device=device
        )
        context = self._draft_incremental_context(
            kv_base,
            state_base,
            prefixes=[0, 8],
            lengths=[8, 4],
            is_mtp_draft=False,
        )
        q = torch.randn(12, 2, self.D, dtype=torch.bfloat16, device=device)
        raw = torch.randn(12, self.D, dtype=torch.bfloat16, device=device)

        def _score(q, weight, pool, table, lengths, *, block_size, max_ctx_len):
            return torch.zeros(
                q.shape[1], max_ctx_len, dtype=torch.float32, device=device
            )

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score."
            "qsa_paged_indexer_score",
            side_effect=_score,
        ):
            selected = context.select_prefix_reuse_prefill_tokens(
                q,
                raw,
                indexer=indexer,
                rope_config=self.rope_config,
            )

        self.assertEqual(selected.shape, (12, 11))
        offset = 0
        for prefix, length in ((0, 8), (8, 4)):
            for local_idx in range(length):
                valid = selected[offset + local_idx]
                valid = valid[valid >= 0]
                self.assertTrue(bool(torch.all(valid < prefix + local_idx + 1)))
            offset += length

    def test_decode_completes_prefill_tail_then_scores_same_projection(self):
        prefill_raw = torch.randn(7, self.D, dtype=torch.bfloat16)
        decode_raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        decode_q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        indexer = self._decode_indexer()
        prefill = self._single_request_context(
            self.kv_base,
            self.state_base,
            is_prefill=True,
            input_length=7,
            sequence_length=0,
            position_ids=[[pos, pos, pos] for pos in range(7)],
        )
        _, prefill_cos, prefill_sin, result = prefill.write_prefill_indexer_cache(
            prefill_raw, indexer=indexer, rope_config=self.rope_config
        )
        self.assertEqual(result["num_kv_writes"], 1)

        decode = self._single_request_context(
            self.kv_base,
            self.state_base,
            is_prefill=False,
            input_length=7,
            sequence_length=7,
            position_ids=[[7, 7, 7]],
        )
        observed = {}

        def _score(q, weight, pool, table, lengths, *, block_size, max_ctx_len):
            observed.update(
                q=q,
                weight=weight,
                pool=pool,
                table=table,
                lengths=lengths,
                block_size=block_size,
                max_ctx_len=max_ctx_len,
            )
            return torch.tensor([[0.25, 3.0]], dtype=torch.float32)

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score."
            "qsa_paged_indexer_score",
            side_effect=_score,
        ):
            selected = decode.select_decode_tokens(
                decode_q,
                decode_raw,
                indexer=indexer,
                rope_config=self.rope_config,
            )

        kv_pool = self.kv_base.view(torch.bfloat16).view(7, 2, self.D)
        state_pool = self.state_base.view(8, 2 * self.RATIO, self.D)
        expected_second = self._expected(
            torch.cat([prefill_raw[4:], decode_raw]),
            prefill_cos[0, 4],
            prefill_sin[0, 4],
            4,
        )
        torch.testing.assert_close(kv_pool[3, 1], expected_second)
        torch.testing.assert_close(state_pool[6, 7], decode_raw[0].float())
        self.assertEqual(observed["q"].shape, (1, 1, 2, self.D))
        torch.testing.assert_close(
            observed["weight"], torch.full((1, 2), 1.0 / self.D**0.5)
        )
        torch.testing.assert_close(
            observed["lengths"], torch.tensor([[2]], dtype=torch.int32)
        )
        self.assertEqual(observed["block_size"], 2)
        self.assertEqual(observed["max_ctx_len"], 2)
        torch.testing.assert_close(
            selected,
            torch.tensor(
                [[4, 5, 6, 7, 0, 1, 2, 3, -1, -1, -1]],
                dtype=torch.int32,
            ),
        )

    def test_paged_score_failure_can_rollback_the_side_cache(self):
        self.kv_base.random_(0, 256)
        self.state_base.normal_()
        kv_before = self.kv_base.clone()
        state_before = self.state_base.clone()
        context = self._single_request_context(
            self.kv_base,
            self.state_base,
            is_prefill=False,
            input_length=7,
            sequence_length=7,
            position_ids=[[7, 7, 7]],
        )

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score."
            "qsa_paged_indexer_score",
            side_effect=RuntimeError("injected paged score failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected paged score failure"):
                context.select_decode_tokens(
                    torch.randn(1, 2, self.D, dtype=torch.bfloat16),
                    torch.randn(1, self.D, dtype=torch.bfloat16),
                    indexer=self._decode_indexer(),
                    rope_config=self.rope_config,
                )

        self.assertFalse(torch.equal(self.kv_base, kv_before))
        self.assertFalse(torch.equal(self.state_base, state_before))
        context.rollback_side_cache()
        self.assertTrue(torch.equal(self.kv_base, kv_before))
        self.assertTrue(torch.equal(self.state_base, state_before))
        context.rollback_side_cache()

    def test_main_attention_failure_after_selection_can_rollback_side_cache(self):
        self.kv_base.random_(0, 256)
        self.state_base.normal_()
        kv_before = self.kv_base.clone()
        state_before = self.state_base.clone()
        context = self._single_request_context(
            self.kv_base,
            self.state_base,
            is_prefill=False,
            input_length=7,
            sequence_length=7,
            position_ids=[[7, 7, 7]],
        )

        def _score(q, weight, pool, table, lengths, *, block_size, max_ctx_len):
            return torch.zeros((1, max_ctx_len), dtype=torch.float32)

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score."
            "qsa_paged_indexer_score",
            side_effect=_score,
        ):
            context.select_decode_tokens(
                torch.randn(1, 2, self.D, dtype=torch.bfloat16),
                torch.randn(1, self.D, dtype=torch.bfloat16),
                indexer=self._decode_indexer(),
                rope_config=self.rope_config,
            )

        self.assertFalse(torch.equal(self.kv_base, kv_before))
        self.assertFalse(torch.equal(self.state_base, state_before))
        try:
            raise RuntimeError("injected main attention failure")
        except RuntimeError:
            context.rollback_side_cache()
        self.assertTrue(torch.equal(self.kv_base, kv_before))
        self.assertTrue(torch.equal(self.state_base, state_before))

    def test_side_cache_transaction_reentry_is_rejected(self):
        raw_keys = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)
        context = self._context()
        context.write_prefill_indexer_cache(
            raw_keys, indexer=self.indexer, rope_config=self.rope_config
        )
        with self.assertRaisesRegex(RuntimeError, "already active"):
            context.write_prefill_indexer_cache(
                raw_keys, indexer=self.indexer, rope_config=self.rope_config
            )
        context.rollback_side_cache()

    def test_actual_finalize_keeps_writes_and_allows_the_next_transaction(self):
        first_raw = torch.randn(self.total_tokens, self.D, dtype=torch.bfloat16)
        second_raw = torch.randn_like(first_raw)
        context = self._context()
        context.write_prefill_indexer_cache(
            first_raw, indexer=self.indexer, rope_config=self.rope_config
        )
        kv_after_first = self.kv_base.clone()
        state_after_first = self.state_base.clone()

        context.finalize_side_cache()
        self.assertTrue(torch.equal(self.kv_base, kv_after_first))
        self.assertTrue(torch.equal(self.state_base, state_after_first))
        context.write_prefill_indexer_cache(
            second_raw, indexer=self.indexer, rope_config=self.rope_config
        )
        context.finalize_side_cache()

        self.assertFalse(torch.equal(self.kv_base, kv_after_first))
        self.assertFalse(torch.equal(self.state_base, state_after_first))

    def test_decode_rejects_target_verify_prefix_and_non_text_mrope(self):
        q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        indexer = self._decode_indexer()
        cases = (
            ("target-verify", dict(target_verify=True)),
            ("prefix reuse", dict(prefix_length=1)),
            ("text-only MRoPE", dict(position_ids=[[7, 8, 7]])),
        )
        for expected, overrides in cases:
            values = dict(
                is_prefill=False,
                input_length=7,
                sequence_length=7,
                position_ids=[[7, 7, 7]],
            )
            values.update(overrides)
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(RuntimeError, expected):
                    self._single_request_context(
                        self.kv_base, self.state_base, **values
                    ).select_decode_tokens(
                        q, raw, indexer=indexer, rope_config=self.rope_config
                    )

    def test_decode_base_rope_factor_one_and_three_rotate_q_identically(self):
        q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        rotated_queries = []

        def _score(q, weight, pool, table, lengths, *, block_size, max_ctx_len):
            rotated_queries.append(q.clone())
            return torch.zeros((1, 1), dtype=torch.float32)

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score."
            "qsa_paged_indexer_score",
            side_effect=_score,
        ):
            for index_factor, position_ids in ((1, [[3]]), (3, [[3, 3, 3]])):
                context = self._single_request_context(
                    torch.zeros_like(self.kv_base),
                    torch.zeros_like(self.state_base),
                    is_prefill=False,
                    input_length=3,
                    sequence_length=3,
                    position_ids=position_ids,
                )
                context.select_decode_tokens(
                    q,
                    raw,
                    indexer=self._decode_indexer(),
                    rope_config=self._base_rope_config(index_factor),
                )

        self.assertEqual(len(rotated_queries), 2)
        torch.testing.assert_close(rotated_queries[0], rotated_queries[1])

    def test_decode_base_rope_rejects_non_logical_position_before_write(self):
        q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        context = self._single_request_context(
            self.kv_base,
            self.state_base,
            is_prefill=False,
            input_length=3,
            sequence_length=3,
            position_ids=[[4]],
        )

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa_runtime.write_indexer_cache"
        ) as writer:
            with self.assertRaisesRegex(RuntimeError, "logical cache positions"):
                context.select_decode_tokens(
                    q,
                    raw,
                    indexer=self._decode_indexer(),
                    rope_config=self._base_rope_config(1),
                )

        writer.assert_not_called()

    def test_decode_rejects_compressed_length_beyond_tag_local_table(self):
        q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        decode = self._single_request_context(
            self.kv_base,
            self.state_base,
            is_prefill=False,
            input_length=11,
            sequence_length=11,
            position_ids=[[11, 11, 11]],
        )

        with self.assertRaisesRegex(RuntimeError, "compressed decode length"):
            decode.select_decode_tokens(
                q,
                raw,
                indexer=self._decode_indexer(),
                rope_config=self.rope_config,
            )

        self.assertEqual(int(torch.count_nonzero(self.kv_base)), 0)
        self.assertEqual(int(torch.count_nonzero(self.state_base)), 0)

    def test_decode_rejects_unallocated_required_kv_block_before_side_write(self):
        q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        self.kv_base.random_(0, 256)
        self.state_base.normal_()
        kv_before = self.kv_base.clone()
        state_before = self.state_base.clone()
        decode = self._single_request_context(
            self.kv_base,
            self.state_base,
            is_prefill=False,
            input_length=7,
            sequence_length=7,
            position_ids=[[7, 7, 7]],
        )
        decode.indexer_kv_inputs.kv_cache_kernel_block_id_device.zero_()

        with self.assertRaisesRegex(RuntimeError, "unallocated"):
            decode.select_decode_tokens(
                q,
                raw,
                indexer=self._decode_indexer(),
                rope_config=self.rope_config,
            )

        self.assertTrue(torch.equal(self.kv_base, kv_before))
        self.assertTrue(torch.equal(self.state_base, state_before))

    @skipUnless(torch.cuda.is_available(), "CUDA is required for the paged scorer")
    def test_cuda_decode_selection_matches_direct_paged_score_math(self):
        device = torch.device("cuda")
        kv_base = torch.zeros(7, 2 * self.D * 2, dtype=torch.uint8, device=device)
        state_base = torch.zeros(
            8, 2 * self.RATIO * self.D, dtype=torch.float32, device=device
        )
        indexer = self._decode_indexer()
        indexer.k_norm_gamma = indexer.k_norm_gamma.to(device)
        prefill_raw = torch.randn(7, self.D, dtype=torch.bfloat16, device=device)
        decode_raw = torch.randn(1, self.D, dtype=torch.bfloat16, device=device)
        decode_q = torch.randn(1, 2, self.D, dtype=torch.bfloat16, device=device)
        self._single_request_context(
            kv_base,
            state_base,
            is_prefill=True,
            input_length=7,
            sequence_length=0,
            position_ids=[[pos, pos, pos] for pos in range(7)],
        ).write_prefill_indexer_cache(
            prefill_raw, indexer=indexer, rope_config=self.rope_config
        )
        decode = self._single_request_context(
            kv_base,
            state_base,
            is_prefill=False,
            input_length=7,
            sequence_length=7,
            position_ids=[[7, 7, 7]],
        )

        selected = decode.select_decode_tokens(
            decode_q,
            decode_raw,
            indexer=indexer,
            rope_config=self.rope_config,
        )

        positions = torch.arange(8, dtype=torch.int32, device=device)
        positions = positions.unsqueeze(1).expand(-1, 3)
        cos, sin = build_interleaved_mrope(
            positions.reshape(-1),
            self.rope_config,
            token_count=8,
            dtype=torch.bfloat16,
            device=device,
        )
        q = apply_partial_rope(decode_q, cos[7].view(1, 1, -1), sin[7].view(1, 1, -1))
        keys = kv_base.view(torch.bfloat16).view(7, 2, self.D)[3]
        logits = torch.relu(q.float() @ keys.float().T).sum(dim=1) / self.D**0.5
        chosen = logits[0].topk(2).indices.tolist()
        expected = []
        for block in chosen:
            expected.extend(range(block * self.RATIO, (block + 1) * self.RATIO))
        expected.extend([-1] * (8 + self.RATIO - 1 - len(expected)))
        torch.testing.assert_close(
            selected, torch.tensor([expected], dtype=torch.int32, device=device)
        )

    def test_paged_topk_expands_blocks_and_appends_partial_tail(self):
        block_logits = torch.tensor(
            [
                [0.2, 1.5, 0.7, float("-inf")],
                [3.0, 20.0, float("-inf"), float("-inf")],
            ],
            dtype=torch.float32,
        )
        token_lengths = torch.tensor([14, 5], dtype=torch.int32)

        selected = select_qsa_paged_tokens(
            block_logits,
            token_lengths,
            compress_ratio=self.RATIO,
            token_budget=8,
        )

        expected = torch.tensor(
            [
                [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1],
                [0, 1, 2, 3, -1, -1, -1, -1, 4, -1, -1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(selected, expected)


if __name__ == "__main__":
    main()
