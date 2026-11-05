import os
from contextlib import nullcontext
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch
from rtp_llm.models_py.model_desc.qwen4_exp_runtime import qsa_prefill
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.models_py.model_desc.test.qwen4_exp.qsa_fixtures import QSARuntimeFixture
from rtp_llm.models_py.modules.qwen4_exp.qsa import (
    cub_topk,
    indexer_paged_score,
    indexer_tensorcore,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import build_qsa_rope


class Qwen4ExpQSAMtpTest(QSARuntimeFixture, TestCase):
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
            "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context.write_indexer_cache"
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
            "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context.write_indexer_cache"
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

        def _score(
            q,
            weight,
            pool,
            table,
            lengths,
            *,
            block_size,
            max_ctx_len,
            validate_block_table,
        ):
            self.assertFalse(validate_block_table)
            observed.update(q=q, lengths=lengths, max_ctx_len=max_ctx_len)
            return torch.tensor(
                [[0.25, 3.0]] * q_len,
                dtype=torch.float32,
                device=q.device,
            )

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score."
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

    @skipUnless(torch.cuda.is_available(), "CUDA is required for target preflight")
    def test_target_preflight_plan_is_consumed_once(self):
        self.kv_base = self.kv_base.cuda()
        self.state_base = self.state_base.cuda()
        self.indexer.k_norm_gamma = self.indexer.k_norm_gamma.cuda()
        q_len = self.RATIO
        context = self._target_verify_context(q_len=q_len, committed_length=7)
        indexer = self._decode_indexer()
        raw_keys = torch.randn(q_len, self.D, dtype=torch.bfloat16, device="cuda")
        q = torch.randn(q_len, 2, self.D, dtype=torch.bfloat16, device="cuda")
        original = Qwen4ExpQSARuntimeContext._target_verify_geometry
        with (
            patch.object(
                Qwen4ExpQSARuntimeContext,
                "_target_verify_geometry",
                autospec=True,
                side_effect=original,
            ) as geometry,
            patch(
                "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score."
                "qsa_paged_indexer_score",
                return_value=torch.zeros(q_len, 2, device="cuda"),
            ),
        ):
            context.validate_before_projection(
                indexer=indexer, token_count=q_len, device=raw_keys.device
            )
            context.select_target_verify_tokens(
                q, raw_keys, indexer=indexer, rope_config=self.rope_config
            )
        self.assertEqual(geometry.call_count, 1)
        self.assertIsNone(context._target_verify_plan)

    @skipUnless(torch.cuda.is_available(), "CUDA is required for paged scoring")
    def test_target_verify_never_overwrites_the_committed_partial_group(self):
        self.indexer.k_norm_gamma = self.indexer.k_norm_gamma.cuda()
        device = self.indexer.k_norm_gamma.device
        q_len = self.RATIO

        def _score(
            q,
            weight,
            pool,
            table,
            lengths,
            *,
            block_size,
            max_ctx_len,
            validate_block_table,
        ):
            self.assertFalse(validate_block_table)
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
                    "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score."
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

        def _score(
            q,
            weight,
            pool,
            table,
            lengths,
            *,
            block_size,
            max_ctx_len,
            validate_block_table,
        ):
            observed.append((q.shape, lengths.clone(), table.clone(), max_ctx_len))
            self.assertFalse(validate_block_table)
            rows = q.shape[0] * q.shape[1]
            return torch.arange(
                rows * max_ctx_len, dtype=torch.float32, device=device
            ).view(rows, max_ctx_len)

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score."
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
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0], (2, self.RATIO, 2, self.D))
        torch.testing.assert_close(
            observed[0][1],
            torch.tensor(
                [[2, 0, 0, 0], [2, 2, 2, 2]], dtype=torch.int32, device=device
            ),
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

    @skipUnless(torch.cuda.is_available(), "CUDA is required for the draft writer")
    def test_short_draft_writer_matches_reference_and_rolls_back(self):
        device = torch.device("cuda")
        self.D = 128
        self.KV_TOKENS_PER_BLOCK = 128
        self.STATE_TOKENS_PER_BLOCK = 128
        self.indexer.k_norm_gamma = (torch.randn(128, device=device) * 0.1).bfloat16()
        indexer = self._decode_indexer()
        rope = self._base_rope_config(3)
        rope.dim = 64
        kv_initial = torch.zeros(12, 32 * 128 * 2, dtype=torch.uint8, device=device)
        state_initial = torch.randn(12, 8 * 128, dtype=torch.float32, device=device)
        kv_table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device=device)
        state_table = torch.tensor([[5, 6], [7, 8]], dtype=torch.int32, device=device)
        for prefixes, lengths in (
            ([5, 127], [4, 4]),
            ([7, 127], [1, 4]),
            ([127, 5], [3, 2]),
        ):
            with self.subTest(prefixes=prefixes, lengths=lengths):
                q = torch.randn(
                    sum(lengths), 2, 128, dtype=torch.bfloat16, device=device
                )
                raw = torch.randn(
                    sum(lengths), 128, dtype=torch.bfloat16, device=device
                )

                def run(fused):
                    kv = kv_initial.clone()
                    state = state_initial.clone()
                    context = self._draft_incremental_context(
                        kv,
                        state,
                        prefixes=prefixes,
                        lengths=lengths,
                        kv_table=kv_table,
                        state_table=state_table,
                    )

                    def score(
                        q, weight, pool, table, lengths, *, max_ctx_len, **kwargs
                    ):
                        return torch.zeros(
                            q.shape[0] * q.shape[1],
                            max_ctx_len,
                            dtype=torch.float32,
                            device=device,
                        )

                    with patch(
                        "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score."
                        "qsa_paged_indexer_score",
                        side_effect=score,
                    ):
                        selected = context.select_draft_incremental_prefill_tokens(
                            q, raw, indexer=indexer, rope_config=rope
                        )
                    return context, selected, kv, state

                reference, selected_ref, kv_ref, state_ref = run(False)
                candidate, selected, kv, state = run(True)
                torch.testing.assert_close(selected, selected_ref, rtol=0, atol=0)
                torch.testing.assert_close(state, state_ref, rtol=0, atol=0)
                torch.testing.assert_close(
                    kv.view(torch.bfloat16),
                    kv_ref.view(torch.bfloat16),
                    rtol=0.01,
                    atol=0.016,
                )
                candidate.rollback_side_cache()
                torch.testing.assert_close(kv, kv_initial, rtol=0, atol=0)
                torch.testing.assert_close(state, state_initial, rtol=0, atol=0)
                reference.finalize_side_cache()

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
            with (
                self.subTest(expected=expected),
                patch(
                    "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context."
                    "write_indexer_cache"
                ) as writer,
            ):
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

        def _score(
            q,
            weight,
            pool,
            table,
            lengths,
            *,
            block_size,
            max_ctx_len,
            validate_block_table,
        ):
            self.assertFalse(validate_block_table)
            return torch.zeros(
                q.shape[0] * q.shape[1], max_ctx_len, dtype=torch.float32, device=device
            )

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score."
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

        def _score(
            q,
            weight,
            pool,
            table,
            lengths,
            *,
            block_size,
            max_ctx_len,
            validate_block_table,
        ):
            self.assertFalse(validate_block_table)
            return torch.zeros(
                q.shape[0] * q.shape[1], max_ctx_len, dtype=torch.float32, device=device
            )

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score."
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

    @skipUnless(torch.cuda.is_available(), "CUDA is required for paged scoring")
    def test_mtp_draft_prefix_reuse_accepts_local_cu_kv_coordinates(self):
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
            prefixes=[8],
            lengths=[8],
            is_mtp_draft=True,
        )
        local_cu_kv = torch.tensor([0, 8], dtype=torch.int32, device=device)
        for inputs in (
            context.main_inputs,
            context.indexer_kv_inputs,
            context.indexer_state_inputs,
        ):
            inputs.cu_kv_seqlens_device = local_cu_kv

        q = torch.randn(8, 2, self.D, dtype=torch.bfloat16, device=device)
        raw = torch.randn(8, self.D, dtype=torch.bfloat16, device=device)

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score."
            "qsa_paged_indexer_score",
            side_effect=lambda q, weight, pool, table, lengths, **kwargs: torch.zeros(
                q.shape[1], kwargs["max_ctx_len"], dtype=torch.float32, device=device
            ),
        ):
            selected = context.select_draft_prefix_reuse_prefill_tokens(
                q,
                raw,
                indexer=indexer,
                rope_config=self._base_rope_config(3),
            )

        self.assertEqual(selected.shape, (8, 8 + self.RATIO - 1))
        for row, visible in enumerate(range(9, 17)):
            valid = selected[row][selected[row] >= 0]
            self.assertTrue(bool(torch.all(valid < visible)))

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
            "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score."
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

        def _score(
            q,
            weight,
            pool,
            table,
            lengths,
            *,
            block_size,
            max_ctx_len,
            validate_block_table,
        ):
            self.assertFalse(validate_block_table)
            return torch.zeros((1, max_ctx_len), dtype=torch.float32)

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score."
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

    def _assert_identity_packing_bytes_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def _identity_packing_fixture(self):
        self.D = self.KV_TOKENS_PER_BLOCK = self.STATE_TOKENS_PER_BLOCK = 128
        self.indexer.k_norm_gamma = torch.randn(
            128, dtype=torch.bfloat16, device="cuda"
        )
        indexer = self._decode_indexer(heads=4)
        indexer.token_budget = 2048
        rope = self.rope_config
        rope.dim = 64
        rope.mrope_dim1, rope.mrope_dim2, rope.mrope_dim3 = 16, 8, 8
        # Physical page zero remains reserved, as in the production side pool.
        kv = torch.randn(129, 32 * 128, dtype=torch.bfloat16, device="cuda").view(
            torch.uint8
        )
        state = torch.randn(129, 8 * 128, dtype=torch.float32, device="cuda")
        q = torch.randn(8192, 4, 128, dtype=torch.bfloat16, device="cuda")
        raw = torch.randn(8192, 128, dtype=torch.bfloat16, device="cuda")
        return indexer, rope, kv, state, q, raw

    def _run_identity_packing_route(
        self,
        fixture,
        *,
        enabled,
        expect_identity,
        ambient_capture=False,
        device_prefix=False,
    ):
        indexer, rope, initial_kv, initial_state, q, raw = fixture
        kv, state = initial_kv.clone(), initial_state.clone()
        table = torch.arange(1, 129, dtype=torch.int32, device="cuda").view(1, -1)
        context = self._draft_incremental_context(
            kv,
            state,
            prefixes=[8192],
            lengths=[8192],
            kv_table=table,
            state_table=table,
            is_mtp_draft=False,
        )
        if not device_prefix:
            prefix = torch.tensor([8192], dtype=torch.int32)
            for inputs in (
                context.main_inputs,
                context.indexer_kv_inputs,
                context.indexer_state_inputs,
            ):
                inputs.prefix_lengths = prefix
        object.__setattr__(context, "metadata_cache", {})
        observed = {}
        real_rope = qsa_prefill.apply_partial_rope
        real_score = indexer_paged_score.qsa_paged_indexer_score
        real_cub = cub_topk.try_cub_qsa_tokens
        real_full = torch.full

        def observe_rope(*args, **kwargs):
            result = real_rope(*args, **kwargs)
            observed["rotated_q"] = result
            return result

        def observe_full(size, fill_value, **kwargs):
            result = real_full(size, fill_value, **kwargs)
            if tuple(size) == (8192, 4) and result.dtype == torch.float32:
                self.assertNotIn("weights_owner", observed)
                observed["weights_owner"] = result
            return result

        def observe_score(score_q, weights, *args, **kwargs):
            observed["score_q"] = score_q
            observed["score_weights"] = weights
            observed["q_before"] = score_q.clone()
            observed["weights_before"] = weights.clone()
            result = real_score(score_q, weights, *args, **kwargs)
            observed["logits"] = result
            return result

        def observe_cub(*args, **kwargs):
            result = real_cub(*args, **kwargs)
            self.assertIsNotNone(result, "real SM120 CUB provider must run")
            self.assertNotIn("selected_owner", observed)
            observed["selected_owner"] = result
            return result

        q_before, raw_before = q.clone(), raw.clone()
        capture_guard = (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=True)
            if ambient_capture
            else nullcontext()
        )
        with (
            patch.dict(
                os.environ,
                {
                    "RTP_QWEN4_PREFILL_IDENTITY_PACKING": str(enabled),
                },
            ),
            patch.object(qsa_prefill, "apply_partial_rope", side_effect=observe_rope),
            patch.object(torch, "full", side_effect=observe_full),
            patch.object(
                indexer_paged_score,
                "qsa_paged_indexer_score",
                side_effect=observe_score,
            ),
            patch.object(
                indexer_tensorcore,
                "paged_indexer_mma",
                wraps=indexer_tensorcore.paged_indexer_mma,
            ) as mma,
            patch.object(
                cub_topk, "try_cub_qsa_tokens", side_effect=observe_cub
            ) as cub,
            capture_guard,
        ):
            selected = context.select_prefix_reuse_prefill_tokens(
                q, raw, indexer=indexer, rope_config=rope
            )
        self.assertEqual(mma.call_count, 1, "no vector/native scorer substitution")
        self.assertEqual(cub.call_count, 1, "no native TopK substitution")
        self.assertEqual(observed["logits"].shape, (8192, 4096))
        self.assertEqual(selected.shape, (8192, 2051))
        self.assertTrue(bool(torch.isneginf(observed["logits"][:, -1]).any()))
        self._assert_identity_packing_bytes_equal(q, q_before)
        self._assert_identity_packing_bytes_equal(raw, raw_before)
        self._assert_identity_packing_bytes_equal(
            observed["score_q"], observed["q_before"]
        )
        self._assert_identity_packing_bytes_equal(
            observed["score_weights"], observed["weights_before"]
        )
        if expect_identity:
            self.assertEqual(
                observed["score_q"].data_ptr(), observed["rotated_q"].data_ptr()
            )
            self.assertIs(observed["score_weights"], observed["weights_owner"])
            self.assertIs(selected, observed["selected_owner"])
        else:
            self.assertNotEqual(
                observed["score_q"].data_ptr(), observed["rotated_q"].data_ptr()
            )
            self.assertIsNot(observed["score_weights"], observed["weights_owner"])
            self.assertIsNot(selected, observed["selected_owner"])
        self.assertIsNotNone(context._side_cache_undo)
        written_kv, written_state = kv.clone(), state.clone()
        context.rollback_side_cache()
        self._assert_identity_packing_bytes_equal(kv, initial_kv)
        self._assert_identity_packing_bytes_equal(state, initial_state)
        context.rollback_side_cache()
        return dict(
            selected=selected,
            logits=observed["logits"],
            kv=written_kv,
            state=written_state,
        )

    @skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
        "real MMA+CUB identity packing qualification requires SM120",
    )
    def test_identity_prefill_packing_real_providers_are_bitexact_and_owned(self):
        fixture = self._identity_packing_fixture()
        baseline = self._run_identity_packing_route(
            fixture, enabled=0, expect_identity=False
        )
        first = self._run_identity_packing_route(
            fixture, enabled=1, expect_identity=True
        )
        for boundary in ("logits", "selected", "kv", "state"):
            self._assert_identity_packing_bytes_equal(
                first[boundary], baseline[boundary]
            )

        indexer, rope, kv, state, q, raw = fixture
        changed = (indexer, rope, kv, state, -q, -raw)
        second_baseline = self._run_identity_packing_route(
            changed, enabled=0, expect_identity=False
        )
        second = self._run_identity_packing_route(
            changed, enabled=1, expect_identity=True
        )
        for boundary in ("logits", "selected", "kv", "state"):
            self._assert_identity_packing_bytes_equal(
                second[boundary], second_baseline[boundary]
            )
            self._assert_identity_packing_bytes_equal(
                first[boundary], baseline[boundary]
            )
        self.assertFalse(torch.equal(first["logits"], second["logits"]))
        self.assertFalse(torch.equal(first["selected"], second["selected"]))
        self.assertNotEqual(first["selected"].data_ptr(), second["selected"].data_ptr())
        # Two live producer outputs must stay independent after another call.
        second["selected"].fill_(-123)
        self._assert_identity_packing_bytes_equal(
            first["selected"], baseline["selected"]
        )

        # Query-only simulation checks the ambient-capture guard; it does not
        # claim that this host-metadata prefill route supports actual capture.
        for options in ({"ambient_capture": True}, {"device_prefix": True}):
            with self.subTest(options=options):
                fallback = self._run_identity_packing_route(
                    fixture, enabled=1, expect_identity=False, **options
                )
                for boundary in ("logits", "selected", "kv", "state"):
                    self._assert_identity_packing_bytes_equal(
                        fallback[boundary], baseline[boundary]
                    )
        noncontiguous_q = q.transpose(0, 1).contiguous().transpose(0, 1)
        self.assertFalse(noncontiguous_q.is_contiguous())
        fallback = self._run_identity_packing_route(
            (indexer, rope, kv, state, noncontiguous_q, raw),
            enabled=1,
            expect_identity=False,
        )
        for boundary in ("logits", "selected", "kv", "state"):
            self._assert_identity_packing_bytes_equal(
                fallback[boundary], baseline[boundary]
            )

    @skipUnless(torch.cuda.is_available(), "CUDA is required for existing fallbacks")
    def test_identity_prefill_packing_reuses_existing_fallbacks(self):
        with patch.dict(os.environ, {"RTP_QWEN4_PREFILL_IDENTITY_PACKING": "1"}):
            for existing in (
                self.test_target_verify_uses_prefix_for_every_query_row,
                self.test_target_prefix_reuse_accepts_mixed_cold_and_hot_rows,
                self.test_mtp_draft_prefix_reuse_accepts_local_cu_kv_coordinates,
                self.test_draft_incremental_prefill_is_ragged_and_visibility_bounded,
            ):
                with self.subTest(existing=existing.__name__):
                    self.setUp()
                    existing()
            self.setUp()
            self._check_draft_graph_replay(4)


if __name__ == "__main__":
    main()
