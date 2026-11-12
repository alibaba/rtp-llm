import math
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.models_py.model_desc.test.qwen4_exp.qsa_fixtures import QSARuntimeFixture
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import build_qsa_rope


class Qwen4ExpQSAMtpTest(QSARuntimeFixture, TestCase):
    @skipUnless(torch.cuda.is_available(), "CUDA is required for target verification")
    def test_eager_fake_target_verify_preserves_reserved_pages_and_returns_padding(
        self,
    ):
        for head_dim in (4, 128):
            for query_len in (1, 4):
                with self.subTest(head_dim=head_dim, query_len=query_len):
                    self.D = head_dim
                    self.KV_TOKENS_PER_BLOCK = self.STATE_TOKENS_PER_BLOCK = 128
                    self.kv_base = torch.randn(
                        7, 32 * head_dim, device="cuda", dtype=torch.bfloat16
                    ).view(torch.uint8)
                    self.state_base = torch.randn(8, 8 * head_dim, device="cuda")
                    self.indexer.k_norm_gamma = torch.randn(
                        head_dim, device="cuda"
                    ).bfloat16()
                    if head_dim == 128:
                        self.rope_config.dim = 64
                        self.rope_config.mrope_dim1 = 12
                        self.rope_config.mrope_dim2 = self.rope_config.mrope_dim3 = 10
                    context = self._target_verify_context(
                        q_len=query_len, committed_length=1
                    )
                    for inputs in (
                        context.main_inputs,
                        context.indexer_kv_inputs,
                        context.indexer_state_inputs,
                    ):
                        inputs.is_s_padded = True
                    context.indexer_kv_inputs.kv_cache_kernel_block_id_device.zero_()
                    context.indexer_state_inputs.kv_cache_block_id_device.zero_()
                    before = self.kv_base.clone(), self.state_base.clone()
                    selected = context.select_target_verify_tokens(
                        torch.randn(
                            query_len, 2, head_dim, device="cuda", dtype=torch.bfloat16
                        ),
                        torch.randn(
                            query_len, head_dim, device="cuda", dtype=torch.bfloat16
                        ),
                        indexer=self._decode_indexer(),
                        rope_config=self.rope_config,
                    )
                    self.assertTrue(bool((selected == -1).all()))
                    torch.testing.assert_close(self.kv_base, before[0], rtol=0, atol=0)
                    torch.testing.assert_close(
                        self.state_base, before[1], rtol=0, atol=0
                    )
                    context.rollback_side_cache()
                    torch.testing.assert_close(self.kv_base, before[0], rtol=0, atol=0)
                    torch.testing.assert_close(
                        self.state_base, before[1], rtol=0, atol=0
                    )

    @skipUnless(torch.cuda.is_available(), "CUDA is required for target preflight")
    def test_eager_target_padding_keeps_one_sided_missing_and_partial_pages_strict(
        self,
    ):
        self.kv_base = self.kv_base.cuda()
        self.state_base = self.state_base.cuda()
        self.indexer.k_norm_gamma = self.indexer.k_norm_gamma.cuda()
        for bad_region in ("kv", "state", "state_tail"):
            with self.subTest(bad_region=bad_region):
                context = self._target_verify_context(q_len=4, committed_length=7)
                for inputs in (
                    context.main_inputs,
                    context.indexer_kv_inputs,
                    context.indexer_state_inputs,
                ):
                    inputs.is_s_padded = True
                if bad_region == "kv":
                    context.indexer_kv_inputs.kv_cache_kernel_block_id_device.zero_()
                elif bad_region == "state":
                    context.indexer_state_inputs.kv_cache_block_id_device.zero_()
                else:
                    context.indexer_state_inputs.kv_cache_block_id_device[0, 1] = 0
                before = self.kv_base.clone(), self.state_base.clone()
                with self.assertRaisesRegex(RuntimeError, "unallocated"):
                    context._target_verify_geometry(
                        indexer=self._decode_indexer(),
                        token_count=4,
                        device=self.kv_base.device,
                    )
                self.assertIsNone(context._side_cache_undo)
                torch.testing.assert_close(self.kv_base, before[0], rtol=0, atol=0)
                torch.testing.assert_close(self.state_base, before[1], rtol=0, atol=0)

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
            "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score."
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
                "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score."
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
                    "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score."
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
            "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score."
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
                        "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score."
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
                RuntimeError,
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
                RuntimeError,
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
                ValueError,
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
                RuntimeError,
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
        for expected, error_type, context, case_q, case_raw in cases:
            with (
                self.subTest(expected=expected),
                patch(
                    "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context."
                    "write_indexer_cache"
                ) as writer,
            ):
                kv_before = context.indexer_kv_cache.kv_cache_base.clone()
                state_before = context.indexer_state_cache.kv_cache_base.clone()
                with self.assertRaisesRegex(error_type, expected):
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
            "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score."
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
    def test_prefix_prefill_score_chunks_preserve_selection_and_bound_storage(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score import (
            qsa_paged_indexer_score,
        )
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime import qsa_prefill
        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
            select_qsa_paged_tokens,
        )

        torch.manual_seed(73)
        device = torch.device("cuda")
        lengths = [257, 3]
        prefixes = [3840, 256]
        rows = max(lengths)
        visible = (
            torch.tensor(prefixes, device=device)[:, None]
            + torch.arange(rows, device=device)
            + 1
        )
        valid = (
            torch.arange(rows, device=device)[None, :]
            < torch.tensor(lengths, device=device)[:, None]
        )
        visible = torch.where(valid, visible, 0).to(torch.int32)
        columns = max((p + n) // 4 for p, n in zip(prefixes, lengths))
        block_size = 32
        table_width = (columns + block_size - 1) // block_size
        table = torch.arange(
            1, 2 * table_width + 1, device=device, dtype=torch.int32
        ).view(2, table_width)
        pool = torch.randn(
            2 * table_width + 1, block_size, 128, device=device, dtype=torch.bfloat16
        )
        q = torch.randn(sum(lengths), 4, 128, device=device, dtype=torch.bfloat16)
        weights = torch.full((sum(lengths), 4), 1 / math.sqrt(128), device=device)
        row_map = torch.cat(
            (
                torch.arange(rows, device=device),
                torch.arange(rows, device=device).clamp_max(2) + lengths[0],
            )
        )
        for token_budget in (8, 2048):
            for zero_scores in (False, True):
                with self.subTest(token_budget=token_budget, zero_scores=zero_scores):
                    row_weights = torch.zeros_like(weights) if zero_scores else weights
                    logits = qsa_paged_indexer_score(
                        q[row_map].view(2, rows, 4, 128),
                        row_weights[row_map],
                        pool.flatten(0, 1),
                        table,
                        visible // 4,
                        block_size=block_size,
                        max_ctx_len=columns,
                    )
                    expected = select_qsa_paged_tokens(
                        logits,
                        visible.flatten(),
                        compress_ratio=4,
                        token_budget=token_budget,
                    )
                    expected = torch.cat(
                        (expected[: lengths[0]], expected[rows : rows + lengths[1]])
                    )
                    plan = SimpleNamespace(
                        lengths=lengths,
                        max_ctx_len=columns,
                        ratio=4,
                        token_budget=token_budget,
                        kv_pool=pool,
                        kv_table=table,
                        compressed_lengths=visible // 4,
                        visible_lengths=visible,
                        kv_entries_per_block=block_size,
                    )
                    budget = 16 * columns * 4
                    calls = []

                    def score(*args, **kwargs):
                        queries = args[0]
                        calls.append(tuple(queries.shape))
                        self.assertEqual(queries.shape[0], 1)
                        self.assertLessEqual(
                            queries.shape[0] * queries.shape[1] * columns * 4, budget
                        )
                        return qsa_paged_indexer_score(*args, **kwargs)

                    with (
                        patch.object(qsa_prefill, "_MAX_SCORE_WORKSPACE_BYTES", budget),
                        patch(
                            "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score.qsa_paged_indexer_score",
                            side_effect=score,
                        ),
                    ):
                        actual = qsa_prefill._select_prefix_prefill_chunks(
                            q, row_weights, plan
                        )
                    self.assertGreater(len(calls), 2)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @skipUnless(torch.cuda.is_available(), "CUDA is required for paged scoring")
    def test_long_prefix_score_keeps_one_mma_output_allocation(self):
        from rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score import (
            qsa_paged_indexer_score,
        )

        device = torch.device("cuda")
        columns = 65536
        q = torch.ones(1, 32, 4, 128, device=device, dtype=torch.bfloat16)
        weights = torch.ones(32, 4, device=device)
        pool = torch.ones(2 * 32, 128, device=device, dtype=torch.bfloat16)
        table = torch.ones(1, columns // 32, device=device, dtype=torch.int32)
        contexts = torch.full((1, 32), columns, device=device, dtype=torch.int32)
        qsa_paged_indexer_score(
            q, weights, pool, table, contexts, block_size=32, max_ctx_len=columns
        )
        with patch("torch.empty", wraps=torch.empty) as allocate:
            logits = qsa_paged_indexer_score(
                q, weights, pool, table, contexts, block_size=32, max_ctx_len=columns
            )
        allocations = [
            call
            for call in allocate.call_args_list
            if call.args and call.args[0] == (32, columns)
        ]
        self.assertEqual(len(allocations), 1)
        torch.testing.assert_close(logits, torch.full_like(logits, 512), rtol=0, atol=0)

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
            return torch.where(
                torch.arange(max_ctx_len, device=device)[None, :]
                < lengths.reshape(-1, 1),
                0.0,
                -float("inf"),
            )

        with patch(
            "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score."
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
            "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score."
            "qsa_paged_indexer_score",
            side_effect=lambda q, weight, pool, table, lengths, **kwargs: torch.where(
                torch.arange(kwargs["max_ctx_len"], device=device)[None, :]
                < lengths.reshape(-1, 1),
                0.0,
                -float("inf"),
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
            "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score."
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
            "rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score."
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
            ("target-verify", RuntimeError, dict(target_verify=True)),
            ("prefix reuse", RuntimeError, dict(prefix_length=1)),
            ("text-only MRoPE", ValueError, dict(position_ids=[[7, 8, 7]])),
        )
        for expected, error_type, overrides in cases:
            values = dict(
                is_prefill=False,
                input_length=7,
                sequence_length=7,
                position_ids=[[7, 7, 7]],
            )
            values.update(overrides)
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(error_type, expected):
                    self._single_request_context(
                        self.kv_base, self.state_base, **values
                    ).select_decode_tokens(
                        q, raw, indexer=indexer, rope_config=self.rope_config
                    )


if __name__ == "__main__":
    main()
