from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import INDEXER_KV_TAG
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_cache_views import (
    _same_tensor_metadata,
    _validate_required_blocks,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
    select_qsa_paged_tokens,
)
from rtp_llm.models_py.model_desc.test.qwen4_exp.qsa_fixtures import QSARuntimeFixture
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    apply_partial_rope,
    build_interleaved_mrope,
)
from rtp_llm.ops import RopeStyle
from rtp_llm.ops.compute_ops import PyCacheStoreInputs


class Qwen4ExpQSACacheTest(QSARuntimeFixture, TestCase):
    def test_metadata_alias_fast_path_preserves_distinct_tensor_validation(self):
        values = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        alias = values.view_as(values)
        with patch.object(torch, "equal", wraps=torch.equal) as equal:
            self.assertTrue(_same_tensor_metadata(values, alias))
            equal.assert_not_called()

            self.assertTrue(_same_tensor_metadata(values, values.clone()))
            self.assertFalse(_same_tensor_metadata(values, values.T))
            self.assertFalse(_same_tensor_metadata(values, values + 1))
            self.assertEqual(equal.call_count, 3)

    def test_required_block_validation_preserves_error_cases(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                table = torch.tensor([[1, 2]], dtype=torch.int32, device=device)
                required = torch.tensor([2], dtype=torch.int32, device=device)
                _validate_required_blocks(
                    table, required, pool_blocks=3, tag=INDEXER_KV_TAG
                )
                _validate_required_blocks(
                    table[:, :0],
                    torch.zeros_like(required),
                    pool_blocks=3,
                    tag=INDEXER_KV_TAG,
                )
                for count, message in (
                    (-1, "negative required-column count"),
                    (3, "does not cover"),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        _validate_required_blocks(
                            table,
                            torch.tensor([count], dtype=torch.int32, device=device),
                            pool_blocks=3,
                            tag=INDEXER_KV_TAG,
                        )
                with self.assertRaisesRegex(RuntimeError, "unallocated"):
                    _validate_required_blocks(
                        torch.tensor([[1, 0]], dtype=torch.int32, device=device),
                        required,
                        pool_blocks=3,
                        tag=INDEXER_KV_TAG,
                    )
                with self.assertRaisesRegex(RuntimeError, "out-of-range physical"):
                    _validate_required_blocks(
                        torch.tensor([[1, 3]], dtype=torch.int32, device=device),
                        torch.tensor([1], dtype=torch.int32, device=device),
                        pool_blocks=3,
                        tag=INDEXER_KV_TAG,
                    )

    def test_token_selection_rejects_invalid_lengths(self):
        logits = torch.zeros(1, 1)
        for length, message in (
            (-1, "non-negative"),
            (8, "do not cover every completed block"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                select_qsa_paged_tokens(
                    logits,
                    torch.tensor([length], dtype=torch.int32),
                    compress_ratio=4,
                    token_budget=4,
                )

    @skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_required_block_validation_handles_strides_and_fallback(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.qsa_validation_triton import (
            is_supported,
            required_blocks_are_valid,
        )

        table = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.int32, device="cuda").T
        required = torch.tensor([2, 0, 2, 0, 2, 0], dtype=torch.int32, device="cuda")[
            ::2
        ]
        self.assertTrue(is_supported(table, required))
        self.assertTrue(required_blocks_are_valid(table, required, pool_blocks=4))
        _validate_required_blocks(table, required, pool_blocks=4, tag=INDEXER_KV_TAG)
        table[1, 1] = 4
        self.assertFalse(required_blocks_are_valid(table, required, pool_blocks=4))
        with self.assertRaisesRegex(RuntimeError, "out-of-range"):
            _validate_required_blocks(
                table, required, pool_blocks=4, tag=INDEXER_KV_TAG
            )

        wide_table = torch.ones(1, 4097, dtype=torch.int32, device="cuda")
        self.assertFalse(is_supported(wide_table, required[:1]))
        _validate_required_blocks(
            wide_table,
            required[:1],
            pool_blocks=4,
            tag=INDEXER_KV_TAG,
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
                    "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context."
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
            "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context.write_indexer_cache"
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

    def test_ordinary_prefill_still_rejects_nonzero_prefix_reuse(self):
        context = self._draft_incremental_context(
            self.kv_base,
            self.state_base,
            prefixes=[7],
            lengths=[1],
            is_mtp_draft=False,
        )
        with patch(
            "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context.write_indexer_cache"
        ) as writer:
            with self.assertRaisesRegex(RuntimeError, "prefix reuse"):
                context.write_prefill_indexer_cache(
                    torch.randn(1, self.D, dtype=torch.bfloat16),
                    indexer=self.indexer,
                    rope_config=self._base_rope_config(3),
                )
        writer.assert_not_called()

    @skipUnless(torch.cuda.is_available(), "CUDA is required for paged metadata")
    def test_mtp_prefix_reuse_uses_cache_positions_for_shifted_native_rows(self):
        from rtp_llm.models_py.model_desc.qwen4_exp_mtp import (
            normalize_mtp_prefill_positions,
        )
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import build_base_rope

        device = torch.device("cuda", torch.cuda.current_device())
        indexer = self._decode_indexer()
        indexer.k_norm_gamma = indexer.k_norm_gamma.to(device)
        config = self._base_rope_config(3)
        for length, published in (
            (length, published)
            for length in (1, 3, 4, 8)
            for published in (False, True)
        ):
            with self.subTest(length=length, published=published):
                context = self._draft_incremental_context(
                    torch.zeros(16, 2 * self.D * 2, dtype=torch.uint8, device=device),
                    torch.zeros(
                        16, 2 * self.RATIO * self.D, dtype=torch.float32, device=device
                    ),
                    prefixes=[8],
                    lengths=[length],
                    is_mtp_draft=True,
                    position_overrides=list(range(9, 8 + length)) + [8 + length - 1],
                )
                shifted = context.main_inputs.combo_position_ids.clone()
                groups = {
                    "full": context.main_inputs,
                    "indexer_kv": context.indexer_kv_inputs,
                    "indexer_state": context.indexer_state_inputs,
                }
                publication = PyCacheStoreInputs()
                shared_positions = shifted.clone()
                for value in groups.values():
                    value.combo_position_ids = shared_positions
                    value.prefix_lengths_device = torch.tensor(
                        [8], dtype=torch.int32, device=device
                    )
                    value.cache_store_inputs = publication if published else None
                    value.cache_store_writer = SimpleNamespace(write=lambda *_: None)
                inputs = SimpleNamespace(
                    input_ids=torch.zeros(length, dtype=torch.int32, device=device),
                    combo_position_ids=context.main_inputs.combo_position_ids,
                    attention_inputs=groups,
                )
                normalize_mtp_prefill_positions(inputs, 3)
                plan = context._draft_incremental_prefill_geometry(
                    indexer=indexer,
                    token_count=length,
                    device=device,
                    rope_config=config,
                    draft=False,
                    draft_prefix_reuse=True,
                )
                positions = torch.arange(
                    8, 8 + length, dtype=torch.int32, device=device
                )
                expected_cos, expected_sin = build_base_rope(
                    positions[:, None].expand(-1, 3).reshape(-1),
                    config,
                    token_count=length,
                    dtype=torch.bfloat16,
                    device=device,
                )
                torch.testing.assert_close(plan.current_cos, expected_cos)
                torch.testing.assert_close(plan.current_sin, expected_sin)
                if length == 3:
                    for value in groups.values():
                        value.combo_position_ids.copy_(shifted)
                        value.cache_store_inputs = None
                    for rope, draft_prefix_reuse in (
                        (config, False),
                        (self.rope_config, True),
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "positions must equal the logical cache positions",
                        ):
                            context._draft_incremental_prefill_geometry(
                                indexer=indexer,
                                token_count=length,
                                device=device,
                                rope_config=rope,
                                draft=False,
                                draft_prefix_reuse=draft_prefix_reuse,
                            )

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
            "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score."
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

    def test_decode_base_rope_factor_one_and_three_rotate_q_identically(self):
        q = torch.randn(1, 2, self.D, dtype=torch.bfloat16)
        raw = torch.randn(1, self.D, dtype=torch.bfloat16)
        rotated_queries = []

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
            rotated_queries.append(q.clone())
            return torch.zeros((1, 1), dtype=torch.float32)

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score."
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
            "rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context.write_indexer_cache"
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

        # The runtime has already checked both length and table capacity. Its
        # decode path must avoid a per-layer device-to-host scalar read here.
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("sync")):
            prevalidated = select_qsa_paged_tokens(
                block_logits,
                token_lengths,
                compress_ratio=self.RATIO,
                token_budget=8,
                validate_lengths=False,
            )
        torch.testing.assert_close(prevalidated, expected)


if __name__ == "__main__":
    main()
