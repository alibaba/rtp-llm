import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.model_desc.generic_moe_mtp import (
    _MTP_INDEXER_ROLE_NORMAL,
    _MTP_INDEXER_ROLE_REUSE,
    _MTP_INDEXER_ROLE_SEED,
    GenericMoeMTPModel,
    _mtp_indexer_share_active,
    _mtp_indexer_share_enabled,
)
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention


class _SparseFmha:
    def is_sparse(self):
        return True


class MtpIndexerShareTest(unittest.TestCase):
    def test_feature_requires_explicit_environment_opt_in(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(_mtp_indexer_share_enabled())
        for value, expected in (("0", False), ("true", False), ("1", True)):
            with (
                self.subTest(value=value),
                patch.dict(
                    "os.environ",
                    {"RTP_LLM_ENABLE_MTP_INDEXER_SHARE": value},
                    clear=True,
                ),
            ):
                self.assertEqual(_mtp_indexer_share_enabled(), expected)

    def test_feature_also_requires_checkpoint_capability(self):
        with patch.dict(
            "os.environ", {"RTP_LLM_ENABLE_MTP_INDEXER_SHARE": "1"}, clear=True
        ):
            unsupported = SimpleNamespace(index_share_for_mtp_iteration=False)
            supported = SimpleNamespace(index_share_for_mtp_iteration=True)
            no_cp = SimpleNamespace(
                prefill_cp_config=SimpleNamespace(is_enabled=lambda: False)
            )
            with_cp = SimpleNamespace(
                prefill_cp_config=SimpleNamespace(is_enabled=lambda: True)
            )
            self.assertFalse(_mtp_indexer_share_active(unsupported, no_cp, 1, 4))
            self.assertTrue(_mtp_indexer_share_active(supported, no_cp, 1, 4))
            self.assertFalse(_mtp_indexer_share_active(supported, no_cp, 2, 4))
            self.assertFalse(_mtp_indexer_share_active(supported, no_cp, 1, 0))
            self.assertFalse(_mtp_indexer_share_active(supported, with_cp, 1, 4))

    def test_cpu_world_control_group_is_created_lazily_and_reused(self):
        old_config = collective_torch._parallelism_config
        old_group = collective_torch._cpu_world_group
        fake_group = object()
        try:
            collective_torch._parallelism_config = SimpleNamespace(world_size=2)
            collective_torch._cpu_world_group = None
            with (
                patch.dict(
                    "os.environ",
                    {"RTP_LLM_ENABLE_MTP_INDEXER_SHARE": "1"},
                    clear=True,
                ),
                patch.object(torch.distributed, "is_gloo_available", return_value=True),
                patch.object(
                    torch.distributed, "new_group", return_value=fake_group
                ) as new_group,
            ):
                self.assertIs(
                    collective_torch._get_or_create_mtp_indexer_cpu_world_group(),
                    fake_group,
                )
                self.assertIs(
                    collective_torch._get_or_create_mtp_indexer_cpu_world_group(),
                    fake_group,
                )
                new_group.assert_called_once()
        finally:
            collective_torch._parallelism_config = old_config
            collective_torch._cpu_world_group = old_group

    def _model(self, enabled=True, topk=4, capacity=3, compress_ratio=1):
        model = object.__new__(GenericMoeMTPModel)
        model._mtp_indexer_share_enabled = enabled
        model._mtp_indexer_role = _MTP_INDEXER_ROLE_NORMAL
        model._mtp_shared_topk_indices = torch.zeros(
            (capacity, topk), dtype=torch.int32
        )
        model.config = SimpleNamespace(
            attn_config=SimpleNamespace(
                indexer_topk=topk, indexer_compress_ratio=compress_ratio
            )
        )
        return model

    def _glm53_model(self, capacity=3):
        model = self._model(topk=4, capacity=capacity, compress_ratio=4)
        model.config.is_glm53_mtp = True
        model._mtp_shared_topk_indices = torch.zeros((capacity, 19), dtype=torch.int32)
        return model

    def test_glm53_freezes_first_step_raw_indices_across_kpool_boundary(self):
        model = self._glm53_model(capacity=2)
        # At raw lengths 11 and 12, the first request has an incomplete tail;
        # the second has three complete pools. Select different query rows.
        pooled = torch.tensor(
            [[0, 1, -1, -1], [2, 0, 1, -1], [1, 0, -1, -1]], dtype=torch.int32
        )
        model._store_mtp_topk_indices(
            pooled,
            torch.tensor([2, 1], dtype=torch.int32),
            torch.tensor([5, 11, 10], dtype=torch.int32),
        )
        expected = torch.tensor(
            [
                [4, 5, 6, 7, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10],
                [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(model.snapshot_mtp_indexer_topk(2), expected)
        for positions in ([11, 12], [12, 13], [15, 16]):
            actual = model._get_mtp_reuse_topk_indices(
                torch.zeros((2, 8)),
                SimpleNamespace(
                    fmha_params=SimpleNamespace(
                        positions_d=torch.tensor(positions, dtype=torch.int32)
                    )
                ),
            )
            torch.testing.assert_close(actual, expected)

    def test_glm53_sparse_attention_does_not_expand_frozen_seed_again(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            SparseMlaOp,
        )

        op = object.__new__(SparseMlaOp)
        op.indexer_group_size, op.indexer_top_k, op.top_k = 4, 4, 19
        seed = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
            dtype=torch.int32,
        )
        actual = op._prepare_local_topk_indices(
            seed, torch.tensor([8], dtype=torch.int32)
        )
        self.assertIs(actual, seed)

    def test_glm53_cp_seed_collects_each_requests_last_valid_query(self):
        from rtp_llm.models_py.distributed.sequence_parallel import token_shard_layout
        from rtp_llm.models_py.distributed.zigzag_token_layout import ZigzagTokenLayout

        q_lens = [3, 17, 33]
        expected = torch.tensor(
            [[2, 0, -1, -1], [19, 0, -1, -1], [52, 0, -1, -1]], dtype=torch.int32
        )
        contributions = []
        for rank in range(8):
            model = self._glm53_model()
            layout = ZigzagTokenLayout(
                q_lens,
                token_shard_layout(sum(q_lens), 8, rank),
                8,
                rank,
                torch.device("cpu"),
            )
            rows = {}
            for fragment in layout.fragments:
                if fragment.cp_rank == rank:
                    for offset in range(fragment.length):
                        rows[fragment.cp_offset + offset] = (
                            fragment.canonical_offset + offset
                        )
            local_ids = sorted(rows)
            topk = torch.tensor(
                [[rows[row], 0, -1, -1] for row in local_ids], dtype=torch.int32
            ).reshape(-1, 4)
            fmha = SimpleNamespace(
                glm53_cp_layout=layout,
                cp_params=SimpleNamespace(
                    total_local_ids=torch.tensor(local_ids, dtype=torch.int32)
                ),
            )

            def reduce_seed(tensor, group):
                contributions.append(tensor.clone())
                return expected

            with patch(
                "rtp_llm.models_py.model_desc.generic_moe_mtp.all_reduce",
                side_effect=reduce_seed,
            ):
                model._store_prefill_cp_seed(
                    topk,
                    fmha,
                    SimpleNamespace(
                        prefix_lengths=torch.tensor([512, 512, 512], dtype=torch.int32)
                    ),
                )
            torch.testing.assert_close(
                model.snapshot_mtp_indexer_topk(3),
                model._expand_mtp_seed(
                    expected, torch.tensor([515, 529, 545], dtype=torch.int32)
                ),
            )
        torch.testing.assert_close(
            torch.stack(contributions).sum(0).to(torch.int32), expected
        )

    def test_cp_position_zero_mask_uses_global_request_boundaries(self):
        embeddings = torch.arange(20).reshape(10, 2)
        result = GenericMoeMTPModel._mask_prefill_position_zero_embeddings(
            embeddings,
            SimpleNamespace(
                cu_seqlens=torch.tensor([0, 3, 7, 10]),
                prefix_lengths=torch.tensor([0, 128, 0]),
            ),
        )
        expected = embeddings.clone()
        expected[[0, 7]] = 0
        torch.testing.assert_close(result, expected)

    def test_role_is_gated_and_validated(self):
        model = self._model()
        model.set_mtp_indexer_role(_MTP_INDEXER_ROLE_SEED)
        self.assertEqual(model._mtp_indexer_role, _MTP_INDEXER_ROLE_SEED)
        model.set_mtp_indexer_role(_MTP_INDEXER_ROLE_REUSE)
        self.assertEqual(model._mtp_indexer_role, _MTP_INDEXER_ROLE_REUSE)

        disabled = self._model(enabled=False)
        disabled.set_mtp_indexer_role(_MTP_INDEXER_ROLE_REUSE)
        self.assertEqual(disabled._mtp_indexer_role, _MTP_INDEXER_ROLE_NORMAL)
        with self.assertRaisesRegex(ValueError, "invalid MTP indexer role"):
            model.set_mtp_indexer_role(3)

    def test_short_context_fills_one_new_causal_position_per_step(self):
        model = self._model(topk=4, capacity=2)
        model._mtp_shared_topk_indices.copy_(
            torch.tensor([[0, 1, -1, -1], [0, 1, 2, -1]], dtype=torch.int32)
        )
        hidden = torch.zeros((2, 8))

        first = model._get_mtp_reuse_topk_indices(
            hidden,
            SimpleNamespace(
                fmha_params=SimpleNamespace(
                    positions_d=torch.tensor([2, 3], dtype=torch.int32)
                )
            ),
        )
        self.assertTrue(
            torch.equal(
                first,
                torch.tensor([[2, 0, 1, -1], [3, 0, 1, 2]], dtype=torch.int32),
            )
        )

        second = model._get_mtp_reuse_topk_indices(
            hidden,
            SimpleNamespace(
                fmha_params=SimpleNamespace(
                    positions_d=torch.tensor([3, 4], dtype=torch.int32)
                )
            ),
        )
        self.assertTrue(
            torch.equal(
                second,
                torch.tensor([[3, 2, 0, 1], [4, 3, 0, 1]], dtype=torch.int32),
            )
        )

    def test_full_topk_keeps_new_causal_positions(self):
        model = self._model(topk=4, capacity=1)
        model._mtp_shared_topk_indices.copy_(
            torch.tensor([[8, 2, 5, 1]], dtype=torch.int32)
        )

        first = model._get_mtp_reuse_topk_indices(
            torch.zeros((1, 8)),
            SimpleNamespace(
                fmha_params=SimpleNamespace(
                    positions_d=torch.tensor([9], dtype=torch.int32)
                )
            ),
        )
        self.assertTrue(
            torch.equal(first, torch.tensor([[9, 8, 2, 5]], dtype=torch.int32))
        )

        second = model._get_mtp_reuse_topk_indices(
            torch.zeros((1, 8)),
            SimpleNamespace(
                fmha_params=SimpleNamespace(
                    positions_d=torch.tensor([10], dtype=torch.int32)
                )
            ),
        )
        self.assertTrue(
            torch.equal(second, torch.tensor([[10, 9, 8, 2]], dtype=torch.int32))
        )
        second_snapshot = second.clone()

        duplicate = model._get_mtp_reuse_topk_indices(
            torch.zeros((1, 8)),
            SimpleNamespace(
                fmha_params=SimpleNamespace(
                    positions_d=torch.tensor([10], dtype=torch.int32)
                )
            ),
        )
        self.assertTrue(torch.equal(duplicate, second_snapshot))

    def test_compressed_reuse_updates_only_at_four_token_boundary(self):
        model = self._model(topk=4, capacity=2, compress_ratio=4)
        model._mtp_shared_topk_indices.copy_(
            torch.tensor([[7, 3, 1, -1], [6, 2, 0, -1]], dtype=torch.int32)
        )
        hidden = torch.zeros((2, 8))

        unchanged = model._get_mtp_reuse_topk_indices(
            hidden,
            SimpleNamespace(
                fmha_params=SimpleNamespace(
                    positions_d=torch.tensor([32, 34], dtype=torch.int32)
                )
            ),
        )
        self.assertTrue(
            torch.equal(
                unchanged,
                torch.tensor([[7, 3, 1, -1], [6, 2, 0, -1]], dtype=torch.int32),
            )
        )

        boundary = model._get_mtp_reuse_topk_indices(
            hidden,
            SimpleNamespace(
                fmha_params=SimpleNamespace(
                    positions_d=torch.tensor([35, 35], dtype=torch.int32)
                )
            ),
        )
        self.assertTrue(
            torch.equal(
                boundary,
                torch.tensor([[8, 7, 3, 1], [8, 6, 2, 0]], dtype=torch.int32),
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_causal_position_update_is_cuda_graph_safe(self):
        model = self._model(topk=4, capacity=1)
        model._mtp_shared_topk_indices = torch.tensor(
            [[8, 2, 5, 1]], dtype=torch.int32, device="cuda"
        )
        hidden = torch.zeros((1, 8), device="cuda")
        positions = torch.tensor([8], dtype=torch.int32, device="cuda")
        fmha_impl = SimpleNamespace(fmha_params=SimpleNamespace(positions_d=positions))

        # Warm lazy operator/allocator state without changing the seed because
        # position 8 is already present.
        model._get_mtp_reuse_topk_indices(hidden, fmha_impl)
        torch.cuda.synchronize()

        positions.fill_(9)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = model._get_mtp_reuse_topk_indices(hidden, fmha_impl)

        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(actual.cpu(), torch.tensor([[9, 8, 2, 5]], dtype=torch.int32))
        )

        positions.fill_(10)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(actual.cpu(), torch.tensor([[10, 9, 8, 2]], dtype=torch.int32))
        )

    def test_reuse_buffer_validation(self):
        model = self._model(capacity=1)
        with self.assertRaisesRegex(RuntimeError, "batch exceeds fixed buffer"):
            model._get_mtp_reuse_topk_indices(
                torch.zeros((2, 8)),
                SimpleNamespace(
                    fmha_params=SimpleNamespace(
                        positions_d=torch.tensor([1, 1], dtype=torch.int32)
                    )
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "requires device positions_d"):
            model._get_mtp_reuse_topk_indices(
                torch.zeros((1, 8)),
                SimpleNamespace(
                    fmha_params=SimpleNamespace(
                        positions_d=torch.tensor([1, 2], dtype=torch.int32)
                    )
                ),
            )

    def test_compute_output_validation_and_store(self):
        model = self._model(capacity=2)
        computed = torch.tensor(
            [[3, 1, 0, -1], [9, 8, 7, 6], [7, 2, 4, 1]], dtype=torch.int32
        )
        seed_rows = torch.tensor([2, 0], dtype=torch.int32)
        model._store_mtp_topk_indices(computed, seed_rows)
        self.assertTrue(torch.equal(model._mtp_shared_topk_indices, computed[[2, 0]]))
        with self.assertRaisesRegex(RuntimeError, "invalid MTP indexer share output"):
            model._store_mtp_topk_indices(computed.to(torch.int64), seed_rows)

    def test_request_order_is_restored_into_batch_staging(self):
        model = self._model(capacity=3)
        request_a = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
        request_b = torch.tensor([[8, 7, 6, 5]], dtype=torch.int32)
        model.load_mtp_indexer_topk(torch.cat([request_b, request_a], dim=0))
        actual = model.snapshot_mtp_indexer_topk(2)
        self.assertTrue(torch.equal(actual, torch.cat([request_b, request_a], dim=0)))

    def test_forced_reuse_skips_indexer(self):
        attention = object.__new__(MlaAttention)
        attention.reuse_topk_indices = False
        attention.layer_idx = 0
        attention.indexer = lambda *args, **kwargs: self.fail("indexer was called")
        shared = torch.tensor([[4, 1, 7, 2]], dtype=torch.int32)
        actual = attention._run_sparse_indexer(
            hidden_states=torch.zeros((1, 8)),
            q_c=None,
            q_view=torch.zeros((1, 1, 8)),
            kv_cache=None,
            fmha_impl=_SparseFmha(),
            prev_topk_indices=shared,
            force_reuse_topk_indices=True,
        )
        self.assertIs(actual, shared)

    def test_forced_reuse_requires_indices(self):
        attention = object.__new__(MlaAttention)
        attention.reuse_topk_indices = False
        attention.layer_idx = 0
        with self.assertRaisesRegex(RuntimeError, "needs previous top-k"):
            attention._run_sparse_indexer(
                hidden_states=torch.zeros((1, 8)),
                q_c=None,
                q_view=torch.zeros((1, 1, 8)),
                kv_cache=None,
                fmha_impl=_SparseFmha(),
                force_reuse_topk_indices=True,
            )


if __name__ == "__main__":
    unittest.main()
