"""CPU layout/budget contracts; mocked transports do not prove NCCL correctness."""

import copy
import threading
import traceback
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    dsa_kv_chunk_execution as planning,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_page_rr_impl as layout,
)
from rtp_llm.models_py.modules.kimi_k3 import dsa_model_adapter as adapter
from rtp_llm.models_py.distributed import collective_torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import (
    DsaWorkspaceOwner,
)


class FlashMlaSparsePageRRLayoutTest(unittest.TestCase):
    def setUp(self):
        def agree(contracts, contract, group):
            contracts[:] = [contract] * len(contracts)

        self.vote_patch = patch.object(
            torch.distributed, "all_gather_object", side_effect=agree
        )
        self.vote = self.vote_patch.start()
        self.addCleanup(self.vote_patch.stop)
        for name in ("current_device", "get_device_properties", "current_stream"):
            guard = patch.object(
                torch.cuda, name, side_effect=AssertionError("CPU test")
            )
            guard.start()
            self.addCleanup(guard.stop)
        profiler = patch.object(
            layout, "profile_scope", side_effect=lambda _: nullcontext()
        )
        profiler.start()
        self.addCleanup(profiler.stop)

    @staticmethod
    def markers(tokens, heads, features):
        # Every coordinate is exactly representable in BF16 for these fixtures.
        value = torch.empty(tokens, heads, features, dtype=torch.bfloat16)
        value[..., 0] = torch.arange(tokens)[:, None]
        value[..., 1] = torch.arange(heads)[None, :]
        value[..., 2:] = (
            torch.arange(tokens)[:, None, None] + torch.arange(heads)[None, :, None]
        )
        return value

    def fixture(self, shards, rank, tokens=16):
        heads, query_dim, value_dim = 96, 4, 3
        rows, local_heads = tokens // shards, heads // shards
        full_q = self.markers(tokens, heads, query_dim)
        source = full_q[:, rank * local_heads : (rank + 1) * local_heads].contiguous()
        wire_q = full_q.transpose(0, 1).contiguous()

        def gather_query(value):
            return wire_q

        comm = SimpleNamespace(
            size=shards,
            rank=rank,
            heads=heads,
            local_heads=local_heads,
            q_dim=query_dim,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            group=object(),
            query_gather=Mock(side_effect=gather_query),
            combine=Mock(side_effect=AssertionError("Prefill must not combine LSE")),
        )
        op = layout.PageRRSparseMlaOp.__new__(layout.PageRRSparseMlaOp)
        op.communicator = comm
        op.num_heads, op.kv_lora_rank, op.qk_rope_head_dim = local_heads, 2, 2
        op.kernel_heads, op.top_k = 128, 2
        op.workspace_owner = SimpleNamespace(
            heads=heads, value_dim=value_dim, sfa_use_fp8=False
        )
        indices = torch.arange(rank * rows, (rank + 1) * rows, dtype=torch.int64)
        # Two requests (3 and 10 rows) with distinct prefixes, then SP tail padding.
        positions = torch.cat(
            (torch.arange(100, 103), torch.arange(17, 27), torch.full((3,), -1))
        )[:tokens]
        requests = torch.tensor([0] * 3 + [1] * 13)[:tokens]
        params = SimpleNamespace(
            is_prefill=True,
            query_cp=False,
            shard_size=shards,
            shard_rank=rank,
            positions_d=positions,
            query_indices=indices,
            query_positions=positions.index_select(0, indices),
            query_requests=requests.index_select(0, indices),
        )
        return op, params, source, full_q, wire_q

    def test_tp_head_to_query_and_back_ragged_padding_and_owned_storage(self):
        for shards in (1, 2, 8):
            for rank in range(shards):
                with self.subTest(shards=shards, rank=rank):
                    op, params, source, full_q, wire_q = self.fixture(shards, rank)
                    self.vote.reset_mock()
                    expected = full_q.index_select(0, params.query_indices)
                    with (
                        patch.object(
                            torch,
                            "arange",
                            side_effect=AssertionError("no index reconstruction"),
                        ),
                        patch.object(
                            torch,
                            "equal",
                            side_effect=AssertionError("no device-value check"),
                        ),
                    ):
                        op.plan(params, None)
                        actual = op.execute(source)
                    self.assertEqual(self.vote.call_count, int(shards > 1))
                    if shards > 1:
                        torch.testing.assert_close(
                            op.communicator.query_gather.call_args.args[0],
                            source.transpose(0, 1).contiguous(),
                            rtol=0,
                            atol=0,
                        )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(actual.is_contiguous())
                    self.assertEqual(
                        actual.untyped_storage().nbytes(),
                        actual.numel() * actual.element_size(),
                    )
                    if shards == 1:
                        # No redistribution: input BMM already owns this compact Q.
                        self.assertIs(actual, source)
                    else:
                        wire_q.fill_(-1)
                        source.fill_(-2)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(
                        op.communicator.query_gather.call_count, int(shards > 1)
                    )

                    full_o = self.markers(16, 96, 3)
                    local = full_o.index_select(0, params.query_indices)
                    h = 96 // shards
                    expected_o = full_o[:, rank * h : (rank + 1) * h].clone()
                    with patch.object(
                        layout, "all_gather", return_value=full_o
                    ) as gather:
                        restored = op._restore_prefill(local)
                    self.assertEqual(gather.call_count, int(shards > 1))
                    self.assertTrue(restored.is_contiguous())
                    self.assertEqual(
                        restored.untyped_storage().nbytes(),
                        restored.numel() * restored.element_size(),
                    )
                    full_o.fill_(-3)
                    local.fill_(-4)
                    torch.testing.assert_close(restored, expected_o, rtol=0, atol=0)
                    op.communicator.combine.assert_not_called()
                    self.assertEqual(self.vote.call_count, int(shards > 1))

    def test_single_row_restore_does_not_alias_large_gather_storage(self):
        for shards in (1, 2, 8):
            op, params, _, _, _ = self.fixture(shards, shards - 1, tokens=shards)
            op.plan(params, None)
            storage = torch.zeros(shards * 10, 96, 3, dtype=torch.bfloat16)
            gathered = storage[:shards]
            local = gathered[-1:].clone()
            with patch.object(layout, "all_gather", return_value=gathered):
                restored = op._restore_prefill(local)
            self.assertEqual(
                restored.untyped_storage().nbytes(), shards * (96 // shards) * 3 * 2
            )
            storage.fill_(1)
            local.fill_(2)
            self.assertEqual(restored.count_nonzero().item(), 0)

    def test_query_cp_identity_is_unchanged(self):
        op, params, _, full_q, _ = self.fixture(2, 0)
        params.query_cp = True
        op.communicator = layout.QueryCpLayout(96)
        self.vote.reset_mock()
        op.plan(params, None)
        self.assertIs(op.execute(full_q), full_q)
        self.assertIs(op._restore_prefill(full_q), full_q)
        self.vote.assert_not_called()

    def test_prefill_decode_prefill_selects_existing_decode_gather(self):
        op, params, source, full_q, _ = self.fixture(8, 3)
        op.plan(params, None)
        torch.testing.assert_close(op.execute(source), full_q[6:8], rtol=0, atol=0)
        decode = copy.copy(params)
        decode.is_prefill = False
        self.vote.reset_mock()
        op.plan(decode, None)
        self.assertIsNone(op._tp_prefill_shape)
        torch.testing.assert_close(op.execute(source), full_q, rtol=0, atol=0)
        self.vote.assert_not_called()
        op.plan(copy.copy(params), None)
        torch.testing.assert_close(op.execute(source), full_q[6:8], rtol=0, atol=0)
        self.assertEqual(self.vote.call_count, 1)

    def test_zero_rows_and_empty_history_restore(self):
        for shards in (1, 2, 8):
            for tokens in (0, 16):
                with self.subTest(shards=shards, tokens=tokens):
                    op, params, source, _, _ = self.fixture(shards, 0, tokens)
                    params.sfa_execution = SimpleNamespace(
                        blocks=[], begin=Mock(), finish=Mock()
                    )
                    self.vote.reset_mock()
                    op.plan(params, None)
                    restored_global = torch.zeros(tokens, 96, 3, dtype=torch.bfloat16)
                    with patch.object(
                        layout, "all_gather", return_value=restored_global
                    ) as gather:
                        result = op.forward(source, None, None)
                    self.assertEqual(result.shape, (tokens, 96 // shards, 3))
                    self.assertEqual(result.count_nonzero().item(), 0)
                    self.assertEqual(gather.call_count, int(shards > 1 and tokens > 0))
                    self.assertEqual(
                        op.communicator.query_gather.call_count,
                        int(shards > 1 and tokens > 0),
                    )
                    params.sfa_execution.finish.assert_called_once()
                    self.assertEqual(self.vote.call_count, int(shards > 1))

    def test_nonempty_history_uses_same_restore_without_workspace_alias(self):
        from rtp_llm.models_py.kernels.cuda import dsa_kv_chunk
        from rtp_llm.models_py.utils import smoke_evidence

        op, params, source, _, _ = self.fixture(8, 3)
        rows = params.query_indices.numel()
        partial = torch.ones(rows, 128, 2, dtype=torch.bfloat16)
        partial[..., 0] = params.query_indices[:, None]
        weights = torch.zeros(96, 2, 3, dtype=torch.bfloat16)
        weights[:, 0, 0] = weights[:, 0, 2] = 1
        weights[:, 1, 1] = weights[:, 1, 2] = torch.arange(96)
        op.workspace_owner.value_weights = {0: weights}
        op._attention = Mock(return_value=(partial, torch.zeros(rows, 128)))
        buffers = {}

        def workspace(name, shape, dtype):
            if name not in buffers:
                buffers[name] = torch.zeros(shape, dtype=dtype)
            return buffers[name]

        execution = SimpleNamespace(
            blocks=[SimpleNamespace(descriptor=object())],
            plan=SimpleNamespace(chunked=False),
            begin=Mock(return_value=workspace),
            finish=Mock(),
        )
        params.sfa_execution = execution
        params.kernel_page_size, params.is_warmup = 2, False
        params.block_table = torch.ones(2, 2, dtype=torch.int32)
        params.cache_adapter = SimpleNamespace(
            read_prefix_chunk=Mock(return_value=torch.zeros(2, 4))
        )
        op.plan(params, params.block_table)
        full_o = self.markers(16, 96, 3)

        def gather(output, group):
            torch.testing.assert_close(output, full_o[6:8], rtol=0, atol=0)
            return full_o

        with (
            patch.object(dsa_kv_chunk, "map_dsa_chunk_slots"),
            patch.object(layout, "check_sparse_lse"),
            patch.object(layout, "zero_sparse_empty_rows"),
            patch.object(smoke_evidence, "completed_dsa_chunks"),
            patch.object(layout, "all_gather", side_effect=gather) as output_gather,
        ):
            actual = op.forward(
                source, torch.zeros(2, 1, 4), torch.zeros(rows, 2, dtype=torch.int32)
            )
        expected = full_o[:, 36:48].clone()
        buffers["output"].fill_(-1)
        full_o.fill_(-2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        execution.finish.assert_called_once()
        op.communicator.combine.assert_not_called()
        op.communicator.query_gather.assert_called_once()
        output_gather.assert_called_once()
        self.assertEqual(self.vote.call_count, 1)

    def test_invalid_geometry_votes_before_payload_collectives(self):
        mutations = (
            lambda p, c: setattr(p, "positions_d", p.positions_d[:-1]),
            lambda p, c: setattr(p, "shard_size", 4),
            lambda p, c: setattr(p, "shard_rank", 7),
            lambda p, c: setattr(p, "query_indices", p.query_indices.reshape(1, -1)),
            lambda p, c: setattr(p, "query_indices", p.query_indices.float()),
            lambda p, c: setattr(p, "query_positions", p.query_positions[:0]),
            lambda p, c: setattr(p, "query_requests", p.query_requests[:0]),
            lambda p, c: setattr(p, "positions_d", p.positions_d.reshape(1, -1)),
        )
        for mutate in mutations:
            op, params, _, _, _ = self.fixture(8, 0)
            mutate(params, op.communicator)
            self.vote.reset_mock()
            with self.assertRaisesRegex(ValueError, "layout disagreement"):
                op.plan(params, None)
            self.assertEqual(self.vote.call_count, 1)
            op.communicator.query_gather.assert_not_called()
        op, params, _, _, _ = self.fixture(8, 0)

        def remote_error(contracts, contract, group):
            contracts[:] = [contract] * len(contracts)
            contracts[-1] = (contract[0], "peer has invalid query interval")

        self.vote.side_effect = remote_error
        with self.assertRaisesRegex(ValueError, "peer has invalid"):
            op.plan(params, None)
        op.communicator.query_gather.assert_not_called()

    def test_standalone_prepare_never_skips_agreement_on_pointer_cache_hit(self):
        op, params, _, _, _ = self.fixture(8, 0)
        inputs = SimpleNamespace(cu_seqlens=torch.tensor([0, 3, 16]))
        op.plan(params, None, inputs)
        self.assertEqual(self.vote.call_count, 1)
        op.workspace_owner._tp_prefill_batch_plan = (inputs.cu_seqlens, op._tp_prefill_shape)
        for field, value in (
            ("query_positions", params.query_positions[:0]),
            ("positions_d", torch.empty(24, dtype=torch.int32)),
            ("query_indices", params.query_indices.int()),
            ("shard_rank", 1),
        ):
            invalid = copy.copy(params)
            setattr(invalid, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "layout disagreement"
            ):
                op.plan(invalid, None, inputs)
            self.assertIsNone(op._tp_prefill_shape)
        self.assertEqual(self.vote.call_count, 5)
        op.communicator.query_gather.assert_not_called()
        op.plan(params, None, inputs)
        self.assertEqual(op._tp_prefill_shape, (16, 2, 96, 12, 4, 3))
        self.assertEqual(self.vote.call_count, 6)

    def test_sparse_lse_check_preserves_empty_sentinels_and_rejects_nan(self):
        layout.check_sparse_lse(torch.tensor([[0.0, float("inf"), -float("inf")]]))
        with self.assertRaisesRegex(RuntimeError, "NaN LSE"):
            layout.check_sparse_lse(torch.tensor([[float("nan")]]))

    def test_prepare_agrees_before_phase_plans_and_keeps_decode_order(self):
        for prefill, graph, phase, expected in (
            (True, False, None, ["indexer", "sfa"]),
            (True, False, "indexer", ["indexer"]),
            (True, False, "sfa", ["sfa"]),
            (False, True, None, ["sfa", "indexer"]),
            (False, False, None, ["indexer", "sfa"]),
        ):
            with self.subTest(prefill=prefill, graph=graph, phase=phase):
                calls = []
                params = SimpleNamespace(
                    is_prefill=prefill, is_cuda_graph=graph, block_table=object(),
                    prepare=Mock(side_effect=lambda *a, **k: calls.append("metadata")),
                    prepare_execution=Mock(side_effect=calls.append),
                )
                impl = SimpleNamespace(
                    page_rr_params=params, execution_phase=phase,
                    fmha_impl=SimpleNamespace(
                        plan=Mock(side_effect=lambda *a: calls.append("layout"))
                    ),
                )
                inputs = object()
                layout.SparseMlaPageRRImpl.prepare(impl, inputs, forbid_realloc=graph)
                self.assertEqual(calls, ["metadata", "layout", *expected])
                self.assertIs(impl.attn_inputs, inputs)
                params.prepare.assert_called_once_with(inputs, forbid_realloc=graph)

    def test_no_attention_inputs_replans_mutable_params_without_cache_access(self):
        op, params, _, _, _ = self.fixture(8, 0)
        self.vote.reset_mock()
        op.plan(params, None)
        self.assertEqual(op._tp_prefill_shape[:2], (16, 2))
        self.assertEqual(self.vote.call_count, 1)
        self.assertFalse(hasattr(op.workspace_owner, "_tp_prefill_batch_plan"))

        # Even a stale entry from the former params-identity fallback is ignored
        # and left untouched when this invocation has no valid batch token.
        stale = (params, op._tp_prefill_shape)
        op.workspace_owner._tp_prefill_batch_plan = stale
        params.positions_d = params.positions_d[:8]
        params.query_indices = params.query_indices[:1]
        params.query_positions = params.query_positions[:1]
        params.query_requests = params.query_requests[:1]
        op.plan(params, None)
        self.assertEqual(op._tp_prefill_shape[:2], (8, 1))
        self.assertEqual(self.vote.call_count, 2)
        self.assertIs(op.workspace_owner._tp_prefill_batch_plan, stale)
        op.communicator.query_gather.assert_not_called()

    def test_independent_empty_batch_tokens_do_not_share_null_pointer_cache(self):
        op, params, _, _, _ = self.fixture(8, 0)
        other, other_params, _, _, _ = self.fixture(8, 0, tokens=8)
        other.workspace_owner = op.workspace_owner
        other.communicator = op.communicator
        first = SimpleNamespace(
            cu_seqlens=torch.empty(0, dtype=torch.int32),
            cu_seqlens_device=torch.empty(0, dtype=torch.int32),
        )
        second = SimpleNamespace(
            cu_seqlens=torch.empty(0, dtype=torch.int32),
            cu_seqlens_device=torch.empty(0, dtype=torch.int32),
        )
        self.assertIsNot(first.cu_seqlens, second.cu_seqlens)
        self.assertIsNot(first.cu_seqlens_device, second.cu_seqlens_device)
        for inputs in (first, second):
            self.assertEqual(inputs.cu_seqlens.data_ptr(), 0)
            self.assertEqual(inputs.cu_seqlens_device.data_ptr(), 0)
        self.vote.reset_mock()
        op.plan(params, None, first)
        self.assertEqual(op._tp_prefill_shape[:2], (16, 2))
        self.assertFalse(hasattr(op.workspace_owner, "_tp_prefill_batch_plan"))
        stale = (first.cu_seqlens_device, op._tp_prefill_shape)
        op.workspace_owner._tp_prefill_batch_plan = stale
        other.plan(other_params, None, second)
        self.assertEqual(other._tp_prefill_shape[:2], (8, 1))
        self.assertEqual(self.vote.call_count, 2)
        self.assertIs(op.workspace_owner._tp_prefill_batch_plan, stale)
        op.communicator.query_gather.assert_not_called()

    def test_accepted_preparation_publication_has_no_per_layer_votes_or_index_reads(
        self,
    ):
        for device_inputs in (False, True):
            with self.subTest(device_inputs=device_inputs):
                op, params, source, _, _ = self.fixture(8, 0)
                indexer, indexer_params, _, _, _ = self.fixture(8, 0)
                indexer.workspace_owner = op.workspace_owner
                indexer.communicator = op.communicator
                cu = torch.tensor([0, 3, 16], dtype=torch.int32)
                next_cu = cu.clone()
                empty_host = torch.empty(0, dtype=torch.int32)
                full_inputs = SimpleNamespace(
                    cu_seqlens=empty_host if device_inputs else cu,
                    cu_seqlens_device=cu,
                )
                # C++ copies the same batch tensors into full/indexer inputs;
                # distinct Python views must still identify the same allocation.
                indexer_inputs = SimpleNamespace(
                    cu_seqlens=empty_host if device_inputs else cu.view(-1),
                    cu_seqlens_device=cu.view(-1),
                )
                next_inputs = SimpleNamespace(
                    cu_seqlens=empty_host if device_inputs else next_cu,
                    cu_seqlens_device=next_cu,
                )
                full_o = self.markers(16, 96, 3)
                local_o = full_o[:2].clone()

                self.vote.reset_mock()
                with (
                    patch.object(
                        torch, "arange", side_effect=AssertionError("no extra arange")
                    ),
                    patch.object(
                        torch,
                        "equal",
                        side_effect=AssertionError("no device-value read"),
                    ),
                    patch.object(
                        torch.Tensor,
                        "item",
                        side_effect=AssertionError("no scalar read"),
                    ),
                    patch.object(
                        torch.Tensor,
                        "index_select",
                        side_effect=AssertionError("use continuous slice"),
                    ),
                    patch.object(
                        layout, "all_gather", return_value=full_o
                    ) as output_gather,
                ):
                    op.plan(params, None, full_inputs)
                    indexer.plan(indexer_params, None, indexer_inputs, tp_prefill_shape=op._tp_prefill_shape)
                    self.assertEqual(self.vote.call_count, 1)
                    for _ in range(3):
                        local_q = op.execute(source)
                        restored = op._restore_prefill(local_o)
                    self.assertEqual(self.vote.call_count, 1)
                    self.assertEqual(op.communicator.query_gather.call_count, 3)
                    self.assertEqual(output_gather.call_count, 3)
                    # A fresh batch allocation must vote again, even for the same shape.
                    op.plan(params, None, next_inputs)
                    indexer.plan(indexer_params, None, next_inputs, tp_prefill_shape=op._tp_prefill_shape)
                    self.assertEqual(self.vote.call_count, 2)
                torch.testing.assert_close(
                    local_q, self.markers(16, 96, 4)[:2], rtol=0, atol=0
                )
                torch.testing.assert_close(restored, full_o[:, :12], rtol=0, atol=0)
                self.assertFalse(hasattr(op.workspace_owner, "_tp_prefill_batch_plan"))
                bad_cu = next_cu.clone()
                bad_inputs = SimpleNamespace(
                    cu_seqlens=empty_host if device_inputs else bad_cu,
                    cu_seqlens_device=bad_cu,
                )
                bad_params = copy.copy(params)
                bad_params.query_positions = params.query_positions[:0]
                with self.assertRaisesRegex(ValueError, "layout disagreement"):
                    op.plan(bad_params, None, bad_inputs)
                self.assertEqual(self.vote.call_count, 3)
                self.assertFalse(hasattr(op.workspace_owner, "_tp_prefill_batch_plan"))
                self.assertEqual(op.communicator.query_gather.call_count, 3)
                self.assertEqual(output_gather.call_count, 3)


class TpPrefillBatchAgreementTest(unittest.TestCase):
    setUp = FlashMlaSparsePageRRLayoutTest.setUp
    markers = staticmethod(FlashMlaSparsePageRRLayoutTest.markers)
    fixture = FlashMlaSparsePageRRLayoutTest.fixture

    def batch_fixture(self, size=2, tokens=16):
        models, batches = [], []
        for rank in range(size):
            model = adapter.KimiK31ModelAdapter.__new__(adapter.KimiK31ModelAdapter)
            model.prefill_query_cp = False
            model.parallelism_config = SimpleNamespace(
                tp_size=size, tp_rank=rank, get_attn_tp_size=lambda: size,
            )
            attention = SimpleNamespace(
                head_num=96 // size, kv_lora_rank=2, rope_head_dim=2,
                mla_fp8_compute=False,
            )
            model.config = SimpleNamespace(
                max_seq_len=256, getAttentionConfigs=Mock(return_value=attention),
            )
            model.weight, model.fmha_config = object(), object()
            model._query_layout_buffers, model._outer_layout_buffers = {}, {}
            model._natural_layout_cache, model._natural_validation_cache = {}, {}
            model._row_exchange = None
            model._get_fmha_group_tags = Mock(return_value=("default", "indexer_kv"))
            groups = {}
            cu = torch.tensor([0, tokens], dtype=torch.int32)
            for tag in ("default", "indexer_kv"):
                op, params, source, _, _ = self.fixture(size, rank, tokens)
                op.communicator.group = rank
                if not groups:
                    owner = op.workspace_owner
                    owner.device, owner.latent_dim, owner.suffix_dim = "cpu", 2, 2
                    model.dsa_workspace = owner
                op.workspace_owner = owner
                params.kv_lens_host = [tokens]
                params.page_size = params.kernel_page_size = 64
                params.block_table = torch.empty(1, 1, dtype=torch.int32)
                groups[tag] = SimpleNamespace(
                    tag=tag, op=op, params=params, source=source,
                    is_prefill=True, is_target_verify=False, is_mtp_draft_update=False,
                    input_lengths=torch.tensor([tokens], dtype=torch.int32),
                    prefix_lengths=torch.tensor([0], dtype=torch.int32),
                    cu_seqlens=cu, cu_seqlens_device=cu,
                    physical_token_count=tokens, logical_token_count=tokens,
                )
            models.append(model)
            batches.append(SimpleNamespace(attention_inputs=groups, input_ids=torch.empty(tokens)))
        return models, batches

    def run_batch(self, models, batches, *, execute=False):
        """A bounded two-sided CPU rendezvous, not a replicated-local mock vote."""
        size = len(models)
        condition, current = threading.Condition(), threading.local()
        receipts, events, results, errors = {}, [], [None] * size, [None] * size
        self.vote.reset_mock()

        def agree(outputs, value, group):
            rank = current.rank
            self.assertEqual(group, rank)
            with condition:
                self.assertNotIn(rank, receipts, "more than one batch vote on a rank")
                receipts[rank] = value
                events.append(("vote", rank))
                condition.notify_all()
                if not condition.wait_for(lambda: len(receipts) == size, timeout=5):
                    raise AssertionError("peer skipped the fixed batch agreement")
                outputs[:] = [receipts[i] for i in range(size)]

        def params_factory(*args, **kwargs):
            params = SimpleNamespace()
            def prepare(inputs):
                events.append(("metadata", current.rank, inputs.tag))
                if getattr(inputs, "prepare_error", False):
                    raise ValueError("rank-local metadata validation failure")
                params.__dict__.update(inputs.params.__dict__)
                params.inputs = inputs
            def prepare_execution(phase):
                events.append(("phase_plan", current.rank, params.inputs.tag, phase))
                if getattr(params.inputs, "plan_error", False):
                    raise ValueError("rank-local workspace capacity failure")
                wire_shape = (size, 1, 64, getattr(params.inputs, "wire_features", 4))
                wire_dtype = getattr(params.inputs, "wire_dtype", torch.bfloat16)
                return SimpleNamespace(
                    plan=SimpleNamespace(page_capacity=2),
                    shapes={"cache.gather": (wire_shape, wire_dtype)},
                )
            params.prepare, params.prepare_execution = prepare, prepare_execution
            return params

        real_preflight = layout.SparseMlaPageRRImpl.prepare_tp_prefill
        def construct(config, parallelism, weights, inputs, fmha, graph, **kwargs):
            rank = current.rank
            self.assertEqual(len(receipts), size, "constructor/scatter ran before agreement")
            self.assertTrue(all(value[1] is None for value in receipts.values()))
            # A constructor is deliberately treated as a payload boundary:
            # a future layout_scatter added there must still run after the vote.
            events.append(("layout_scatter", rank, inputs.tag))
            prepared = kwargs["prepared_tp_prefill"]
            self.assertIsNotNone(prepared)
            inputs.op.plan(
                prepared.params, prepared.params.block_table, inputs,
                tp_prefill_shape=prepared.shape,
            )
            return inputs.op

        factory = Mock(side_effect=construct)
        factory.prepare_tp_prefill = real_preflight
        self.vote.side_effect = agree
        def worker(rank):
            current.rank = rank
            def host_metadata(inputs):
                events.append(("host_metadata", rank))
                if getattr(models[rank], "host_error", False):
                    raise ValueError("rank-local host metadata failure")
            try:
                results[rank] = models[rank]._prepare_dsa(
                    batches[rank], False, prepare_host_metadata=host_metadata,
                )
                if execute:
                    for _ in range(3):
                        for tag, op in results[rank].items():
                            events.append(("query_gather", rank, tag))
                            op.execute(batches[rank].attention_inputs[tag].source)
            except Exception as exc:
                errors[rank] = exc

        with (
            patch.object(adapter, "get_attention_inputs_value", side_effect=lambda inputs: inputs.attention_inputs),
            patch.object(adapter, "get_primary_attention_inputs", side_effect=AssertionError("no primary before vote")),
            patch.object(collective_torch, "get_process_group", side_effect=lambda group: current.rank),
            patch.object(layout, "DsaPageRRParams", side_effect=params_factory),
            patch.object(layout, "SparseMlaPageRRImpl", factory),
            patch.object(torch.Tensor, "item", side_effect=AssertionError("no scalar synchronization")),
        ):
            threads = [threading.Thread(target=worker, args=(rank,), daemon=True) for rank in range(size)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
            self.assertFalse(any(thread.is_alive() for thread in threads), "CPU rendezvous hung")
        self.assertEqual(set(receipts), set(range(size)))
        self.assertEqual(self.vote.call_count, size)
        return results, errors, events

    def test_asymmetric_errors_and_tag_disagreement_reject_all_ranks_before_payload(self):
        for failure in (
            "cache_hit_geometry", "metadata", "workspace", "host_metadata",
            "empty_mapping", "empty_tags", "missing_tag", "tag_order",
            "cross_tag_geometry", "cross_rank_geometry", "mode",
            "transport_feature_width", "transport_wire_dtype",
        ):
            with self.subTest(failure=failure):
                models, batches = self.batch_fixture()
                # Hit on one rank, miss on the other, with the same batch token
                # as the malformed tag. Neither state may alter vote participation.
                models[0].dsa_workspace._tp_prefill_batch_plan = (
                    batches[0].attention_inputs["default"].cu_seqlens, (16, 8, 96, 48, 4, 3),
                )
                bad = batches[0].attention_inputs["indexer_kv"]
                if failure == "cache_hit_geometry":
                    bad.params.query_positions = bad.params.query_positions[:0]
                elif failure == "metadata":
                    bad.prepare_error = True
                elif failure == "workspace":
                    bad.plan_error = True
                elif failure == "host_metadata":
                    models[0].host_error = True
                elif failure == "empty_mapping":
                    batches[0].attention_inputs = {}
                elif failure == "empty_tags":
                    models[0]._get_fmha_group_tags.return_value = ()
                elif failure == "missing_tag":
                    del batches[0].attention_inputs["indexer_kv"]
                elif failure == "tag_order":
                    models[0]._get_fmha_group_tags.return_value = ("indexer_kv", "default")
                elif failure == "cross_tag_geometry":
                    bad.prefix_lengths.add_(1)
                elif failure == "cross_rank_geometry":
                    for group in batches[0].attention_inputs.values():
                        group.prefix_lengths.add_(1)
                elif failure == "transport_feature_width":
                    bad.wire_features = 656
                elif failure == "transport_wire_dtype":
                    bad.wire_dtype = torch.uint8
                elif failure == "mode":
                    for group in batches[0].attention_inputs.values():
                        group.is_prefill = False
                results, errors, events = self.run_batch(models, batches)
                self.assertEqual(results, [None, None])
                self.assertTrue(all(isinstance(error, ValueError) for error in errors), errors)
                self.assertEqual(str(errors[0]), str(errors[1]))
                self.assertIn("layout disagreement", str(errors[0]))
                self.assertFalse(any(event[0] in ("layout_scatter", "query_gather") for event in events))

    def test_fatal_and_unknown_errors_keep_identity_traceback_and_skip_agreement(self):
        for stage in ("host_metadata", "preflight"):
            for error_type, message in (
                (MemoryError, "host allocation failed"),
                (torch.cuda.OutOfMemoryError, "CUDA out of memory"),
                (RuntimeError, "CUDA error: an illegal memory access was encountered"),
                (RuntimeError, "unexpected native preparation failure"),
                (TypeError, "unexpected programming error"),
                (AssertionError, "unexpected invariant failure"),
            ):
                with self.subTest(stage=stage, error_type=error_type, message=message):
                    models, batches = self.batch_fixture()
                    failure = error_type(message)
                    # Seed a previous-batch hit as well: a fatal must escape this
                    # call, not be cached/stringified and blamed on a later batch.
                    sentinel = object()
                    models[0].dsa_workspace._tp_prefill_batch_plan = sentinel
                    def raise_original(*args, **kwargs):
                        raise failure
                    factory = Mock(side_effect=AssertionError("must not publish"))
                    factory.prepare_tp_prefill.side_effect = raise_original
                    with (
                        patch.object(adapter, "get_attention_inputs_value", side_effect=lambda inputs: inputs.attention_inputs),
                        patch.object(collective_torch, "get_process_group", side_effect=AssertionError("must not start fatal recovery vote")),
                        patch.object(layout, "agree_tp_prefill_batch") as agreement,
                        patch.object(layout, "SparseMlaPageRRImpl", factory),
                    ):
                        try:
                            models[0]._prepare_dsa(
                                batches[0], False,
                                prepare_host_metadata=raise_original if stage == "host_metadata" else None,
                            )
                        except error_type as caught:
                            self.assertIs(caught, failure)
                            self.assertEqual(str(caught), message)
                            self.assertIsNone(caught.__cause__)
                            self.assertEqual(traceback.extract_tb(caught.__traceback__)[-1].name, "raise_original")
                        else:
                            self.fail("fatal/unknown exception was swallowed")
                        agreement.assert_not_called()
                        factory.assert_not_called()
                    self.assertIs(models[0].dsa_workspace._tp_prefill_batch_plan, sentinel)
                    self.vote.assert_not_called()

    def test_one_vote_per_batch_despite_asymmetric_cache_state_and_layer_reuse(self):
        for size, tokens in ((2, 16), (8, 16), (2, 0)):
            with self.subTest(size=size, tokens=tokens):
                models, batches = self.batch_fixture(size, tokens)
                models[0].dsa_workspace._tp_prefill_batch_plan = object()
                # Reuse the very same cu-seqlens allocation in a second batch.
                # Agreement is governed by the explicit call, not pointer lifetime.
                for _ in range(2):
                    results, errors, events = self.run_batch(models, batches, execute=True)
                    self.assertEqual(errors, [None] * size)
                    for rank in range(size):
                        local = [event[0] for event in events if event[1] == rank]
                        self.assertEqual(local.count("vote"), 1)
                        self.assertLess(local.index("vote"), local.index("layout_scatter"))
                        self.assertEqual(local.count("metadata"), 2)
                        self.assertEqual(local.count("phase_plan"), 2)
                        self.assertEqual(local.count("query_gather"), 6)
                        self.assertEqual(set(results[rank]), {"default", "indexer_kv"})

    def test_accepted_constructor_reuses_params_and_does_not_prepare_twice(self):
        op, params, _, _, _ = self.fixture(2, 0)
        params.block_table = object()
        shape = (16, 8, 96, 48, 4, 3)
        prepared = layout.DsaTpPrefillPreparation(params, shape, ())
        impl = SimpleNamespace(
            page_rr_params=params, _prepared_tp_prefill=prepared,
            fmha_impl=op, prepare=Mock(side_effect=AssertionError("already prepared")),
        )
        inputs = object()
        layout.SparseMlaPageRRImpl.create_params(impl, inputs)
        self.assertIs(impl.fmha_params, params)
        self.assertIs(impl.cp_params, params)
        self.assertIsNone(impl._prepared_tp_prefill)
        self.assertEqual(op._tp_prefill_shape, shape)
        self.vote.assert_not_called()


class TpPrefillBudgetTest(unittest.TestCase):
    def test_full_backing_peaks_not_local_views(self):
        for shards in (1, 2, 8):
            for tokens in (0, 16, 65536):
                for value_dim in (128, 4096):
                    rows, heads, query_dim = tokens // shards, 96, 576
                    source = tokens * (heads // shards) * query_dim * 2
                    compact = rows * heads * query_dim * 2
                    padded = rows * 128 * query_dim * 2
                    local_o = rows * heads * value_dim * 2
                    restored = tokens * (heads // shards) * value_dim * 2
                    gather_peak = (
                        source + tokens * heads * query_dim * 2 + max(source, compact)
                        if shards > 1
                        else source + compact
                    )
                    restore_peak = source + padded + local_o + restored
                    if shards > 1:
                        restore_peak += tokens * heads * value_dim * 2
                    previous = rows * (3 * heads + 128) * query_dim * 2 + local_o * 2
                    expected = max(
                        0,
                        max(gather_peak, source + compact + padded, restore_peak)
                        - previous,
                    )
                    self.assertEqual(
                        planning._tp_prefill_layout_extra_bytes(
                            tokens, rows, heads, shards, query_dim, value_dim
                        ),
                        expected,
                    )

    def fixture(
        self, *, prefill=True, query_cp=False, tokens=65536, shards=8, **owner_options
    ):
        options = dict(
            device="cpu",
            tp_size=shards,
            indexer_use_fp8=False,
            sfa_budget_bytes=1,
            indexer_budget_bytes=1,
        )
        options.update(owner_options)
        owner = DsaWorkspaceOwner(**options)
        owner.prefill_query_cp = query_cp
        rows = tokens // shards if prefill and not query_cp else tokens
        params = SimpleNamespace(
            workspace_owner=owner,
            is_prefill=prefill,
            query_cp=query_cp,
            is_cuda_graph=False,
            positions_d=torch.empty(tokens, dtype=torch.int32),
            query_positions=torch.empty(rows, dtype=torch.int32),
            shard_size=shards,
            shard_rank=0,
            page_size=64,
            kernel_page_size=64,
            block_table=torch.empty(1, 0, dtype=torch.int32),
            kv_lens_host=[0],
            decode=SimpleNamespace(local_causal_lens=torch.empty(tokens, 1)),
        )
        return owner, params

    def real_geometry_fixture(self, *, tokens=256, lengths=None, anchors=8):
        # Exercise the actual default, independent of the runner's env. No
        # model tensors, payload collectives or vector-chain numerics run here.
        with patch.dict("os.environ", {}, clear=True):
            owner, params = self.fixture(
                tokens=tokens,
                hidden_dim=7168,
                q_rank=1536,
                attn_res_blocks=anchors,
                sfa_budget_bytes=None,
            )
        params.kv_lens_host = list(lengths if lengths is not None else (tokens,))
        columns = max((n + 511) // 512 for n in params.kv_lens_host)
        params.block_table = torch.empty(
            len(params.kv_lens_host), columns, dtype=torch.int32
        )
        return owner, params

    def plan_only(self, params, phase="sfa"):
        # Stop at the real planner's return, before descriptors, streams or
        # phase buffers are materialized. The real accounting callback runs.
        class Planned(Exception):
            pass

        result = []
        planner = planning.plan_dsa_kv_chunks

        def stop_after_plan(*args, **kwargs):
            result.append(planner(*args, **kwargs))
            raise Planned

        with patch.object(planning, "plan_dsa_kv_chunks", side_effect=stop_after_plan):
            with self.assertRaises(Planned):
                planning.DsaKVExecution(params, phase)
        return result[0]

    def fixed_bytes_at_rejection(self, params, phase="sfa"):
        with patch.object(
            planning, "plan_dsa_kv_chunks", wraps=planning.plan_dsa_kv_chunks
        ) as plan:
            with self.assertRaisesRegex(ValueError, "fixed live data exceeds"):
                planning.DsaKVExecution(params, phase)
        return plan.call_args.kwargs["fixed_bytes"]

    def test_new_peak_is_charged_and_existing_budget_rejects(self):
        owner, params = self.fixture()
        with patch.object(planning, "_tp_prefill_layout_extra_bytes", return_value=0):
            previous = self.fixed_bytes_at_rejection(params)
        extra = planning._tp_prefill_layout_extra_bytes(65536, 8192, 96, 8, 576, 128)
        self.assertGreater(extra, 0)
        owner.budgets["sfa"] = previous
        self.assertEqual(self.fixed_bytes_at_rejection(params), previous + extra)

    def test_query_cp_decode_and_indexer_budget_unchanged(self):
        for prefill, query_cp, phase in (
            (True, True, "sfa"),
            (False, False, "sfa"),
            (True, False, "indexer"),
        ):
            with self.subTest(prefill=prefill, query_cp=query_cp, phase=phase):
                _, params = self.fixture(prefill=prefill, query_cp=query_cp, tokens=16)
                with patch.object(
                    planning,
                    "_tp_prefill_layout_extra_bytes",
                    side_effect=AssertionError("unchanged path"),
                ) as extra:
                    self.fixed_bytes_at_rejection(params, phase)
                    extra.assert_not_called()

    def test_phase_diagnostics_preserve_existing_non_tp_prefill_charges(self):
        for prefill, query_cp, phase in (
            (True, True, "sfa"),
            (False, False, "sfa"),
            (True, False, "indexer"),
        ):
            with self.subTest(prefill=prefill, query_cp=query_cp, phase=phase):
                owner, params = self.fixture(
                    prefill=prefill,
                    query_cp=query_cp,
                    tokens=16,
                    hidden_dim=7168,
                    q_rank=1536,
                    attn_res_blocks=8,
                )
                rows = (
                    params.query_positions.numel()
                    if prefill
                    else params.positions_d.numel()
                )
                expected = owner.external_projection_bytes(
                    16, indexer=phase == "indexer"
                )
                if phase == "indexer":
                    expected += rows * 32 * (128 * 2 + 4)
                else:
                    expected += rows * (
                        (3 * 96 + 128) * 576 * 2
                        + 128 * (512 * 2 + 8)
                        + 96 * 128 * 4
                        + 2048 * 4
                    )
                expected += 64 * 1024 * 1024
                with self.assertRaises(ValueError) as caught:
                    planning.DsaKVExecution(params, phase)
                message = str(caught.exception)
                for field in (
                    f"fixed_bytes={expected},",
                    f"lower_bound_bytes={expected},",
                    "min_required_bytes=unknown,",
                    "kv_costs=not_computed,",
                    f"phase={phase}/{'prefill' if prefill else 'decode'},",
                    f"query_cp={query_cp},",
                    "total_physical_query_tokens=16,",
                ):
                    self.assertIn(field, message)

    def test_default_real_geometry_64k_rejects_with_actionable_capacity(self):
        owner, params = self.real_geometry_fixture(tokens=65536)
        mib = 1024 * 1024
        self.assertEqual(owner.budgets["sfa"], 12288 * mib)
        with self.assertRaises(ValueError) as caught:
            planning.DsaKVExecution(params, "sfa")
        message = str(caught.exception)
        # Fixed overflow is sufficient to reject; do not price the 1024
        # capacities just to diagnose a request that cannot fit.
        expected = {
            "budget_bytes": 12288 * mib,
            "fixed_bytes": 12529 * mib + 2048,
            "remaining_after_fixed_bytes": -241 * mib - 2048,
            "lower_bound_bytes": 12529 * mib + 2048,
            "shortfall_at_least_bytes": 241 * mib + 2048,
            "total_physical_query_tokens": 65536,
            "local_query_rows": 8192,
            "history_tokens": 65536,
            "table_entries": 128,
            "attn_res_blocks": 8,
        }
        for key, value in expected.items():
            self.assertIn(f"{key}={value}", message)
        self.assertIn("phase=sfa/prefill", message)
        self.assertIn("query_cp=False", message)
        for field in (
            "diagnostic_reason=fixed_exceeds_budget,",
            "min_required_bytes=unknown,",
            "min_page_capacity=unknown,",
            "additional_workspace_bytes=unknown,",
            "kv_costs=not_computed,",
            "one_page_temporary_bytes=unknown,",
            "one_page_required_bytes=unknown;",
        ):
            self.assertIn(field, message)
        self.assertIsNone(caught.exception.__cause__.min_required_bytes)
        self.assertIn(
            "re-evaluate DSA_SFA_WORKSPACE_MB with full KV/workspace accounting",
            message,
        )
        self.assertIn("fixed-only lower bound is not a sufficient budget", message)
        self.assertNotIn("DSA_SFA_WORKSPACE_MB>=", message)
        self.assertIn("not a GPU-memory guarantee", message)

    def test_real_geometry_small_and_exact_budget_boundary(self):
        owner, params = self.real_geometry_fixture()
        plan = self.plan_only(params)
        self.assertEqual(
            plan.fixed_bytes, 199424 * 256 + 64 * 1024 * 1024 + 256 * 16 + 16
        )
        self.assertEqual(plan.required_bytes, plan.fixed_bytes + 1220608)
        self.assertEqual(plan.page_capacity, 4)
        owner.budgets["sfa"] = plan.required_bytes
        self.assertEqual(self.plan_only(params).required_bytes, plan.required_bytes)
        owner.budgets["sfa"] -= 1
        with self.assertRaisesRegex(ValueError, "shortfall_bytes=1,"):
            planning.DsaKVExecution(params, "sfa")
        # Fixed-only headroom is not an allowance for a runnable batch.
        owner.budgets["sfa"] = plan.fixed_bytes
        with self.assertRaisesRegex(ValueError, "one physical KV page"):
            planning.DsaKVExecution(params, "sfa")

    def test_tp_aligned_batch_boundary_not_a_per_sequence_limit(self):
        owner, params = self.real_geometry_fixture(tokens=256)
        budget = self.plan_only(params).required_bytes
        owner.budgets["sfa"] = budget
        self.assertEqual(self.plan_only(params).page_capacity, 4)
        owner, params = self.real_geometry_fixture(tokens=264)
        owner.budgets["sfa"] = budget
        with self.assertRaisesRegex(ValueError, "total_physical_query_tokens=264,"):
            planning.DsaKVExecution(params, "sfa")
        _, params = self.real_geometry_fixture(tokens=255)
        with self.assertRaisesRegex(ValueError, "matching global/local rows"):
            planning.DsaKVExecution(params, "sfa")

    def test_default_budget_tp_aligned_boundary_includes_workspace(self):
        # Only this single-history, 8-anchor, nonresident geometry has this
        # boundary. Neither value is a global or per-request token limit.
        _, params = self.real_geometry_fixture(tokens=62968)
        plan = self.plan_only(params)
        self.assertEqual(
            plan.required_bytes, 203536 * 62968 + 64 * 1024 * 1024 + 123 * 16
        )
        self.assertEqual(plan.budget_bytes - plan.required_bytes, 1536208)
        owner, params = self.real_geometry_fixture(tokens=62976)
        with self.assertRaises(ValueError) as caught:
            planning.DsaKVExecution(params, "sfa")
        message = str(caught.exception)
        self.assertIn("shortfall_bytes=92080,", message)
        self.assertIn("total_physical_query_tokens=62976,", message)
        self.assertIn("one physical KV page", message)
        error = caught.exception.__cause__
        self.assertLess(error.fixed_bytes, owner.budgets["sfa"])
        self.assertEqual(error.min_required_bytes, owner.budgets["sfa"] + 92080)

    def test_ragged_history_and_padding_charge_physical_query_batch(self):
        for tokens, lengths in (
            (248, (121, 120)),
            (256, (121, 120)),
            (256, (129, 127)),
        ):
            with self.subTest(tokens=tokens, lengths=lengths):
                owner, params = self.real_geometry_fixture(
                    tokens=tokens, lengths=lengths
                )
                plan = self.plan_only(params)
                self.assertEqual(
                    plan.fixed_bytes,
                    199424 * tokens
                    + 64 * 1024 * 1024
                    + sum(lengths) * 16
                    + params.block_table.numel() * 16,
                )
                owner.budgets["sfa"] = plan.fixed_bytes - 1
                with self.assertRaises(ValueError) as caught:
                    planning.DsaKVExecution(params, "sfa")
                self.assertIn(
                    f"total_physical_query_tokens={tokens},", str(caught.exception)
                )
                self.assertIn(f"history_tokens={sum(lengths)},", str(caught.exception))
                self.assertIn("history_requests=2,", str(caught.exception))

    def test_anchor_and_resident_capacity_shift_the_actual_budget(self):
        _, params = self.real_geometry_fixture()
        baseline = self.plan_only(params)
        for anchors in (0, 4, 9):
            with self.subTest(anchors=anchors):
                owner, params = self.real_geometry_fixture(anchors=anchors)
                owner.layout_bytes = 37
                other = owner.phase("indexer")
                other.buffers["resident"] = torch.empty(17, dtype=torch.uint8)
                other.metadata_budget_bytes = 19
                owner.phase("sfa").buffers["resident"] = torch.empty(
                    13, dtype=torch.uint8
                )
                plan = self.plan_only(params)
                delta = (anchors - 8) * 32 * 7168 * 2 + 37 + 17 + 19
                self.assertEqual(plan.fixed_bytes, baseline.fixed_bytes + delta)
                self.assertEqual(
                    plan.required_bytes, baseline.required_bytes + delta + 13
                )
                owner.budgets["sfa"] = plan.required_bytes - 1
                with self.assertRaises(ValueError) as caught:
                    planning.DsaKVExecution(params, "sfa")
                message = str(caught.exception)
                for field in (
                    f"attn_res_blocks={anchors},",
                    "other_resident_bytes=36,",
                    "phase_resident_bytes=13;",
                    "shortfall_bytes=1,",
                ):
                    self.assertIn(field, message)

    def test_one_page_real_geometry_still_needs_workspace(self):
        owner, params = self.real_geometry_fixture(tokens=256, lengths=(1,))
        plan = self.plan_only(params)
        self.assertEqual(plan.page_capacity, 1)
        self.assertGreater(plan.required_bytes, plan.fixed_bytes)
        owner.budgets["sfa"] = plan.required_bytes - 1
        with self.assertRaises(ValueError) as caught:
            planning.DsaKVExecution(params, "sfa")
        for field in (
            f"min_required_bytes={plan.required_bytes},",
            f"one_page_required_bytes={plan.required_bytes};",
            "shortfall_bytes=1,",
        ):
            self.assertIn(field, str(caught.exception))

    def test_prefill_kv_cap_is_independent_of_full_phase_budget(self):
        for phase, fp8 in (("sfa", False), ("sfa", True), ("indexer", False), ("indexer", True)):
            with self.subTest(phase=phase, fp8=fp8):
                owner, params = self.fixture(
                    tokens=16, sfa_budget_bytes=2**30, indexer_budget_bytes=2**30,
                    sfa_use_fp8=fp8 and phase == "sfa",
                    indexer_use_fp8=fp8 and phase == "indexer",
                )
                params.kv_lens_host = [128, 128]
                params.block_table = torch.empty(2, 2, dtype=torch.int32)
                with patch.object(planning, "prefill_kv_workspace_budget_bytes", return_value=2**30):
                    full = self.plan_only(params, phase)
                with patch.object(planning, "prefill_kv_workspace_budget_bytes", return_value=1):
                    with self.assertRaises(ValueError) as caught:
                        planning.DsaKVExecution(params, phase)
                error = caught.exception.__cause__
                self.assertEqual(error.budget_scope, "prefill_kv")
                self.assertIn("RTP_PREFILL_KV_WORKSPACE_BUDGET_GIB>=", str(caught.exception))
                self.assertGreater(error.min_required_bytes, 1)
                with patch.object(
                    planning, "prefill_kv_workspace_budget_bytes", return_value=error.min_required_bytes
                ):
                    bounded = self.plan_only(params, phase)
                self.assertLess(bounded.page_capacity, full.page_capacity)
                self.assertLessEqual(bounded.required_bytes, owner.budgets[phase])
                self.assertEqual(sum(b.tokens for b in bounded.blocks), 256)

    def test_planning_key_tracks_both_prefill_budgets(self):
        owner, params = self.fixture(tokens=16)
        with patch.object(planning, "prefill_kv_workspace_budget_bytes", return_value=100):
            before = planning.DsaKVExecution.planning_key(params, "sfa")
            owner.budgets["sfa"] += 1
            self.assertNotEqual(before, planning.DsaKVExecution.planning_key(params, "sfa"))
            owner.budgets["sfa"] -= 1
        with patch.object(planning, "prefill_kv_workspace_budget_bytes", return_value=101):
            self.assertNotEqual(before, planning.DsaKVExecution.planning_key(params, "sfa"))

    def test_megatopk_falls_back_only_before_graph_algorithm_is_frozen(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_chunk_plan import DsaKVBlock

        class Planned(Exception):
            pass

        for frozen_mode in (None, False, True):
            with self.subTest(frozen_mode=frozen_mode):
                owner, params = self.fixture(prefill=False, tokens=16, indexer_budget_bytes=2**30)
                # Planner-only policy test: no FP8 score/kernel construction.
                owner.mega_topk = True
                params.block_table = torch.empty(16, 2, dtype=torch.int32)
                params.is_cuda_graph = frozen_mode is not None
                if params.is_cuda_graph:
                    owner.phase("indexer", graph_key=(16, 1)).graph_capacity = {
                        "batch": 16, "geometry": ((2,), 64, 64, 8, 0),
                        "pages": 2, "mega_topk": frozen_mode,
                    }
                block = DsaKVBlock((0,), (0,), (128,))
                seen = []
                with patch.object(
                    planning, "indexer_candidate_workspace_shapes",
                    wraps=planning.indexer_candidate_workspace_shapes,
                ) as candidates:
                    def plan(*args, **kwargs):
                        candidates.reset_mock()
                        kwargs["plan_temporary_bytes"]((block,))
                        mega = not candidates.called
                        seen.append(mega)
                        if mega:
                            raise planning.DsaKVCapacityError(
                                budget_bytes=10, fixed_bytes=11, page_tokens=64,
                            )
                        raise Planned

                    with patch.object(planning, "plan_dsa_kv_chunks", side_effect=plan):
                        if frozen_mode is True:
                            with self.assertRaisesRegex(ValueError, "fixed live data exceeds"):
                                planning.DsaKVExecution(params, "indexer")
                        else:
                            with self.assertRaises(Planned):
                                planning.DsaKVExecution(params, "indexer")
                self.assertEqual(seen, [True, False] if frozen_mode is None else [frozen_mode])

    def test_decode_direct_page_metadata_refresh_preserves_capture_storage(self):
        execution = planning.DsaKVExecution.__new__(planning.DsaKVExecution)
        execution.fp8_indexer, execution.prefill = True, False
        execution.batch, execution.queries, execution.rows = 2, 2, 4
        execution.direct_native_page_size, execution.num_sms = 64, 1
        execution.metadata_ready = None
        score = SimpleNamespace(prepare_paged_metadata=Mock(return_value=torch.zeros(2, dtype=torch.int32)))
        execution.owner = SimpleNamespace(device="cpu", indexer_score=score)
        block = planning.DsaPreparedKVBlock(None, 64, 128, None)
        execution.blocks = [block]
        params = SimpleNamespace(
            forbid_realloc=False,
            positions_d=torch.tensor([0, 1, 2, -1]),
            block_table=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32),
            decode=SimpleNamespace(local_causal_lens=torch.tensor([[65, 80], [200, 200]])),
        )
        execution.update_indexer_metadata(params)
        pointers = [t.data_ptr() for t in (block.native_context, block.native_table, block.native_schedule)]
        torch.testing.assert_close(block.native_context, torch.tensor([[1, 16], [128, 0]], dtype=torch.int32))
        torch.testing.assert_close(block.native_table, params.block_table[:, 1:3])
        params.forbid_realloc = True
        params.block_table.add_(10)
        score.prepare_paged_metadata.return_value = torch.ones(2, dtype=torch.int32)
        execution.update_indexer_metadata(params)
        self.assertEqual(pointers, [t.data_ptr() for t in (block.native_context, block.native_table, block.native_schedule)])
        torch.testing.assert_close(block.native_table, params.block_table[:, 1:3])
        torch.testing.assert_close(block.native_schedule, torch.ones(2, dtype=torch.int32))
        self.assertEqual(score.prepare_paged_metadata.call_args.kwargs["page_size"], 64)
        score.prepare_paged_metadata.return_value = torch.ones(3, dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "scheduling capacity changed"):
            execution.update_indexer_metadata(params)

    def test_planning_key_tracks_layout_and_invalid_geometry_is_rejected(self):
        owner, params = self.fixture(tokens=16)
        before = planning.DsaKVExecution.planning_key(params, "sfa")
        params.query_cp = True
        self.assertNotEqual(before, planning.DsaKVExecution.planning_key(params, "sfa"))
        params.query_cp = False
        owner.value_dim = 256
        self.assertNotEqual(before, planning.DsaKVExecution.planning_key(params, "sfa"))
        for geometry in (
            (15, 2, 96, 8),
            (16, 2, 95, 8),
            (16, 2, 96, 0),
            (-8, -1, 96, 8),
        ):
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                planning._tp_prefill_layout_extra_bytes(*geometry, 576, 128)


if __name__ == "__main__":
    unittest.main()
