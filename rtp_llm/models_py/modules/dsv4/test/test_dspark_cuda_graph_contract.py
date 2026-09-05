import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import rtp_llm.models_py.model_desc.deepseek_v4_dspark_model as dspark_model_module
from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import DeepSeekV4DSparkModel
from rtp_llm.models_py.model_desc.deepseek_v4_model import DeepSeekV4Model
from rtp_llm.models_py.modules.factory.attention.attn_factory import (
    CudaGraphSelectionMode,
    PrefillCudaGraphUnsupportedBackend,
)
from rtp_llm.models_py.speculative.dspark_proposer_mixin import map_context_rows
from rtp_llm.ops.compute_ops import PyModelInputs


def _dspark_harness(gamma: int = 5) -> DeepSeekV4DSparkModel:
    model = DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel)
    model._gen_num_per_cycle = gamma
    model._dspark_commit_cp_enabled = False
    model._dspark_kv_cache_sharded = False
    model.tp_size = 2
    model.tp_rank = 0
    model._v4_args = type(
        "Args", (), {"window_size": 128, "dim": 8, "vocab_size": 17}
    )()
    return model


class DSparkCudaGraphContractTest(unittest.TestCase):
    def test_dsv4_descriptors_reject_prefill_graph_selection(self) -> None:
        for model in (
            DeepSeekV4Model.__new__(DeepSeekV4Model),
            DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel),
        ):
            with self.subTest(model=type(model).__name__):
                with self.assertRaises(PrefillCudaGraphUnsupportedBackend):
                    model.prepare_fmha_impl(
                        inputs=None,
                        is_cuda_graph=True,
                        cuda_graph_selection_mode=CudaGraphSelectionMode.PREFILL_GRAPH,
                    )

    def test_forward_uses_fixed_role_entrypoints(self) -> None:
        model = _dspark_harness(gamma=3)
        model.v4 = SimpleNamespace(
            embed=SimpleNamespace(weight=torch.empty((1, 8), dtype=torch.bfloat16))
        )
        model.kv_cache = None
        model._dspark_width = 3
        model._dspark_hidden_dim = 8
        propose_inputs = PyModelInputs()
        propose_inputs.input_ids = torch.zeros(3, dtype=torch.int32)

        with self.assertRaises(RuntimeError):
            model.forward(propose_inputs)

        propose_outputs = model.forward_propose(propose_inputs)
        self.assertEqual(tuple(propose_outputs.hidden_states.shape), (3, 8))
        self.assertEqual(propose_outputs.hidden_states.dtype, torch.bfloat16)

        commit_inputs = PyModelInputs()
        commit_inputs.input_ids = torch.zeros(4, dtype=torch.int32)
        commit_outputs = model.forward_commit(commit_inputs)
        self.assertEqual(tuple(commit_outputs.hidden_states.shape), (0, 8))

    def test_padded_bucket_rows_have_no_attention_work(self) -> None:
        model = _dspark_harness()
        prefix_lengths = torch.tensor([10, 0], dtype=torch.int32)
        active = torch.tensor([True, False])
        # Row zero owns one physical block. Row one is CUDA-graph padding
        # and deliberately has no block allocation.
        block_table = torch.tensor([[1], [0]], dtype=torch.int32)

        indices, topk_length = model._build_noncausal_indices(
            prefix_lengths,
            active,
            block_table,
            entries_per_block=256,
            tokens_per_block=256,
        )

        self.assertEqual(tuple(indices.shape), (10, 256))
        self.assertEqual(topk_length.tolist(), [15, 0])
        self.assertTrue(torch.all(indices[5:] == -1))
        self.assertTrue(torch.all(indices[:5, :15] >= 0))

    def test_runtime_gamma_controls_query_width(self) -> None:
        model = _dspark_harness(gamma=3)
        prefix_lengths = torch.tensor([10, 0], dtype=torch.int32)
        active = torch.tensor([True, False])
        block_table = torch.tensor([[1], [0]], dtype=torch.int32)

        indices, topk_length = model._build_noncausal_indices(
            prefix_lengths,
            active,
            block_table,
            entries_per_block=256,
            tokens_per_block=256,
        )

        self.assertEqual(tuple(indices.shape), (6, 256))
        self.assertEqual(topk_length.tolist(), [13, 0])

    def test_runtime_gamma_controls_output_width(self) -> None:
        model = _dspark_harness(gamma=3)
        model.init_dspark_proposer(
            width=3,
            noise_token_id=1,
            aux_feature_dim=24,
            hidden_dim=8,
        )

        outputs = model.dspark_empty_outputs(2, torch.device("cpu"))

        self.assertEqual(tuple(outputs.hidden_states.shape), (6, 8))

    def test_padded_graph_slot_maps_no_rows_with_prefix_sum_starts(self) -> None:
        # A graph bucket of two requests replays a one-request batch. Rows
        # are front-packed, so the padded slot's derived start equals the
        # packed span end and its zero length maps no rows; the buffer's
        # padding tail rows must all stay invalid.
        lengths = torch.tensor([3, 0], dtype=torch.int32)
        starts = (lengths.cumsum(0) - lengths).to(torch.int32)
        prefix = torch.tensor([10, 0], dtype=torch.int32)

        req, positions = map_context_rows(starts, lengths, prefix, row_count=12)

        self.assertEqual(req[:3].tolist(), [0, 0, 0])
        self.assertEqual(positions[:3].tolist(), [7, 8, 9])
        self.assertTrue(torch.all(req[3:] == -1))
        self.assertTrue(torch.all(positions[3:] == -1))

    def test_dense_verify_rows_map_full_fixed_width_blocks(self) -> None:
        # Decode-tail commit preserves every target verify row, independent
        # of rejection. Rows past the fixed-width batch span are graph-bucket
        # padding and must stay invalid.
        lengths = torch.tensor([4, 4], dtype=torch.int32)
        starts = (lengths.cumsum(0) - lengths).to(torch.int32)
        committed_ends = torch.tensor([123, 367], dtype=torch.int32)

        req, positions = map_context_rows(starts, lengths, committed_ends, row_count=12)

        expected_req = torch.full((12,), -1, dtype=torch.int32)
        expected_req[:4] = 0
        expected_req[4:8] = 1
        expected_positions = torch.full((12,), -1, dtype=torch.int32)
        expected_positions[:4] = torch.tensor([119, 120, 121, 122], dtype=torch.int32)
        expected_positions[4:8] = torch.tensor([363, 364, 365, 366], dtype=torch.int32)
        self.assertTrue(torch.equal(req, expected_req))
        self.assertTrue(torch.equal(positions, expected_positions))

    def test_cp_row_map_derives_from_forward_cp_metadata(self) -> None:
        model = _dspark_harness(gamma=3)
        model._dspark_commit_cp_enabled = True
        starts = torch.tensor([0], dtype=torch.int32)
        lengths = torch.tensor([4], dtype=torch.int32)
        committed_ends = torch.tensor([14], dtype=torch.int32)
        cp_ctx = SimpleNamespace(
            global_positions=torch.tensor([10, 13], dtype=torch.int64),
            local_is_real=torch.tensor([True, True]),
            req_id_per_token=torch.tensor([0, 0], dtype=torch.int32),
            cp_size=2,
            cp_rank=0,
            kv_cache_sharded=False,
        )
        cp_inputs = SimpleNamespace(
            attention_inputs=SimpleNamespace(
                context_parallel_info=SimpleNamespace(
                    prefill_qkv_padding_mask=torch.ones(4, dtype=torch.bool)
                ),
                prefix_lengths=torch.tensor([10], dtype=torch.int32),
            )
        )

        with patch.object(
            dspark_model_module, "build_cp_context_for_forward", return_value=cp_ctx
        ) as build_mock:
            req, positions, commit_ctx = model.map_commit_rows(
                starts, lengths, committed_ends, row_count=2, inputs=cp_inputs
            )

        self.assertEqual(req.tolist(), [0, 0])
        self.assertEqual(positions.tolist(), [10, 13])
        # The derived ctx travels as a return value, never as model state.
        self.assertIs(commit_ctx, cp_ctx)
        # The ctx is derived from THIS forward's metadata, sized by its rows.
        self.assertEqual(build_mock.call_args.args[3], 2)

        # A dense forward (no CP-split prefill stream) uses the packed layout
        # even on a CP-enabled engine.
        dense_inputs = SimpleNamespace(
            attention_inputs=SimpleNamespace(
                context_parallel_info=None,
                prefix_lengths=torch.tensor([10], dtype=torch.int32),
            )
        )
        req, positions, commit_ctx = model.map_commit_rows(
            starts, lengths, committed_ends, row_count=4, inputs=dense_inputs
        )

        self.assertEqual(req.tolist(), [0, 0, 0, 0])
        self.assertEqual(positions.tolist(), [10, 11, 12, 13])
        self.assertIsNone(commit_ctx)

    def test_cuda_graph_metadata_owner_is_persistent(self) -> None:
        model = _dspark_harness()
        metadata = model.prepare_fmha_impl(inputs=None, is_cuda_graph=True)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.sched_meta_cache, {})
        self.assertTrue(metadata.support_cuda_graph())
        metadata.prepare_cuda_graph(None)

    def test_commit_write_uses_request_ids_positions_and_committed_ends(self) -> None:
        """The DSpARK commit call must not reconstruct positions from a QSL."""

        class FakeAttention:
            compress_ratio = 0
            rope_head_dim = 2
            head_dim = 4
            eps = 1e-6

            def __init__(self) -> None:
                self._kv_cache = None
                self._block_tables_by_type = {}
                self._cp_ctx = None
                self.freqs_cis = torch.zeros((32, 2), dtype=torch.float32)
                self.wkv = object()
                self.kv_norm = object()

            def _ensure_freqs_cis_bound(self) -> None:
                pass

            def _pool_entries_per_block(self, _region: str) -> int:
                return 134

            def _swa_entries_per_block(self) -> int:
                return 134

            def _swa_cp_byte_sliced(self) -> bool:
                return False

            def _pool_view_3d_fp8(self, _region: str) -> torch.Tensor:
                return torch.zeros((2, 134, 1), dtype=torch.uint8)

            def _lin(self, _weight: object, x: torch.Tensor) -> torch.Tensor:
                return x

        model = _dspark_harness(gamma=1)
        attention = FakeAttention()
        model.v4 = SimpleNamespace(
            layers=[SimpleNamespace(attn=attention)],
        )
        model.kv_cache = object()

        context_req_ids = torch.tensor([0, 0, -1, -1], dtype=torch.int32)
        context_positions = torch.tensor([10, 11, -1, -1], dtype=torch.int32)
        prefix_lengths = torch.tensor([12], dtype=torch.int32)
        expected_slots = torch.tensor([144, 145, -1, -1], dtype=torch.long)

        with (
            patch.object(
                dspark_model_module,
                "fused_rmsnorm_rope",
                return_value=torch.zeros((4, 4), dtype=torch.bfloat16),
            ),
            patch.object(
                dspark_model_module,
                "compute_swa_slot_mapping_from_positions",
                return_value=expected_slots,
            ) as slot_mapper,
            patch.object(
                dspark_model_module,
                "decode_write_swa_fp8",
            ) as cache_writer,
        ):
            model._commit_layer_features(
                layer_idx=0,
                main_x=torch.zeros((4, 4), dtype=torch.bfloat16),
                context_req_ids=context_req_ids,
                context_positions=context_positions,
                committed_ends=prefix_lengths,
                block_table=torch.tensor([[1]], dtype=torch.int32),
                tokens_per_block=256,
                batch_size=1,
            )

        kwargs = slot_mapper.call_args.kwargs
        self.assertTrue(torch.equal(kwargs["req_id_per_token"], context_req_ids))
        self.assertTrue(torch.equal(kwargs["positions"], context_positions))
        self.assertTrue(torch.equal(kwargs["seq_lens"], prefix_lengths))
        self.assertTrue(
            torch.equal(kwargs["block_table"], torch.tensor([[1]], dtype=torch.int32))
        )
        self.assertEqual(kwargs["num_tokens"], 4)
        self.assertEqual(kwargs["pool_entries_per_block"], 134)
        self.assertEqual(kwargs["tokens_per_block_for_block_table"], 256)
        self.assertEqual(kwargs["ring_entries"], 134)
        self.assertTrue(
            torch.equal(cache_writer.call_args.kwargs["slot_mapping"], expected_slots)
        )

    def test_cp_project_then_gather_selects_cache_writer(self) -> None:
        """CP gathers projected KV and honors replicated/byte-sliced pools."""

        class FakeAttention:
            compress_ratio = 0
            rope_head_dim = 2
            head_dim = 4
            eps = 1e-6

            def __init__(self, byte_sliced: bool) -> None:
                self._byte_sliced = byte_sliced
                self._kv_cache = None
                self._block_tables_by_type = {}
                self._cp_ctx = None
                self.freqs_cis = torch.zeros((32, 2), dtype=torch.float32)
                self.wkv = object()
                self.kv_norm = object()
                self.raw_pool = torch.zeros((64,), dtype=torch.uint8)
                self.compaction = object()

            def _ensure_freqs_cis_bound(self) -> None:
                pass

            def _swa_entries_per_block(self) -> int:
                return 16

            def _swa_cp_byte_sliced(self) -> bool:
                return self._byte_sliced

            def _pool_view_3d_fp8(self, _region: str) -> torch.Tensor:
                return torch.zeros((4, 16, 1), dtype=torch.uint8)

            def _pool_raw_u8(self, _region: str) -> torch.Tensor:
                return self.raw_pool

            def _build_swa_cp_byte_compaction(self, *_args, **_kwargs):
                return self.compaction

            def _lin(self, _weight: object, x: torch.Tensor) -> torch.Tensor:
                return x

        local_kv = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
        gathered_kv = torch.cat((local_kv, local_kv + 10), dim=0)
        gathered_req_ids = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
        gathered_positions = torch.tensor([10, 12, 11, 13], dtype=torch.int32)
        expected_slots = torch.tensor([26, 28, 27, 29], dtype=torch.long)

        for byte_sliced in (False, True):
            with self.subTest(byte_sliced=byte_sliced):
                model = _dspark_harness(gamma=1)
                attention = FakeAttention(byte_sliced)
                model.v4 = SimpleNamespace(layers=[SimpleNamespace(attn=attention)])
                model.kv_cache = object()
                commit_cp_ctx = SimpleNamespace(cp_rank=1, cp_size=2)

                with (
                    patch.object(
                        dspark_model_module,
                        "fused_rmsnorm_rope",
                        return_value=local_kv,
                    ),
                    patch(
                        "rtp_llm.models_py.distributed.collective_torch.all_gather",
                        return_value=gathered_kv,
                    ) as all_gather,
                    patch.object(
                        dspark_model_module,
                        "compute_swa_slot_mapping_from_positions",
                        return_value=expected_slots,
                    ) as slot_mapper,
                    patch.object(
                        dspark_model_module, "decode_write_swa_fp8"
                    ) as regular_writer,
                    patch(
                        "rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton."
                        "quantize_and_insert_k_cache_cp_byte_sliced"
                    ) as sliced_writer,
                ):
                    model._commit_layer_features(
                        layer_idx=0,
                        main_x=torch.zeros((2, 4), dtype=torch.bfloat16),
                        context_req_ids=torch.tensor([0, 0], dtype=torch.int32),
                        context_positions=torch.tensor([10, 12], dtype=torch.int32),
                        committed_ends=torch.tensor([14], dtype=torch.int32),
                        block_table=torch.tensor([[1]], dtype=torch.int32),
                        tokens_per_block=16,
                        batch_size=1,
                        gathered_req_ids=gathered_req_ids,
                        gathered_positions=gathered_positions,
                        cp_ctx=commit_cp_ctx,
                    )

                all_gather.assert_called_once()
                self.assertTrue(
                    torch.equal(
                        slot_mapper.call_args.kwargs["positions"],
                        gathered_positions,
                    )
                )
                if byte_sliced:
                    regular_writer.assert_not_called()
                    sliced_writer.assert_called_once()
                    kwargs = sliced_writer.call_args.kwargs
                    self.assertEqual(kwargs["cp_rank"], 1)
                    self.assertEqual(kwargs["cp_size"], 2)
                    self.assertIs(kwargs["compaction"], attention.compaction)
                    self.assertTrue(
                        torch.equal(sliced_writer.call_args.args[2], expected_slots)
                    )
                else:
                    sliced_writer.assert_not_called()
                    regular_writer.assert_called_once()
                    self.assertTrue(
                        torch.equal(regular_writer.call_args.kwargs["kv"], gathered_kv)
                    )


if __name__ == "__main__":
    unittest.main()
