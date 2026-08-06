import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import rtp_llm.models_py.model_desc.deepseek_v4_dspark_model as dspark_model_module
from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import DeepSeekV4DSparkModel
from rtp_llm.models_py.speculative.dspark_proposer_mixin import map_context_rows


def _dspark_harness(gamma: int = 5) -> DeepSeekV4DSparkModel:
    model = DeepSeekV4DSparkModel.__new__(DeepSeekV4DSparkModel)
    model._gen_num_per_cycle = gamma
    model._v4_args = type(
        "Args", (), {"window_size": 128, "dim": 8, "vocab_size": 17}
    )()
    return model


class DSparkCudaGraphContractTest(unittest.TestCase):
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
            vocab_size=17,
        )

        outputs = model.dspark_empty_outputs(2, torch.device("cpu"))

        self.assertEqual(tuple(outputs.hidden_states.shape), (6, 8))
        self.assertEqual(tuple(outputs.draft_tokens.shape), (2, 3))

    def test_padded_graph_slot_maps_no_rows_with_prefix_sum_starts(self) -> None:
        # A graph bucket of two requests replays a one-request batch. Rows
        # are front-packed, so the padded slot's derived start equals the
        # packed span end and its zero length maps no rows; the buffer's
        # padding tail rows must all stay invalid.
        lengths = torch.tensor([3, 0], dtype=torch.int32)
        starts = (lengths.cumsum(0) - lengths).to(torch.int32)
        prefix = torch.tensor([10, 0], dtype=torch.int32)

        req, positions = map_context_rows(
            starts, lengths, prefix, row_count=12
        )

        self.assertEqual(req[:3].tolist(), [0, 0, 0])
        self.assertEqual(positions[:3].tolist(), [7, 8, 9])
        self.assertTrue(torch.all(req[3:] == -1))
        self.assertTrue(torch.all(positions[3:] == -1))

    def test_packed_verify_rows_map_accepted_prefixes(self) -> None:
        # Decode-tail rows arrive front-packed (accepted rows only); rows
        # past the packed span are scatter padding and must stay invalid.
        lengths = torch.tensor([3, 1], dtype=torch.int32)
        starts = (lengths.cumsum(0) - lengths).to(torch.int32)
        committed_ends = torch.tensor([123, 367], dtype=torch.int32)

        req, positions = map_context_rows(
            starts, lengths, committed_ends, row_count=12
        )

        expected_req = torch.full((12,), -1, dtype=torch.int32)
        expected_req[:3] = 0
        expected_req[3] = 1
        expected_positions = torch.full((12,), -1, dtype=torch.int32)
        expected_positions[:3] = torch.tensor([120, 121, 122], dtype=torch.int32)
        expected_positions[3] = 366
        self.assertTrue(torch.equal(req, expected_req))
        self.assertTrue(torch.equal(positions, expected_positions))

    def test_cuda_graph_metadata_owner_is_persistent(self) -> None:
        model = _dspark_harness()
        metadata = model.prepare_fmha_impl(inputs=None, is_cuda_graph=True)

        self.assertTrue(model._should_capture_cuda_graph(None, False))
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
                self.freqs_cis = torch.zeros((32, 2), dtype=torch.float32)
                self.wkv = object()
                self.kv_norm = object()

            def _ensure_freqs_cis_bound(self) -> None:
                pass

            def _pool_entries_per_block(self, _region: int) -> int:
                return 134

            def _pool_view_3d_fp8(self, _region: int) -> torch.Tensor:
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


if __name__ == "__main__":
    unittest.main()
