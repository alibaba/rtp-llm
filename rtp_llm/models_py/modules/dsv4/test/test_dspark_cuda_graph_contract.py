import unittest

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_dspark_model import (
    DeepSeekV4DSparkModel,
)


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

        outputs = model._empty_outputs(batch_size=2, device=torch.device("cpu"))

        self.assertEqual(tuple(outputs.hidden_states.shape), (6, 8))
        self.assertEqual(tuple(outputs.draft_tokens.shape), (2, 3))
        self.assertEqual(tuple(outputs.draft_probs.shape), (2, 3, 17))

    def test_padded_context_start_is_monotonic_sentinel(self) -> None:
        # A graph bucket of two requests replays a one-request batch. The
        # padded start must be the row-capacity sentinel (12), not zero: a
        # repeated zero would make searchsorted(right=True) assign row zero to
        # the padded request.
        starts = torch.tensor([0, 12], dtype=torch.int32)
        lengths = torch.tensor([3, 0], dtype=torch.int32)
        prefix = torch.tensor([10, 0], dtype=torch.int32)

        req, positions = DeepSeekV4DSparkModel._map_context_rows(
            starts, lengths, prefix, row_count=12
        )

        self.assertEqual(req[:3].tolist(), [0, 0, 0])
        self.assertEqual(positions[:3].tolist(), [7, 8, 9])
        self.assertTrue(torch.all(req[3:] == -1))
        self.assertTrue(torch.all(positions[3:] == -1))

    def test_cuda_graph_metadata_owner_is_persistent(self) -> None:
        model = _dspark_harness()
        metadata = model.prepare_fmha_impl(inputs=None, is_cuda_graph=True)

        self.assertTrue(model._should_capture_cuda_graph(None, False))
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.sched_meta_cache, {})
        self.assertTrue(metadata.support_cuda_graph())
        metadata.prepare_cuda_graph(None)


if __name__ == "__main__":
    unittest.main()
