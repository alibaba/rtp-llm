import unittest

import torch

from benchmark.topology_kv_candidate_schedule import (
    BlockCandidateConfig,
    benchmark_decode_attention,
    block_schedule_to_token_indices,
    build_key_block_centroids,
    build_block_candidate_schedule,
    build_topology_candidate_token_indices,
    dense_decode_attention,
    format_benchmark_results,
    run_decode_attention_grid,
    sparse_decode_attention,
)


class TopologyKVCandidateScheduleTest(unittest.TestCase):
    def test_schedule_keeps_sink_local_and_salient_causal_blocks(self):
        centroids = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.9],
                [0.1, 0.8],
            ]
        )
        config = BlockCandidateConfig(
            block_size=64,
            sink_blocks=1,
            local_blocks=2,
            salience_blocks=2,
            max_candidate_blocks=5,
        )

        schedule = build_block_candidate_schedule(centroids, config)

        row = schedule[5].tolist()
        self.assertEqual(row, [0, 2, 3, 4, 5])
        self.assertTrue(all(block <= 5 for block in row))

    def test_schedule_pads_short_rows_without_future_blocks(self):
        centroids = torch.eye(3)
        config = BlockCandidateConfig(
            block_size=32,
            sink_blocks=1,
            local_blocks=2,
            salience_blocks=2,
            max_candidate_blocks=5,
        )

        schedule = build_block_candidate_schedule(centroids, config)

        self.assertEqual(schedule[0].tolist(), [0, -1, -1, -1, -1])
        self.assertEqual(schedule[1].tolist(), [0, 1, -1, -1, -1])
        self.assertEqual(schedule[2].tolist(), [0, 1, 2, -1, -1])

    def test_sparse_attention_matches_dense_when_all_tokens_are_selected(self):
        torch.manual_seed(0)
        query = torch.randn(1, 2, 1, 16)
        key = torch.randn(1, 2, 128, 16)
        value = torch.randn(1, 2, 128, 16)
        candidate_indices = torch.arange(128).view(1, 128)

        dense = dense_decode_attention(query, key, value)
        sparse = sparse_decode_attention(query, key, value, candidate_indices)

        torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-5)

    def test_sparse_attention_rejects_duplicate_candidate_indices(self):
        query = torch.randn(1, 2, 1, 16)
        key = torch.randn(1, 2, 128, 16)
        value = torch.randn(1, 2, 128, 16)
        candidate_indices = torch.tensor([[0, 1, 1, 2]])

        with self.assertRaisesRegex(ValueError, "duplicate tokens"):
            sparse_decode_attention(query, key, value, candidate_indices)

    def test_key_block_centroids_average_batch_heads_and_tail_blocks(self):
        key = torch.tensor(
            [
                [
                    [[1.0, 1.0], [3.0, 3.0], [10.0, 0.0], [14.0, 4.0], [5.0, 7.0]],
                    [[5.0, 5.0], [7.0, 7.0], [20.0, 2.0], [24.0, 6.0], [9.0, 11.0]],
                ]
            ]
        )

        centroids = build_key_block_centroids(key, block_size=2)

        expected = torch.tensor([[4.0, 4.0], [17.0, 3.0], [7.0, 9.0]])
        torch.testing.assert_close(centroids, expected)

    def test_key_block_centroids_accept_2d_and_3d_inputs(self):
        key_2d = torch.tensor(
            [[1.0, 1.0], [3.0, 3.0], [10.0, 0.0], [14.0, 4.0], [5.0, 7.0]]
        )
        key_3d = key_2d.unsqueeze(0)

        expected = torch.tensor([[2.0, 2.0], [12.0, 2.0], [5.0, 7.0]])

        torch.testing.assert_close(build_key_block_centroids(key_2d, 2), expected)
        torch.testing.assert_close(build_key_block_centroids(key_3d, 2), expected)

    def test_block_schedule_to_token_indices_expands_blocks_and_masks_tail(self):
        schedule = torch.tensor([[0, 2, -1]])

        token_indices = block_schedule_to_token_indices(
            schedule,
            block_size=4,
            seq_len=10,
        )

        self.assertEqual(
            token_indices.tolist(),
            [[0, 1, 2, 3, 8, 9, -1, -1, -1, -1, -1, -1]],
        )

    def test_topology_candidate_indices_use_schedule_and_keep_latest_block(self):
        key = torch.arange(16, dtype=torch.float32).view(1, 1, 8, 2)

        token_indices = build_topology_candidate_token_indices(
            key,
            selected_tokens=4,
            block_size=2,
        )

        self.assertEqual(token_indices.tolist(), [0, 1, 6, 7])

    def test_topology_candidate_indices_keep_latest_block_with_one_block_budget(self):
        key = torch.arange(16, dtype=torch.float32).view(1, 1, 8, 2)

        token_indices = build_topology_candidate_token_indices(
            key,
            selected_tokens=1,
            block_size=2,
        )

        self.assertEqual(token_indices.tolist(), [6])

    def test_topology_candidate_indices_accept_2d_and_3d_inputs(self):
        key_2d = torch.arange(16, dtype=torch.float32).view(8, 2)
        key_3d = key_2d.unsqueeze(0)

        self.assertEqual(
            build_topology_candidate_token_indices(
                key_2d,
                selected_tokens=1,
                block_size=2,
            ).tolist(),
            [6],
        )
        self.assertEqual(
            build_topology_candidate_token_indices(
                key_3d,
                selected_tokens=1,
                block_size=2,
            ).tolist(),
            [6],
        )

    def test_benchmark_grid_and_markdown_format_are_reproducible(self):
        results = run_decode_attention_grid(
            seq_lens=[128],
            selected_tokens=[32, 64],
            heads=2,
            head_dim=16,
            rounds=1,
            warmup=0,
            dtype=torch.float32,
            device="cpu",
        )

        table = format_benchmark_results(results)

        self.assertEqual(len(results), 2)
        self.assertIn("| seq_len | selected_tokens | dense_sdpa_ms |", table)
        self.assertIn("| 128 | 32 |", table)
        self.assertIn("| 128 | 64 |", table)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for speed testing")
    def test_sparse_attention_cuda_benchmark_runs_with_topology_schedule(self):
        result = benchmark_decode_attention(
            seq_len=16384,
            selected_tokens=512,
            heads=16,
            head_dim=64,
            rounds=60,
            warmup=20,
            dtype=torch.float16,
            device="cuda",
        )

        self.assertEqual(result.seq_len, 16384)
        self.assertEqual(result.selected_tokens, 512)
        self.assertGreater(result.dense_ms, 0.0)
        self.assertGreater(result.sparse_ms, 0.0)
        self.assertGreater(result.speedup, 0.0)


if __name__ == "__main__":
    unittest.main()
