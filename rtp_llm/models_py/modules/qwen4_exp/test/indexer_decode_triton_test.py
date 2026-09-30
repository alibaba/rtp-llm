import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.indexer_compressor import (
    restore_indexer_cache,
    write_indexer_cache,
)
from rtp_llm.models_py.modules.qwen4_exp.indexer_decode_triton import (
    write_decode_key_,
    write_decode_key_with_undo_,
    write_draft_window_with_undo_,
    write_target_window_with_undo_,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class IndexerDecodeTritonTest(unittest.TestCase):
    def test_ragged_draft_window_graph_replay_and_rollback(self):
        raw, prefixes, cos, sin, gamma, kv, kt, state, st = self._target_case(
            [7, 127]
        )
        raw, cos, sin = raw[:5].contiguous(), cos[:5].contiguous(), sin[:5].contiguous()
        lengths = torch.tensor([1, 4], dtype=torch.int32, device=raw.device)
        cu = torch.tensor([0, 1, 5], dtype=torch.int32, device=raw.device)
        original_kv, original_state = kv.clone(), state.clone()

        def write():
            return write_draft_window_with_undo_(
                raw, cu, prefixes, lengths, cos, sin, gamma, kv, kt, state, st,
                norm_eps=1e-6,
            )

        write()
        reference_kv, reference_state = original_kv.clone(), original_state.clone()
        write_indexer_cache(
            raw, cu, prefixes, cos, sin, gamma, reference_kv, kt,
            reference_state, st, ratio=4, kv_tokens_per_block=128,
            state_tokens_per_block=128, norm_eps=1e-6,
            rope_is_token_aligned=True,
        )
        torch.testing.assert_close(state, reference_state, rtol=0, atol=0)
        torch.testing.assert_close(kv, reference_kv, rtol=0.01, atol=0.016)

        kv.copy_(original_kv)
        state.copy_(original_state)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            undo = write()

        kv.copy_(original_kv)
        state.copy_(original_state)
        lengths.copy_(torch.tensor([3, 2], dtype=torch.int32, device=raw.device))
        cu.copy_(torch.tensor([0, 3, 5], dtype=torch.int32, device=raw.device))
        prefixes.copy_(torch.tensor([5, 125], dtype=torch.int32, device=raw.device))
        raw.copy_(torch.randn_like(raw.float()).bfloat16())
        kt[:, 0], kt[:, 1] = kt[:, 1].clone(), kt[:, 0].clone()
        st[:, 0], st[:, 1] = st[:, 1].clone(), st[:, 0].clone()
        graph.replay()

        reference_kv, reference_state = original_kv.clone(), original_state.clone()
        write_indexer_cache(
            raw, cu, prefixes, cos, sin, gamma, reference_kv, kt,
            reference_state, st, ratio=4, kv_tokens_per_block=128,
            state_tokens_per_block=128, norm_eps=1e-6,
            rope_is_token_aligned=True,
        )
        torch.testing.assert_close(state, reference_state, rtol=0, atol=0)
        torch.testing.assert_close(kv, reference_kv, rtol=0.01, atol=0.016)
        restore_indexer_cache(undo)
        torch.testing.assert_close(state, original_state, rtol=0, atol=0)
        torch.testing.assert_close(kv, original_kv, rtol=0, atol=0)

    def _target_case(self, prefixes):
        device = torch.device("cuda")
        batch = len(prefixes)
        raw = torch.randn(batch * 4, 128, device=device).to(torch.bfloat16)
        starts = torch.tensor(prefixes, device=device, dtype=torch.int32)
        cosine = torch.randn(batch * 4, 64, device=device).to(torch.bfloat16)
        sine = torch.randn(batch * 4, 64, device=device).to(torch.bfloat16)
        gamma = (torch.randn(128, device=device) * 0.1).to(torch.bfloat16)
        table = torch.tensor(
            [[1 + 2 * b, 2 + 2 * b] for b in range(batch)],
            dtype=torch.int32,
            device=device,
        )
        pool_blocks = 2 * batch + 1
        kv = torch.randn(
            pool_blocks, 32, 128, device=device, dtype=torch.float32
        ).to(torch.bfloat16)
        state = torch.randn(
            pool_blocks, 8, 128, device=device, dtype=torch.float32
        ).to(torch.bfloat16).float()
        return raw, starts, cosine, sine, gamma, kv, table, state, table.clone()

    def test_target_window_parity_and_rollback(self):
        for prefixes in ([5], [7], [127], [128], [5, 127]):
            with self.subTest(prefixes=prefixes):
                raw, starts, cos, sin, gamma, kv, kt, state, st = self._target_case(prefixes)
                original_kv = kv.clone()
                original_state = state.clone()
                reference_kv = kv.clone()
                reference_state = state.clone()
                cu = torch.arange(
                    0, (len(prefixes) + 1) * 4, 4, dtype=torch.int32, device=raw.device
                )
                write_indexer_cache(
                    raw, cu, starts, cos, sin, gamma, reference_kv, kt,
                    reference_state, st, ratio=4, kv_tokens_per_block=128,
                    state_tokens_per_block=128, norm_eps=1e-6,
                    rope_is_token_aligned=True,
                )
                undo = write_target_window_with_undo_(
                    raw, starts, cos, sin, gamma, kv, kt, state, st,
                    norm_eps=1e-6,
                )
                torch.testing.assert_close(state, reference_state, rtol=0, atol=0)
                torch.testing.assert_close(kv, reference_kv, rtol=0.01, atol=0.016)
                restore_indexer_cache(undo)
                torch.testing.assert_close(state, original_state, rtol=0, atol=0)
                torch.testing.assert_close(kv, original_kv, rtol=0, atol=0)

    def test_target_window_graph_replay_reads_updated_prefix(self):
        raw, starts, cos, sin, gamma, kv, kt, state, st = self._target_case([5])
        initial_kv = kv.clone()
        initial_state = state.clone()
        write_target_window_with_undo_(
            raw, starts, cos, sin, gamma, kv, kt, state, st, norm_eps=1e-6
        )
        kv.copy_(initial_kv)
        state.copy_(initial_state)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            undo = write_target_window_with_undo_(
                raw, starts, cos, sin, gamma, kv, kt, state, st, norm_eps=1e-6
            )
        kv.copy_(initial_kv)
        state.copy_(initial_state)
        starts.fill_(127)
        kt[:, 0], kt[:, 1] = kt[:, 1].clone(), kt[:, 0].clone()
        st[:, 0], st[:, 1] = st[:, 1].clone(), st[:, 0].clone()
        raw.copy_(torch.randn_like(raw.float()).to(torch.bfloat16))
        graph.replay()
        reference_kv = initial_kv.clone()
        reference_state = initial_state.clone()
        cu = torch.tensor([0, 4], dtype=torch.int32, device=raw.device)
        write_indexer_cache(
            raw, cu, starts, cos, sin, gamma, reference_kv, kt,
            reference_state, st, ratio=4, kv_tokens_per_block=128,
            state_tokens_per_block=128, norm_eps=1e-6,
            rope_is_token_aligned=True,
        )
        torch.testing.assert_close(state, reference_state, rtol=0, atol=0)
        torch.testing.assert_close(kv, reference_kv, rtol=0.01, atol=0.016)
        restore_indexer_cache(undo)
        torch.testing.assert_close(state, initial_state, rtol=0, atol=0)
        torch.testing.assert_close(kv, initial_kv, rtol=0, atol=0)

    def _case(self, positions):
        device = torch.device("cuda")
        batch = len(positions)
        raw = torch.randn(batch, 128, device=device).to(torch.bfloat16)
        starts = torch.tensor(positions, device=device, dtype=torch.int32)
        gamma = torch.randn(128, device=device).to(torch.bfloat16) * 0.1
        cosine = torch.randn(batch, 64, device=device).to(torch.bfloat16)
        sine = torch.randn(batch, 64, device=device).to(torch.bfloat16)
        kv_table = torch.tensor(
            [[1 + 2 * b, 2 + 2 * b] for b in range(batch)],
            dtype=torch.int32,
            device=device,
        )
        state_table = torch.tensor(
            [[1 + 2 * b, 2 + 2 * b] for b in range(batch)],
            dtype=torch.int32,
            device=device,
        )
        pool_blocks = 2 * batch + 1
        kv_pool = torch.zeros(
            pool_blocks, 32, 128, device=device, dtype=torch.bfloat16
        )
        state_pool = torch.zeros(
            pool_blocks, 8, 128, device=device, dtype=torch.float32
        )
        for b, position in enumerate(positions):
            if (position + 1) % 4 == 0:
                for previous in range(position - 3, position):
                    block = int(state_table[b, previous // 128])
                    key = torch.randn(128, device=device).to(torch.bfloat16)
                    state_pool[block, previous % 8] = key.float()
        return raw, starts, cosine, sine, gamma, kv_pool, kv_table, state_pool, state_table

    def _compare(self, tensors):
        raw, starts, cosine, sine, gamma, kv, kv_table, state, state_table = tensors
        reference_kv = kv.clone()
        reference_state = state.clone()
        cu = torch.arange(raw.shape[0] + 1, dtype=torch.int32, device=raw.device)
        write_indexer_cache(
            raw, cu, starts, cosine, sine, gamma, reference_kv, kv_table,
            reference_state, state_table, ratio=4, kv_tokens_per_block=128,
            state_tokens_per_block=128, norm_eps=1e-6,
            rope_is_token_aligned=True,
        )
        write_decode_key_(
            raw, starts, cosine, sine, gamma, kv, kv_table, state, state_table,
            norm_eps=1e-6,
        )
        torch.testing.assert_close(state, reference_state, rtol=0, atol=0)
        torch.testing.assert_close(kv, reference_kv, rtol=0.01, atol=0.016)

    def test_parity_across_partial_group_and_page_boundary(self):
        for positions in ([5], [7], [127], [128], [131], [7, 131]):
            with self.subTest(positions=positions):
                self._compare(self._case(positions))

    def test_graph_replay_reads_new_keys_and_positions(self):
        tensors = self._case([5])
        raw, starts, cosine, sine, gamma, kv, kv_table, state, state_table = tensors
        initial_kv = kv.clone()
        initial_state = state.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            write_decode_key_(
                raw, starts, cosine, sine, gamma, kv, kv_table, state,
                state_table, norm_eps=1e-6,
            )
        torch.cuda.current_stream().wait_stream(stream)
        kv.copy_(initial_kv)
        state.copy_(initial_state)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write_decode_key_(
                raw, starts, cosine, sine, gamma, kv, kv_table, state,
                state_table, norm_eps=1e-6,
            )
        kv.copy_(initial_kv)
        state.copy_(initial_state)
        raw.copy_(torch.randn_like(raw.float()).to(torch.bfloat16))
        starts.fill_(7)
        graph.replay()
        cu = torch.tensor([0, 1], dtype=torch.int32, device=raw.device)
        reference_kv = initial_kv.clone()
        reference_state = initial_state.clone()
        write_indexer_cache(
            raw, cu, starts, cosine, sine, gamma, reference_kv, kv_table,
            reference_state, state_table, ratio=4, kv_tokens_per_block=128,
            state_tokens_per_block=128, norm_eps=1e-6,
            rope_is_token_aligned=True,
        )
        torch.testing.assert_close(state, reference_state, rtol=0, atol=0)
        torch.testing.assert_close(kv, reference_kv, rtol=0.01, atol=0.016)

    def test_transactional_graph_replay_restores_mutated_slots(self):
        raw, starts, cosine, sine, gamma, kv, kv_table, state, state_table = (
            self._case([7, 5])
        )
        initial_kv = kv.clone()
        initial_state = state.clone()

        def write():
            return write_decode_key_with_undo_(
                raw, starts, cosine, sine, gamma, kv, kv_table, state,
                state_table, norm_eps=1e-6,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            write()
        torch.cuda.current_stream().wait_stream(stream)
        kv.copy_(initial_kv)
        state.copy_(initial_state)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            undo = write()
        kv.copy_(initial_kv)
        state.copy_(initial_state)
        starts.copy_(torch.tensor([5, 7], device=starts.device, dtype=starts.dtype))
        graph.replay()
        self.assertFalse(torch.equal(state, initial_state))
        restore_indexer_cache(undo)
        torch.testing.assert_close(state, initial_state, rtol=0, atol=0)
        torch.testing.assert_close(kv, initial_kv, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
