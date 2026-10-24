import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import (
    restore_indexer_cache,
    write_indexer_cache,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode_triton import (
    write_decode_key_,
    write_decode_key_with_undo_,
    write_draft_window_with_undo_,
    write_target_window_with_undo_,
)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class IndexerDecodeTritonTest(unittest.TestCase):
    PAGE_SIZE = 128

    @unittest.skipIf(
        torch.version.hip is not None,
        "ROCm multi-token target/draft writers are unsupported",
    )
    def test_ragged_draft_window_graph_replay_and_rollback(self):
        raw, prefixes, cos, sin, gamma, kv, kt, state, st = self._target_case(
            [7, self.PAGE_SIZE - 1]
        )
        raw, cos, sin = raw[:5].contiguous(), cos[:5].contiguous(), sin[:5].contiguous()
        lengths = torch.tensor([1, 4], dtype=torch.int32, device=raw.device)
        cu = torch.tensor([0, 1, 5], dtype=torch.int32, device=raw.device)
        original_kv, original_state = kv.clone(), state.clone()

        def write():
            return write_draft_window_with_undo_(
                raw,
                cu,
                prefixes,
                lengths,
                cos,
                sin,
                gamma,
                kv,
                kt,
                state,
                st,
                norm_eps=1e-6,
            )

        write()
        reference_kv, reference_state = original_kv.clone(), original_state.clone()
        write_indexer_cache(
            raw,
            cu,
            prefixes,
            cos,
            sin,
            gamma,
            reference_kv,
            kt,
            reference_state,
            st,
            ratio=4,
            kv_tokens_per_block=self.PAGE_SIZE,
            state_tokens_per_block=self.PAGE_SIZE,
            norm_eps=1e-6,
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
        prefixes.copy_(
            torch.tensor([5, self.PAGE_SIZE - 3], dtype=torch.int32, device=raw.device)
        )
        raw.copy_(torch.randn_like(raw.float()).bfloat16())
        kt[:, 0], kt[:, 1] = kt[:, 1].clone(), kt[:, 0].clone()
        st[:, 0], st[:, 1] = st[:, 1].clone(), st[:, 0].clone()
        graph.replay()

        reference_kv, reference_state = original_kv.clone(), original_state.clone()
        write_indexer_cache(
            raw,
            cu,
            prefixes,
            cos,
            sin,
            gamma,
            reference_kv,
            kt,
            reference_state,
            st,
            ratio=4,
            kv_tokens_per_block=self.PAGE_SIZE,
            state_tokens_per_block=self.PAGE_SIZE,
            norm_eps=1e-6,
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
            pool_blocks, self.PAGE_SIZE // 4, 128, device=device, dtype=torch.float32
        ).to(torch.bfloat16)
        state = (
            torch.randn(pool_blocks, 8, 128, device=device, dtype=torch.float32)
            .to(torch.bfloat16)
            .float()
        )
        return raw, starts, cosine, sine, gamma, kv, table, state, table.clone()

    @unittest.skipIf(
        torch.version.hip is not None,
        "ROCm multi-token target/draft writers are unsupported",
    )
    def test_target_window_parity_and_rollback(self):
        for query_len in (1, 2, 3, 4):
            for prefixes in (
                [4],
                [5],
                [7],
                [self.PAGE_SIZE - 1],
                [self.PAGE_SIZE],
                [5, self.PAGE_SIZE - 1],
            ):
                with self.subTest(query_len=query_len, prefixes=prefixes):
                    raw, starts, cos, sin, gamma, kv, kt, state, st = self._target_case(
                        prefixes
                    )
                    batch = len(prefixes)
                    raw = (
                        raw.view(batch, 4, 128)[:, :query_len]
                        .reshape(-1, 128)
                        .contiguous()
                    )
                    cos = (
                        cos.view(batch, 4, 64)[:, :query_len]
                        .reshape(-1, 64)
                        .contiguous()
                    )
                    sin = (
                        sin.view(batch, 4, 64)[:, :query_len]
                        .reshape(-1, 64)
                        .contiguous()
                    )
                    original_kv = kv.clone()
                    original_state = state.clone()
                    reference_kv = kv.clone()
                    reference_state = state.clone()
                    cu = torch.arange(
                        0,
                        (batch + 1) * query_len,
                        query_len,
                        dtype=torch.int32,
                        device=raw.device,
                    )
                    write_indexer_cache(
                        raw,
                        cu,
                        starts,
                        cos,
                        sin,
                        gamma,
                        reference_kv,
                        kt,
                        reference_state,
                        st,
                        ratio=4,
                        kv_tokens_per_block=self.PAGE_SIZE,
                        state_tokens_per_block=self.PAGE_SIZE,
                        norm_eps=1e-6,
                        rope_is_token_aligned=True,
                    )
                    undo = write_target_window_with_undo_(
                        raw,
                        starts,
                        cos,
                        sin,
                        gamma,
                        kv,
                        kt,
                        state,
                        st,
                        norm_eps=1e-6,
                    )
                    torch.testing.assert_close(state, reference_state, rtol=0, atol=0)
                    torch.testing.assert_close(kv, reference_kv, rtol=0.01, atol=0.016)
                    restore_indexer_cache(undo)
                    torch.testing.assert_close(state, original_state, rtol=0, atol=0)
                    torch.testing.assert_close(kv, original_kv, rtol=0, atol=0)

    @unittest.skipIf(
        torch.version.hip is not None,
        "ROCm multi-token target/draft writers are unsupported",
    )
    def test_target_window_graph_replay_reads_updated_prefix(self):
        for query_len in (1, 2, 3, 4):
            with self.subTest(query_len=query_len):
                raw, starts, cos, sin, gamma, kv, kt, state, st = self._target_case(
                    [5, self.PAGE_SIZE - 2]
                )
                raw = raw.view(2, 4, 128)[:, :query_len].reshape(-1, 128).contiguous()
                cos = cos.view(2, 4, 64)[:, :query_len].reshape(-1, 64).contiguous()
                sin = sin.view(2, 4, 64)[:, :query_len].reshape(-1, 64).contiguous()
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
                for prefixes in (
                    [self.PAGE_SIZE - 1, 5],
                    [self.PAGE_SIZE, 4],
                    [self.PAGE_SIZE - 3, self.PAGE_SIZE - 1],
                ):
                    kv.copy_(initial_kv)
                    state.copy_(initial_state)
                    starts.copy_(
                        torch.tensor(prefixes, dtype=torch.int32, device=raw.device)
                    )
                    kt[:, 0], kt[:, 1] = kt[:, 1].clone(), kt[:, 0].clone()
                    st[:, 0], st[:, 1] = st[:, 1].clone(), st[:, 0].clone()
                    raw.copy_(torch.randn_like(raw.float()).to(torch.bfloat16))
                    graph.replay()
                    reference_kv = initial_kv.clone()
                    reference_state = initial_state.clone()
                    cu = torch.arange(
                        0,
                        3 * query_len,
                        query_len,
                        dtype=torch.int32,
                        device=raw.device,
                    )
                    write_indexer_cache(
                        raw,
                        cu,
                        starts,
                        cos,
                        sin,
                        gamma,
                        reference_kv,
                        kt,
                        reference_state,
                        st,
                        ratio=4,
                        kv_tokens_per_block=self.PAGE_SIZE,
                        state_tokens_per_block=self.PAGE_SIZE,
                        norm_eps=1e-6,
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
            pool_blocks, self.PAGE_SIZE // 4, 128, device=device, dtype=torch.bfloat16
        )
        state_pool = torch.zeros(
            pool_blocks, 8, 128, device=device, dtype=torch.float32
        )
        for b, position in enumerate(positions):
            if (position + 1) % 4 == 0:
                for previous in range(position - 3, position):
                    block = int(state_table[b, previous // self.PAGE_SIZE])
                    key = torch.randn(128, device=device).to(torch.bfloat16)
                    state_pool[block, previous % 8] = key.float()
        return (
            raw,
            starts,
            cosine,
            sine,
            gamma,
            kv_pool,
            kv_table,
            state_pool,
            state_table,
        )

    def _compare(self, tensors):
        raw, starts, cosine, sine, gamma, kv, kv_table, state, state_table = tensors
        reference_kv = kv.clone()
        reference_state = state.clone()
        cu = torch.arange(raw.shape[0] + 1, dtype=torch.int32, device=raw.device)
        write_indexer_cache(
            raw,
            cu,
            starts,
            cosine,
            sine,
            gamma,
            reference_kv,
            kv_table,
            reference_state,
            state_table,
            ratio=4,
            kv_tokens_per_block=self.PAGE_SIZE,
            state_tokens_per_block=self.PAGE_SIZE,
            norm_eps=1e-6,
            rope_is_token_aligned=True,
        )
        write_decode_key_(
            raw,
            starts,
            cosine,
            sine,
            gamma,
            kv,
            kv_table,
            state,
            state_table,
            norm_eps=1e-6,
        )
        torch.testing.assert_close(state, reference_state, rtol=0, atol=0)
        torch.testing.assert_close(kv, reference_kv, rtol=0.01, atol=0.016)

    def test_parity_across_partial_group_and_page_boundary(self):
        for positions in (
            [5],
            [7],
            [self.PAGE_SIZE - 1],
            [self.PAGE_SIZE],
            [self.PAGE_SIZE + 3],
            [7, self.PAGE_SIZE + 3],
        ):
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
                raw,
                starts,
                cosine,
                sine,
                gamma,
                kv,
                kv_table,
                state,
                state_table,
                norm_eps=1e-6,
            )
        torch.cuda.current_stream().wait_stream(stream)
        kv.copy_(initial_kv)
        state.copy_(initial_state)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write_decode_key_(
                raw,
                starts,
                cosine,
                sine,
                gamma,
                kv,
                kv_table,
                state,
                state_table,
                norm_eps=1e-6,
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
            raw,
            cu,
            starts,
            cosine,
            sine,
            gamma,
            reference_kv,
            kv_table,
            reference_state,
            state_table,
            ratio=4,
            kv_tokens_per_block=self.PAGE_SIZE,
            state_tokens_per_block=self.PAGE_SIZE,
            norm_eps=1e-6,
            rope_is_token_aligned=True,
        )
        torch.testing.assert_close(state, reference_state, rtol=0, atol=0)
        torch.testing.assert_close(kv, reference_kv, rtol=0.01, atol=0.016)

    def test_transactional_graph_replay_restores_mutated_slots(self):
        raw, starts, cosine, sine, gamma, kv, kv_table, state, state_table = self._case(
            [7, 5]
        )
        initial_kv = kv.clone()
        initial_state = state.clone()

        def write():
            return write_decode_key_with_undo_(
                raw,
                starts,
                cosine,
                sine,
                gamma,
                kv,
                kv_table,
                state,
                state_table,
                norm_eps=1e-6,
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

    @unittest.skipUnless(torch.version.hip is not None, "ROCm contract")
    def test_rocm_single_token_graph_mask_page_and_undo(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode_triton import (
            is_supported,
        )

        for batch in (1, 16):
            with self.subTest(batch=batch):
                tensors = self._case([127] * batch)
                raw, starts, cos, sin, gamma, kv, kt, state, st = tensors
                kv.normal_()
                state.normal_()
                old_kv, old_state = kv.clone(), state.clone()
                valid = torch.ones(batch, dtype=torch.bool, device=raw.device)
                if batch > 1:
                    valid[-1] = False
                    kt[-2].zero_()
                    st[-2].zero_()
                # The real device passes; an unvalidated architecture/page
                # must not be opened by this qualified ROCm path.
                args = (*tensors[:5], kv, kt, state, st)
                self.assertTrue(
                    is_supported(
                        *args,
                        kv_tokens_per_block=128,
                        state_tokens_per_block=128,
                        ratio=4
                    )
                )
                self.assertFalse(
                    is_supported(
                        *args,
                        kv_tokens_per_block=64,
                        state_tokens_per_block=64,
                        ratio=4
                    )
                )
                with patch.object(
                    torch.cuda,
                    "get_device_properties",
                    return_value=SimpleNamespace(gcnArchName="gfx950"),
                ):
                    self.assertFalse(
                        is_supported(
                            *args,
                            kv_tokens_per_block=128,
                            state_tokens_per_block=128,
                            ratio=4
                        )
                    )

                def write():
                    write_decode_key_(
                        raw,
                        starts,
                        cos,
                        sin,
                        gamma,
                        kv,
                        kt,
                        state,
                        st,
                        norm_eps=1e-6,
                        valid=valid,
                    )

                write()
                kv.copy_(old_kv)
                state.copy_(old_state)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    write()
                for position in (127, 128, 131):
                    kv.copy_(old_kv)
                    state.copy_(old_state)
                    starts.fill_(position)
                    raw.normal_()
                    cos.normal_()
                    sin.normal_()
                    kt.copy_(kt.flip(1))
                    st.copy_(st.flip(1))
                    graph.replay()
                    expected_kv, expected_state = old_kv.clone(), old_state.clone()
                    for row in range(batch):
                        if not bool(valid[row]) or not bool(kt[row].any()):
                            continue
                        write_indexer_cache(
                            raw[row : row + 1],
                            torch.tensor([0, 1], device=raw.device, dtype=torch.int32),
                            starts[row : row + 1],
                            cos[row : row + 1],
                            sin[row : row + 1],
                            gamma,
                            expected_kv,
                            kt[row : row + 1],
                            expected_state,
                            st[row : row + 1],
                            ratio=4,
                            kv_tokens_per_block=128,
                            state_tokens_per_block=128,
                            norm_eps=1e-6,
                            rope_is_token_aligned=True,
                        )
                    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
                    torch.testing.assert_close(kv, expected_kv, rtol=0.01, atol=0.016)
                    state_changed = (expected_state != old_state).any(-1)
                    kv_changed = (expected_kv != old_kv).any(-1)
                    torch.testing.assert_close(
                        state[~state_changed], old_state[~state_changed], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        kv[~kv_changed], old_kv[~kv_changed], rtol=0, atol=0
                    )
                # Transactional entry point uses fixed-size Torch snapshots on
                # HIP; capture/replay/rollback must preserve the same slots.
                if batch > 1:
                    kt[-2] = torch.tensor(
                        [batch * 2 - 3, batch * 2 - 2], device=kt.device
                    )
                    st[-2].copy_(kt[-2])
                kv.copy_(old_kv)
                state.copy_(old_state)

                def transactional():
                    return write_decode_key_with_undo_(
                        raw, starts, cos, sin, gamma, kv, kt, state, st, norm_eps=1e-6
                    )

                transactional()
                kv.copy_(old_kv)
                state.copy_(old_state)
                undo_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(undo_graph):
                    undo = transactional()
                kv.copy_(old_kv)
                state.copy_(old_state)
                starts.fill_(127)
                raw.normal_()
                kt.copy_(kt.flip(1))
                st.copy_(st.flip(1))
                undo_graph.replay()
                restore_indexer_cache(undo)
                torch.testing.assert_close(state, old_state, rtol=0, atol=0)
                torch.testing.assert_close(kv, old_kv, rtol=0, atol=0)


@unittest.skipIf(
    torch.version.hip is not None,
    "ROCm side-cache writer is qualified only for page128",
)
class IndexerDecode64TokenTritonTest(IndexerDecodeTritonTest):
    PAGE_SIZE = 64


if __name__ == "__main__":
    unittest.main()
