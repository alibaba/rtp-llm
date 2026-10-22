"""Pinned cuLA checkpoints versus an independent FP32 recurrent reference."""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

setup_cutlass_import_path()
from cula.kda import chunk_kda

from rtp_llm.models_py.modules.factory.attention.cuda_kda_impl import (
    cula_kda_paged_prefill,
    plan_state_sequences,
)


def recurrent_reference(
    q, k, v, gate, beta, state, alog, bias, checkpoint_indices=None
):
    q = F.normalize(q.float(), dim=-1) * 128**-0.5
    k = F.normalize(k.float(), dim=-1)
    decay = (-5.0 * torch.sigmoid(alog.exp()[None, :, None] *
                                (gate.float() + bias.reshape(-1, 128)))).exp()
    beta = beta.float().sigmoid()
    state = state.clone()
    outputs, checkpoints = [], []
    for t in range(q.shape[0]):
        state = state * decay[t, :, :, None]
        correction = (state * k[t, :, :, None]).sum(dim=-2)
        state = state + k[t, :, :, None] * ((v[t].float() - correction) * beta[t, :, None])[:, None, :]
        outputs.append((q[t, :, :, None] * state).sum(dim=-2))
        if checkpoint_indices is None or t in checkpoint_indices:
            checkpoints.append(state.clone())
    return torch.stack(outputs), checkpoints


class NativeKdaCheckpointTest(unittest.TestCase):
    def test_bounded_gate_and_every_checkpoint(self):
        original_empty = torch.empty
        poisoned = []

        def empty_with_poison(*args, **kwargs):
            result = original_empty(*args, **kwargs)
            if result.ndim == 5 and result.shape[-2:] == (128, 128) and result.dtype == torch.float32:
                result.fill_(float("nan"))
                poisoned.append(result.shape)
            return result

        def checked_chunk(*args, **kwargs):
            initial = kwargs["initial_state"].clone()
            result = chunk_kda(*args, **kwargs)
            torch.testing.assert_close(kwargs["initial_state"], initial, rtol=0, atol=0)
            return result

        for heads, length, prefix in ((12, 1, 0), (12, 63, 0), (12, 64, 0),
                                      (24, 65, 0), (12, 131, 0), (24, 129, 31),
                                      (12, 401, 0)):
            with self.subTest(heads=heads, length=length, prefix=prefix):
                torch.manual_seed(5700 + heads + length)
                shape = (length, heads, 128)
                q, k, v, gate = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
                beta = torch.randn(length, heads, device="cuda", dtype=torch.bfloat16)
                alog = torch.randn(heads, device="cuda") * 0.1
                bias = torch.randn(heads * 128, device="cuda") * 0.01
                initial = torch.randn(heads, 128, 128, device="cuda") * 0.02
                blocks = (prefix + length + 63) // 64
                table = list(range(1, blocks + 1))
                pool = torch.full((blocks + 3, heads, 128, 128), -77.0, device="cuda")
                if prefix:
                    pool[table[(prefix - 1) // 64]].copy_(initial)
                else:
                    initial.zero_()
                before = pool.clone()
                expected, states = recurrent_reference(q, k, v, gate, beta, initial, alog, bias)
                with patch("torch.empty", side_effect=empty_with_poison), patch("cula.kda.chunk_kda", side_effect=checked_chunk):
                    actual = cula_kda_paged_prefill(
                        q, k, v, gate, beta, alog, bias, -5.0, pool,
                        [0, length], [prefix], [table], 64,
                    )
                # Preserve the existing bounded-KDA operator tolerances.
                torch.testing.assert_close(actual.float(), expected, rtol=0.03, atol=0.003)
                written = set()
                for segment in plan_state_sequences([0, length], [prefix], [table], 64)[0].segments:
                    torch.testing.assert_close(pool[segment.cache_block], states[segment.end - 1], rtol=0.03, atol=0.003)
                    self.assertTrue(torch.isfinite(pool[segment.cache_block]).all().item())
                    written.add(segment.cache_block)
                untouched = [i for i in range(pool.shape[0]) if i not in written]
                torch.testing.assert_close(pool[untouched], before[untouched], rtol=0, atol=0)
        self.assertTrue(poisoned)

    def test_successive_chunks_consume_published_checkpoint_state(self):
        # Resetting the prefix state, reading the wrong physical block, or
        # publishing only the last checkpoint must break this continuation.
        torch.manual_seed(6107)
        heads, length, block_size = 12, 401, 64
        shape = (length, heads, 128)
        q, k, v = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        # Keep enough history to distinguish continuation from a fresh start.
        gate = (torch.randn(shape, device="cuda") * 0.1 - 4.0).to(torch.bfloat16)
        beta = torch.randn(length, heads, device="cuda", dtype=torch.bfloat16)
        alog = torch.randn(heads, device="cuda") * 0.1
        bias = torch.randn(heads * 128, device="cuda") * 0.01
        for boundary in (128, 256):
            q[boundary].copy_(k[boundary - 1])

        initial = torch.zeros(heads, 128, 128, device="cuda")
        expected, states = recurrent_reference(q, k, v, gate, beta, initial, alog, bias)
        # Logical pages deliberately do not match their physical pool rows.
        table = [6, 2, 8, 4, 1, 7, 3]
        checkpoint_rows = (
            (64, 6),
            (128, 2),
            (192, 8),
            (256, 4),
            (320, 1),
            (384, 7),
            (401, 3),
        )
        pool = torch.full((10, heads, 128, 128), -77.0, device="cuda")
        outputs = []
        start = 0
        for end in (128, 256, 401):
            with self.subTest(start=start, end=end):
                before = pool.clone()
                if start:
                    # This fixture must fail if a continuation starts at zero.
                    # The oracle here uses no production planning or cuLA code.
                    reset_output, _ = recurrent_reference(
                        q[start : start + 1],
                        k[start : start + 1],
                        v[start : start + 1],
                        gate[start : start + 1],
                        beta[start : start + 1],
                        initial,
                        alog,
                        bias,
                    )
                    self.assertGreater(
                        (expected[start] - reset_output[0]).abs().max().item(), 0.01
                    )
                actual = cula_kda_paged_prefill(
                    q[start:end],
                    k[start:end],
                    v[start:end],
                    gate[start:end],
                    beta[start:end],
                    alog,
                    bias,
                    -5.0,
                    pool,
                    [0, end - start],
                    [start],
                    [table],
                    block_size,
                )
                outputs.append(actual)
                torch.testing.assert_close(
                    actual.float(), expected[start:end], rtol=0.03, atol=0.003
                )
                written = set()
                for position, row in checkpoint_rows:
                    if start < position <= end:
                        torch.testing.assert_close(
                            pool[row], states[position - 1], rtol=0.03, atol=0.003
                        )
                        self.assertTrue(torch.isfinite(pool[row]).all().item())
                        written.add(row)
                # In particular, the consumed prefix checkpoint must survive.
                untouched = [row for row in range(pool.shape[0]) if row not in written]
                torch.testing.assert_close(
                    pool[untouched], before[untouched], rtol=0, atol=0
                )
            start = end
        torch.testing.assert_close(
            torch.cat(outputs).float(), expected, rtol=0.03, atol=0.003
        )

    def test_real_geometry_sp_padding_matches_chunked_checkpoint_continuation(self):
        torch.manual_seed(6121)
        heads, length, boundary, block_size = 12, 33311, 32768, 4096
        physical_length = length + 1
        tail_length = length - boundary
        shape = (physical_length, heads, 128)
        q, k, v = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        gate = (torch.randn(shape, device="cuda") * 0.1 - 4.0).to(torch.bfloat16)
        beta = torch.randn(physical_length, heads, device="cuda", dtype=torch.bfloat16)
        alog = torch.randn(heads, device="cuda") * 0.1
        bias = torch.randn(heads * 128, device="cuda") * 0.01
        # Make the first continuation output sensitive to the prefix state.
        q[boundary].copy_(k[boundary - 1])
        # SP padding is a virtual request. None of its inputs may enter KDA.
        for inputs in (q, k, v, gate, beta):
            inputs[-1].fill_(float("nan"))

        # Nine logical pages use unrelated physical rows. Row 0 is reserved
        # for virtual requests, and spare physical rows must never change.
        table = [6, 2, 10, 4, 1, 11, 3, 8, 5]
        virtual_table = [0] * len(table)
        state_words = heads * 128 * 128
        # RTP stores BF16 q/k/v convolution history after each FP32 state.
        # Three history tokens for kernel size 4 leave this many FP32 words.
        conv_words = 3 * 3 * heads * 128 * 2 // 4
        block_stride = state_words + conv_words
        pool_shape = (14, heads, 128, 128)
        pool_strides = (block_stride, 128 * 128, 128, 1)

        def make_pool():
            storage = torch.full(
                (14, block_stride), -77.0, device="cuda", dtype=torch.float32
            )
            return storage, storage.as_strided(pool_shape, pool_strides)

        def assert_cache_writes(storage, before, written):
            untouched = [row for row in range(14) if row not in written]
            torch.testing.assert_close(
                storage[untouched], before[untouched], rtol=0, atol=0
            )
            torch.testing.assert_close(
                storage[:, state_words:], before[:, state_words:], rtol=0, atol=0
            )

        baseline_storage, baseline_pool = make_pool()
        pristine_storage = baseline_storage.clone()
        # 33311 real tokens plus one SP dummy produce two sequences. The real
        # request therefore uses bounded groups rather than the sole-sequence
        # packed path used by the aligned 32768-token first chunk below.
        baseline = cula_kda_paged_prefill(
            q,
            k,
            v,
            gate,
            beta,
            alog,
            bias,
            -5.0,
            baseline_pool,
            [0, length, physical_length],
            [0, 0],
            [table, virtual_table],
            block_size,
        )
        self.assertTrue(torch.isfinite(baseline[:length]).all().item())
        torch.testing.assert_close(
            baseline[length], torch.zeros_like(baseline[length]), rtol=0, atol=0
        )
        self.assertTrue(torch.isfinite(baseline_pool[table]).all().item())
        assert_cache_writes(baseline_storage, pristine_storage, set(table))

        chunk_storage, chunk_pool = make_pool()
        first = cula_kda_paged_prefill(
            q[:boundary],
            k[:boundary],
            v[:boundary],
            gate[:boundary],
            beta[:boundary],
            alog,
            bias,
            -5.0,
            chunk_pool,
            [0, boundary],
            [0],
            [table],
            block_size,
        )
        torch.testing.assert_close(first, baseline[:boundary], rtol=0.03, atol=0.003)
        torch.testing.assert_close(
            chunk_pool[table[:8]], baseline_pool[table[:8]], rtol=0.03, atol=0.003
        )
        assert_cache_writes(chunk_storage, pristine_storage, set(table[:8]))

        before_tail = chunk_storage.clone()
        tail = cula_kda_paged_prefill(
            q[boundary:],
            k[boundary:],
            v[boundary:],
            gate[boundary:],
            beta[boundary:],
            alog,
            bias,
            -5.0,
            chunk_pool,
            [0, tail_length, tail_length + 1],
            [boundary, 0],
            [table, virtual_table],
            block_size,
        )
        torch.testing.assert_close(
            tail[:tail_length], baseline[boundary:length], rtol=0.03, atol=0.003
        )
        torch.testing.assert_close(
            tail[tail_length], torch.zeros_like(tail[tail_length]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            chunk_pool[table], baseline_pool[table], rtol=0.03, atol=0.003
        )
        # Consuming the eighth checkpoint must not overwrite it, the dummy
        # row, other physical rows, or convolution-history bytes in any block.
        assert_cache_writes(chunk_storage, before_tail, {table[8]})

        # Independent FP32 recurrence checks only the 543-token continuation,
        # conditional on cuLA's baseline checkpoint at token 32768. It does not
        # establish independent FP32 correctness of the preceding 32768 tokens.
        # Save only the last state instead of retaining 543 full checkpoints.
        expected_tail, expected_states = recurrent_reference(
            q[boundary:length],
            k[boundary:length],
            v[boundary:length],
            gate[boundary:length],
            beta[boundary:length],
            baseline_pool[table[7]],
            alog,
            bias,
            checkpoint_indices={tail_length - 1},
        )
        torch.testing.assert_close(
            tail[:tail_length].float(), expected_tail, rtol=0.03, atol=0.003
        )
        torch.testing.assert_close(
            chunk_pool[table[8]], expected_states[0], rtol=0.03, atol=0.003
        )

        # Run the same CUDA continuation with its published initial state
        # erased: the fixture must detect a lost or ignored prefix checkpoint.
        reset_storage = before_tail.clone()
        reset_pool = reset_storage.as_strided(pool_shape, pool_strides)
        reset_pool[table[7]].zero_()
        reset_tail = cula_kda_paged_prefill(
            q[boundary:],
            k[boundary:],
            v[boundary:],
            gate[boundary:],
            beta[boundary:],
            alog,
            bias,
            -5.0,
            reset_pool,
            [0, tail_length, tail_length + 1],
            [boundary, 0],
            [table, virtual_table],
            block_size,
        )
        self.assertGreater(
            (tail[0].float() - reset_tail[0].float()).abs().max().item(), 0.01
        )


if __name__ == "__main__":
    unittest.main()
