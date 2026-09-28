"""Check long single-sequence cuLA Prefill checkpoint publication."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SOURCE = Path(__file__).resolve().parents[1] / "native_kda.py"
SPEC = importlib.util.spec_from_file_location("native_kda_under_test", SOURCE)
NATIVE_KDA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = NATIVE_KDA
SPEC.loader.exec_module(NATIVE_KDA)


class PackedCulaPrefillTest(unittest.TestCase):
    def test_long_aligned_sequence_uses_one_call_and_publishes_each_page(self):
        block_size = 64
        length = 6 * block_size + 3
        shape = (length, 1, 128)
        q = torch.zeros(shape, dtype=torch.bfloat16)
        k = torch.zeros_like(q)
        v = torch.arange(length, dtype=torch.bfloat16)[:, None, None].expand(shape)
        g = torch.zeros_like(q)
        beta = torch.zeros((length, 1), dtype=torch.bfloat16)
        state_cache = torch.zeros((8, 1, 128, 128), dtype=torch.float32)
        segments = tuple(
            NATIVE_KDA.StateSegment(
                start=page * block_size,
                end=min((page + 1) * block_size, length),
                cache_block=page + 1,
            ) for page in range(7)
        )
        sequence = NATIVE_KDA.StateSequence(None, segments)
        calls = []

        def cula(q_arg, k_arg, v_arg, g_arg, beta_arg, **kwargs):
            calls.append((q_arg.shape[1], kwargs["checkpoint_states"].shape[1]))
            self.assertEqual(kwargs["checkpoint_interval"], block_size)
            self.assertIsNone(kwargs["checkpoint_offsets"])
            self.assertEqual(kwargs["initial_state"].dtype, torch.float32)
            if not calls or len(calls) == 1:
                self.assertTrue(torch.count_nonzero(kwargs["initial_state"]) == 0)
            for page, checkpoint in enumerate(kwargs["checkpoint_states"][0]):
                checkpoint.fill_(page + 1)
            return v_arg, None, kwargs["checkpoint_states"]

        output = NATIVE_KDA._cula_paged_prefill(
            cula, q, k, v, g, beta,
            torch.zeros(1), torch.zeros(1), -20.0,
            state_cache, (sequence,), block_size,
        )
        self.assertEqual(calls, [(length, len(segments))])
        self.assertTrue(torch.equal(output, v))
        for page in range(7):
            self.assertTrue(torch.all(state_cache[page + 1] == page + 1))

    def test_unaligned_reused_prefix_keeps_segmented_calls(self):
        block_size = 64
        segments = (NATIVE_KDA.StateSegment(0, 3, 2),) + tuple(
            NATIVE_KDA.StateSegment(3 + page * block_size,
                                    3 + (page + 1) * block_size, page + 3)
            for page in range(5)
        )
        length = segments[-1].end
        shape = (length, 1, 128)
        inputs = [torch.zeros(shape, dtype=torch.bfloat16) for _ in range(4)]
        beta = torch.zeros((length, 1), dtype=torch.bfloat16)
        cache = torch.zeros((8, 1, 128, 128), dtype=torch.float32)
        cache[1].fill_(0.25)
        calls = []

        def cula(q_arg, k_arg, v_arg, g_arg, beta_arg, **kwargs):
            calls.append((q_arg.shape[1], kwargs["checkpoint_states"].shape[1]))
            kwargs["checkpoint_states"].fill_(0.25)
            return v_arg, None, kwargs["checkpoint_states"]

        NATIVE_KDA._cula_paged_prefill(
            cula, *inputs, beta, torch.zeros(1), torch.zeros(1), -5.0,
            cache, (NATIVE_KDA.StateSequence(1, segments),), block_size,
        )
        self.assertEqual(calls, [(3, 1), (256, 4), (64, 1)])


if __name__ == "__main__":
    unittest.main()
