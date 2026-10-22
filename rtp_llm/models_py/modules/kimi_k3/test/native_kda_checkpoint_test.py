"""Shared cuLA checkpoints for K3 and KimiLinear versus an FP32 reference."""
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


def recurrent_reference(q, k, v, gate, beta, state, alog, bias, lower_bound=-5.0):
    q = F.normalize(q.float(), dim=-1) * 128**-0.5
    k = F.normalize(k.float(), dim=-1)
    gate_input = gate.float() + bias.reshape(-1, 128)
    if lower_bound is None:
        log_decay = -alog.exp()[None, :, None] * F.softplus(gate_input)
    else:
        log_decay = lower_bound * torch.sigmoid(
            alog.exp()[None, :, None] * gate_input
        )
    decay = log_decay.exp()
    beta = beta.float().sigmoid()
    state = state.clone()
    outputs, checkpoints = [], []
    for t in range(q.shape[0]):
        state = state * decay[t, :, :, None]
        correction = (state * k[t, :, :, None]).sum(dim=-2)
        state = state + k[t, :, :, None] * ((v[t].float() - correction) * beta[t, :, None])[:, None, :]
        outputs.append((q[t, :, :, None] * state).sum(dim=-2))
        checkpoints.append(state.clone())
    return torch.stack(outputs), checkpoints


class NativeKdaCheckpointTest(unittest.TestCase):
    def test_gate_and_every_checkpoint(self):
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

        cases = ((12, 1, 0), (12, 63, 0), (12, 64, 0), (24, 65, 0),
                 (12, 131, 0), (24, 129, 31), (12, 401, 0))
        for lower_bound, heads, length, prefix in (
            (bound, *case) for bound in (-5.0, None) for case in cases
        ):
            with self.subTest(lower_bound=lower_bound, heads=heads, length=length, prefix=prefix):
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
                expected, states = recurrent_reference(
                    q, k, v, gate, beta, initial, alog, bias, lower_bound
                )
                with patch("torch.empty", side_effect=empty_with_poison), patch("cula.kda.chunk_kda", side_effect=checked_chunk) as cula_call:
                    actual = cula_kda_paged_prefill(
                        q, k, v, gate, beta, alog, bias, lower_bound, pool,
                        [0, length], [prefix], [table], 64,
                    )
                self.assertEqual(bool(cula_call.call_count), lower_bound is not None)
                # Preserve the existing KDA operator tolerances for both models.
                torch.testing.assert_close(actual.float(), expected, rtol=0.03, atol=0.003)
                written = set()
                for segment in plan_state_sequences([0, length], [prefix], [table], 64)[0].segments:
                    torch.testing.assert_close(pool[segment.cache_block], states[segment.end - 1], rtol=0.03, atol=0.003)
                    self.assertTrue(torch.isfinite(pool[segment.cache_block]).all().item())
                    written.add(segment.cache_block)
                untouched = [i for i in range(pool.shape[0]) if i not in written]
                torch.testing.assert_close(pool[untouched], before[untouched], rtol=0, atol=0)
        self.assertTrue(poisoned)


if __name__ == "__main__":
    unittest.main()
