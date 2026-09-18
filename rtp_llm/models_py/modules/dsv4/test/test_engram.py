"""CPU-only checks for V4.1 history, checkpoint format, and gate semantics."""

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

# Load this self-contained module without importing platform-specific CUDA ops.
_spec = importlib.util.spec_from_file_location(
    "engram_cpu_test", Path(__file__).resolve().parents[1] / "engram.py"
)
engram = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engram)


def _config():
    return dict(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[1000, 1000],
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=32,
        engram_pad_token_id=2,
        engram_vocab_size=13,
    )


class EngramTest(unittest.TestCase):
    def setUp(self):
        self.layout = engram.EngramLayout(_config())
        self.hash = engram.NgramHashState(self.layout, token_map=list(range(32)))

    def test_hash_matches_scalar_reference_with_boundary_and_dead_tokens(self):
        windows = torch.tensor([[7, 6, 5, 4], [4, 3, -1, -1], [8, 7, 6, 5]])
        dead = torch.zeros_like(windows, dtype=torch.bool)
        dead[-1, 1] = True
        actual = self.hash(windows, dead)
        expected = torch.empty_like(actual)
        for row, tokens in enumerate(windows.tolist()):
            for layer in range(2):
                rolling, blocked = 0, False
                for shift, token in enumerate(tokens):
                    blocked |= token < 0 or bool(dead[row, shift])
                    value = 2 if blocked else token
                    rolling ^= value * int(self.hash.multipliers[layer, shift])
                    if shift:
                        for head in range(2):
                            col = (shift - 1) * 2 + head
                            expected[row, layer, col] = rolling % self.layout.primes[
                                layer
                            ][shift - 1][head] + int(self.layout.offsets[layer, col])
        torch.testing.assert_close(actual, expected)

    def test_cp_chunks_equal_full_sequence_hashing(self):
        ids = torch.arange(3, 27)
        complete = engram.make_token_windows(
            ids, torch.tensor([0, 24]), torch.tensor([[-1, -1, -1]])
        )
        full_hash = self.hash(complete)
        shards = []
        for start in range(0, 24, 6):
            previous = [
                int(ids[p]) if p >= 0 else -1 for p in range(start - 1, start - 4, -1)
            ]
            windows = engram.make_token_windows(
                ids[start : start + 6], torch.tensor([0, 6]), torch.tensor([previous])
            )
            shards.append(self.hash(windows))
        torch.testing.assert_close(torch.cat(shards), full_hash)

    def test_packed_requests_and_speculative_rollback(self):
        history = torch.tensor([[6, 5, 4], [12, 11, 10]])
        # Two unrelated requests, each with an accepted token and two candidates.
        windows = engram.make_token_windows(
            torch.tensor([7, 8, 9, 13, 14, 15]), torch.tensor([0, 3, 6]), history
        )
        self.assertEqual(windows[3].tolist(), [13, 12, 11, 10])
        first = self.hash(windows)
        # Reject candidate 8 and replace it by 20. No cached future token survives.
        retried = engram.make_token_windows(
            torch.tensor([20, 21]), torch.tensor([0, 2]), torch.tensor([[7, 6, 5]])
        )
        self.assertEqual(retried.tolist(), [[20, 7, 6, 5], [21, 20, 7, 6]])
        self.assertFalse(torch.equal(self.hash(retried)[0], first[1]))
        torch.testing.assert_close(self.hash(windows), first)

    def test_empty_request_does_not_steal_neighbor_history(self):
        windows = engram.make_token_windows(
            torch.tensor([4]),
            torch.tensor([0, 0, 1]),
            torch.tensor([[9, 8, 7], [3, 2, 1]]),
        )
        self.assertEqual(windows.tolist(), [[4, 3, 2, 1]])

    def test_host_lookup_dequantizes_fp8_bytes_and_e8m0_exponents(self):
        weight = torch.arange(128).reshape(4, 32).float().to(torch.float8_e4m3fn)
        scales = torch.tensor([[127], [128], [126], [129]], dtype=torch.uint8)
        lookup = engram.HostEngramEmbedding(weight, scales)
        ids = torch.tensor([[2, 0], [1, -1]])
        rows = lookup(ids, "cpu")
        self.assertEqual(rows.dtype, torch.bfloat16)
        torch.testing.assert_close(rows[0, 0], (weight[2].float() * 0.5).bfloat16())
        torch.testing.assert_close(rows[1, 0], (weight[1].float() * 2).bfloat16())
        self.assertEqual(rows[1, 1].count_nonzero(), 0)
        self.assertEqual(lookup.weight.data_ptr(), weight.data_ptr())
        self.assertEqual(lookup.weight.device.type, "cpu")

    def test_gate_has_per_lane_key_shared_value_and_exact_mask(self):
        torch.manual_seed(41)
        hidden = torch.randn(3, 4, 32).bfloat16()
        kv = torch.randn(3, 5 * 32).bfloat16()
        q, k = torch.randn(4, 32).bfloat16(), torch.randn(4, 32).bfloat16()
        output = engram.gated_engram_residual(
            hidden, kv, q, k, 1e-20, torch.tensor([True, False, True])
        )
        torch.testing.assert_close(output[1], hidden[1], rtol=0, atol=0)
        for token in (0, 2):
            for lane in range(4):
                h = hidden[token, lane].float()
                key = kv[token, lane * 32 : (lane + 1) * 32].float()
                value = kv[token, 128:].float()
                dot = (h * q[lane].float() * k[lane].float() * key).sum()
                dot /= torch.sqrt(h.square().mean() + 1e-20)
                dot /= torch.sqrt(key.square().mean() + 1e-20) * math.sqrt(32)
                gate = torch.sigmoid(torch.sign(dot) * dot.abs().clamp_min(1e-6).sqrt())
                expected = (h + gate * value).bfloat16()
                torch.testing.assert_close(output[token, lane], expected)

    def test_zero_gate_dot_uses_positive_clamped_sqrt(self):
        hidden = torch.zeros(1, 1, 32)
        kv = torch.cat((torch.zeros(1, 32), torch.ones(1, 32)), dim=1)
        output = engram.gated_engram_residual(
            hidden, kv, torch.ones(1, 32), torch.ones(1, 32), 1e-20
        )
        torch.testing.assert_close(
            output, torch.full_like(hidden, torch.sigmoid(torch.tensor(0.001)))
        )

    def test_mxfp8_reference_matches_independent_numpy_rounding(self):
        torch.manual_seed(7)
        x = torch.randn(7, 64).bfloat16()
        blocks = x.float().numpy().reshape(7, 2, 32)
        exponent = np.ceil(
            np.log2(np.maximum(np.abs(blocks).max(-1), np.finfo(np.float32).tiny) / 448)
        )
        scale = torch.from_numpy(np.exp2(exponent))[..., None]
        expected = (torch.from_numpy(blocks) / scale).to(
            torch.float8_e4m3fn
        ).float() * scale
        torch.testing.assert_close(
            engram.mxfp8_activation_reference(x), expected.reshape(7, 64).bfloat16()
        )

    def test_checkpoint_load_keeps_table_out_of_module_buffers(self):
        from safetensors.torch import save_file

        prefix = "layers.1.engram."
        tensors = {
            prefix + "embed.weight": torch.ones(1000, 32).to(torch.float8_e4m3fn),
            prefix + "embed.scale": torch.full((1000, 1), 127, dtype=torch.uint8),
            prefix + "wkv.weight": torch.ones(160, 192).to(torch.float8_e4m3fn),
            prefix + "wkv.scale": torch.full((5, 6), 127, dtype=torch.uint8),
            prefix + "q_weight": torch.ones(4, 32, dtype=torch.bfloat16),
            prefix + "k_weight": torch.ones(4, 32, dtype=torch.bfloat16),
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_file(tensors, root / "weights.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {key: "weights.safetensors" for key in tensors}}
                )
            )
            model = engram.Engram.from_checkpoint(
                {**_config(), "rms_norm_eps": 1e-20}, 1, folder, "cpu"
            )
            pointer = model.embed_tokens.weight.data_ptr()
            self.assertTrue(
                all("embed" not in name for name, _ in model.named_buffers())
            )
            model.to(device="cpu")
            self.assertEqual(model.embed_tokens.weight.data_ptr(), pointer)
            hashes = self.hash(torch.tensor([[7, 6, 5, 4]]))[:, 0]
            hidden = torch.ones(1, 4, 32, dtype=torch.bfloat16)
            output = model(hidden, hashes)
            self.assertEqual(output.shape, hidden.shape)
            self.assertTrue(bool(output.isfinite().all()))


if __name__ == "__main__":
    unittest.main()
