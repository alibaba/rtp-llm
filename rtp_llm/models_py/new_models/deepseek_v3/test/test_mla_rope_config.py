import math
import unittest

import torch

from rtp_llm.models_py.new_models.deepseek_v3.rope_cache import build_rope_cache


class MlaRopeConfigTest(unittest.TestCase):
    def test_local_rope_cache_preserves_reference_numerics(self):
        max_seq_len = 32
        rope_dim = 8
        base = 10000.0
        plain = build_rope_cache(
            {"qk_rope_head_dim": rope_dim, "rope_theta": base},
            max_seq_len,
            torch.device("cpu"),
        )
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=inv_freq.dtype), inv_freq)
        self.assertTrue(
            torch.equal(
                plain,
                torch.cat([freqs.cos(), freqs.sin()], dim=-1),
            )
        )

        factor = 8.0
        original_max = 16
        beta_fast = 32.0
        beta_slow = 1.0
        mscale = 1.0
        mscale_all_dim = 1.0
        yarn = build_rope_cache(
            {
                "qk_rope_head_dim": rope_dim,
                "rope_theta": base,
                "rope_scaling": {
                    "factor": factor,
                    "original_max_position_embeddings": original_max,
                    "beta_fast": beta_fast,
                    "beta_slow": beta_slow,
                    "mscale": mscale,
                    "mscale_all_dim": mscale_all_dim,
                },
            },
            max_seq_len,
            torch.device("cpu"),
        )
        freq_extra = 1.0 / (
            base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
        )
        freq_inter = freq_extra / factor

        def correction_dim(num_rotations):
            return (
                rope_dim * math.log(original_max / (num_rotations * 2 * math.pi))
            ) / (2 * math.log(base))

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), rope_dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp(
            (torch.arange(rope_dim // 2, dtype=torch.float32) - low) / (high - low),
            0,
            1,
        )
        mask = 1.0 - ramp
        yarn_inv_freq = freq_inter * (1 - mask) + freq_extra * mask
        yarn_freqs = torch.outer(
            torch.arange(max_seq_len, dtype=torch.float32),
            yarn_inv_freq,
        )

        def yarn_mscale(value):
            if factor <= 1:
                return 1.0
            return 0.1 * value * math.log(factor) + 1.0

        scale = yarn_mscale(mscale) / yarn_mscale(mscale_all_dim)
        self.assertTrue(
            torch.equal(
                yarn,
                torch.cat(
                    [
                        yarn_freqs.cos() * scale,
                        yarn_freqs.sin() * scale,
                    ],
                    dim=-1,
                ),
            )
        )

        rope_parameters_yarn = build_rope_cache(
            {
                "qk_rope_head_dim": rope_dim,
                "rope_parameters": {
                    "rope_type": "yarn",
                    "rope_theta": base,
                    "factor": factor,
                    "original_max_position_embeddings": original_max,
                    "beta_fast": beta_fast,
                    "beta_slow": beta_slow,
                    "mscale": mscale,
                    "mscale_all_dim": mscale_all_dim,
                },
            },
            max_seq_len,
            torch.device("cpu"),
        )
        self.assertTrue(torch.equal(rope_parameters_yarn, yarn))

        zero_all_dim = build_rope_cache(
            {
                "qk_rope_head_dim": rope_dim,
                "rope_theta": base,
                "rope_scaling": {
                    "factor": factor,
                    "original_max_position_embeddings": original_max,
                    "beta_fast": beta_fast,
                    "beta_slow": beta_slow,
                    "mscale": mscale,
                    "mscale_all_dim": 0.0,
                },
            },
            max_seq_len,
            torch.device("cpu"),
        )
        zero_all_dim_scale = yarn_mscale(mscale)
        torch.testing.assert_close(
            zero_all_dim,
            torch.cat(
                [
                    yarn_freqs.cos() * zero_all_dim_scale,
                    yarn_freqs.sin() * zero_all_dim_scale,
                ],
                dim=-1,
            ),
        )

    def test_rope_cache_rejects_ambiguous_or_incomplete_scaling(self):
        base_config = {"qk_rope_head_dim": 8}
        with self.assertRaisesRegex(ValueError, "unsupported.*rope_type"):
            build_rope_cache(
                {
                    **base_config,
                    "rope_parameters": {
                        "rope_type": "linear",
                        "factor": 2.0,
                    },
                },
                32,
                torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "missing keys"):
            build_rope_cache(
                {
                    **base_config,
                    "rope_parameters": {
                        "rope_type": "yarn",
                        "factor": 2.0,
                    },
                },
                32,
                torch.device("cpu"),
            )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            build_rope_cache(
                {
                    **base_config,
                    "rope_scaling": {
                        "factor": 2.0,
                        "original_max_position_embeddings": 16,
                        "mscale": 1.0,
                        "mscale_all_dim": 1.0,
                    },
                    "rope_parameters": {
                        "rope_type": "yarn",
                        "factor": 4.0,
                        "original_max_position_embeddings": 16,
                        "mscale": 1.0,
                        "mscale_all_dim": 1.0,
                    },
                },
                32,
                torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
