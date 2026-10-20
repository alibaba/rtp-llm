import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    Qwen4ExpQSAIndexer,
    apply_partial_rope,
    build_base_rope,
    build_qsa_rope_from_logical_positions,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.rope_triton import (
    try_apply_partial_rope,
    try_build_text_rope,
)
from rtp_llm.models_py.modules.qwen4_exp.test import qsa_runtime_test

MODULE = "rtp_llm.models_py.modules.qwen4_exp.qsa.rope_triton"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class IndexerRoPETritonTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(271)
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 fast path")

    def config(self, style="Mrope", **kwargs):
        config = dict(
            style=style,
            dim=64,
            base=1000000.0,
            scale=1.0,
            index_factor=3,
            mrope_interleaved=True,
            mrope_dim1=16,
            mrope_dim2=8,
            mrope_dim3=8,
            indexer_is_neox_style=True,
        )
        config.update(kwargs)
        return SimpleNamespace(**config)

    def rope(self, positions, config, fused, dtype=torch.bfloat16):
        args = dict(token_count=positions.numel(), dtype=dtype, device=positions.device)
        frequency = float(config.base) ** (
            -2.0 * torch.arange(32, device=positions.device, dtype=torch.float32) / 64
        )
        if fused:
            self.assertIsNotNone(
                try_build_text_rope(
                    positions,
                    config,
                    config.style,
                    dtype=dtype,
                    device=positions.device,
                    inv_freq=frequency,
                )
            )
            return build_qsa_rope_from_logical_positions(
                positions, config, inv_freq=frequency, **args
            )
        with patch(MODULE + ".try_build_text_rope", return_value=None):
            return build_qsa_rope_from_logical_positions(positions, config, **args)

    def test_canonical_positions_match_eager_bytes(self):
        for style in ("Base", "Mrope"):
            for rows in (1, 3, 4, 8, 16, 33, 4093, 4096, 12288):
                positions = torch.randint(0, 32769, (rows,), device="cuda")
                for base, scale in ((1000000.0, 1.0), (10000.0, 3.0), (500000.0, 0.5)):
                    with self.subTest(style=style, rows=rows, base=base, scale=scale):
                        config = self.config(style, base=base, scale=scale)
                        expected = self.rope(positions, config, False)
                        actual = self.rope(positions, config, True)
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_strides_large_and_negative_positions(self):
        positions = torch.tensor(
            [-1, 0, 3, 127, 128, 32768, 131071, 16777217], device="cuda"
        )
        for value in (positions, positions.repeat_interleave(2)[::2].to(torch.int32)):
            for a, b in zip(
                self.rope(value, self.config(), True),
                self.rope(value, self.config(), False),
            ):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_base_transport_frequency_keeps_axis_checks_and_live_positions(self):
        for factor in (1, 3):
            config = self.config("Base", index_factor=factor, base=10000000.0)
            frequency = float(config.base) ** (
                -2.0 * torch.arange(32, device="cuda", dtype=torch.float32) / 64
            )
            positions = torch.arange(16, device="cuda", dtype=torch.int32)
            transported = positions.repeat_interleave(factor)
            kwargs = dict(token_count=16, dtype=torch.bfloat16, device=positions.device)
            for prefix in (0, 127, 4093, 32749):
                transported.copy_((positions + prefix).repeat_interleave(factor))
                expected = build_base_rope(transported, config, **kwargs)
                actual = build_base_rope(
                    transported, config, inv_freq=frequency, **kwargs
                )
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            # Equality is already checked by the native draft graph owner;
            # the explicit opt-out is unchanged from the old Graph path.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = build_base_rope(
                    transported,
                    config,
                    inv_freq=frequency,
                    validate_position_axes=False,
                    **kwargs
                )
            for prefix in (3, 128, 32700):
                transported.copy_((positions + prefix).repeat_interleave(factor))
                graph.replay()
                expected = build_base_rope(transported, config, **kwargs)
                for a, b in zip(captured, expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            if factor == 3:
                transported[1] += 1
                with self.assertRaisesRegex(ValueError, "three axes"):
                    build_base_rope(transported, config, inv_freq=frequency, **kwargs)

    def test_rotation_preserves_bf16_products_and_tail(self):
        for batch, query, heads in ((1, 1, 32), (1, 4, 32), (4, 4, 32), (7, 4, 8)):
            x = torch.randn(
                batch, query, heads, 128, device="cuda", dtype=torch.bfloat16
            )
            cos = torch.randn(batch, query, 1, 64, device="cuda", dtype=torch.bfloat16)
            sin = torch.randn_like(cos)
            self.assertIsNotNone(try_apply_partial_rope(x, cos, sin))
            with patch(MODULE + ".try_apply_partial_rope", return_value=None):
                expected = apply_partial_rope(x, cos, sin)
            actual = apply_partial_rope(x, cos, sin)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(actual[..., 64:], x[..., 64:]))

    def test_graph_refreshes_positions_and_values(self):
        positions = torch.arange(16, device="cuda")
        x = torch.randn(4, 4, 32, 128, device="cuda", dtype=torch.bfloat16)
        config = self.config()
        frequency = float(config.base) ** (
            -2.0 * torch.arange(32, device="cuda", dtype=torch.float32) / 64
        )
        for _ in range(3):
            cos, sin = self.rope(positions, config, True)
            apply_partial_rope(x, cos.view(4, 4, 1, 64), sin.view(4, 4, 1, 64))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            cos, sin = build_qsa_rope_from_logical_positions(
                positions,
                config,
                token_count=16,
                dtype=x.dtype,
                device=x.device,
                inv_freq=frequency,
            )
            actual = apply_partial_rope(x, cos.view(4, 4, 1, 64), sin.view(4, 4, 1, 64))
        for prefix in (127, 4093, 32749):
            positions.copy_(torch.arange(prefix, prefix + 16, device="cuda"))
            x.normal_()
            graph.replay()
            cos, sin = self.rope(positions, config, False)
            with patch(MODULE + ".try_apply_partial_rope", return_value=None):
                expected = apply_partial_rope(
                    x, cos.view(4, 4, 1, 64), sin.view(4, 4, 1, 64)
                )
            self.assertTrue(torch.equal(actual, expected))

    def test_frequency_is_model_owned_and_config_changes_fall_back(self):
        config = self.config()
        indexer = Qwen4ExpQSAIndexer(
            torch.randn(384, 32, device="cuda", dtype=torch.bfloat16),
            torch.zeros(128, device="cuda", dtype=torch.bfloat16),
            torch.zeros(128, device="cuda", dtype=torch.bfloat16),
            n_heads=2,
            kv_heads=1,
            head_dim=128,
            token_budget=2048,
            compress_ratio=4,
            norm_eps=1e-6,
            rope_config=config,
        )
        expected = float(config.base) ** (
            -2.0 * torch.arange(32, device="cuda", dtype=torch.float32) / 64
        )
        self.assertTrue(torch.equal(indexer.rope_inv_freq(config), expected))
        self.assertNotIn("_rope_inv_freq", indexer.state_dict())
        self.assertNotIn("_score_weight_buffer", indexer.state_dict())
        for rows in (1, 3, 4, 8, 16, 32, 128):
            weights = indexer.score_weights(rows, expected.device)
            reference = torch.full(
                (rows, 2), 1.0 / 128**0.5, dtype=torch.float32, device=expected.device
            )
            self.assertTrue(torch.equal(weights, reference))
        self.assertIsNone(indexer.score_weights(129, expected.device))
        indexer.n_heads = 3
        self.assertIsNone(indexer.score_weights(4, expected.device))
        self.assertIsNone(indexer.rope_inv_freq(self.config(base=10000.0)))

    def test_indexer_rmsnorm_matches_native_reduction_and_strides(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.norm import exact_head_rms_norm
        from rtp_llm.models_py.modules.qwen4_exp.qsa.norm_triton import (
            try_indexer_rmsnorm,
        )

        for rows in (1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 127, 128):
            for dtype in (torch.bfloat16, torch.float32):
                gamma = torch.randn(128, dtype=dtype, device="cuda") * 0.1
                for x in (
                    torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16),
                    torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)[
                        :, :128
                    ],
                ):
                    self.assertIsNotNone(try_indexer_rmsnorm(x, gamma, 1e-6))
                    with patch(
                        "rtp_llm.models_py.modules.qwen4_exp.qsa.norm_triton.try_indexer_rmsnorm",
                        return_value=None,
                    ):
                        expected = exact_head_rms_norm(x, gamma, 1e-6)
                    self.assertTrue(
                        torch.equal(exact_head_rms_norm(x, gamma, 1e-6), expected),
                        (rows, dtype),
                    )
        for rows in (1, 3, 4, 8, 16, 32):
            x = torch.randn(rows, 640, device="cuda", dtype=torch.bfloat16)[
                :, :512
            ].reshape(rows, 4, 128)
            gamma = torch.full((128,), 0.003, device="cuda", dtype=torch.bfloat16)
            with patch(
                "rtp_llm.models_py.modules.qwen4_exp.qsa.norm_triton.try_indexer_rmsnorm",
                return_value=None,
            ):
                expected = exact_head_rms_norm(x, gamma, 1e-6)
            self.assertTrue(
                torch.equal(exact_head_rms_norm(x, gamma, 1e-6), expected), rows
            )

    def test_indexer_norm_graph_uses_fresh_values_and_gains(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.norm import exact_head_rms_norm
        from rtp_llm.models_py.modules.qwen4_exp.qsa.norm_triton import (
            try_indexer_rmsnorm,
        )

        x = torch.randn(4, 4, 4, 128, device="cuda", dtype=torch.bfloat16)
        gamma = torch.randn(128, device="cuda", dtype=torch.float32) * 0.1
        for _ in range(3):
            exact_head_rms_norm(x, gamma, 1e-6)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = exact_head_rms_norm(x, gamma, 1e-6)
        for _ in range(3):
            x.normal_()
            gamma.normal_(0, 0.1)
            graph.replay()
            with patch(
                "rtp_llm.models_py.modules.qwen4_exp.qsa.norm_triton.try_indexer_rmsnorm",
                return_value=None,
            ):
                expected = exact_head_rms_norm(x, gamma, 1e-6)
            self.assertTrue(torch.equal(actual, expected))
        self.assertIsNone(try_indexer_rmsnorm(x.float(), gamma, 1e-6))
        gamma.requires_grad_(True)
        self.assertIsNone(try_indexer_rmsnorm(x, gamma, 1e-6))

    def test_target_selection_cache_and_undo_match_eager(self):
        config = self.config()
        indexer = Qwen4ExpQSAIndexer(
            torch.randn(640, 32, device="cuda", dtype=torch.bfloat16),
            torch.zeros(128, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.1,
            n_heads=4,
            kv_heads=1,
            head_dim=128,
            token_budget=2048,
            compress_ratio=4,
            norm_eps=1e-6,
            rope_config=config,
        )
        for prefix in (7, 127, 129, 4093, 32749):
            columns = (prefix + 4 + 127) // 128
            kv = torch.randn(
                columns + 1, 32 * 128, device="cuda", dtype=torch.bfloat16
            ).view(torch.uint8)
            state = torch.randn(
                columns + 1, 8 * 128, device="cuda", dtype=torch.float32
            )
            q = torch.randn(4, 4, 128, device="cuda", dtype=torch.bfloat16)
            raw = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
            table = torch.arange(1, columns + 1, dtype=torch.int32, device="cuda").view(
                1, -1
            )
            fixture = qsa_runtime_test.Qwen4ExpQSARuntimeTest()
            fixture.D = 128
            fixture.KV_TOKENS_PER_BLOCK = fixture.STATE_TOKENS_PER_BLOCK = 128
            fixture.setUp()

            def context():
                fixture.kv_base, fixture.state_base = kv.clone(), state.clone()
                value = fixture._target_verify_context(committed_length=prefix)
                value.indexer_kv_inputs.kv_cache_kernel_block_id_device = table
                value.indexer_state_inputs.kv_cache_block_id_device = table
                return value

            eager, fused = context(), context()
            with patch(MODULE + ".try_build_text_rope", return_value=None), patch(
                MODULE + ".try_apply_partial_rope", return_value=None
            ), patch(
                "rtp_llm.models_py.modules.qwen4_exp.qsa.target_writer_triton."
                "try_write_target_window",
                return_value=None,
            ):
                expected = eager.select_target_verify_tokens(
                    q, raw, indexer=indexer, rope_config=config
                )
            actual = fused.select_target_verify_tokens(
                q, raw, indexer=indexer, rope_config=config
            )
            self.assertTrue(torch.equal(actual, expected), prefix)
            self.assertTrue(
                torch.equal(
                    eager.indexer_kv_cache.kv_cache_base,
                    fused.indexer_kv_cache.kv_cache_base,
                ),
                prefix,
            )
            self.assertTrue(
                torch.equal(
                    eager.indexer_state_cache.kv_cache_base,
                    fused.indexer_state_cache.kv_cache_base,
                ),
                prefix,
            )
            fused.rollback_side_cache()
            self.assertTrue(
                torch.equal(fused.indexer_kv_cache.kv_cache_base, kv), prefix
            )
            self.assertTrue(
                torch.equal(fused.indexer_state_cache.kv_cache_base, state), prefix
            )

    def test_invalid_layout_keeps_validation_and_unsupported_falls_back(self):
        positions = torch.arange(4, device="cuda")
        for updates in (
            dict(indexer_is_neox_style=False),
            dict(scale=0),
            dict(mrope_interleaved=False),
            dict(mrope_dim2=31),
        ):
            config = self.config(**updates)
            with self.assertRaises(ValueError):
                build_qsa_rope_from_logical_positions(
                    positions,
                    config,
                    token_count=4,
                    dtype=torch.bfloat16,
                    device=positions.device,
                )
        config = self.config()
        self.assertIsNone(
            try_build_text_rope(
                positions,
                config,
                config.style,
                dtype=torch.float32,
                device=positions.device,
            )
        )
        with patch(MODULE + ".try_build_text_rope", return_value=None):
            expected = build_qsa_rope_from_logical_positions(
                positions,
                config,
                token_count=4,
                dtype=torch.float32,
                device=positions.device,
            )
        actual = build_qsa_rope_from_logical_positions(
            positions,
            config,
            token_count=4,
            dtype=torch.float32,
            device=positions.device,
        )
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))


if __name__ == "__main__":
    unittest.main()
