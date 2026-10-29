import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    ZeroCenteredFusedQKRMSNorm,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.main_qk_norm_rope import (
    try_main_qk_norm_rope,
)
from rtp_llm.ops import AttentionConfigs, RopeStyle
from rtp_llm.ops.fused_rope_kvcache_op import (
    FusedRopeAttnParams,
    FusedRopeKVCachePrefillOpQKVOut,
    _get_fused_rope_kvcache,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MainQkNormRopeTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 native math contract")
        torch.manual_seed(81002)

    def setup_case(self, heads, rows, prefix=0):
        cfg = AttentionConfigs()
        cfg.head_num, cfg.kv_head_num, cfg.size_per_head = heads, 1, 256
        cfg.kernel_tokens_per_block, cfg.max_seq_len = 128, 32768
        cfg.use_logn_attn = False
        cfg.rope_config.style = RopeStyle.Mrope
        cfg.rope_config.index_factor, cfg.rope_config.dim = 3, 64
        cfg.rope_config.base, cfg.rope_config.scale = 10000000.0, 1.0
        (
            cfg.rope_config.mrope_dim1,
            cfg.rope_config.mrope_dim2,
            cfg.rope_config.mrope_dim3,
        ) = (11, 11, 10)
        plain = AttentionConfigs()
        for name in (
            "head_num",
            "kv_head_num",
            "size_per_head",
            "kernel_tokens_per_block",
            "max_seq_len",
            "use_logn_attn",
        ):
            setattr(plain, name, getattr(cfg, name))
        plain.rope_config.style = RopeStyle.No
        positions = torch.randint(0, 32768, (rows, 3), device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, rows], device="cuda", dtype=torch.int32)
        length = torch.tensor([rows], dtype=torch.int32).pin_memory()
        prefixes = torch.tensor([prefix], dtype=torch.int32).pin_memory()
        pages = (prefix + rows + 127) // 128
        table = torch.arange(1, pages + 1, device="cuda", dtype=torch.int32).reshape(
            1, pages
        )
        offset = _get_fused_rope_kvcache().convert_offset_to_block_array(table)
        params = FusedRopeAttnParams(
            offset,
            None,
            None,
            positions.flatten(),
            cu,
            cu,
            length,
            prefixes,
            length,
            rows,
            prefix,
            rows + prefix,
            False,
            torch.bfloat16,
        )
        norm = ZeroCenteredFusedQKRMSNorm(
            torch.randn(256, device="cuda") * 0.07,
            torch.randn(256, device="cuda") * 0.07,
            heads,
            1,
            256,
        )
        source = torch.randn(
            rows, (heads + 2) * 256, device="cuda", dtype=torch.bfloat16
        )
        return (
            cfg,
            norm,
            source,
            positions,
            params,
            FusedRopeKVCachePrefillOpQKVOut(cfg),
            FusedRopeKVCachePrefillOpQKVOut(plain),
            pages + 1,
        )

    def test_native_output_heads_gamma_and_v_bytes(self):
        for heads in (3, 6, 12):
            for rows in (1, 4, 16, 4096):
                cfg, norm, source, positions, params, original, plain, _ = (
                    self.setup_case(heads, rows)
                )
                saved = source.clone()
                expected = original.forward(norm(source.clone()), None, params).clone()
                actual = plain.forward(
                    try_main_qk_norm_rope(
                        source,
                        positions,
                        norm,
                        cfg.rope_config,
                        position_base=params.max_prefix_length,
                    ),
                    None,
                    params,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(source, saved, rtol=0, atol=0)
                torch.testing.assert_close(
                    actual[:, -256:], source[:, -256:], rtol=0, atol=0
                )

    def test_writer_mutates_fp8_cache_and_preserves_every_scale(self):
        for rows in (4, 4096):
            cfg, norm, source, positions, params, original, plain, pages = (
                self.setup_case(12, rows, 127)
            )
            raw = (torch.randn(pages, 2, 1, 128, 256, device="cuda") * 0.2).to(
                torch.float8_e4m3fn
            )
            scales = torch.rand(pages, 256, device="cuda")
            old_cache = SimpleNamespace(
                kv_cache_base=raw.clone(), kv_scale_base=scales.clone()
            )
            new_cache = SimpleNamespace(
                kv_cache_base=raw.clone(), kv_scale_base=scales.clone()
            )
            # Exercise actual native cache writes, rather than only checking
            # equality of two unchanged caches from a QKV-only writer.
            expected = original.forward(norm(source.clone()), old_cache, params).clone()
            actual = plain.forward(
                try_main_qk_norm_rope(
                    source,
                    positions,
                    norm,
                    cfg.rope_config,
                    position_base=params.max_prefix_length,
                ),
                new_cache,
                params,
            )
            self.assertTrue(
                bool(
                    (
                        old_cache.kv_cache_base.view(torch.uint8)
                        != raw.view(torch.uint8)
                    ).any()
                )
            )
            for a, b in (
                (actual, expected),
                (
                    new_cache.kv_cache_base.view(torch.uint8),
                    old_cache.kv_cache_base.view(torch.uint8),
                ),
                (new_cache.kv_scale_base, old_cache.kv_scale_base),
            ):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_nonpositive_axes_preserve_native_prefix_fallback(self):
        for prefix in (0, 127):
            for coordinate in (0, -1):
                cfg, norm, source, positions, params, original, plain, pages = (
                    self.setup_case(12, 4, prefix)
                )
                values = torch.tensor(
                    [[0, 0, 0], [0, 11, 13], [11, 0, 13], [11, 13, 0]],
                    device="cuda",
                    dtype=torch.int32,
                )
                positions.copy_(torch.where(values == 0, coordinate, values))
                raw = torch.zeros(
                    pages, 2, 1, 128, 256, device="cuda", dtype=torch.float8_e4m3fn
                )
                sf = torch.ones(pages, 256, device="cuda")
                old_cache = SimpleNamespace(
                    kv_cache_base=raw.clone(), kv_scale_base=sf.clone()
                )
                new_cache = SimpleNamespace(
                    kv_cache_base=raw.clone(), kv_scale_base=sf.clone()
                )
                expected = original.forward(
                    norm(source.clone()), old_cache, params
                ).clone()
                prepared = try_main_qk_norm_rope(
                    source, positions, norm, cfg.rope_config, position_base=prefix
                )
                actual = plain.forward(prepared, new_cache, params)
                for a, reference in (
                    (actual, expected),
                    (
                        new_cache.kv_cache_base.view(torch.uint8),
                        old_cache.kv_cache_base.view(torch.uint8),
                    ),
                    (new_cache.kv_scale_base, old_cache.kv_scale_base),
                ):
                    torch.testing.assert_close(a, reference, rtol=0, atol=0)

    def test_graph_refreshes_positions_and_raw_gamma(self):
        cfg, norm, source, positions, params, original, plain, _ = self.setup_case(
            12, 4
        )

        def call():
            return plain.forward(
                try_main_qk_norm_rope(
                    source,
                    positions,
                    norm,
                    cfg.rope_config,
                    position_base=params.max_prefix_length,
                ),
                None,
                params,
            )

        for _ in range(3):
            call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = call()
        for _ in range(4):
            source.normal_()
            positions.copy_(
                torch.randint(
                    0, 32768, positions.shape, device="cuda", dtype=torch.int32
                )
            )
            norm.q_weight.normal_(0, 0.1)
            norm.k_weight.normal_(0, 0.1)
            graph.replay()
            expected = original.forward(norm(source.clone()), None, params)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
