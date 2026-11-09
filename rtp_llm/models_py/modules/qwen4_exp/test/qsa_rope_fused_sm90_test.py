"""SM90 dispatch of the fused QSA indexer text-RoPE kernel.

The fused Triton text kernel (``_text_rope``, one launch producing the
``[rows, 64]`` bf16 cos/sin pair) must engage on SM90 exactly like it does
on SM120 once the step plan passes the shared cached ``inv_freq`` table:

* the production step-plan text call sites launch ``_text_rope`` and drop
  the eager aten ``arange/pow/div/cos/sin/cat`` chain from the window;
* fused and eager cos/sin agree bit-exactly (bf16) for Base and equal-axis
  text MRoPE transports, including position 0, 1, large positions and
  non-unit scales;
* the shared ``inv_freq`` cache returns one constant table per geometry;
* multimodal 3-axis MRoPE transports keep the eager MRoPE builder.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime import qsa_step_plan
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_context import (
    Qwen4ExpQSARuntimeContext,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    build_base_rope,
    build_qsa_rope,
    build_qsa_rope_from_logical_positions,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.rope_triton import (
    _text_rope,
    try_build_text_rope,
)
from rtp_llm.ops import RopeStyle


def _variant_total(fn):
    """Compiled variant count across devices (triton>=3.6 ``device_caches``)."""
    total = 0
    caches = getattr(fn, "device_caches", None) or {}
    for cache in caches.values():
        entry = cache[0] if isinstance(cache, tuple) else cache
        total += len(entry)
    return total

MODULE = "rtp_llm.models_py.modules.qwen4_exp.qsa.rope_triton"
# Positions incl. 0, 1, scale-multiples, power-of-two boundaries, 1e5-class
# large positions and the 2**24 + 1 float32-rounding boundary.
POSITIONS = (0, 1, 2, 3, 4, 5, 63, 64, 65, 127, 128, 4095, 4096, 65535, 100000, 100003, 262143, 16777217)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaRopeFusedSM90Test(unittest.TestCase):
    D = 128
    RATIO = 4
    PAGE = 8

    def setUp(self):
        torch.manual_seed(271)
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 fast path")
        # Concrete device index: the step-plan build compares tensor
        # devices against this value, and cuda (no index) != cuda:0.
        self.device = torch.device("cuda", torch.cuda.current_device())
        qsa_step_plan.clear_step_plan_cache()
        # Hermetic strict flag: the profiled step-plan build needs the
        # production default (validation demoted), like qsa_step_plan_test.
        strict_off = patch.dict(
            "os.environ", {"RTP_LLM_QSA_STRICT_VALIDATION": "0"}
        )
        strict_off.start()
        self.addCleanup(strict_off.stop)
        self.indexer = SimpleNamespace(
            head_dim=self.D,
            n_heads=2,
            compress_ratio=self.RATIO,
            token_budget=8,
            k_norm_gamma=torch.zeros(self.D, device=self.device),
            norm_eps=1e-6,
        )

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

    def frequency(self, config):
        return float(config.base) ** (
            -2.0 * torch.arange(32, device=self.device, dtype=torch.float32) / 64
        )

    def rope_pair(self, positions, config, *, fused):
        args = dict(
            token_count=positions.numel(), dtype=torch.bfloat16, device=positions.device
        )
        frequency = self.frequency(config)
        if fused:
            self.assertIsNotNone(
                try_build_text_rope(
                    positions,
                    config,
                    config.style,
                    dtype=torch.bfloat16,
                    device=positions.device,
                    inv_freq=frequency,
                ),
                "the fused text rope kernel declined SM90",
            )
            return build_qsa_rope_from_logical_positions(
                positions, config, inv_freq=frequency, **args
            )
        with patch(MODULE + ".try_build_text_rope", return_value=None):
            return build_qsa_rope_from_logical_positions(positions, config, **args)

    def test_fused_text_rope_matches_eager_bytes_on_sm90(self):
        positions = torch.tensor(POSITIONS, device="cuda", dtype=torch.int64)
        for style in ("Base", "Mrope"):
            for base, scale in ((1000000.0, 1.0), (10000.0, 3.0), (500000.0, 0.5)):
                with self.subTest(style=style, base=base, scale=scale):
                    config = self.config(style, base=base, scale=scale)
                    eager = self.rope_pair(positions, config, fused=False)
                    actual = self.rope_pair(positions, config, fused=True)
                    for a, b in zip(actual, eager):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_fused_rope_reuses_one_variant_across_row_counts(self):
        # Serving prefill plans hit a new token count on nearly every step
        # (draft tail included); a per-ROWS specialization pays a ~100ms
        # Triton JIT compile per new shape (the v18 draft-tail windows
        # compiled the fused rope six times and inflated TTFB p50 by
        # ~240ms).  One warmed variant must serve every row count.
        config = self.config("Base", index_factor=1)
        frequency = self.frequency(config)
        for count in (33, 5, 129, 1024, 1525):
            positions = torch.arange(count, device=self.device, dtype=torch.int32)
            fused = try_build_text_rope(
                positions,
                config,
                config.style,
                dtype=torch.bfloat16,
                device=self.device,
                inv_freq=frequency,
            )
            self.assertIsNotNone(
                fused, f"the fused text rope declined row count {count}"
            )
        variants = _variant_total(_text_rope)
        for count in (65, 777, 2221, 4093):
            positions = torch.arange(count, device=self.device, dtype=torch.int32)
            try_build_text_rope(
                positions,
                config,
                config.style,
                dtype=torch.bfloat16,
                device=self.device,
                inv_freq=frequency,
            )
        self.assertEqual(
            _variant_total(_text_rope),
            variants,
            "unseen row counts must reuse the warmed fused-rope variant",
        )

    def test_base_transport_matches_eager_chain(self):
        for factor in (1, 3):
            config = self.config("Base", index_factor=factor)
            frequency = self.frequency(config)
            positions = torch.arange(16, device="cuda", dtype=torch.int32)
            transported = positions.repeat_interleave(factor)
            kwargs = dict(
                token_count=16, dtype=torch.bfloat16, device=positions.device
            )
            for prefix in (0, 127, 4093, 32749):
                transported.copy_((positions + prefix).repeat_interleave(factor))
                eager = build_base_rope(transported, config, **kwargs)
                actual = build_base_rope(
                    transported, config, inv_freq=frequency, **kwargs
                )
                for a, b in zip(actual, eager):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            if factor == 3:
                # The fused dispatch keeps the equal-axes transport check.
                transported[1] += 1
                with self.assertRaisesRegex(ValueError, "three axes"):
                    build_base_rope(
                        transported, config, inv_freq=frequency, **kwargs
                    )

    # ---- production step-plan call path ----

    def _inputs(self, prefixes, lengths, positions, *, device):
        batch = len(lengths)
        input_lengths = torch.tensor(lengths, dtype=torch.int32)
        prefix_lengths = torch.tensor(prefixes, dtype=torch.int32)
        cu = torch.zeros(batch + 1, dtype=torch.int32, device=device)
        cu[1:] = input_lengths.cumsum(0)
        cu_kv = torch.zeros(batch + 1, dtype=torch.int32, device=device)
        cu_kv[1:] = (prefix_lengths + input_lengths).cumsum(0)
        kv_table = torch.arange(1, batch * 3 + 1, dtype=torch.int32, device=device).view(
            batch, 3
        )

        def make():
            return SimpleNamespace(
                is_prefill=True,
                is_target_verify=False,
                is_cuda_graph=False,
                is_s_padded=False,
                context_parallel_info=None,
                cache_store_inputs=None,
                input_lengths=input_lengths,
                prefix_lengths=prefix_lengths,
                sequence_lengths=torch.empty(0, dtype=torch.int32),
                cu_seqlens_device=cu,
                cu_kv_seqlens_device=cu_kv,
                combo_position_ids=positions.reshape(-1),
                kv_cache_kernel_block_id_device=kv_table,
                kv_cache_block_id_device=kv_table.clone(),
            )

        return make

    def _context(
        self, prefixes, lengths, *, is_mtp_draft=False, positions=None, layer=0
    ):
        device = self.device
        if positions is None:
            rows = []
            for prefix, length in zip(prefixes, lengths):
                rows.extend([pos, pos, pos] for pos in range(prefix, prefix + length))
            positions = torch.tensor(rows, dtype=torch.int32)
        factory = self._inputs(prefixes, lengths, positions, device=device)
        return Qwen4ExpQSARuntimeContext(
            main_cache=SimpleNamespace(tag="full"),
            main_inputs=factory(),
            indexer_kv_cache=SimpleNamespace(
                tag=INDEXER_KV_TAG,
                kv_cache_base=torch.zeros(
                    8 + layer,
                    (self.PAGE // self.RATIO) * self.D * 2,
                    dtype=torch.uint8,
                    device=device,
                ),
                seq_size_per_block=self.PAGE,
            ),
            indexer_kv_inputs=factory(),
            indexer_state_cache=SimpleNamespace(
                tag=INDEXER_STATE_TAG,
                kv_cache_base=torch.zeros(
                    8 + layer,
                    2 * self.RATIO * self.D,
                    dtype=torch.float32,
                    device=device,
                ),
                seq_size_per_block=self.PAGE,
            ),
            indexer_state_inputs=factory(),
            is_mtp_draft=is_mtp_draft,
        )

    def _geometry(
        self, context, token_count, rope_config, *, draft=False, draft_prefix_reuse=False
    ):
        return context._draft_incremental_prefill_geometry(
            indexer=self.indexer,
            token_count=token_count,
            device=self.device,
            rope_config=rope_config,
            draft=draft,
            draft_prefix_reuse=draft_prefix_reuse,
        )

    def _profiled_names(self, build):
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            build()
            torch.cuda.synchronize()
        return [event.name for event in prof.events()]

    def test_step_plan_text_path_launches_the_fused_kernel(self):
        config = self.config("Base")
        # Warm the shared inv_freq table and the Triton compile with one
        # full build; the profiled build below uses fresh inputs objects,
        # so it is a plan-cache miss that must rebuild fused-only.
        warm = self._context([8], [5])
        self._geometry(warm, 5, config)
        context = self._context([8], [5])
        names = self._profiled_names(lambda: self._geometry(context, 5, config))
        self.assertTrue(
            any("_text_rope" in name for name in names),
            "the fused text rope kernel did not launch on SM90",
        )
        for forbidden in ("aten::pow", "aten::cos", "aten::sin"):
            self.assertNotIn(forbidden, names)

    def test_step_plan_text_path_matches_the_eager_rebuild(self):
        # Target prefix reuse with Base RoPE: both the current-position rope
        # and the block-start rope take the fused kernel.
        config = self.config("Base")
        fused = self._geometry(self._context([8], [5]), 5, config)
        with patch(MODULE + ".try_build_text_rope", return_value=None):
            eager = self._geometry(self._context([8], [5]), 5, config)
        for name in ("current_cos", "current_sin", "rope_cos", "rope_sin"):
            torch.testing.assert_close(
                getattr(fused, name), getattr(eager, name), rtol=0, atol=0
            )
        # MTP draft prefix reuse (equal-axis text MRoPE): the block-start
        # rope goes through build_qsa_rope_from_logical_positions fused.
        mrope = self.config("Mrope")
        fused_draft = self._geometry(
            self._context([8], [4], is_mtp_draft=True),
            4,
            mrope,
            draft_prefix_reuse=True,
        )
        with patch(MODULE + ".try_build_text_rope", return_value=None):
            eager_draft = self._geometry(
                self._context([8], [4], is_mtp_draft=True),
                4,
                mrope,
                draft_prefix_reuse=True,
            )
        for name in ("current_cos", "current_sin", "rope_cos", "rope_sin"):
            torch.testing.assert_close(
                getattr(fused_draft, name),
                getattr(eager_draft, name),
                rtol=0,
                atol=0,
            )

    def test_multimodal_target_mrope_keeps_the_eager_builder(self):
        config = self.config("Mrope")
        # Image-like head rows with unequal T/H/W axes and a consecutive
        # equal-axis text tail ending at the maximum position.
        rows = [[8, 1, 2], [9, 3, 4], [10, 5, 6], [11, 11, 11], [12, 12, 12]]
        positions = torch.tensor(rows, dtype=torch.int32)
        warm = self._context([8], [5], positions=positions.clone())
        self._geometry(warm, 5, config)
        context = self._context([8], [5], positions=positions)
        names = self._profiled_names(lambda: self._geometry(context, 5, config))
        self.assertFalse(
            any("_text_rope" in name for name in names),
            "multimodal MRoPE positions must keep the eager builder",
        )
        self.assertIn("aten::cos", names)
        self.assertIn("aten::sin", names)

    def test_inv_freq_cache_returns_one_shared_table(self):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import cached_inv_freq

        config = self.config()
        device = self.device
        first = cached_inv_freq(config, device)
        self.assertIs(cached_inv_freq(config, device), first)
        expected = float(config.base) ** (
            -2.0 * torch.arange(32, device=device, dtype=torch.float32) / 64
        )
        self.assertTrue(torch.equal(first, expected))
        self.assertEqual(tuple(first.shape), (32,))
        self.assertEqual(first.dtype, torch.float32)
        self.assertTrue(first.is_cuda)
        self.assertTrue(first.is_contiguous())
        # A different geometry caches a separate table.
        other = cached_inv_freq(self.config(base=10000.0), device)
        self.assertIsNot(other, first)
        self.assertTrue(
            torch.equal(
                other,
                10000.0
                ** (-2.0 * torch.arange(32, device=device, dtype=torch.float32) / 64),
            )
        )


if __name__ == "__main__":
    unittest.main()
