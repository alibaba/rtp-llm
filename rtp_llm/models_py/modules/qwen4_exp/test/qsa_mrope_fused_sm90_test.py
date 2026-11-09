"""SM90 fused dispatch of the QSA indexer MRoPE text rope.

``build_qsa_rope``'s MRoPE branch reaches the existing fused text-rope
kernel (``_text_rope``, one launch producing the ``[rows, 64]`` bf16
cos/sin pair) once the transported rows are proven text-shaped:

* equal-axis text transports produce fused cos/sin that are byte-equal
  (``torch.equal``) to the eager ``build_interleaved_mrope`` chain across
  section configurations (including zero-width axes), bases, scales and
  int32/int64 transports;
* multimodal rows never reach the fused kernel and keep the eager builder;
* the equal-axes proof never synchronizes the device: host-resident rows
  compare their axes on the CPU, device-resident rows without a caller
  proof keep the eager builder, and ``validate_position_axes=False``
  carries a caller-side proof (the plan window's suffix verdict);
* the production target-MRoPE plan window launches the fused kernel for
  both rope pairs of a text step and drops the eager aten
  ``pow/cos/sin/cat`` chain, while a multimodal step keeps the eager
  builder for both pairs;
* ``cached_mrope_inv_freq`` returns one constant table per
  (base, sections, device) geometry with the exact eager construction.
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
    build_interleaved_mrope,
    build_qsa_rope,
    cached_inv_freq,
    cached_mrope_inv_freq,
    mrope_equal_axis_verdict,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.rope_triton import (
    try_build_text_rope,
)

MODULE = "rtp_llm.models_py.modules.qwen4_exp.qsa.rope_triton"
# Positions incl. 0, 1, scale-multiples, power-of-two boundaries, 1e5-class
# large positions and the 2**24 + 1 float32-rounding boundary.
POSITIONS = (0, 1, 2, 3, 4, 5, 63, 64, 65, 127, 128, 4095, 4096, 65535, 100000, 100003, 262143, 16777217)
# Valid interleaved section layouts: the released geometry, wide temporal,
# maxed H/W slots and zero-width axes.
SECTIONS = ((16, 8, 8), (24, 4, 4), (11, 11, 10), (21, 11, 0), (22, 0, 10), (32, 0, 0))


def _config(**kwargs):
    config = dict(
        style="Mrope",
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


def _equal_axis_transport(positions):
    """Token-major flat [3T] transport whose three axes agree."""
    return positions.view(-1, 1).expand(-1, 3).reshape(-1)


class QsaMropeHostVerdictTest(unittest.TestCase):
    """CPU-side pieces of the MRoPE text proof (no GPU required)."""

    def test_verdict_reads_only_host_rows(self):
        rows = torch.tensor([[4, 4, 4], [5, 5, 5]], dtype=torch.int32)
        self.assertIs(
            mrope_equal_axis_verdict(rows.view(-1), token_count=2, index_factor=3),
            True,
        )
        image = torch.tensor([[4, 4, 4], [5, 1, 2]], dtype=torch.int32)
        self.assertIs(
            mrope_equal_axis_verdict(image.view(-1), token_count=2, index_factor=3),
            False,
        )

    def test_verdict_declines_device_rows_and_malformed_counts(self):
        rows = torch.tensor([[4, 4, 4]], dtype=torch.int32)
        self.assertIs(
            mrope_equal_axis_verdict(
                rows.view(-1).cuda(), token_count=1, index_factor=3
            ),
            None,
        )
        # Malformed counts return None so the eager builder keeps its exact
        # legacy errors instead of a reshape failure.
        self.assertIs(
            mrope_equal_axis_verdict(rows.view(-1), token_count=2, index_factor=3),
            None,
        )

    def test_table_cache_is_keyed_by_base_sections_and_device(self):
        device = torch.device("cpu")
        config = _config()
        first = cached_mrope_inv_freq(config, device)
        self.assertIs(cached_mrope_inv_freq(config, device), first)
        expected = float(config.base) ** (
            -2.0 * torch.arange(32, dtype=torch.float32, device=device) / 64
        )
        self.assertTrue(torch.equal(first, expected))
        self.assertEqual(tuple(first.shape), (32,))
        self.assertEqual(first.dtype, torch.float32)
        self.assertTrue(first.is_contiguous())
        other_sections = cached_mrope_inv_freq(
            _config(mrope_dim1=24, mrope_dim2=4, mrope_dim3=4), device
        )
        self.assertIsNot(other_sections, first)
        other_base = cached_mrope_inv_freq(_config(base=10000.0), device)
        self.assertIsNot(other_base, first)
        self.assertTrue(
            torch.equal(
                other_base,
                10000.0
                ** (-2.0 * torch.arange(32, dtype=torch.float32, device=device) / 64),
            )
        )
        # The equal-axes table reduces to the slot-indexed base table the
        # eager MRoPE builder multiplies in.
        self.assertTrue(torch.equal(first, cached_inv_freq(config, device)))

    def test_eager_mrope_equals_base_rope_for_equal_axes(self):
        # Documents the derivation behind the fused table: with all three
        # axes equal, positions[:, axes] equals positions[:, 0] for every
        # interleaved slot, so the eager MRoPE chain and the one-axis Base
        # chain produce identical bytes for every section layout.
        positions = torch.tensor(POSITIONS, dtype=torch.int32)
        transported = _equal_axis_transport(positions)
        for sections in SECTIONS:
            for base, scale in ((1000000.0, 1.0), (10000.0, 3.0), (500000.0, 0.5)):
                with self.subTest(sections=sections, base=base, scale=scale):
                    mrope = _config(
                        base=base, scale=scale,
                        mrope_dim1=sections[0], mrope_dim2=sections[1],
                        mrope_dim3=sections[2],
                    )
                    base_style = _config(
                        style="Base", base=base, scale=scale,
                        mrope_dim1=sections[0], mrope_dim2=sections[1],
                        mrope_dim3=sections[2],
                    )
                    kwargs = dict(
                        token_count=positions.numel(),
                        dtype=torch.float32,
                        device=torch.device("cpu"),
                    )
                    mrope_cos, mrope_sin = build_interleaved_mrope(
                        transported, mrope, **kwargs
                    )
                    base_cos, base_sin = build_base_rope(transported, base_style, **kwargs)
                    self.assertTrue(torch.equal(mrope_cos, base_cos))
                    self.assertTrue(torch.equal(mrope_sin, base_sin))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QsaMropeFusedSM90Test(unittest.TestCase):
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

    def _fused_calls(self, build, *args, **kwargs):
        """Run one build, counting the fused text-rope dispatch attempts."""
        calls = []

        def spy(*call_args, **call_kwargs):
            calls.append(call_args)
            return try_build_text_rope(*call_args, **call_kwargs)

        with patch(MODULE + ".try_build_text_rope", side_effect=spy):
            result = build(*args, **kwargs)
        return result, calls

    def test_fused_mrope_matches_eager_bytes_across_sections_and_scales(self):
        for sections in SECTIONS:
            for base, scale in ((1000000.0, 1.0), (10000.0, 3.0), (500000.0, 0.5)):
                for transport_dtype in (torch.int32, torch.int64):
                    with self.subTest(
                        sections=sections,
                        base=base,
                        scale=scale,
                        dtype=transport_dtype,
                    ):
                        config = _config(
                            base=base,
                            scale=scale,
                            mrope_dim1=sections[0],
                            mrope_dim2=sections[1],
                            mrope_dim3=sections[2],
                        )
                        positions = torch.tensor(POSITIONS, dtype=transport_dtype)
                        transported = _equal_axis_transport(positions)
                        kwargs = dict(
                            token_count=positions.numel(),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                        actual, calls = self._fused_calls(
                            build_qsa_rope, transported, config, **kwargs
                        )
                        self.assertEqual(
                            len(calls), 1, "the fused text rope was not engaged"
                        )
                        with patch(MODULE + ".try_build_text_rope", return_value=None):
                            eager = build_qsa_rope(transported, config, **kwargs)
                        self.assertTrue(torch.equal(actual[0], eager[0]))
                        self.assertTrue(torch.equal(actual[1], eager[1]))

    def test_multimodal_rows_keep_the_eager_builder(self):
        config = _config()
        # Image-like rows with unequal T/H/W axes plus an equal-axis text tail.
        rows = torch.tensor(
            [[8, 1, 2], [9, 3, 4], [10, 5, 6], [11, 11, 11], [12, 12, 12]],
            dtype=torch.int32,
        )
        kwargs = dict(
            token_count=rows.shape[0], dtype=torch.bfloat16, device=self.device
        )
        actual, calls = self._fused_calls(build_qsa_rope, rows.view(-1), config, **kwargs)
        self.assertEqual(calls, [], "multimodal rows must not reach the fused kernel")
        with patch(MODULE + ".try_build_text_rope", return_value=None):
            eager = build_qsa_rope(rows.view(-1), config, **kwargs)
        self.assertTrue(torch.equal(actual[0], eager[0]))
        self.assertTrue(torch.equal(actual[1], eager[1]))

    def test_logical_positions_proof_engages_the_fused_kernel(self):
        config = _config()
        positions = torch.tensor(POSITIONS, dtype=torch.int32)
        logical = positions.to(torch.int64)
        transported = _equal_axis_transport(positions)
        kwargs = dict(
            token_count=positions.numel(), dtype=torch.bfloat16, device=self.device
        )
        actual, calls = self._fused_calls(
            build_qsa_rope, transported, config, logical_positions=logical, **kwargs
        )
        self.assertEqual(len(calls), 1, "the validated transport did not fuse")
        with patch(MODULE + ".try_build_text_rope", return_value=None):
            eager = build_qsa_rope(
                transported, config, logical_positions=logical, **kwargs
            )
        self.assertTrue(torch.equal(actual[0], eager[0]))
        self.assertTrue(torch.equal(actual[1], eager[1]))

    def test_device_rows_need_a_caller_proof_to_fuse(self):
        config = _config()
        positions = torch.tensor(POSITIONS, dtype=torch.int32)
        transported = _equal_axis_transport(positions).to(self.device)
        kwargs = dict(
            token_count=positions.numel(), dtype=torch.bfloat16, device=self.device
        )
        # Device-resident rows have no sync-free verdict: without a caller
        # proof the eager builder keeps them.
        unproven, calls = self._fused_calls(build_qsa_rope, transported, config, **kwargs)
        self.assertEqual(calls, [], "device rows fused without a proof")
        proven, calls = self._fused_calls(
            build_qsa_rope,
            transported,
            config,
            validate_position_axes=False,
            **kwargs,
        )
        self.assertEqual(len(calls), 1, "the caller proof did not fuse")
        self.assertTrue(torch.equal(proven[0], unproven[0]))
        self.assertTrue(torch.equal(proven[1], unproven[1]))

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

    def test_step_plan_text_step_fuses_both_rope_pairs(self):
        config = _config()
        # Warm the shared tables and the Triton compile with one full build;
        # the profiled build below uses fresh inputs objects, so it is a
        # plan-cache miss that must rebuild fused-only.
        warm = self._context([8], [5])
        self._geometry(warm, 5, config)
        context = self._context([8], [5])
        names = self._profiled_names(lambda: self._geometry(context, 5, config))
        self.assertEqual(
            sum("_text_rope" in name for name in names),
            2,
            "both rope pairs (current positions and block starts) must fuse",
        )
        for forbidden in ("aten::pow", "aten::cos", "aten::sin"):
            self.assertNotIn(forbidden, names)

    def test_step_plan_text_step_matches_the_eager_rebuild(self):
        config = _config()
        fused = self._geometry(self._context([8], [5]), 5, config)
        with patch(MODULE + ".try_build_text_rope", return_value=None):
            eager = self._geometry(self._context([8], [5]), 5, config)
        for name in ("current_cos", "current_sin", "rope_cos", "rope_sin"):
            self.assertTrue(
                torch.equal(getattr(fused, name), getattr(eager, name)),
                f"{name} differs between the fused and eager rebuilds",
            )

    def test_step_plan_multimodal_step_keeps_both_rope_pairs_eager(self):
        config = _config()
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


if __name__ == "__main__":
    unittest.main()
