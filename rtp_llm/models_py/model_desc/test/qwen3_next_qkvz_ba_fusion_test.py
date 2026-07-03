"""Equivalence tests for Qwen3NextGatedDeltaNet qkvz+ba fusion.

The fusion concatenates the in_proj_qkvz and in_proj_ba weights along the
output dim and runs a single GEMM, then slices the output to recover the
two original projections. This must be:
  (1) numerically equivalent to running the two GEMMs separately, and
  (2) bypassed when linear_attn_qkvz_s is present (FP8 path).
"""

import unittest
from unittest.mock import MagicMock, patch

import torch


class TestQwen3NextQkvzBaFusion(unittest.TestCase):
    """Validates fusion correctness against the 2-GEMM baseline."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        self.device = torch.device("cuda:0")
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

    # ---- (1) low-level math equivalence ----

    def test_fused_slice_equals_separate_gemms(self) -> None:
        """cat([qkvz_w, ba_w]) @ x sliced must match (qkvz_w @ x, ba_w @ x).

        Pure linear-algebra invariant; does not exercise Qwen3NextGatedDeltaNet
        but locks in the math contract the fusion relies on.
        """
        # Dimensions are intentionally large enough that hipBLASLt picks a real
        # tile (small shapes can fall to a vendor reference path that hides
        # accumulation-order differences).
        M, K = 256, 1024
        qkvz_dim, ba_dim = 1024, 16
        dtype = torch.bfloat16

        x = torch.randn(M, K, dtype=dtype, device=self.device)
        qkvz_w = torch.randn(K, qkvz_dim, dtype=dtype, device=self.device)
        ba_w = torch.randn(K, ba_dim, dtype=dtype, device=self.device)

        out_qkvz = x @ qkvz_w
        out_ba = x @ ba_w

        fused_w = torch.cat([qkvz_w, ba_w], dim=1).contiguous()
        out_fused = x @ fused_w

        # bf16 GEMM accumulation order can differ between GEMM impls; tolerance
        # follows the project's RmsNormGated test (atol=rtol=1e-2).
        torch.testing.assert_close(
            out_fused[..., :qkvz_dim], out_qkvz, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            out_fused[..., qkvz_dim:], out_ba, atol=1e-2, rtol=1e-2
        )

    # ---- (2) Qwen3NextGatedDeltaNet end-to-end ----

    def _build_module(self, weights_extra=None, hw_kernel_config=None, num_v_heads=4):
        """Construct a small Qwen3NextGatedDeltaNet with random weights.

        Mirrors the setup pattern in
        rtp_llm/models_py/modules/factory/attention/cuda_cp_impl/test/test_cp_linear_attn.py
        but at the smallest legal sizes so the test runs fast.

        num_v_heads controls the BA out-dim (= 2 * num_v_heads): 4 -> 8
        (not 16-aligned), 8 -> 16 (aligned). hw_kernel_config is forwarded to
        the module so the in_proj_ba swizzle/NoSwizzle decision can be observed.
        """
        from rtp_llm.models_py.model_desc.qwen3_next import Qwen3NextGatedDeltaNet
        from rtp_llm.ops import DataType, LinearAttentionConfig, ParallelismConfig
        from rtp_llm.utils.model_weight import W

        num_k_heads = 2
        head_k_dim, head_v_dim = 32, 32
        hidden_size, conv_kernel_dim = 128, 4

        cfg = LinearAttentionConfig()
        cfg.linear_num_key_heads = num_k_heads
        cfg.linear_num_value_heads = num_v_heads
        cfg.linear_key_head_dim = head_k_dim
        cfg.linear_value_head_dim = head_v_dim
        cfg.linear_conv_kernel_dim = conv_kernel_dim
        cfg.ssm_state_dtype = DataType.TYPE_BF16
        cfg.conv_state_dtype = DataType.TYPE_BF16

        par = ParallelismConfig()
        par.tp_size = 1
        par.tp_rank = 0

        qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
        z_dim = head_v_dim * num_v_heads
        qkvz_dim = qkv_dim + z_dim
        ba_dim = num_v_heads * 2

        bf16 = torch.bfloat16
        dev = self.device
        weights = {
            W.linear_attn_conv1d_w: torch.randn(
                qkv_dim, 1, conv_kernel_dim, dtype=bf16, device=dev
            ),
            W.linear_attn_dt_b: torch.randn(num_v_heads, dtype=bf16, device=dev),
            W.linear_attn_alog: torch.randn(num_v_heads, dtype=bf16, device=dev),
            W.linear_attn_norm_w: torch.randn(head_v_dim, dtype=bf16, device=dev),
            W.linear_attn_qkvz_w: torch.randn(
                hidden_size, qkvz_dim, dtype=bf16, device=dev
            ),
            W.linear_attn_qkvz_s: None,
            W.linear_attn_ba_w: torch.randn(
                hidden_size, ba_dim, dtype=bf16, device=dev
            ),
            W.linear_attn_out_w: torch.randn(
                num_v_heads * head_v_dim, hidden_size, dtype=bf16, device=dev
            ),
            W.linear_attn_out_s: None,
        }
        if weights_extra:
            weights.update(weights_extra)

        return Qwen3NextGatedDeltaNet(
            cfg, par, weights, layernorm_eps=1e-6, hw_kernel_config=hw_kernel_config
        ).to(dev)

    def test_bf16_path_takes_fusion(self) -> None:
        """When linear_attn_qkvz_s is None (BF16), fusion is enabled."""
        module = self._build_module()
        self.assertTrue(module._qkvz_ba_fused, "BF16 path must enable fusion")
        self.assertIsNotNone(module.in_proj_fused)
        self.assertIsNone(module.in_proj_qkvz)
        self.assertIsNone(module.in_proj_ba)

    def test_quantized_path_falls_back_in_constructor(self) -> None:
        """When linear_attn_qkvz_s is set (FP8 path) the constructor must
        take the 2-GEMM fallback branch. Real FP8 strategy selection needs
        a fully-quantized weight + quant_config setup, so we mock the
        Linear factory to bypass strategy lookup and directly assert which
        branch ran.

        Asserts:
          - _qkvz_ba_fused == False
          - in_proj_fused is None
          - both in_proj_qkvz and in_proj_ba were constructed
          - factory was invoked with the qkvz weight + qkvz scale keys
        """
        from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
        from rtp_llm.utils.model_weight import W

        # Sentinel scale tensor; only its presence triggers the fallback.
        scale = torch.randn(8, dtype=torch.float32, device=self.device)

        # Mock LinearFactory.create_linear_from_weights for this construction
        # only. The mock returns a fresh MagicMock per call so the module's
        # in_proj_qkvz, in_proj_ba, out_proj attributes are truthy and
        # distinguishable.
        with patch.object(
            LinearFactory,
            "create_linear_from_weights",
            side_effect=lambda *a, **kw: MagicMock(name="MockLinear"),
        ) as mock_factory:
            module = self._build_module(weights_extra={W.linear_attn_qkvz_s: scale})

        self.assertFalse(module._qkvz_ba_fused, "qkvz_s presence must disable fusion")
        self.assertIsNone(module.in_proj_fused, "fused Linear must NOT be constructed")
        self.assertIsNotNone(
            module.in_proj_qkvz, "in_proj_qkvz must be constructed in fallback"
        )
        self.assertIsNotNone(
            module.in_proj_ba, "in_proj_ba must be constructed in fallback"
        )

        # Verify the factory was invoked for qkvz with the scale key (i.e.
        # the fallback path actually wired the FP8 scales through).
        qkvz_calls = [
            c
            for c in mock_factory.call_args_list
            if len(c.args) >= 2 and c.args[1] == W.linear_attn_qkvz_w
        ]
        self.assertEqual(
            len(qkvz_calls),
            1,
            "fallback must call factory once for in_proj_qkvz",
        )
        self.assertEqual(
            qkvz_calls[0].args[2],
            W.linear_attn_qkvz_s,
            "fallback must pass linear_attn_qkvz_s as scale_key",
        )

    def _ba_hw_kernel_config_in_fallback(self, num_v_heads):
        """Build the FP8 fallback (non-fused) branch with swizzle enabled and
        return the hw_kernel_config the factory received for in_proj_ba.

        Mocks the Linear factory so no real GEMM/strategy lookup runs; we only
        inspect which hw_kernel_config was wired for the BA projection.
        """
        from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
        from rtp_llm.ops import HWKernelConfig
        from rtp_llm.utils.model_weight import W

        hw = HWKernelConfig()
        hw.use_swizzleA = True
        # qkvz_s presence forces the 2-GEMM (non-fused) branch where in_proj_ba
        # is created standalone.
        scale = torch.randn(8, dtype=torch.float32, device=self.device)

        with patch.object(
            LinearFactory,
            "create_linear_from_weights",
            side_effect=lambda *a, **kw: MagicMock(name="MockLinear"),
        ) as mock_factory:
            self._build_module(
                weights_extra={W.linear_attn_qkvz_s: scale},
                hw_kernel_config=hw,
                num_v_heads=num_v_heads,
            )

        ba_calls = [
            c
            for c in mock_factory.call_args_list
            if len(c.args) >= 2 and c.args[1] == W.linear_attn_ba_w
        ]
        self.assertEqual(len(ba_calls), 1, "fallback must build in_proj_ba once")
        return hw, ba_calls[0].kwargs.get("hw_kernel_config")

    def test_in_proj_ba_no_swizzle_when_unaligned(self) -> None:
        """BA out-dim 8 (= 2*4, not 16-aligned, mirrors TP=4's 24): in_proj_ba
        must receive hw_kernel_config=None so dispatch picks NoSwizzle,
        consistent with device_impl skipping the swizzle. This is the crash fix."""
        _hw, ba_cfg = self._ba_hw_kernel_config_in_fallback(num_v_heads=4)
        self.assertIsNone(
            ba_cfg,
            "unaligned BA must pass hw_kernel_config=None (NoSwizzle dispatch)",
        )

    def test_in_proj_ba_keeps_swizzle_when_aligned(self) -> None:
        """BA out-dim 16 (= 2*8, 16-aligned, mirrors TP=1/2): the swizzle
        speedup is preserved — in_proj_ba keeps the real hw_kernel_config."""
        hw, ba_cfg = self._ba_hw_kernel_config_in_fallback(num_v_heads=8)
        self.assertIs(
            ba_cfg,
            hw,
            "aligned BA must keep the real hw_kernel_config (WithSwizzle dispatch)",
        )

    def test_input_project_helper_shapes(self) -> None:
        """_input_project must return (projected_qkvz, projected_ba) with
        the right shapes regardless of which dispatch branch runs.

        Ensures forward() and the CP test can share one stable API for
        running the input projection — see the projection helper in
        Qwen3NextGatedDeltaNet.
        """
        module = self._build_module()
        hidden_size = 128  # mirrors _build_module
        M = 16
        x = torch.randn(M, hidden_size, dtype=torch.bfloat16, device=self.device)
        with torch.no_grad():
            qkvz, ba = module._input_project(x)
        # qkvz: (M, qkvz_dim), ba: (M, ba_dim)
        # qkvz_dim = 2*key_dim + value_dim*2; ba_dim = 2*v_heads
        # With test config: key_dim=2*32=64, value_dim=4*32=128, v_heads=4
        # qkvz_dim = 2*64 + 128 + 128 = 384
        # ba_dim = 2*4 = 8
        self.assertEqual(qkvz.shape, (M, 384))
        self.assertEqual(ba.shape, (M, 8))
        # Both must be on the right device/dtype
        self.assertEqual(qkvz.device, x.device)
        self.assertEqual(qkvz.dtype, x.dtype)

    def test_dict_entries_are_views_into_fused_buffer(self) -> None:
        """After fusion, the qkvz / ba dict entries must be VIEWS into the
        fused buffer, not separate tensors. This:
          (a) avoids the ~1.16GB redundant weight memory across 24 GDN
              layers (originals are GC'd when init returns), and
          (b) keeps online weight updates working: WeightManager calls
              ori_tensor.copy_(data) on the dict entry, which must land
              inside the fused buffer for in_proj_fused to see the update.
        """
        from rtp_llm.utils.model_weight import W

        module = self._build_module()
        fused_buf = module.in_proj_fused.weight
        qkvz_view = module.weights[W.linear_attn_qkvz_w]
        ba_view = module.weights[W.linear_attn_ba_w]

        # Same underlying storage as in_proj_fused.weight (the fused buffer).
        self.assertEqual(qkvz_view.data_ptr(), fused_buf.data_ptr())
        self.assertEqual(
            ba_view.data_ptr(),
            fused_buf.data_ptr() + module._qkvz_size * fused_buf.element_size(),
        )

    def test_online_weight_update_changes_forward_output(self) -> None:
        """End-to-end check: simulate WeightManager's in-place copy_() onto
        the qkvz dict entry, then re-run the projection. The next forward
        must reflect the new qkvz weight; ba must be untouched.

        This verifies via forward output (layout-agnostic) rather than
        slicing in_proj_fused.weight directly. The CUDA Linear strategy
        (CudaF16Linear) stores self.weight = weight.T, so direct slicing
        of in_proj_fused.weight depends on the backend, but the projection
        result does not.
        """
        from rtp_llm.utils.model_weight import W

        module = self._build_module()
        hidden_size = 128  # mirrors _build_module
        x = torch.randn(8, hidden_size, dtype=torch.bfloat16, device=self.device)

        # Capture the projection output before any update.
        with torch.no_grad():
            qkvz_before, ba_before = module._input_project(x)

        # Replace qkvz with a deliberately distinct value via the in-place
        # path that WeightManager.update_layer_weight uses (ori.copy_(data)).
        qkvz_entry = module.weights[W.linear_attn_qkvz_w]
        new_qkvz_w = torch.full_like(qkvz_entry, 0.5)
        with torch.inference_mode():
            qkvz_entry.copy_(new_qkvz_w)

        # Re-run the projection — qkvz output must change, ba output must not.
        with torch.no_grad():
            qkvz_after, ba_after = module._input_project(x)

        self.assertFalse(
            torch.equal(qkvz_before, qkvz_after),
            "qkvz projection must reflect the in-place weight update",
        )
        torch.testing.assert_close(
            ba_before,
            ba_after,
            atol=0,
            rtol=0,
            msg="ba projection must be untouched by a qkvz-only update",
        )

        # The new qkvz output should equal x @ new_qkvz_w within bf16
        # tolerance — this confirms the update landed on the actual GEMM
        # weight, not just on a stray copy.
        expected = x @ new_qkvz_w
        torch.testing.assert_close(qkvz_after, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
