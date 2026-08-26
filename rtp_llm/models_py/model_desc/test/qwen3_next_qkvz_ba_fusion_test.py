"""Equivalence tests for Qwen3NextGatedDeltaNet qkvz+ba fusion.

The fusion concatenates the in_proj_qkvz and in_proj_ba weights along the
output dim and runs a single GEMM, then slices the output to recover the
two original projections. This must be:
  (1) numerically equivalent to running the two GEMMs separately, and
  (2) bypassed for FP8 qkvz or a ROCm swizzle-incompatible fused layout.
"""

import unittest
from types import SimpleNamespace
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

    def test_rocm_ba_loader_rewrite_follows_local_shape(self) -> None:
        from rtp_llm.device.device_impl import RocmImpl
        from rtp_llm.utils.model_weight import W
        from rtp_llm.utils.swizzle_utils import swizzle_tensor

        fake_impl = SimpleNamespace(
            _is_gfx950=lambda: False,
            py_env_configs=SimpleNamespace(
                py_hw_kernel_config=SimpleNamespace(use_swizzleA=True)
            ),
        )
        aligned_ba = torch.randn(2048, 16, dtype=torch.bfloat16, device=self.device)
        expected_aligned = swizzle_tensor(aligned_ba.t(), False).t()

        rewritten_aligned = RocmImpl.maybe_rewrite_weight_by_key(
            fake_impl,
            W.linear_attn_ba_w,
            aligned_ba,
        )
        self.assertTrue(torch.equal(rewritten_aligned, expected_aligned))

        unaligned_ba = torch.randn(5120, 24, dtype=torch.bfloat16, device=self.device)
        unaligned_before = unaligned_ba.clone()
        rewritten_unaligned = RocmImpl.maybe_rewrite_weight_by_key(
            fake_impl,
            W.linear_attn_ba_w,
            unaligned_ba,
        )
        self.assertIs(rewritten_unaligned, unaligned_ba)
        self.assertTrue(torch.equal(rewritten_unaligned, unaligned_before))

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

    def _build_with_mocked_linears(
        self,
        *,
        is_rocm: bool,
        use_swizzle: bool,
        num_v_heads: int,
        quantized_qkvz: bool = False,
    ):
        """Construct a module while recording only its linear dispatch policy."""
        from rtp_llm.models_py.modules.factory.linear.factory import LinearFactory
        from rtp_llm.ops import HWKernelConfig
        from rtp_llm.utils.model_weight import W

        hw = HWKernelConfig()
        hw.use_swizzleA = use_swizzle
        weights_extra = None
        if quantized_qkvz:
            weights_extra = {
                W.linear_attn_qkvz_s: torch.randn(
                    8, dtype=torch.float32, device=self.device
                )
            }

        hip_version = "test-rocm" if is_rocm else None
        with patch.object(torch.version, "hip", hip_version, create=True), patch.object(
            LinearFactory,
            "create_linear",
            side_effect=lambda *a, **kw: MagicMock(name="MockFusedLinear"),
        ) as fused_factory, patch.object(
            LinearFactory,
            "create_linear_from_weights",
            side_effect=lambda *a, **kw: MagicMock(name="MockLinear"),
        ) as weights_factory:
            module = self._build_module(
                weights_extra=weights_extra,
                hw_kernel_config=hw,
                num_v_heads=num_v_heads,
            )

        return module, hw, fused_factory, weights_factory

    @staticmethod
    def _factory_call_for_weight(mock_factory, weight_key):
        calls = [
            call
            for call in mock_factory.call_args_list
            if len(call.args) >= 2 and call.args[1] == weight_key
        ]
        if len(calls) != 1:
            raise AssertionError(
                f"expected one factory call for {weight_key}, got {len(calls)}"
            )
        return calls[0]

    def test_bf16_path_takes_fusion(self) -> None:
        """When linear_attn_qkvz_s is None (BF16), fusion is enabled."""
        module = self._build_module()
        self.assertTrue(module._qkvz_ba_fused, "BF16 path must enable fusion")
        self.assertIsNotNone(module.in_proj_fused)
        self.assertIsNone(module.in_proj_qkvz)
        self.assertIsNone(module.in_proj_ba)

    def test_rocm_swizzle_unaligned_bf16_uses_two_gemms(self) -> None:
        """A swizzled qkvz plus raw BA must never become one WithSwizzle GEMM."""
        from rtp_llm.utils.model_weight import W

        module, hw, fused_factory, weights_factory = self._build_with_mocked_linears(
            is_rocm=True,
            use_swizzle=True,
            num_v_heads=4,
        )

        self.assertFalse(module._qkvz_ba_fused)
        fused_factory.assert_not_called()
        qkvz_call = self._factory_call_for_weight(weights_factory, W.linear_attn_qkvz_w)
        ba_call = self._factory_call_for_weight(weights_factory, W.linear_attn_ba_w)
        self.assertIs(qkvz_call.kwargs["hw_kernel_config"], hw)
        self.assertIsNone(ba_call.kwargs["hw_kernel_config"])

    def test_rocm_swizzle_aligned_bf16_keeps_fusion(self) -> None:
        from rtp_llm.utils.model_weight import W

        module, hw, fused_factory, weights_factory = self._build_with_mocked_linears(
            is_rocm=True,
            use_swizzle=True,
            num_v_heads=8,
        )

        self.assertTrue(module._qkvz_ba_fused)
        fused_factory.assert_called_once()
        self.assertIs(
            fused_factory.call_args.kwargs["hw_kernel_config"],
            hw,
        )
        self.assertFalse(
            any(
                len(call.args) >= 2
                and call.args[1] in (W.linear_attn_qkvz_w, W.linear_attn_ba_w)
                for call in weights_factory.call_args_list
            )
        )

    def test_rocm_without_swizzle_keeps_unaligned_bf16_fusion(self) -> None:
        module, hw, fused_factory, _ = self._build_with_mocked_linears(
            is_rocm=True,
            use_swizzle=False,
            num_v_heads=4,
        )

        self.assertTrue(module._qkvz_ba_fused)
        fused_factory.assert_called_once()
        self.assertIs(fused_factory.call_args.kwargs["hw_kernel_config"], hw)

    def test_cuda_keeps_unaligned_bf16_fusion(self) -> None:
        module, hw, fused_factory, _ = self._build_with_mocked_linears(
            is_rocm=False,
            use_swizzle=True,
            num_v_heads=4,
        )

        self.assertTrue(module._qkvz_ba_fused)
        fused_factory.assert_called_once()
        self.assertIs(fused_factory.call_args.kwargs["hw_kernel_config"], hw)

    def test_fp8_qkvz_with_aligned_ba_keeps_ba_swizzle(self) -> None:
        from rtp_llm.utils.model_weight import W

        module, hw, fused_factory, weights_factory = self._build_with_mocked_linears(
            is_rocm=True,
            use_swizzle=True,
            num_v_heads=8,
            quantized_qkvz=True,
        )

        self.assertFalse(module._qkvz_ba_fused)
        fused_factory.assert_not_called()
        qkvz_call = self._factory_call_for_weight(weights_factory, W.linear_attn_qkvz_w)
        ba_call = self._factory_call_for_weight(weights_factory, W.linear_attn_ba_w)
        self.assertEqual(qkvz_call.args[2], W.linear_attn_qkvz_s)
        self.assertIs(qkvz_call.kwargs["hw_kernel_config"], hw)
        self.assertIs(ba_call.kwargs["hw_kernel_config"], hw)

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

    @unittest.skipUnless(
        torch.version.hip is not None,
        "ROCm-only BA swizzle dispatch",
    )
    def test_in_proj_ba_no_swizzle_when_unaligned(self) -> None:
        """BA out-dim 8 (= 2*4, not 16-aligned, mirrors TP=4's 24): in_proj_ba
        must receive hw_kernel_config=None so dispatch picks NoSwizzle,
        consistent with device_impl skipping the swizzle. This is the crash fix."""
        _hw, ba_cfg = self._ba_hw_kernel_config_in_fallback(num_v_heads=4)
        self.assertIsNone(
            ba_cfg,
            "unaligned BA must pass hw_kernel_config=None (NoSwizzle dispatch)",
        )

    @unittest.skipUnless(
        torch.version.hip is not None,
        "ROCm-only BA swizzle dispatch",
    )
    def test_in_proj_ba_keeps_swizzle_when_aligned(self) -> None:
        """Quantized qkvz does not disable swizzle for an aligned BF16 BA."""
        hw, ba_cfg = self._ba_hw_kernel_config_in_fallback(num_v_heads=8)
        self.assertIs(
            ba_cfg,
            hw,
            "aligned BA must keep WithSwizzle dispatch",
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
        # ROCm uses column-major layout (cat in [N,K] then .t()),
        # CUDA uses row-major (torch.empty + copy_).
        _is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        K = fused_buf.shape[0]
        if _is_rocm:
            expected_offset = module._qkvz_size * K * fused_buf.element_size()
        else:
            expected_offset = module._qkvz_size * fused_buf.element_size()
        self.assertEqual(
            ba_view.data_ptr(),
            fused_buf.data_ptr() + expected_offset,
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
