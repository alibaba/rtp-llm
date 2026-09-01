"""
Unit tests for pack_ue8m0 kernel implementations.

Tests verify that our Triton pack_ue8m0 kernel implementations produce
identical results to deep_gemm's get_mn_major_tma_aligned_packed_ue8m0_tensor.
"""

from unittest import SkipTest, TestCase, main

import torch


def _is_sm100() -> bool:
    """Check if current GPU is SM100 (Blackwell)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


# ============================================================================
# Test Parameters
# ============================================================================

# Shapes for basic pack kernel tests: (B, M, K)
# B = batch/expert, M = token, K = scale groups
PACK_SHAPES_GRAN1 = [
    # Small shapes - M is the token dimension, K is scale groups
    (2, 128, 16),
    (4, 256, 32),
    # Medium shapes (typical MoE scenarios)
    (8, 512, 64),
    (16, 1024, 128),
]

# Shapes for fused kernel tests: (E, T, H) where input is (E, T, 2*H)
FUSED_SHAPES = [
    # E=experts, T=tokens, H=hidden_dim/2
    # H must be divisible by 128 (group_size) and H/128 must be divisible by 4 (for packing)
    (4, 128, 2560),  # 2560 / 128 = 20 groups, 20 / 4 = 5 packed groups
    (8, 256, 5120),  # 5120 / 128 = 40 groups, 40 / 4 = 10 packed groups
]


# ============================================================================
# Helper Functions
# ============================================================================


def create_test_scale_for_reference(E: int, T: int, num_groups: int) -> torch.Tensor:
    """
    Create a test scale tensor matching what deep_gemm reference expects.
    Shape: (E, T, num_groups) with positive values rounded to powers of 2 (UE8M0).
    """
    # Generate random positive scales
    scale = (
        torch.randn(E, T, num_groups, device="cuda", dtype=torch.float32).abs() + 1e-6
    )
    # Round to power of 2 (UE8M0 format)
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    return scale


# ============================================================================
# Tests for pack_ue8m0_kernel_launcher
# ============================================================================


class TestPackUe8m0KernelLauncher(TestCase):
    """Tests for pack_ue8m0_kernel_launcher from deepgemm_wrapper.py."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")

    def test_pack_kernel_gran1_matches_reference(self):
        """
        Test pack_ue8m0_kernel_gran1 produces same packed values as reference.
        Uses gran_mn=1 (per-token scales).
        """
        from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor

        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            pack_ue8m0_kernel_launcher,
        )

        for shape in PACK_SHAPES_GRAN1:
            with self.subTest(shape=shape):
                E, T, K = shape
                gran_mn = 1

                # Create scale tensor - shape (E, T, K) for per-token scales
                scale = create_test_scale_for_reference(E, T, K)

                # Reference from deep_gemm
                ref_output = get_mn_major_tma_aligned_packed_ue8m0_tensor(scale)

                # Our implementation
                our_output = pack_ue8m0_kernel_launcher(scale, gran_mn)

                # Both should have same shape
                self.assertEqual(
                    ref_output.shape,
                    our_output.shape,
                    f"Shape mismatch: ref={ref_output.shape}, ours={our_output.shape}",
                )

                # Values should match exactly
                diff = (ref_output != our_output).sum().item()
                self.assertEqual(
                    diff,
                    0,
                    f"Values mismatch: {diff} elements differ out of {ref_output.numel()}",
                )

    def test_pack_kernel_2d_input_gran1(self):
        """Test pack kernel with 2D input (no batch dimension), gran_mn=1."""
        from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor

        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            pack_ue8m0_kernel_launcher,
        )

        T = 256
        K = 32
        gran_mn = 1

        # 2D scale tensor
        scale = torch.randn(T, K, device="cuda", dtype=torch.float32).abs() + 1e-6
        scale = torch.exp2(torch.ceil(torch.log2(scale)))

        # Reference
        ref_output = get_mn_major_tma_aligned_packed_ue8m0_tensor(scale)

        # Our implementation
        our_output = pack_ue8m0_kernel_launcher(scale, gran_mn)

        # Verify
        self.assertEqual(
            ref_output.shape,
            our_output.shape,
            f"Shape mismatch: ref={ref_output.shape}, ours={our_output.shape}",
        )
        diff = (ref_output != our_output).sum().item()
        self.assertEqual(diff, 0, f"Values mismatch: {diff} elements differ")


# ============================================================================
# Tests for create_packed_scale_tensor
# ============================================================================


class TestCreatePackedScaleTensor(TestCase):
    """Tests for create_packed_scale_tensor from activation.py."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")

    def test_tensor_shape(self):
        """Test that created tensor has correct shape."""
        from rtp_llm.models_py.triton_kernels.common.activation import (
            create_packed_scale_tensor,
        )

        for shape in FUSED_SHAPES:
            with self.subTest(shape=shape):
                E, T, H = shape
                hidden_dim = 2 * H
                group_size = 128
                G = H // group_size
                G_packed = G // 4

                tensor = create_packed_scale_tensor(
                    E, T, hidden_dim, group_size, "cuda"
                )

                self.assertEqual(
                    tensor.shape,
                    (E, T, G_packed),
                    f"Shape mismatch: expected {(E, T, G_packed)}, got {tensor.shape}",
                )
                self.assertEqual(
                    tensor.dtype, torch.int32, f"Expected int32, got {tensor.dtype}"
                )

    def test_tensor_strides(self):
        """Test that created tensor has column-major layout for K dimension."""
        from rtp_llm.models_py.triton_kernels.common.activation import (
            create_packed_scale_tensor,
        )

        for shape in FUSED_SHAPES:
            with self.subTest(shape=shape):
                E, T, H = shape
                hidden_dim = 2 * H
                group_size = 128
                G = H // group_size
                G_packed = G // 4

                tensor = create_packed_scale_tensor(
                    E, T, hidden_dim, group_size, "cuda"
                )

                # Expected strides for column-major K: (G_packed * T, 1, T)
                expected_strides = (G_packed * T, 1, T)
                self.assertEqual(
                    tensor.stride(),
                    expected_strides,
                    f"Stride mismatch: expected {expected_strides}, got {tensor.stride()}",
                )


# ============================================================================
# Tests for fused SiLU+Mul+Quant+Pack kernel
# ============================================================================


class TestSiluAndMulPostQuantPackedFwd(TestCase):
    """Tests for silu_and_mul_masked_post_quant_packed_fwd from activation.py."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")

    def test_fp8_output_matches_separate_kernel(self):
        """
        Test that fused packed kernel produces same FP8 output as
        separate quant kernel (ignoring scale packing).
        """
        from rtp_llm.models_py.triton_kernels.common.activation import (
            create_packed_scale_tensor,
            silu_and_mul_masked_post_quant_fwd,
            silu_and_mul_masked_post_quant_packed_fwd,
        )

        for shape in FUSED_SHAPES:
            with self.subTest(shape=shape):
                E, T, H = shape
                group_size = 128
                G = H // group_size

                # Create input (ensure same random seed for both paths)
                torch.manual_seed(42)
                input_tensor = torch.randn(
                    E, T, 2 * H, device="cuda", dtype=torch.bfloat16
                )
                masked_m = torch.full((E,), T, device="cuda", dtype=torch.int32)

                # ---- Reference path: separate quant kernel ----
                ref_output = torch.empty(
                    E, T, H, device="cuda", dtype=torch.float8_e4m3fn
                )
                ref_scale = torch.empty(E, T, G, device="cuda", dtype=torch.float32)

                silu_and_mul_masked_post_quant_fwd(
                    input_tensor.clone(),
                    ref_output,
                    ref_scale,
                    group_size,
                    masked_m,
                    scale_ue8m0=True,
                )

                # ---- Our path: fused packed kernel ----
                our_output = torch.empty(
                    E, T, H, device="cuda", dtype=torch.float8_e4m3fn
                )
                our_scale_packed = create_packed_scale_tensor(
                    E, T, 2 * H, group_size, "cuda"
                )

                silu_and_mul_masked_post_quant_packed_fwd(
                    input_tensor.clone(),
                    our_output,
                    our_scale_packed,
                    group_size,
                    masked_m,
                )

                # ---- Verify FP8 outputs match ----
                output_diff = (ref_output != our_output).sum().item()
                self.assertEqual(
                    output_diff,
                    0,
                    f"FP8 output mismatch: {output_diff} elements differ",
                )

    def test_packed_scale_matches_reference(self):
        """
        Test that the packed scales from fused kernel match what we get from
        separate quant + reference pack.
        """
        from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor

        from rtp_llm.models_py.triton_kernels.common.activation import (
            create_packed_scale_tensor,
            silu_and_mul_masked_post_quant_fwd,
            silu_and_mul_masked_post_quant_packed_fwd,
        )

        for shape in FUSED_SHAPES:
            with self.subTest(shape=shape):
                E, T, H = shape
                group_size = 128
                G = H // group_size

                # Create input
                torch.manual_seed(42)
                input_tensor = torch.randn(
                    E, T, 2 * H, device="cuda", dtype=torch.bfloat16
                )
                masked_m = torch.full((E,), T, device="cuda", dtype=torch.int32)

                # ---- Reference path: separate quant + reference pack ----
                ref_output = torch.empty(
                    E, T, H, device="cuda", dtype=torch.float8_e4m3fn
                )
                ref_scale = torch.empty(E, T, G, device="cuda", dtype=torch.float32)

                silu_and_mul_masked_post_quant_fwd(
                    input_tensor.clone(),
                    ref_output,
                    ref_scale,
                    group_size,
                    masked_m,
                    scale_ue8m0=True,
                )

                # Pack using reference
                ref_scale_packed = get_mn_major_tma_aligned_packed_ue8m0_tensor(
                    ref_scale
                )

                # ---- Our path: fused packed kernel ----
                our_output = torch.empty(
                    E, T, H, device="cuda", dtype=torch.float8_e4m3fn
                )
                our_scale_packed = create_packed_scale_tensor(
                    E, T, 2 * H, group_size, "cuda"
                )

                silu_and_mul_masked_post_quant_packed_fwd(
                    input_tensor.clone(),
                    our_output,
                    our_scale_packed,
                    group_size,
                    masked_m,
                )

                # ---- Compare packed scales ----
                # Shapes should match
                self.assertEqual(
                    ref_scale_packed.shape,
                    our_scale_packed.shape,
                    f"Packed scale shape mismatch: ref={ref_scale_packed.shape}, ours={our_scale_packed.shape}",
                )

                # Values should match
                scale_diff = (ref_scale_packed != our_scale_packed).sum().item()
                self.assertEqual(
                    scale_diff,
                    0,
                    f"Packed scale mismatch: {scale_diff} elements differ",
                )

    def test_partial_tokens(self):
        """Test with partial tokens (masked_m < T)."""
        from deep_gemm import get_mn_major_tma_aligned_packed_ue8m0_tensor

        from rtp_llm.models_py.triton_kernels.common.activation import (
            create_packed_scale_tensor,
            silu_and_mul_masked_post_quant_fwd,
            silu_and_mul_masked_post_quant_packed_fwd,
        )

        E, T, H = 4, 128, 2560
        group_size = 128
        G = H // group_size

        # Create input
        torch.manual_seed(42)
        input_tensor = torch.randn(E, T, 2 * H, device="cuda", dtype=torch.bfloat16)
        # Partial tokens: varying number of valid tokens per expert
        masked_m = torch.tensor([32, 64, 96, 128], device="cuda", dtype=torch.int32)

        # ---- Reference path ----
        ref_output = torch.empty(E, T, H, device="cuda", dtype=torch.float8_e4m3fn)
        ref_scale = torch.empty(E, T, G, device="cuda", dtype=torch.float32)

        silu_and_mul_masked_post_quant_fwd(
            input_tensor.clone(),
            ref_output,
            ref_scale,
            group_size,
            masked_m,
            scale_ue8m0=True,
        )

        # Pack reference scale for comparison
        ref_scale_packed = get_mn_major_tma_aligned_packed_ue8m0_tensor(ref_scale)

        # ---- Our path ----
        our_output = torch.empty(E, T, H, device="cuda", dtype=torch.float8_e4m3fn)
        our_scale_packed = create_packed_scale_tensor(E, T, 2 * H, group_size, "cuda")

        silu_and_mul_masked_post_quant_packed_fwd(
            input_tensor.clone(),
            our_output,
            our_scale_packed,
            group_size,
            masked_m,
        )

        # ---- Verify FP8 output and packed scales for valid tokens only ----
        self.assertEqual(
            ref_scale_packed.shape,
            our_scale_packed.shape,
            f"Packed scale shape mismatch: ref={ref_scale_packed.shape}, ours={our_scale_packed.shape}",
        )

        for e in range(E):
            valid_tokens = masked_m[e].item()

            # Check FP8 output
            ref_valid = ref_output[e, :valid_tokens, :]
            our_valid = our_output[e, :valid_tokens, :]
            diff = (ref_valid != our_valid).sum().item()
            self.assertEqual(
                diff,
                0,
                f"Expert {e}: FP8 output mismatch in valid region: {diff} elements differ",
            )

            # Check packed scales (only for valid tokens)
            ref_scale_valid = ref_scale_packed[e, :valid_tokens, :]
            our_scale_valid = our_scale_packed[e, :valid_tokens, :]
            scale_diff = (ref_scale_valid != our_scale_valid).sum().item()
            self.assertEqual(
                scale_diff,
                0,
                f"Expert {e}: Packed scale mismatch in valid region: {scale_diff} elements differ",
            )


# ============================================================================
# Tests for end-to-end DeepGEMM compatibility
# ============================================================================


class TestDeepGemmIntegration(TestCase):
    """
    End-to-end tests verifying that pre-packed scales work correctly
    with deep_gemm's m_grouped_fp8_gemm_nt_masked.
    """

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")

    def test_gemm_with_our_pack_vs_reference_pack(self):
        """
        Test that GEMM with our pre-packed scales produces same result as
        GEMM with reference-packed scales.
        """
        if not _is_sm100():
            raise SkipTest("Requires SM100 (Blackwell) GPU")

        from deep_gemm import (
            get_mn_major_tma_aligned_packed_ue8m0_tensor,
            m_grouped_fp8_gemm_nt_masked,
        )

        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            pack_ue8m0_kernel_launcher,
        )

        # Test dimensions
        E = 4
        T = 128
        K = 2560
        N = 5120
        group_size = 128

        torch.manual_seed(42)

        # Create FP8 activation and scale
        a = torch.randn(E, T, K, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        a_scale = (
            torch.randn(E, T, K // group_size, device="cuda", dtype=torch.float32).abs()
            + 1e-6
        )
        # Round to power of 2 (UE8M0)
        a_scale = torch.exp2(torch.ceil(torch.log2(a_scale)))

        # Create weight and scale
        w = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        w_scale = (
            torch.randn(
                E, N // group_size, K // group_size, device="cuda", dtype=torch.float32
            ).abs()
            + 1e-6
        )
        w_scale = torch.exp2(torch.ceil(torch.log2(w_scale)))

        masked_m = torch.full((E,), T, device="cuda", dtype=torch.int32)

        # ---- Reference path: use deep_gemm's pack function ----
        a_scale_packed_ref = get_mn_major_tma_aligned_packed_ue8m0_tensor(a_scale)

        output_ref = torch.empty(E, T, N, device="cuda", dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (a, a_scale_packed_ref),
            (w, w_scale),
            output_ref,
            masked_m,
            T,
        )

        # ---- Our path: use our pack function ----
        a_scale_packed_ours = pack_ue8m0_kernel_launcher(a_scale, gran_mn=1)

        output_ours = torch.empty(E, T, N, device="cuda", dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (a, a_scale_packed_ours),
            (w, w_scale),
            output_ours,
            masked_m,
            T,
        )

        # ---- Verify packed scales match ----
        self.assertEqual(
            a_scale_packed_ref.shape,
            a_scale_packed_ours.shape,
            f"Packed scale shape mismatch: ref={a_scale_packed_ref.shape}, ours={a_scale_packed_ours.shape}",
        )
        scale_diff = (a_scale_packed_ref != a_scale_packed_ours).sum().item()
        self.assertEqual(
            scale_diff, 0, f"Packed scale mismatch: {scale_diff} elements differ"
        )

        # ---- Verify GEMM outputs are identical ----
        max_diff = (output_ref - output_ours).abs().max().item()
        mean_diff = (output_ref - output_ours).abs().mean().item()

        self.assertLess(max_diff, 1e-5, f"GEMM output max diff: {max_diff}")
        self.assertLess(mean_diff, 1e-6, f"GEMM output mean diff: {mean_diff}")

    def test_fused_kernel_e2e_gemm(self):
        """
        End-to-end test: fused quant+pack kernel output used in GEMM
        should match separate quant + reference pack + GEMM.
        """
        if not _is_sm100():
            raise SkipTest("Requires SM100 (Blackwell) GPU")

        from deep_gemm import (
            get_mn_major_tma_aligned_packed_ue8m0_tensor,
            m_grouped_fp8_gemm_nt_masked,
        )

        from rtp_llm.models_py.triton_kernels.common.activation import (
            create_packed_scale_tensor,
            silu_and_mul_masked_post_quant_fwd,
            silu_and_mul_masked_post_quant_packed_fwd,
        )

        # Test dimensions
        E = 4
        T = 128
        K = 2560  # down projection input dim
        N = 5120  # down projection output dim
        group_size = 128

        torch.manual_seed(42)

        # Create weight (w2) and its scale
        w2 = torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        w2_scale = (
            torch.randn(
                E, N // group_size, K // group_size, device="cuda", dtype=torch.float32
            ).abs()
            + 1e-6
        )
        w2_scale = torch.exp2(torch.ceil(torch.log2(w2_scale)))

        # Create activation input (simulating output from gate/up GEMM)
        upgate_output = torch.randn(E, T, 2 * K, device="cuda", dtype=torch.bfloat16)
        masked_m = torch.full((E,), T, device="cuda", dtype=torch.int32)

        # ---- Reference path: separate quant + reference pack + GEMM ----
        ref_fp8 = torch.empty(E, T, K, device="cuda", dtype=torch.float8_e4m3fn)
        ref_scale = torch.empty(
            E, T, K // group_size, device="cuda", dtype=torch.float32
        )

        silu_and_mul_masked_post_quant_fwd(
            upgate_output.clone(),
            ref_fp8,
            ref_scale,
            group_size,
            masked_m,
            scale_ue8m0=True,
        )

        ref_scale_packed = get_mn_major_tma_aligned_packed_ue8m0_tensor(ref_scale)

        output_ref = torch.empty(E, T, N, device="cuda", dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (ref_fp8, ref_scale_packed),
            (w2, w2_scale),
            output_ref,
            masked_m,
            T,
        )

        # ---- Our path: fused kernel + GEMM ----
        our_fp8 = torch.empty(E, T, K, device="cuda", dtype=torch.float8_e4m3fn)
        our_scale_packed = create_packed_scale_tensor(E, T, 2 * K, group_size, "cuda")

        silu_and_mul_masked_post_quant_packed_fwd(
            upgate_output.clone(),
            our_fp8,
            our_scale_packed,
            group_size,
            masked_m,
        )

        output_ours = torch.empty(E, T, N, device="cuda", dtype=torch.bfloat16)
        m_grouped_fp8_gemm_nt_masked(
            (our_fp8, our_scale_packed),
            (w2, w2_scale),
            output_ours,
            masked_m,
            T,
        )

        # ---- Verify FP8 outputs match ----
        fp8_diff = (ref_fp8 != our_fp8).sum().item()
        self.assertEqual(
            fp8_diff, 0, f"FP8 quant output mismatch: {fp8_diff} elements differ"
        )

        # ---- Verify packed scales match ----
        self.assertEqual(
            ref_scale_packed.shape,
            our_scale_packed.shape,
            f"Packed scale shape mismatch: ref={ref_scale_packed.shape}, ours={our_scale_packed.shape}",
        )
        scale_diff = (ref_scale_packed != our_scale_packed).sum().item()
        self.assertEqual(
            scale_diff, 0, f"Packed scale mismatch: {scale_diff} elements differ"
        )

        # ---- Verify GEMM outputs match ----
        max_diff = (output_ref - output_ours).abs().max().item()
        mean_diff = (output_ref - output_ours).abs().mean().item()

        # Should be exactly equal since inputs are identical
        self.assertLess(max_diff, 1e-5, f"GEMM output max diff: {max_diff}")
        self.assertLess(mean_diff, 1e-6, f"GEMM output mean diff: {mean_diff}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    main()
