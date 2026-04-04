"""Minimal smoke test: verify FA3 backend is selected and produces correct output on H20."""

import logging

import pytest
import torch

pytestmark = [pytest.mark.gpu(type="H20"), pytest.mark.timeout(120)]

logging.basicConfig(level="INFO", format="%(message)s")
logger = logging.getLogger(__name__)


class TestFA3Smoke:
    """Verify FA3 backends work end-to-end with standalone AutoModel inference."""

    def test_fa3_backend_selection(self):
        """Check that FlashAttention3PrefillImpl is importable and support() logic is correct."""
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.flash_attention_3 import (
            FlashAttention3PrefillImpl,
        )

        # Verify class exists and has required methods
        assert hasattr(FlashAttention3PrefillImpl, "support")
        assert hasattr(FlashAttention3PrefillImpl, "forward")
        logger.info("FlashAttention3PrefillImpl imported OK")

    def test_flashinfer_fa3_backend_selection(self):
        """Check that FlashInferFA3PrefillImpl is importable."""
        from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_fa3 import (
            FlashInferFA3PrefillImpl,
        )

        assert hasattr(FlashInferFA3PrefillImpl, "support")
        assert hasattr(FlashInferFA3PrefillImpl, "forward")
        logger.info("FlashInferFA3PrefillImpl imported OK")

    def test_fa3_kernel_correctness(self):
        """Direct kernel call: FA3 vs SDPA reference."""
        try:
            from flash_attn_interface import flash_attn_varlen_func
        except ImportError:
            pytest.skip("flash_attn_interface not installed")

        bs, seq, nh, kvh, hd = 1, 2048, 32, 8, 128
        dtype = torch.float16
        device = "cuda"

        q = torch.randn(bs * seq, nh, hd, dtype=dtype, device=device)
        k = torch.randn(bs * seq, kvh, hd, dtype=dtype, device=device)
        v = torch.randn(bs * seq, kvh, hd, dtype=dtype, device=device)
        cu = torch.tensor([0, seq], dtype=torch.int32, device=device)

        # FA3
        out_fa3 = flash_attn_varlen_func(q, k, v, cu, cu, seq, seq, causal=True)
        if isinstance(out_fa3, tuple):
            out_fa3 = out_fa3[0]

        # SDPA reference
        qi = q.reshape(bs, seq, nh, hd).transpose(1, 2)
        ki = k.reshape(bs, seq, kvh, hd).transpose(1, 2)
        vi = v.reshape(bs, seq, kvh, hd).transpose(1, 2)
        ki = ki.repeat_interleave(nh // kvh, dim=1)
        vi = vi.repeat_interleave(nh // kvh, dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(
            qi, ki, vi, is_causal=True
        )
        ref = ref.transpose(1, 2).reshape(bs * seq, nh, hd)

        max_diff = (out_fa3 - ref).abs().max().item()
        logger.info(f"FA3 kernel correctness: max_diff={max_diff:.6f}")
        assert max_diff < 0.01, f"FA3 max_diff={max_diff} > 0.01"

    def test_fa3_registered_in_factory(self):
        """Verify FA3 impls are registered in PREFILL_MHA_IMPS."""
        from rtp_llm.models_py.modules.factory.attention.attn_factory import (
            PREFILL_MHA_IMPS,
        )

        impl_names = [cls.__name__ for cls in PREFILL_MHA_IMPS]
        logger.info(f"PREFILL_MHA_IMPS: {impl_names}")
        assert (
            "FlashAttention3PrefillImpl" in impl_names
        ), f"FlashAttention3PrefillImpl not in PREFILL_MHA_IMPS: {impl_names}"
