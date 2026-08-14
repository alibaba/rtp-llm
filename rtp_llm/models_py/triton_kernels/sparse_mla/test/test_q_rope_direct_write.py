"""Correctness tests for GLM5 FP8 Q-RoPE direct output."""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rope_emb_new import (
    NewMlaRotaryEmbeddingOp,
)
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_qk_rope_cat_cache_mla import (
    fused_qk_rope_cat_cache_mla,
    supports_q_rope_direct_write,
)

HEADS = 64
NOPE = 192
ROPE = 64
KV_LORA = 512
QK_DIM = NOPE + ROPE
Q_OUT_DIM = KV_LORA + ROPE
BLOCK_SIZE = 64
FP8_SLOT_BYTES = 656


def _cos_sin_cache(device: torch.device) -> torch.Tensor:
    inv = 1.0 / (
        10000.0
        ** (
            torch.arange(0, ROPE, 2, device=device, dtype=torch.float32)
            / ROPE
        )
    )
    positions = torch.arange(16384.0, device=device)
    return torch.cat(
        [torch.outer(positions, inv).cos(), torch.outer(positions, inv).sin()],
        dim=-1,
    )


def _make_inputs(tokens: int, seed: int = 0) -> tuple[torch.Tensor, ...]:
    device = torch.device("cuda")
    torch.manual_seed(seed)
    pages = max(1, (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE)
    q = torch.randn(
        tokens,
        HEADS,
        QK_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    compressed_kv = torch.randn(
        tokens,
        KV_LORA,
        dtype=torch.bfloat16,
        device=device,
    )
    k_pe = torch.randn(
        tokens,
        ROPE,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.full(
        (pages, BLOCK_SIZE, FP8_SLOT_BYTES),
        0xAA,
        dtype=torch.uint8,
        device=device,
    )
    slot_mapping = torch.arange(tokens, dtype=torch.int64, device=device)
    if tokens:
        slot_mapping[::7] = -1
    positions = torch.randint(
        0,
        16383,
        (tokens,),
        dtype=torch.int32,
        device=device,
    )
    return (
        q,
        compressed_kv,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        _cos_sin_cache(device),
    )


def _run(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_transformed: torch.Tensor | None = None,
    is_neox_style: bool = True,
) -> None:
    fused_qk_rope_cat_cache_mla(
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        kv_lora_rank=KV_LORA,
        rope_head_dim=ROPE,
        is_neox_style=is_neox_style,
        kv_cache_type="fp8_ds_mla",
        q_transformed=q_transformed,
    )


class QRoPEDirectWriteContractTest(unittest.TestCase):
    def test_rejects_non_cuda_contract(self) -> None:
        q = torch.empty((1, HEADS, QK_DIM), dtype=torch.bfloat16)
        output = torch.empty((1, HEADS, Q_OUT_DIM), dtype=torch.bfloat16)
        self.assertFalse(
            supports_q_rope_direct_write(
                q,
                output,
                kv_lora_rank=KV_LORA,
                rope_head_dim=ROPE,
                is_neox_style=True,
                kv_cache_type="fp8_ds_mla",
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class QRoPEDirectWriteCudaTest(unittest.TestCase):
    def _check_exact(self, tokens: int, is_neox_style: bool) -> None:
        inputs = _make_inputs(tokens, seed=20260730 + tokens)
        q, compressed_kv, k_pe, kv_cache, slot_mapping, positions, cache = inputs
        q_before = q.clone()

        q_ref = q.clone()
        k_pe_ref = k_pe.clone()
        kv_cache_ref = kv_cache.clone()
        _run(
            q_ref,
            compressed_kv,
            k_pe_ref,
            kv_cache_ref,
            slot_mapping,
            positions,
            cache,
            is_neox_style=is_neox_style,
        )

        q_direct = q.clone()
        k_pe_direct = k_pe.clone()
        kv_cache_direct = kv_cache.clone()
        q_output = torch.empty(
            (tokens, HEADS, Q_OUT_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        )
        _run(
            q_direct,
            compressed_kv,
            k_pe_direct,
            kv_cache_direct,
            slot_mapping,
            positions,
            cache,
            q_transformed=q_output,
            is_neox_style=is_neox_style,
        )

        weight = torch.randn(
            HEADS,
            NOPE,
            KV_LORA,
            dtype=torch.bfloat16,
            device=q.device,
        )
        q_output_ref = torch.empty_like(q_output)
        q_output_ref[..., KV_LORA:] = q_ref[..., NOPE:]
        if tokens:
            torch.bmm(
                q_ref[..., :NOPE].transpose(0, 1),
                weight,
                out=q_output_ref[..., :KV_LORA].transpose(0, 1),
            )
            torch.bmm(
                q_direct[..., :NOPE].transpose(0, 1),
                weight,
                out=q_output[..., :KV_LORA].transpose(0, 1),
            )
        torch.cuda.synchronize()

        torch.testing.assert_close(q_direct, q_before, rtol=0, atol=0)
        torch.testing.assert_close(k_pe_direct, k_pe_ref, rtol=0, atol=0)
        torch.testing.assert_close(kv_cache_direct, kv_cache_ref, rtol=0, atol=0)
        torch.testing.assert_close(q_output, q_output_ref, rtol=0, atol=0)

    def test_exact_small_prime_and_glm5_shapes(self) -> None:
        for is_neox_style in (False, True):
            for tokens in (0, 1, 17, 257):
                with self.subTest(
                    tokens=tokens, is_neox_style=is_neox_style
                ):
                    self._check_exact(tokens, is_neox_style)

    def test_flashinfer_interleaved_output_view_exact(self) -> None:
        tokens = 257
        inputs = _make_inputs(tokens, seed=20260814)
        q, _, k_pe, _, _, positions, cache = inputs
        q_pe = q[..., NOPE:]
        q_before = q_pe.clone()
        q_ref = q_pe.clone()
        k_ref = k_pe.clone()
        rope_op = NewMlaRotaryEmbeddingOp(cache, is_neox_style=False)
        rope_op.forward(q_ref, k_ref, None, precomputed_pos_ids=positions)

        q_direct = q_pe.clone()
        k_direct = k_pe.clone()
        q_output = torch.full(
            (tokens, HEADS, Q_OUT_DIM),
            1.0,
            dtype=torch.bfloat16,
            device=q.device,
        )
        rope_op.forward(
            q_direct,
            k_direct,
            None,
            precomputed_pos_ids=positions,
            q_rope_output=q_output[..., KV_LORA:],
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(q_direct, q_before, rtol=0, atol=0)
        torch.testing.assert_close(k_direct, k_ref, rtol=0, atol=0)
        torch.testing.assert_close(
            q_output[..., KV_LORA:], q_ref, rtol=0, atol=0
        )
        torch.testing.assert_close(
            q_output[..., :KV_LORA],
            torch.ones_like(q_output[..., :KV_LORA]),
            rtol=0,
            atol=0,
        )

    def test_cuda_graph_capture_and_replay(self) -> None:
        inputs = _make_inputs(17, seed=20260731)
        q, compressed_kv, k_pe, kv_cache, slot_mapping, positions, cache = inputs
        # Direct-write owns only the RoPE suffix. Initialize the BMM prefix
        # with a finite sentinel so the full-tensor replay check also verifies
        # that capture/replay does not overwrite memory outside that suffix.
        output = torch.full(
            (17, HEADS, Q_OUT_DIM),
            1.0,
            dtype=torch.bfloat16,
            device=q.device,
        )

        _run(*inputs, q_transformed=output, is_neox_style=False)
        torch.cuda.synchronize()
        output_ptr = output.data_ptr()
        expected = output.clone()
        q_before = q.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _run(*inputs, q_transformed=output, is_neox_style=False)
        graph.replay()
        torch.cuda.synchronize()

        self.assertEqual(output.data_ptr(), output_ptr)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(q, q_before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
