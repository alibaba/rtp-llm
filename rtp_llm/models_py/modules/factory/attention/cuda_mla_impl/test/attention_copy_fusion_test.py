from types import SimpleNamespace
from unittest import TestCase, main

import torch
from flash_mla import flash_mla_sparse_fwd

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.cp_topk_offsets_triton import (
    offset_cp_topk,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rope_emb_new import (
    NewMlaRotaryEmbeddingOp,
)


class AttentionCopyFusionTest(TestCase):
    def assert_bits(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            )
        )

    def test_real_rope_contiguous_output_and_input_alias_contract(self) -> None:
        torch.manual_seed(124)
        for neox in (False, True):
            angles = torch.randn((64, 32), device="cuda")
            cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
            rope = NewMlaRotaryEmbeddingOp(cache, neox)
            q = torch.randn((17, 8, 256), device="cuda", dtype=torch.bfloat16)
            fused = torch.randn((17, 576), device="cuda", dtype=torch.bfloat16)
            before = fused.clone()
            q_ref, fused_ref = q.clone(), fused.clone()
            ids = torch.arange(17, device="cuda", dtype=torch.int32) * 3
            params = SimpleNamespace(positions_d=ids)
            rope.forward(q_ref[..., 192:], fused_ref[:, 512:], params)
            out = torch.empty((17, 64), device="cuda", dtype=q.dtype)
            rope.forward(q[..., 192:], fused[:, 512:], params, key_output=out)
            self.assert_bits(q, q_ref)
            self.assert_bits(out, fused_ref[:, 512:])
            self.assert_bits(fused, before)
            self.assertTrue(out.is_contiguous())

    def test_real_flashmla_remapped_history_and_graph_refresh_exact(self) -> None:
        torch.manual_seed(125)
        q = torch.randn((4, 64, 576), device="cuda", dtype=torch.bfloat16) * 0.1
        raw = torch.randn((256, 1, 576), device="cuda", dtype=torch.bfloat16) * 0.1
        mapping = torch.cat((torch.arange(48), torch.arange(64, 128))).to(
            device="cuda", dtype=torch.int64
        )
        offsets = torch.tensor([0, 48, 0, 48], device="cuda", dtype=torch.int32)
        topk = torch.arange(128, device="cuda", dtype=torch.int32).repeat(4, 1)
        topk[:, 5] = -1
        topk[:, 9] = 200

        def reference() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            indices = topk + offsets[:, None]
            indices.masked_fill_(topk < 0, -1)
            return flash_mla_sparse_fwd(q, raw[mapping], indices[:, None], 576**-0.5)

        def candidate() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            indices = offset_cp_topk(topk, offsets, mapping, raw.shape[0])
            return flash_mla_sparse_fwd(q, raw, indices[:, None], 576**-0.5)

        for _ in range(3):
            candidate()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = candidate()
        for lengths in ((48, 64), (64, 48)):
            mapping.copy_(
                torch.cat(
                    (torch.arange(lengths[0]), torch.arange(64, 64 + lengths[1]))
                ).to(mapping)
            )
            offsets.copy_(
                torch.tensor(
                    [0, lengths[0], 0, lengths[0]], device="cuda", dtype=torch.int32
                )
            )
            raw.normal_()
            expected = reference()
            graph.replay()
            torch.cuda.synchronize()
            for got, want in zip(actual, expected):
                self.assert_bits(got, want)
            mapped = offset_cp_topk(topk, offsets, mapping, raw.shape[0])
            logical = topk.to(torch.int64) + offsets[:, None]
            valid = (topk >= 0) & (logical >= 0) & (logical < mapping.numel())
            expected_indices = torch.full_like(topk, -1)
            expected_indices[valid] = mapping[logical[valid]].to(torch.int32)
            self.assertTrue(torch.equal(mapped, expected_indices))


if __name__ == "__main__":
    main()
