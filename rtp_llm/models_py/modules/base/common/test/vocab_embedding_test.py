import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.borrowed_parameters import (
    BorrowedParameter,
    FixedHostCheckpointStorage,
    bind_borrowed_parameters,
    validate_borrowed_parameters,
)
from rtp_llm.models_py.layers.embedding import VocabParallelEmbedding
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.base.common.fp8_embedding import Fp8VocabEmbedding


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Fp8VocabEmbeddingTest(unittest.TestCase):
    def test_output_element_count_crossing_signed_int32_boundary(self):
        reference, module = self.make_embedding()
        # N itself still fits int32, while N * D crosses 2**31. Inspect every
        # output element in bounded chunks without allocating a second 4 GiB tensor.
        for rows in (2**31 // module.hidden_size, 2**31 // module.hidden_size + 1):
            with self.subTest(rows=rows):
                ids = torch.arange(rows, device="cuda", dtype=torch.int32) % 257
                mask = ids % 2 == 0
                output = module(ids, text_tokens_mask=mask)
                for begin in range(0, rows, 4096):
                    end = min(begin + 4096, rows)
                    expected = F.embedding(ids[begin:end].long(), reference)
                    expected[~mask[begin:end]] = 0
                    torch.testing.assert_close(
                        output[begin:end], expected, rtol=0, atol=0
                    )
                del output

    def make_embedding(self, width=2560, tp=1):
        torch.manual_seed(671)
        weight = torch.randn(257, width, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.tensor([0.17], device="cuda", dtype=torch.float32)
        module = Fp8VocabEmbedding(weight, tp, weight.device, weight_scale=scale)
        self.assertEqual(module.weight.data_ptr(), weight.data_ptr())
        # The accepted lookup ABI rounds the FP8 value and scale to BF16
        # before multiplication, matching the checkpoint BF16 compute path.
        return weight.to(torch.bfloat16) * scale.to(torch.bfloat16), module

    def test_eager_column_tiles_match_independent_dequantized_reference(self):
        for width in (2560, 1280, 640, 320, 257):
            with self.subTest(width=width):
                reference, module = self.make_embedding(width)
                ids = torch.tensor(
                    [0, 256, 17, 17, 128], device="cuda", dtype=torch.int32
                )
                torch.testing.assert_close(
                    module(ids), F.embedding(ids.long(), reference), rtol=0, atol=0
                )

    def test_graph_replay_uses_live_ids_and_masks(self):
        reference, module = self.make_embedding()
        ids = torch.tensor([0, 1, 17, 256], device="cuda", dtype=torch.int32)
        mask = torch.ones_like(ids, dtype=torch.bool)
        for _ in range(3):
            module(ids, text_tokens_mask=mask)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module(ids, text_tokens_mask=mask)
        for next_ids in ([5, 6, 9, 10], [256, 17, 0, 1]):
            ids.copy_(torch.tensor(next_ids, device="cuda", dtype=torch.int32))
            mask.copy_(torch.tensor([True, False, True, False], device="cuda"))
            graph.replay()
            expected = F.embedding(ids.long(), reference)
            expected = torch.where(
                mask.unsqueeze(-1), expected, torch.zeros_like(expected)
            )
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_target_embedding_preserves_bf16_mask_and_column_contract(self):
        reference, module = self.make_embedding(1280, tp=2)
        ids = torch.tensor([5, 17, 0, 256], device="cuda", dtype=torch.int32)
        mask = torch.tensor([True, False, True, False], device="cuda")

        def gather(local, group):
            return torch.cat((local, (local.float() + 0.25).to(local.dtype)), 0)

        with patch(
            "rtp_llm.models_py.modules.base.common.fp8_embedding.all_gather",
            side_effect=gather,
        ):
            expected = module(ids, text_tokens_mask=mask).repeat(1, 4)
            actual = module.forward_hyper(ids, branches=4, text_tokens_mask=mask)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_token_rows_are_zero(self):
        _, module = self.make_embedding()
        invalid = torch.tensor([-7, -1, 257, 900], device="cuda", dtype=torch.int64)
        torch.testing.assert_close(
            module(invalid),
            torch.zeros(4, 2560, device="cuda", dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )

    def test_strided_ids_masks_empty_and_tail(self):
        reference, module = self.make_embedding(1280)
        for dtype in (torch.int32, torch.int64):
            ids = torch.tensor(
                [1, 0, -1, 0, 256, 0, 257, 0, 17, 0], device="cuda", dtype=dtype
            )[::2]
            mask = torch.tensor(
                [1, 3, 1, 3, 0, 3, 1, 3, -1, 3], device="cuda", dtype=torch.int32
            )[::2]
            valid = (ids >= 0) & (ids < 257) & mask.bool()
            expected = F.embedding(ids.clamp(0, 256).long(), reference)
            expected = torch.where(valid[:, None], expected, 0)
            torch.testing.assert_close(
                module(ids, text_tokens_mask=mask), expected, rtol=0, atol=0
            )
            self.assertEqual(module(ids[:0]).shape, (0, 1280))

    def test_tp_order_matches_native_embedding(self):
        reference, module = self.make_embedding(1280, tp=2)
        ids = torch.tensor([17, 128], device="cuda", dtype=torch.int32)
        peer = reference + 1

        def gather(local, group):
            return torch.cat((local, F.embedding(ids.long(), peer)), dim=0)

        with patch(
            "rtp_llm.models_py.modules.base.common.fp8_embedding.all_gather", gather
        ):
            output = module(ids)
        expected = torch.cat(
            (F.embedding(ids.long(), reference), F.embedding(ids.long(), peer)), dim=-1
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_target_draft_aliases_use_gpu_storage_on_replay(self):
        _, target = self.make_embedding()
        weight = target.weight
        draft = Fp8VocabEmbedding(weight, 1, weight.device, target.weight_scale)
        self.assertEqual(draft.weight.data_ptr(), target.weight.data_ptr())
        ids = torch.tensor([0, 17, 256], device="cuda", dtype=torch.int32)
        for _ in range(3):
            target(ids)
            draft(ids)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            target_output = target(ids)
            draft_output = draft(ids)
        # Direct mutation is a graph alias test; serving weight updates require restart.
        weight.fill_(7)
        graph.replay()
        expected = F.embedding(
            ids.long(),
            weight.to(torch.bfloat16) * target.weight_scale.to(torch.bfloat16),
        )
        torch.testing.assert_close(target_output, expected, rtol=0, atol=0)
        torch.testing.assert_close(draft_output, expected, rtol=0, atol=0)

    def test_cpu_or_non_fp8_tables_are_rejected(self):
        for weight in (
            torch.randn(7, 8).pin_memory(),
            torch.randn(7, 8, device="cuda"),
        ):
            with self.assertRaises(ValueError):
                Fp8VocabEmbedding(
                    weight, 1, torch.device("cuda", 0), torch.ones(1, device="cuda")
                )


class _HostEmbedding(FixedHostCheckpointStorage, VocabParallelEmbedding):
    pass


class _VocabOwner(RtpModule):
    def __init__(self):
        super().__init__()
        self.embedding = _HostEmbedding(7, 8, params_dtype=torch.bfloat16)
        self.head = VocabParallelEmbedding(7, 8, params_dtype=torch.bfloat16)

    def prepare_borrowed_parameters(self, bindings):
        if set(bindings) != {"embedding.weight", "head.weight"}:
            raise ValueError("borrow the complete vocabulary")


class WeightResidencyContractTest(unittest.TestCase):
    def test_fixed_host_embedding_does_not_migrate_with_parent(self):
        owner = _VocabOwner()
        values = torch.arange(56).reshape(7, 8).bfloat16()
        owner.load_weights({"embedding.weight": values, "head.weight": values + 1})
        before = owner.embedding.weight.data_ptr()
        owner.to("meta")  # Verify allocator dispatch without claiming CUDA coverage.
        self.assertTrue(owner.head.weight.is_meta)
        self.assertEqual(owner.embedding.weight.device.type, "cpu")
        self.assertEqual(owner.embedding.weight.data_ptr(), before)
        torch.testing.assert_close(owner.embedding.weight, values)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cpu_embedding_cuda_head_loan_preserves_both_owners(self):
        owner = _VocabOwner()
        values = torch.arange(56).reshape(7, 8).bfloat16()
        owner.load_weights({"embedding.weight": values, "head.weight": values + 1})
        owner.to("cuda")
        self.assertEqual(owner.embedding.weight.device.type, "cpu")
        self.assertTrue(owner.head.weight.is_cuda)
        draft = _VocabOwner()
        loans = {
            key: BorrowedParameter(owner, key)
            for key in ("embedding.weight", "head.weight")
        }
        bind_borrowed_parameters(draft, loans, "cuda")
        draft.to("cuda")
        validate_borrowed_parameters(draft, "cuda")
        self.assertIs(draft.head.weight, owner.head.weight)
        self.assertIs(draft.embedding.weight, owner.embedding.weight)
        loans["head.weight"] = BorrowedParameter(owner, "embedding.weight")
        with self.assertRaisesRegex(ValueError, "residency"):
            bind_borrowed_parameters(_VocabOwner(), loans, "cuda")


if __name__ == "__main__":
    unittest.main()
