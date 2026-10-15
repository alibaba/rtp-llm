import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.weight_manager import WeightManager
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models_py.modules.qwen4_exp.vocab_embedding import Qwen4Fp8VocabEmbedding


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class Fp8VocabEmbeddingTest(unittest.TestCase):
    def make_embedding(self, width=2560, tp=1):
        torch.manual_seed(671)
        weight = torch.randn(257, width, device="cuda").to(torch.float8_e4m3fn)
        scale = torch.tensor([0.17], device="cuda", dtype=torch.float32)
        module = Qwen4Fp8VocabEmbedding(weight, tp, weight.device, weight_scale=scale)
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

    def test_invalid_token_rows_are_zero(self):
        _, module = self.make_embedding()
        invalid = torch.tensor([-7, -1, 257, 900], device="cuda", dtype=torch.int64)
        torch.testing.assert_close(
            module(invalid),
            torch.zeros(4, 2560, device="cuda", dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )

    def test_tp_order_matches_native_embedding(self):
        reference, module = self.make_embedding(1280, tp=2)
        ids = torch.tensor([17, 128], device="cuda", dtype=torch.int32)
        peer = reference + 1

        def gather(local, group):
            return torch.cat((local, F.embedding(ids.long(), peer)), dim=0)

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.vocab_embedding.all_gather", gather
        ):
            output = module(ids)
        expected = torch.cat(
            (F.embedding(ids.long(), reference), F.embedding(ids.long(), peer)), dim=-1
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_target_draft_aliases_use_gpu_storage_on_replay(self):
        _, target = self.make_embedding()
        weight = target.weight
        draft = Qwen4Fp8VocabEmbedding(weight, 1, weight.device, target.weight_scale)
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
                Qwen4Fp8VocabEmbedding(
                    weight, 1, torch.device("cuda", 0), torch.ones(1, device="cuda")
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class WeightResidencyContractTest(unittest.TestCase):

    def test_loader_accepts_pinned_cpu_embedding_and_cuda_lm_head_aliases(self):
        loader = ModelLoader.__new__(ModelLoader)
        loader._model_weights_info = SimpleNamespace(
            weights=[
                SimpleNamespace(
                    name="embedding", resident_device="cpu", pinned_memory=True
                ),
                SimpleNamespace(
                    name="lm_head", resident_device=None, pinned_memory=False
                ),
            ]
        )
        loader._load_config = SimpleNamespace(is_ft_style_weight=True)
        sentinel = object()
        loader._load_from_ft_style = lambda device: sentinel
        loader._load_dynamic_weights = lambda *args: None
        loader._init_eplb_weight = lambda *args: None
        embedding = torch.randn(7, 8, dtype=torch.bfloat16).pin_memory()
        head = torch.randn(7, 8, dtype=torch.bfloat16, device="cuda")
        self.assertIs(
            loader.load_weights("cuda:0", {"embedding": embedding, "lm_head": head}),
            sentinel,
        )
        self.assertEqual(
            loader._global_weight_aliases["embedding"].data_ptr(), embedding.data_ptr()
        )
        with self.assertRaisesRegex(ValueError, "expected cuda"):
            loader.load_weights(
                "cuda:0", {"embedding": embedding, "lm_head": embedding}
            )
        with self.assertRaisesRegex(ValueError, "pinned CPU storage"):
            loader.load_weights("cuda:0", {"embedding": torch.randn(7, 8)})

    def test_descriptor_load_materializes_pinned_owner_without_cuda_staging(self):
        class PinnedWeight(AtomicWeight):
            resident_device = "cpu"
            pinned_memory = True

            @classmethod
            def support(cls, quant_config, src_weight_info):
                return False

            def _load_raw_tensor(self, source, layer_id, device, load_config):
                self.received_device = device
                return {"embedding": torch.randn(7, 8, dtype=torch.bfloat16)}

            def _split(self, tensors, load_config):
                return tensors

            def _postprocess(self, tensors, device, load_config):
                return tensors

        descriptor = PinnedWeight("embedding", weights=[])
        before = torch.cuda.memory_allocated()
        result = descriptor.load(
            None, None, "cuda:0", SimpleNamespace(merge_lora=False)
        )
        self.assertEqual(descriptor.received_device, "cpu")
        self.assertTrue(result["embedding"].is_pinned())
        self.assertEqual(torch.cuda.memory_allocated(), before)

    def test_online_host_weight_update_rejects_before_ipc_decode(self):
        manager = WeightManager.__new__(WeightManager)
        descriptor = AtomicWeight("embedding", weights=[])
        descriptor.resident_device = "cpu"
        descriptor.pinned_memory = True
        manager._weight_module = SimpleNamespace(weights=[descriptor])
        with self.assertRaisesRegex(PermissionError, "service restart"):
            manager.update(
                dict(name="embed_tokens.weight", method="shm", desc="not decoded")
            )


if __name__ == "__main__":
    unittest.main()
