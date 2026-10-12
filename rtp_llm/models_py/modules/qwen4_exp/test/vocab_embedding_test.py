import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from rtp_llm.model_loader.loader import ModelLoader
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.weight_manager import WeightManager
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models_py.modules.qwen4_exp.vocab_embedding import (
    Qwen4CpuVocabEmbedding,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CpuVocabEmbeddingTest(unittest.TestCase):
    def make_embedding(self, width=2560, dtype=torch.bfloat16, tp=1):
        torch.manual_seed(671)
        weight = torch.randn(257, width, dtype=dtype).pin_memory()
        module = Qwen4CpuVocabEmbedding(weight, tp, torch.device("cuda", 0))
        self.assertTrue(module.weight.is_pinned())
        self.assertEqual(module.weight.device.type, "cpu")
        self.assertEqual(module.weight.data_ptr(), weight.data_ptr())
        return weight, module

    def test_exact_eager_column_tiles_and_dtypes(self):
        for width in (2560, 1280, 640, 320, 257):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(width=width, dtype=dtype):
                    weight, module = self.make_embedding(width, dtype)
                    ids = torch.tensor(
                        [0, 256, 17, 17, 128], device="cuda", dtype=torch.int32
                    )
                    torch.testing.assert_close(
                        module(ids),
                        F.embedding(ids.cpu().long(), weight).cuda(),
                        rtol=0,
                        atol=0,
                    )

    def test_graph_replay_uses_live_token_ids_and_mask(self):
        weight, module = self.make_embedding()
        ids = torch.tensor([0, 1, 17, 256], device="cuda", dtype=torch.int32)
        mask = torch.ones_like(ids, dtype=torch.bool)
        for _ in range(3):
            module(ids, text_tokens_mask=mask)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = module(ids, text_tokens_mask=mask)
        for values, masks in [
            ([128, 256, 17, 0], [True, False, True, True]),
            ([256, 2, 0, 255], [False, True, True, False]),
        ]:
            ids.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
            mask.copy_(torch.tensor(masks, device="cuda"))
            graph.replay()
            torch.cuda.synchronize()
            expected = F.embedding(torch.tensor(values), weight).cuda()
            expected[~mask] = 0
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_target_draft_alias_and_weight_update_share_pinned_owner(self):
        weight, target = self.make_embedding()
        draft = Qwen4CpuVocabEmbedding(weight, 1, torch.device("cuda", 0))
        self.assertEqual(draft.weight.data_ptr(), target.weight.data_ptr())
        ids = torch.tensor([0, 17, 256], device="cuda", dtype=torch.int32)
        for _ in range(3):
            target(ids)
            draft(ids)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            target_output = target(ids)
            draft_output = draft(ids)
        weight.fill_(7)
        graph.replay()
        torch.cuda.synchronize()
        expected = F.embedding(ids.cpu().long(), weight).cuda()
        torch.testing.assert_close(target_output, expected, rtol=0, atol=0)
        torch.testing.assert_close(draft_output, expected, rtol=0, atol=0)

    def test_side_stream_empty_and_invalid_ids(self):
        weight, module = self.make_embedding()
        ids = torch.tensor([17, 128, 256], device="cuda", dtype=torch.int64)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            output = module(ids)
        torch.cuda.current_stream().wait_stream(side)
        torch.testing.assert_close(
            output, F.embedding(ids.cpu(), weight).cuda(), rtol=0, atol=0
        )
        self.assertEqual(tuple(module(ids[:0]).shape), (0, 2560))
        invalid = torch.tensor([-1, 257, -(2**63), 2**63 - 1], device="cuda")
        torch.testing.assert_close(
            module(invalid),
            torch.zeros(4, 2560, device="cuda", dtype=weight.dtype),
            rtol=0,
            atol=0,
        )

    def test_tp_order_matches_native_embedding(self):
        weight, module = self.make_embedding(1280, tp=2)
        ids = torch.tensor([17, 128], device="cuda", dtype=torch.int32)
        peer = (weight + 1).cuda()

        def gather(local, group):
            return torch.cat((local, F.embedding(ids.long(), peer)), dim=0)

        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.vocab_embedding.all_gather", gather
        ):
            output = module(ids)
        expected = torch.cat(
            (
                F.embedding(ids.cpu().long(), weight).cuda(),
                F.embedding(ids.long(), peer),
            ),
            dim=-1,
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_rejects_invalid_layout(self):
        for weight in (torch.randn(7, 8, dtype=torch.float16), torch.randn(7, 8).t()):
            with self.assertRaises(ValueError):
                Qwen4CpuVocabEmbedding(weight, 1, torch.device("cuda", 0))

    def test_constructor_keeps_table_out_of_gpu_memory(self):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        _, module = self.make_embedding()
        torch.cuda.synchronize()
        self.assertLess(torch.cuda.max_memory_allocated() - before, 128 * 1024)
        self.assertGreater(
            module.weight.numel() * module.weight.element_size(), 1024 * 1024
        )

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

    def test_ft_materialization_pins_cpu_embedding_and_keeps_head_cuda(self):
        loader = ModelLoader.__new__(ModelLoader)
        embedding = AtomicWeight("embedding", weights=[])
        embedding.resident_device = "cpu"
        embedding.pinned_memory = True
        head = AtomicWeight("lm_head", weights=[])
        loader._model_weights_info = SimpleNamespace(
            weights=[embedding, head], layer_weights=[]
        )
        loader._misc_weights_info = []
        loader._global_weight_aliases = {}
        source = torch.randn(7, 8, dtype=torch.bfloat16)

        def load_prefix(prefixes, device, direct_io):
            self.assertEqual(device, "cpu")
            self.assertFalse(direct_io)
            prefix = ModelWeights.global_weight_prefix(0, 0, 0)
            return {
                prefix + "embedding": [source],
                prefix + "lm_head": [source.clone()],
            }

        loader._load_config = SimpleNamespace(
            num_layers=0,
            tp_rank=0,
            dp_rank=0,
            ep_rank=0,
            compute_dtype=torch.bfloat16,
            exported_device=SimpleNamespace(support_dio_load=True),
            database=SimpleNamespace(load_tensors_by_prefix=load_prefix),
        )
        weights = loader._load_from_ft_style("cuda:0")
        self.assertTrue(weights.global_weights["embedding"].is_pinned())
        self.assertEqual(weights.global_weights["embedding"].device.type, "cpu")
        self.assertEqual(weights.global_weights["lm_head"].device.type, "cuda")
        torch.testing.assert_close(
            weights.global_weights["embedding"], source, rtol=0, atol=0
        )

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
