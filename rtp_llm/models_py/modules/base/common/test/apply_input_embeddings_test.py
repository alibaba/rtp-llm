import types
from unittest import SkipTest, TestCase, main

import torch

from rtp_llm.models_py.model_desc.module_base import GptModelBase


class _OverlayModel(GptModelBase):
    def __init__(self):
        torch.nn.Module.__init__(self)

    def apply(self, inputs_embeds, input_embeddings, input_embeddings_locs):
        inputs = types.SimpleNamespace(
            input_embeddings=input_embeddings,
            input_embeddings_locs=input_embeddings_locs,
        )
        return self.apply_input_embeddings(inputs_embeds.clone(), inputs)


class ApplyInputEmbeddingsTest(TestCase):
    def test_multiple_embeddings_scatter_and_preserve_gaps(self):
        model = _OverlayModel()
        inputs_embeds = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        emb1 = torch.full((2, 4), 101.0)
        emb2 = torch.full((1, 4), 202.0)

        result = model.apply(
            inputs_embeds,
            input_embeddings=[emb1, emb2],
            input_embeddings_locs=torch.tensor([1, 5], dtype=torch.int64),
        )

        self.assertTrue(torch.equal(result[0:1], inputs_embeds[0:1]))
        self.assertTrue(torch.equal(result[1:3], emb1))
        self.assertTrue(torch.equal(result[3:5], inputs_embeds[3:5]))
        self.assertTrue(torch.equal(result[5:6], emb2))

    def test_1d_embedding_overlays_single_token(self):
        model = _OverlayModel()
        inputs_embeds = torch.zeros(3, 4)
        emb = torch.tensor([1.0, 2.0, 3.0, 4.0])

        result = model.apply(
            inputs_embeds,
            input_embeddings=[emb],
            input_embeddings_locs=torch.tensor([2], dtype=torch.int64),
        )

        self.assertTrue(torch.equal(result[0:2], inputs_embeds[0:2]))
        self.assertTrue(torch.equal(result[2:3], emb.unsqueeze(0)))

    def test_same_device_copy_converts_dtype(self):
        model = _OverlayModel()
        inputs_embeds = torch.zeros(3, 4, dtype=torch.float32)
        emb = torch.full((1, 4), 7.5, dtype=torch.float16)

        result = model.apply(
            inputs_embeds,
            input_embeddings=[emb],
            input_embeddings_locs=torch.tensor([1], dtype=torch.int64),
        )

        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.equal(result[1:2], emb.to(torch.float32)))


class ApplyInputEmbeddingsCudaTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")

    def test_cpu_embedding_copies_to_cuda_target_and_converts_dtype(self):
        model = _OverlayModel()
        inputs_embeds = torch.zeros(3, 4, device="cuda", dtype=torch.float16)
        emb = torch.full((2, 4), 3.25, device="cpu", dtype=torch.float32)

        result = model.apply(
            inputs_embeds,
            input_embeddings=[emb],
            input_embeddings_locs=torch.tensor([1], dtype=torch.int64),
        )

        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(
            torch.equal(result[1:3], emb.to(device=result.device, dtype=result.dtype))
        )

    def test_cuda_embedding_copies_to_target_cuda_device(self):
        if torch.cuda.device_count() < 2:
            raise SkipTest("requires at least two CUDA devices")

        model = _OverlayModel()
        inputs_embeds = torch.zeros(3, 4, device="cuda:1", dtype=torch.float16)
        emb = torch.full((1, 4), 9.5, device="cuda:0", dtype=torch.float32)

        result = model.apply(
            inputs_embeds,
            input_embeddings=[emb],
            input_embeddings_locs=torch.tensor([1], device="cuda:0", dtype=torch.int32),
        )

        self.assertEqual(result.device, inputs_embeds.device)
        self.assertEqual(result.dtype, inputs_embeds.dtype)
        self.assertTrue(
            torch.equal(result[1:2], emb.to(device=result.device, dtype=result.dtype))
        )


if __name__ == "__main__":
    main()
