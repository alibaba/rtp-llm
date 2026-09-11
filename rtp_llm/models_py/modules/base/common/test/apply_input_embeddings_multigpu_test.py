from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.base.common.test.apply_input_embeddings_test import (
    _OverlayModel,
)


class ApplyInputEmbeddingsMultiGpuTest(TestCase):
    def test_cuda_embedding_copies_to_target_cuda_device(self):
        self.assertGreaterEqual(
            torch.cuda.device_count(), 2, "target requires two CUDA devices"
        )

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
