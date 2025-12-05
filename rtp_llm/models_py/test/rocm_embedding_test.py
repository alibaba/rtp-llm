import itertools
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules import Embedding
from rtp_llm.models_py.modules.base.common.embedding import EmbeddingTorch


class EmbedingTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_embeding_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        w = torch.randn(131072, hidden_size, dtype=dtype)
        embeding = Embedding(
            GptInitModelParameters(
                head_num=1, size_per_head=1, layer_num=1, max_seq_len=1, vocab_size=1
            ),
            w,
        )
        embeding_torch = EmbeddingTorch(w)
        x = torch.randint(0, hidden_size, (num_tokens,), dtype=torch.int32)
        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     for _ in range(10):
        #         out = embeding(x)
        #         # out = embeding_torch(x)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
        self.assertTrue(
            torch.allclose(embeding_torch(x), embeding(x), atol=1e-2, rtol=1e-2)
        )

    def test_embeding(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0], hidden_size=params[1], dtype=params[2]
            ):
                self._run_embeding_test(*params)


if __name__ == "__main__":
    main()
