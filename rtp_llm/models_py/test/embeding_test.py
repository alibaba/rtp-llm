import torch
import itertools
from unittest import TestCase, main, SkipTest
from rtp_llm.models_py.modules import EmbeddingRTP, Embedding
from torch import dtype as _dtype
from torch.profiler import profile, ProfilerActivity, record_function


class EmbedingTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096, 5120]
    HIDDEN_SIZES = [768, 2048, 4096, 5120, 8192]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_embeding_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        w = torch.randn(hidden_size, hidden_size, dtype=dtype)
        embeding = EmbeddingRTP(w)
        embeding_torch = Embedding(w)
        x = torch.randint(0, hidden_size, (num_tokens, ), dtype=torch.int32)
        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     for _ in range(10):
        #         out = embeding(x)
        #         # out = embeding_torch(x)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
        self.assertTrue(torch.allclose(embeding_torch(x), embeding(x), atol=1e-2, rtol=1e-2))

    def test_embeding(self):
        for params in itertools.product(
                self.NUM_TOKENS,
                self.HIDDEN_SIZES,
                self.DTYPES,
        ):
            with self.subTest(
                    num_tokens=params[0],
                    hidden_size=params[1],
                    dtype=params[2]
            ):
                self._run_embeding_test(*params)


if __name__ == '__main__':
    main()
