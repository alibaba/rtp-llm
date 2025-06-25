import torch
import itertools
from unittest import TestCase, main, SkipTest
from rtp_llm.models_py.modules import EmbeddingTorch, Embedding
from torch import dtype as _dtype


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
        w = torch.randn(hidden_size, hidden_size, dtype=dtype)
        embeding = Embedding(w)
        embeding_torch = EmbeddingTorch(w)
        x = torch.randint(0, hidden_size, (num_tokens, ), dtype=torch.int32)
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
