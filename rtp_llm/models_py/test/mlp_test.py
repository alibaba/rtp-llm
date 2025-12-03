import itertools
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules import DenseMLP, FusedSiluActDenseMLP
from rtp_llm.utils.model_weight import W


class MLPTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096, 5120]
    HIDDEN_SIZES = [768, 2048, 4096, 5120, 8192]

    #
    # DTYPES = [torch.bfloat16]
    # NUM_TOKENS = [5120]
    # HIDDEN_SIZES = [512]
    #

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_mlp_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        model_param = GptInitModelParameters(1, 128, 1, 1, 5120)
        model_param.activation_type = "SiGLU"
        weights = {}
        weights[W.ffn_w1] = torch.randn(hidden_size, 4 * hidden_size, dtype=dtype)
        torch.nn.init.xavier_uniform_(weights[W.ffn_w1])
        weights[W.ffn_w3] = torch.randn(hidden_size, 4 * hidden_size, dtype=dtype)
        torch.nn.init.xavier_uniform_(weights[W.ffn_w3])
        weights[W.ffn_w2] = torch.randn(4 * hidden_size, hidden_size, dtype=dtype)
        torch.nn.init.xavier_uniform_(weights[W.ffn_w2])

        qwen3_mlp = DenseMLP(model_param, weights)
        qwen3_mlp_fused = FusedSiluActDenseMLP(model_param, weights)

        x = torch.randn(num_tokens, hidden_size, dtype=dtype)

        # for _ in range(5):
        #     out = qwen3_mlp(x)
        #     out = qwen3_mlp_fused(x)
        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        # for _ in range(10):
        #     out = qwen3_mlp(x)
        #     out = qwen3_mlp_fused(x)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

        self.assertTrue(
            torch.allclose(qwen3_mlp(x), qwen3_mlp_fused(x), atol=1e-2, rtol=1e-2)
        )

    def test_mlp(self):
        for params in itertools.product(
            self.NUM_TOKENS, self.HIDDEN_SIZES, self.DTYPES
        ):
            with self.subTest(
                num_tokens=params[0], hidden_size=params[1], dtype=params[2]
            ):
                self._run_mlp_test(*params)


if __name__ == "__main__":
    main()
