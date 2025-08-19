import torch
import itertools
from unittest import TestCase, main, SkipTest
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.utils.model_weight import W
from rtp_llm.models_py.modules.ep.topk import select_experts
from torch import dtype as _dtype
import torch.nn.functional as F
import math
from torch.profiler import profile, ProfilerActivity, record_function


class MoESelectTopkOpTest(TestCase):
    DTYPES = [torch.float32, torch.float16]
    NUM_TOKENS = [7, 83, 4096, 5120]
    HIDDEN_DIMS = [768, 2048]
    NUM_EXPERT = [64, 128]
    TOP_K = [2, 5, 10, 32, 128]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")


    def _run_select_topk_op_test(self, num_tokens: int, num_expert: int, hidden_dim: int, top_k: int, dtype: _dtype):
        torch.manual_seed(2)
        hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype).to('cuda')
        router_logits = torch.randn(num_tokens, num_expert, dtype=dtype).to('cuda')
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(10):
                cuda_topk_weights, cuda_topk_ids = select_experts(hidden_states, router_logits, top_k=top_k, renormalize=False, use_grouped_topk=False, torch_native=False)
                torch_topk_weights, torch_topk_ids = select_experts(hidden_states, router_logits, top_k=top_k, renormalize=False, use_grouped_topk=False, torch_native=True)
                
                torch.set_printoptions(threshold=float('inf'))
                
                self.assertTrue(torch.allclose(cuda_topk_weights, torch_topk_weights, atol=1e-1, rtol=1e-1))
                # self.assertTrue(torch.allclose(cuda_topk_ids.long(), torch_topk_ids.long(), atol=1e-1, rtol=1e-1))
                
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))


    def test_select_topk(self):
        for params in itertools.product(
                self.NUM_TOKENS,
                self.NUM_EXPERT,
                self.HIDDEN_DIMS,
                self.TOP_K,
                self.DTYPES
        ):
            with self.subTest(
                    num_tokens=params[0],
                    num_expert=params[1],
                    hidden_dim=params[2],
                    top_k=params[3],
                    dtype=params[4]
            ):
                top_k, num_expert = params[3], params[1]
                if top_k > num_expert:
                    continue
                self._run_select_topk_op_test(*params)


if __name__ == '__main__':
    main()
