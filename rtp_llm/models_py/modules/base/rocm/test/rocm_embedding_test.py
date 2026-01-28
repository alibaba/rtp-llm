import torch
from torch import dtype as _dtype

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules import Embedding
from rtp_llm.models_py.modules.base.common.embedding import EmbeddingTorch
from rtp_llm.ops import ParallelismConfig


import pytest
from pytest import mark

NUM_TOKENS = [7, 83, 4096]
HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
DTYPES = [torch.half, torch.bfloat16]

@mark.MI308X
@mark.rocm
@mark.gpu
class EmbedingTest:
    DTYPES = DTYPES
    NUM_TOKENS = NUM_TOKENS
    HIDDEN_SIZES = HIDDEN_SIZES

    def _run_embeding_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        w = torch.randn(131072, hidden_size, dtype=dtype)
        model_config = ModelConfig()
        model_config.attn_config.head_num = 1
        model_config.attn_config.size_per_head = 1
        model_config.num_layers = 1
        model_config.max_seq_len = 1
        model_config.vocab_size = 1
        
        parallelism_config = ParallelismConfig()
        parallelism_config.tp_size = 1
        parallelism_config.tp_rank = 0
        
        embeding = Embedding(model_config, parallelism_config, w)
        embeding_torch = EmbeddingTorch(w)
        x = torch.randint(0, hidden_size, (num_tokens,), dtype=torch.int32)
        # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     for _ in range(10):
        #         out = embeding(x)
        #         # out = embeding_torch(x)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
        out_torch = embeding_torch(x)
        out = embeding(x)
        torch.cuda.synchronize()
        
        if not torch.allclose(out_torch, out, atol=1e-2, rtol=1e-2):
            diff = (out_torch - out).abs()
            # Report where the largest diff occurs to aid debugging.
            flat_index = int(diff.reshape(-1).argmax().item())
            max_diff = diff.reshape(-1)[flat_index].item()
            diff_index = tuple(torch.unravel_index(torch.tensor(flat_index), diff.shape))
            print(f"max diff: {max_diff} at index {diff_index}")
            print(
                f"out_torch{diff_index}: {out_torch[diff_index]}, out{diff_index}: {out[diff_index]}"
            )
            print(f"embeding_torch(x): {out_torch}")
            print(f"embeding(x): {out}")
            pytest.fail("Embedding outputs mismatch")

    @mark.parametrize(
        "num_tokens",
        NUM_TOKENS,
        ids=[f"tokens={num_tokens}" for num_tokens in NUM_TOKENS],
    )
    @mark.parametrize(
        "hidden_size",
        HIDDEN_SIZES,
        ids=[f"hidden={hidden_size}" for hidden_size in HIDDEN_SIZES],
    )
    @mark.parametrize(
        "dtype",
        DTYPES,
        ids=[f"dtype={dtype}" for dtype in DTYPES],
    )
    def test_embeding(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        with torch.device("cuda"):
            self._run_embeding_test(num_tokens, hidden_size, dtype)
