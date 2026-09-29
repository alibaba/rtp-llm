import runpy
import unittest
from pathlib import Path

import torch


merge_mla_states_in_place = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "mla_state_merge.py")
)["merge_mla_states_in_place"]


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class MlaChunkAttentionTest(unittest.TestCase):
    def test_causal_current_plus_noncausal_prefix_matches_full_attention(self):
        from tokenspeed_mla.mla_prefill import tokenspeed_mla_prefill

        torch.manual_seed(37)
        device = "cuda:0"
        q_len, prefix_len, heads = 64, 256, 12
        q = (torch.randn(q_len, heads, 192, device=device) * 0.3).to(
            torch.float8_e4m3fn
        )
        prefix_k = (torch.randn(prefix_len, heads, 192, device=device) * 0.3).to(
            torch.float8_e4m3fn
        )
        prefix_v = (torch.randn(prefix_len, heads, 128, device=device) * 0.3).to(
            torch.float8_e4m3fn
        )
        current_k = (torch.randn(q_len, heads, 192, device=device) * 0.3).to(
            torch.float8_e4m3fn
        )
        current_v = (torch.randn(q_len, heads, 128, device=device) * 0.3).to(
            torch.float8_e4m3fn
        )
        scale = 1 / (192**0.5)

        def attention(key, value, causal):
            n = key.shape[0]
            output, lse = tokenspeed_mla_prefill(
                query=q, key=key, value=value,
                seq_lens=torch.tensor([n], dtype=torch.int32, device=device),
                cum_seq_lens=torch.tensor([0, n], dtype=torch.int32, device=device),
                max_seq_len=n, batch_size=1, softmax_scale=scale,
                is_causal=causal, return_lse=True,
                cum_seq_lens_q=torch.tensor([0, q_len], dtype=torch.int32, device=device),
                max_seq_len_q=q_len, enable_pdl=False,
            )
            return output, lse

        expected, expected_lse = attention(
            torch.cat((prefix_k, current_k)),
            torch.cat((prefix_v, current_v)), True,
        )
        actual, actual_lse = attention(current_k, current_v, True)
        for start in range(0, prefix_len, 128):
            partial, partial_lse = attention(
                prefix_k[start : start + 128], prefix_v[start : start + 128], False
            )
            merge_mla_states_in_place(actual, actual_lse, partial, partial_lse)
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
        torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
