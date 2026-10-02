import unittest
import inspect

import torch

from rtp_llm.models_py.modules.kimi_k3.native_mla_decode import NativeMlaDecode
import flashinfer
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla


class KimiK3Fp8DecodeTest(unittest.TestCase):
    def test_plain_e4m3_cache_and_query_produce_bf16_output(self):
        if not torch.cuda.is_available():
            self.fail("FP8 MLA Decode requires a CUDA GPU")
        torch.manual_seed(67)
        self.assertTrue(
            {"cum_seq_lens_q", "max_q_len", "backend", "out"}.issubset(
                inspect.signature(trtllm_batch_decode_with_kv_cache_mla).parameters
            ), f"flashinfer={flashinfer.__file__}; signature={inspect.signature(trtllm_batch_decode_with_kv_cache_mla)}"
        )
        device = "cuda"
        heads = 12
        workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
        op = NativeMlaDecode(
            num_heads=heads, kv_lora_rank=512, nope_dim=128, pe_dim=64,
            page_size=128, softmax_extra_scale=1.0, workspace=workspace,
            max_batch=1, fp8_compute=True,
        )
        q = torch.randn((1, heads, 192), dtype=torch.bfloat16, device=device) * 0.2
        kv = torch.randn((1, 512), dtype=torch.bfloat16, device=device) * 0.2
        pe = torch.randn((1, 64), dtype=torch.bfloat16, device=device) * 0.2
        kc = torch.randn((heads, 128, 512), dtype=torch.bfloat16, device=device) * 0.02
        vc = torch.randn((heads, 512, 128), dtype=torch.bfloat16, device=device) * 0.02
        cache = torch.zeros((1, 128, 576), dtype=torch.float8_e4m3fn, device=device)
        slots = torch.tensor([0], dtype=torch.int64, device=device)
        block_tables = torch.tensor([[0]], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([1], dtype=torch.int32, device=device)
        cu_query_lens = torch.tensor([0, 1], dtype=torch.int32, device=device)
        query = op.write_cache(
            q, kv, pe, cache, slots, kc
        )
        self.assertEqual(query.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(query.shape), (1, heads, 576))
        expected_cache = torch.cat((kv, pe), dim=-1).to(torch.float8_e4m3fn)
        torch.testing.assert_close(cache[0, 0].float(), expected_cache[0].float(), rtol=0, atol=0)
        output = op.attend(
            query, cache, vc,
            block_tables=block_tables,
            seq_lens=seq_lens,
            cu_query_lens=cu_query_lens,
            max_query_len=1, max_seq_len=1,
        )
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(tuple(output.shape), (1, heads, 128))
        self.assertTrue(torch.isfinite(output.float()).all().item())
        reference = torch.bmm(
            cache[0, 0, :512].float().view(1, 1, 512).expand(heads, -1, -1),
            vc.float(),
        ).transpose(0, 1)
        torch.testing.assert_close(output.float(), reference, rtol=0.08, atol=0.02)

        # The decode adapter reserves its query buffer before graph capture.
        # Replay must read fresh input values and overwrite the same cache slot.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_query = op.write_cache(
                q, kv, pe, cache, slots, kc
            )
            graph_output = op.attend(
                graph_query, cache, vc,
                block_tables=block_tables,
                seq_lens=seq_lens,
                cu_query_lens=cu_query_lens,
                max_query_len=1, max_seq_len=1,
            )
        kv.normal_(0, 0.1)
        pe.normal_(0, 0.1)
        graph.replay()
        torch.cuda.synchronize()
        replay_reference = torch.bmm(
            cache[0, 0, :512].float().view(1, 1, 512).expand(heads, -1, -1),
            vc.float(),
        ).transpose(0, 1)
        torch.testing.assert_close(graph_output.float(), replay_reference, rtol=0.08, atol=0.02)
        graph.reset()


if __name__ == "__main__":
    unittest.main()
