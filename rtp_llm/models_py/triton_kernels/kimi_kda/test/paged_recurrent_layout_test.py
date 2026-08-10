import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3KDA
from rtp_llm.models_py.triton_kernels.kimi_kda.fused_recurrent import (
    fused_recurrent_kda,
)
from rtp_llm.utils.model_weight import W


class KimiK3PagedRecurrentLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        torch.manual_seed(20260731)

    @staticmethod
    def _block_map(batch: int, pages: int) -> torch.Tensor:
        return torch.arange(
            1,
            batch * pages + 1,
            dtype=torch.int32,
            device="cuda",
        ).reshape(batch, pages)

    def test_model_paged_decode_matches_canonical_layout_eager_and_graph(
        self,
    ) -> None:
        batch = 4
        # K3 TP8 decode uses 12 local KDA heads with 128-dimensional K/V.
        heads = 12
        head_dim = 128
        page_size = 8
        pages = 3
        block_map = self._block_map(batch, pages)
        lengths_plus_one = torch.tensor(
            [2, 8, 9, 10], dtype=torch.int32, device="cuda"
        )
        cu_seqlens = torch.arange(
            batch + 1, dtype=torch.int32, device="cuda"
        )
        flat_shape = (batch, heads * head_dim)
        q = torch.randn(*flat_shape, dtype=torch.bfloat16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        raw_gate = torch.randn_like(q)
        raw_beta = torch.randn(
            batch, heads, dtype=torch.float32, device="cuda"
        )
        a_log = torch.randn(heads, dtype=torch.float32, device="cuda")
        dt_bias = torch.randn(
            heads * head_dim, dtype=torch.float32, device="cuda"
        )
        initial_cache = torch.randn(
            batch * pages + 1,
            heads,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device="cuda",
        )
        layer = SimpleNamespace(
            local_heads=heads,
            head_dim=head_dim,
            gate_lower_bound=-20.0,
            weights={
                W.linear_attn_alog: a_log,
                W.linear_attn_dt_b_kda: dt_bias,
            },
        )

        host_lengths = lengths_plus_one.cpu().tolist()
        host_block_map = block_map.cpu().tolist()
        read_blocks = [
            host_block_map[index][(length - 2) // page_size]
            for index, length in enumerate(host_lengths)
        ]
        write_blocks = [
            host_block_map[index][(length - 1) // page_size]
            for index, length in enumerate(host_lengths)
        ]
        head_shape = (1, batch, heads, head_dim)

        # The canonical RTP cache is physically [H,K,V].  The non-paged model
        # path transposes it to the kernel's V-first layout and transposes the
        # final state back before publishing it to the cache.
        gathered_v_first = (
            initial_cache[read_blocks].transpose(-1, -2).contiguous()
        )
        expected_output, expected_final_v_first = fused_recurrent_kda(
            q.reshape(head_shape),
            k.reshape(head_shape),
            v.reshape(head_shape),
            raw_gate.reshape(head_shape),
            raw_beta.reshape(1, batch, heads),
            initial_state=gathered_v_first,
            A_log=a_log,
            dt_bias=dt_bias,
            inplace_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=-20.0,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )
        expected_cache = initial_cache.clone()
        expected_cache[write_blocks] = expected_final_v_first.transpose(-1, -2)

        # Keep an explicit negative control so square K/V dimensions cannot
        # make the test compare two identically misinterpreted states again.
        wrong_layout_output, _ = fused_recurrent_kda(
            q.reshape(head_shape),
            k.reshape(head_shape),
            v.reshape(head_shape),
            raw_gate.reshape(head_shape),
            raw_beta.reshape(1, batch, heads),
            initial_state=initial_cache.clone(),
            A_log=a_log,
            dt_bias=dt_bias,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=-20.0,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
            block_map=block_map,
            seq_size_per_block=page_size,
            sequence_lengths=lengths_plus_one,
        )
        self.assertFalse(
            torch.equal(wrong_layout_output, expected_output),
            "test fixture must distinguish V-first from canonical K-first cache",
        )

        def run_paged(cache: torch.Tensor) -> torch.Tensor:
            return KimiK3KDA._paged_decode_core(
                layer,
                q,
                k,
                v,
                raw_gate,
                raw_beta,
                cu_seqlens,
                cache,
                block_map,
                lengths_plus_one,
                page_size,
            )

        eager_cache = initial_cache.clone()
        eager_output = run_paged(eager_cache)
        torch.testing.assert_close(eager_output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(
            eager_cache, expected_cache, rtol=1e-5, atol=2e-7
        )

        # Compile before capture, then verify the same cache ABI survives a
        # captured launch and replay.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            run_paged(initial_cache.clone())
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph_cache = initial_cache.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = run_paged(graph_cache)
        graph_cache.copy_(initial_cache)

        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(
            graph_cache, expected_cache, rtol=1e-5, atol=2e-7
        )


if __name__ == "__main__":
    unittest.main()
