"""Real 32-head FlashMLA outputs against independent attention reductions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.flash_mla_compat import flash_mla_sparse_fwd


class TpWorkspaceTest(unittest.TestCase):
    def test_bound_workspace_matches_local_attention_projection(self):
        from rtp_llm.models_py.model_desc.deepseek_v4_model import (
            DeepSeekV4Model,
            Dsv4SharedRuntimeBufferStore,
        )
        from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace

        for local_heads, cp_size in ((32, 1), (128, 1), (128, 4)):
            with self.subTest(local_heads=local_heads, cp_size=cp_size):
                bind = Mock()
                model = SimpleNamespace(
                    v4=SimpleNamespace(
                        layers=[
                            SimpleNamespace(attn=SimpleNamespace(n_heads=local_heads))
                        ],
                        _bind_prefill_workspace_dims=bind,
                    ),
                    _v4_args=SimpleNamespace(n_heads=128, head_dim=512),
                    _prefill_cp_size=cp_size,
                    _resolve_prefill_q_token_capacity=lambda: 3,
                    _resolve_prefill_ws_gather_widths=lambda: (2048, 512),
                )
                with patch.object(
                    Dsv4SharedRuntimeBufferStore,
                    "mtp_hidden_requested",
                    return_value=False,
                ), patch.object(
                    Dsv4SharedRuntimeBufferStore, "get_or_create", return_value=Mock()
                ):
                    DeepSeekV4Model._bind_runtime_buffers(model, torch.device("cpu"))
                rows, width, cp_rows, main_w, idx_w = bind.call_args.args
                workspace = PrefillWorkspace(
                    torch.device("cpu"),
                    q_rows=rows,
                    q_dim=width,
                    reserve_cp=cp_size > 1,
                    cp_rows=cp_rows,
                    main_w=main_w,
                    idx_w=idx_w,
                    align_bytes=1,
                )
                projected_q = workspace.prefill_q(3).view(3, local_heads, 512)
                self.assertEqual(projected_q.shape, (3, local_heads, 512))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class FlashMlaTp4Test(unittest.TestCase):
    def test_prefill_sink_mask_and_graph(self):
        torch.manual_seed(53)
        q = torch.randn(3, 32, 512, device="cuda", dtype=torch.bfloat16) * 0.2
        kv = torch.randn(128, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.2
        sink = torch.randn(32, device="cuda")
        indices = torch.arange(128, device="cuda", dtype=torch.int32).repeat(3, 1, 1)
        lengths = torch.tensor([1, 37, 128], device="cuda", dtype=torch.int32)

        def run():
            return flash_mla_sparse_fwd(
                q, kv, indices, 512**-0.5, attn_sink=sink, topk_length=lengths
            )[0]

        for _ in range(3):
            output = run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        q.mul_(0.5)
        graph.replay()
        for row, count in enumerate((1, 37, 128)):
            scores = q[row].float() @ kv[:count, 0].float().T * 512**-0.5
            logits = torch.cat((scores, sink[:, None]), dim=-1)
            weights = logits.softmax(-1)[:, :count]
            expected = weights @ kv[:count, 0].float()
            torch.testing.assert_close(
                output[row].float(), expected, rtol=0.02, atol=0.002
            )


if __name__ == "__main__":
    unittest.main()
