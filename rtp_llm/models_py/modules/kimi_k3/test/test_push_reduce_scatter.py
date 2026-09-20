"""K3 standalone RS dispatch; the distributed GPU test checks the real kernel."""

import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.kimi_k3 import gemm_reduce_scatter as rs


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class K3PushDispatchTest(unittest.TestCase):
    def test_misaligned_view_keeps_group_selected_push_backend(self):
        device = torch.device("cuda", torch.cuda.current_device())
        group = Mock(size=Mock(return_value=8))
        push = Mock()
        state = rs._GemmReduceScatterState(
            group, device, 8, 8, 7168, use_fused=False, push=push
        )
        storage = torch.arange(8 * 7168 + 1, device=device).bfloat16()
        x = storage[1:].view(8, 7168)
        self.assertTrue(x.is_contiguous())
        self.assertNotEqual(x.data_ptr() % 16, 0)
        with patch.dict(
            rs._STATES, {(group, device.index): state}, clear=True
        ), patch.object(rs.dist, "reduce_scatter_tensor") as nccl:
            out = rs.reduce_scatter(x, group)
        nccl.assert_not_called()
        push.reduce_scatter.assert_called_once()
        aligned, output = push.reduce_scatter.call_args.args
        self.assertIs(out, output)
        self.assertEqual(aligned.data_ptr() % 16, 0)
        torch.testing.assert_close(aligned, x, rtol=0, atol=0)

    def test_push_for_small_prefill_and_all_decode_sizes(self):
        device = torch.device("cuda", torch.cuda.current_device())
        group = Mock()
        group.size.return_value = 8
        weight = torch.ones((16, 7168), device=device, dtype=torch.bfloat16)
        for prefill in (True, False):
            fused = Mock()
            state = rs._GemmReduceScatterState(
                group,
                device,
                8,
                8192,
                7168,
                deep_gemm=Mock(bf16_gemm_rs_nn=fused),
                workspace=object(),
                use_fused=prefill,
            )
            push = Mock()
            state.push = push
            for m in (0, 1, 8, 511, 512, 513, 8192):
                with self.subTest(prefill=prefill, m=m), patch.dict(
                    rs._STATES, {(group, device.index): state}, clear=True
                ), patch.object(rs.dist, "reduce_scatter_tensor") as nccl:
                    push.reset_mock()
                    fused.reset_mock()
                    x = torch.ones((m, 16), device=device, dtype=torch.bfloat16)
                    out = rs.gemm_reduce_scatter(x, weight, group, pad_rows=True)
                    self.assertEqual(out.shape, ((m + 7) // 8, 7168))
                    use_push = m > 0 and (not prefill or m < 512)
                    self.assertEqual(push.reduce_scatter.call_count, int(use_push))
                    self.assertEqual(fused.call_count, int(prefill and m >= 512))
                    nccl.assert_not_called()
                    if use_push:
                        partial, output = push.reduce_scatter.call_args.args
                        self.assertIs(output, out)
                        torch.testing.assert_close(partial[:m], x @ weight)
                        self.assertEqual(torch.count_nonzero(partial[m:]).item(), 0)

    def test_dense_rs_uses_same_policy_and_nccl_when_push_unavailable(self):
        device = torch.device("cuda", torch.cuda.current_device())
        group = Mock(size=Mock(return_value=8))
        for prefill in (True, False):
            for push in (None, Mock()):
                state = rs._GemmReduceScatterState(
                    group, device, 8, 1024, 7168, use_fused=prefill, push=push
                )
                for m in (0, 8, 504, 512, 1024):
                    with self.subTest(
                        prefill=prefill, enabled=push is not None, m=m
                    ), patch.dict(
                        rs._STATES, {(group, device.index): state}, clear=True
                    ), patch.object(
                        rs.dist, "reduce_scatter_tensor"
                    ) as nccl:
                        if push is not None:
                            push.reset_mock()
                        x = torch.empty((m, 7168), device=device, dtype=torch.bfloat16)
                        out = rs.reduce_scatter(x, group)
                        self.assertEqual(out.shape, (m // 8, 7168))
                        use_push = (
                            push is not None and m > 0 and (not prefill or m < 512)
                        )
                        self.assertEqual(nccl.call_count, int(m > 0 and not use_push))
                        if push is not None:
                            self.assertEqual(
                                push.reduce_scatter.call_count, int(use_push)
                            )


if __name__ == "__main__":
    unittest.main()
