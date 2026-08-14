import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from rtp_llm.models_py.modules.kimi_k3.moe import KimiK3LatentMoE


class KimiK3MegaBarrierUnitTest(unittest.TestCase):
    @staticmethod
    def _module() -> KimiK3LatentMoE:
        module = KimiK3LatentMoE.__new__(KimiK3LatentMoE)
        nn.Module.__init__(module)
        module._mega_group = object()
        module.layer_idx = 11
        return module

    def test_barrier_is_noop_by_default(self) -> None:
        module = self._module()

        with patch("torch.distributed.barrier") as barrier:
            module._maybe_pre_kernel_barrier(torch.device("cpu"), 512)

        barrier.assert_not_called()

    def test_barrier_synchronizes_cuda_stream_and_ranks(self) -> None:
        module = self._module()
        stream = MagicMock()

        with (
            patch.dict(
                "os.environ",
                {"DSV4_MEGA_MOE_PRE_KERNEL_BARRIER": "1"},
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.device"),
            patch("torch.cuda.current_stream", return_value=stream),
            patch("torch.cuda.current_device", return_value=3),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=3),
            patch("torch.distributed.get_world_size", return_value=8),
            patch("torch.distributed.barrier") as barrier,
        ):
            module._maybe_pre_kernel_barrier(torch.device("cuda:3"), 512)

        stream.synchronize.assert_called_once_with()
        barrier.assert_called_once_with(group=module._mega_group, device_ids=[3])

    def test_barrier_rejects_cuda_graph_capture(self) -> None:
        module = self._module()

        with (
            patch.dict(
                "os.environ",
                {"DSV4_MEGA_MOE_PRE_KERNEL_BARRIER": "1"},
            ),
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            self.assertRaisesRegex(RuntimeError, "CUDA graph capture"),
        ):
            module._maybe_pre_kernel_barrier(torch.device("cuda:0"), 512)

    def test_expert_sum_guards_symmetric_buffer_reuse_and_peer_kernel(self) -> None:
        module = self._module()
        module._mega_buf = SimpleNamespace(num_max_tokens_per_rank=512)
        module._mega_input_packer = SimpleNamespace(pack=MagicMock())
        module._mega_y = torch.empty((512, 4), dtype=torch.bfloat16)
        module._mega_l1_w = object()
        module._mega_l1_sf = object()
        module._mega_l2_w = object()
        module._mega_l2_sf = object()
        module.beta = 1.0
        module.linear_beta = None
        barrier = MagicMock()
        peer_kernel = MagicMock()
        calls = MagicMock()
        calls.attach_mock(module._mega_input_packer.pack, "pack")
        calls.attach_mock(barrier, "barrier")
        calls.attach_mock(peer_kernel, "peer_kernel")
        routed_input = torch.empty((3, 4), dtype=torch.bfloat16)
        expert_ids = torch.zeros((3, 2), dtype=torch.int64)
        routing_weights = torch.ones((3, 2), dtype=torch.float32)

        with (
            patch.object(module, "_maybe_pre_kernel_barrier", barrier),
            patch.dict(
                sys.modules,
                {"deep_gemm": SimpleNamespace(fp8_fp4_mega_moe=peer_kernel)},
            ),
        ):
            output = module._deep_gemm_mega_expert_sum(
                routed_input,
                expert_ids,
                routing_weights,
            )

        self.assertEqual(
            [item[0] for item in calls.method_calls],
            ["barrier", "pack", "barrier", "peer_kernel"],
        )
        self.assertEqual(barrier.call_count, 2)
        self.assertEqual(tuple(output.shape), (3, 4))


if __name__ == "__main__":
    unittest.main()
