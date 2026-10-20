import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
from rtp_llm.models_py.modules.qwen4_exp.head_ipc_gather import (
    Qwen4HeadIpcGather,
    prepare_head_ipc_gather,
)


class HeadIpcGatherTest(unittest.TestCase):
    def test_default_off_and_unsupported_layouts_allocate_nothing(self):
        config = SimpleNamespace(_qwen4_full_fp8_nvfp4=True)
        pc = SimpleNamespace(
            role_type="DECODE", tp_size=2, dp_size=1, local_world_size=2
        )
        head = SimpleNamespace(
            weight=torch.empty(124160, 2560, device="meta", dtype=torch.float8_e4m3fn)
        )
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.head_ipc_gather.Qwen4HeadIpcGather",
            side_effect=AssertionError("unexpected allocation"),
        ):
            with patch.dict("os.environ", {"QWEN4_HEAD_IPC_GATHER": "0"}):
                self.assertIsNone(prepare_head_ipc_gather(config, pc, head))
            with patch.dict("os.environ", {"QWEN4_HEAD_IPC_GATHER": "1"}):
                for key, value in [
                    ("role_type", "PREFILL"),
                    ("tp_size", 4),
                    ("dp_size", 2),
                    ("local_world_size", 4),
                ]:
                    changed = SimpleNamespace(**{**vars(pc), key: value})
                    self.assertIsNone(prepare_head_ipc_gather(config, changed, head))
                self.assertIsNone(prepare_head_ipc_gather(config, pc, head))

    def test_invalid_tensor_domains_do_not_touch_native_ops(self):
        workspace = object.__new__(Qwen4HeadIpcGather)
        workspace.closed = False
        workspace.device = torch.device("cuda", torch.cuda.current_device())
        x = torch.empty(4, 124160, device=workspace.device, dtype=torch.bfloat16)
        invalid = [
            None,
            x.cpu(),
            x.float(),
            x[:0],
            x[:, :320],
            x.t(),
            x.unsqueeze(0),
            torch.empty(33, 124160, device=workspace.device, dtype=torch.bfloat16),
        ]
        for tensor in invalid:
            self.assertIsNone(workspace.try_gather(tensor))
        workspace.closed = True
        self.assertIsNone(workspace.try_gather(x))

    def test_graph_capture_defers_without_allocating_or_changing_owner(self):
        workspace = object.__new__(Qwen4HeadIpcGather)
        workspace.closed = False
        workspace.device = torch.device("cuda", torch.cuda.current_device())
        x = torch.zeros(4, 124160, device=workspace.device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.assertIsNone(workspace.try_gather(x))
            output = x + 1
        graph.replay()
        self.assertTrue(torch.equal(output, x + 1))
        graph.reset()

    def test_model_hook_preserves_optional_fallback(self):
        local = torch.ones(4, 8)
        model = SimpleNamespace()
        self.assertIsNone(Qwen4ExpModel.try_gather_fp8_lm_head_for_tp(model, local))
        expected = torch.ones(4, 16)
        seen = []
        model._head_ipc_gather = SimpleNamespace(
            try_gather=lambda x: (seen.append(x), expected)[1]
        )
        self.assertIs(
            Qwen4ExpModel.try_gather_fp8_lm_head_for_tp(model, local), expected
        )
        self.assertIs(seen[0], local)


if __name__ == "__main__":
    unittest.main()
