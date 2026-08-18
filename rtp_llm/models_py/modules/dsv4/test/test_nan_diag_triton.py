import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4 import _nan_diag_triton as nan_diag


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class NanDiagTritonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.enabled = patch.object(nan_diag, "ENABLED", True)
        self.enabled.start()
        self.addCleanup(self.enabled.stop)
        nan_diag.reset("cuda")

    def test_disabled_is_noop(self) -> None:
        outputs = SimpleNamespace(hidden_states=torch.zeros(1, device="cuda"))
        with patch.object(nan_diag, "ENABLED", False):
            self.assertIs(nan_diag.attach_event_buffers(outputs), outputs)
            nan_diag.report((outputs.hidden_states,), nan_diag.SOURCE_MOE_INPUT, 1)
        self.assertFalse(hasattr(outputs, "nan_diag_events"))

    def test_records_all_boundaries_once_and_is_read_only(self) -> None:
        tensors = [
            torch.zeros((2, 513), dtype=torch.bfloat16, device="cuda") for _ in range(4)
        ]
        for index, tensor in enumerate(tensors):
            tensor[index % 2, 300 + index] = float("nan" if index % 2 == 0 else "inf")
        before = [tensor.clone() for tensor in tensors]
        state, events = nan_diag._state("cuda")
        nan_diag.report(tuple(tensors), nan_diag.SOURCE_ATTENTION_INPUT, 17)
        nan_diag.report(tuple(tensors), nan_diag.SOURCE_ATTENTION_INPUT, 18)
        torch.cuda.synchronize()

        self.assertEqual(int(state[0].item()), 4)
        rows = {row[0]: row for row in events[:4].cpu().tolist()}
        self.assertEqual(set(rows), {1, 2, 3, 4})
        self.assertEqual({row[1] for row in rows.values()}, {17})
        self.assertEqual(rows[1][2:], [300, 1, 0, 2, 2, 513])
        for actual, expected in zip(tensors, before):
            torch.testing.assert_close(actual, expected, equal_nan=True)

    def test_cuda_graph_replay_resets_and_reports(self) -> None:
        x = torch.zeros((2, 513), dtype=torch.bfloat16, device="cuda")
        state, events = nan_diag._state(x.device)
        nan_diag.report((x, x, x, x), nan_diag.SOURCE_ATTENTION_INPUT, 23)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            nan_diag.reset(x.device)
            nan_diag.report((x, x, x, x), nan_diag.SOURCE_ATTENTION_INPUT, 23)

        x[1, 300] = float("nan")
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(int(state[0].item()), 4)
            self.assertEqual(set(events[:4, 0].cpu().tolist()), {1, 2, 3, 4})
            self.assertEqual(set(events[:4, 1].cpu().tolist()), {23})
