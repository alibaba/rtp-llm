import unittest
from unittest.mock import patch
import importlib.util
from pathlib import Path

import torch

_source = Path(__file__).resolve().parents[1] / "nccl_fp8_projection_overlap.py"
_spec = importlib.util.spec_from_file_location("nccl_fp8_projection_overlap", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
all_gather_project_local_overlap = _module.all_gather_project_local_overlap


class _Projection:
    N = 2

    def __init__(self, events):
        self.events = events

    def __call__(self, x, out=None):
        self.events.append(("project", x.shape[0]))
        result = x @ torch.tensor([[1., 2.], [3., 4.]])
        out.copy_(result)
        return out


class _Work:
    def __init__(self, output, local, events):
        self.output = output
        self.local = local
        self.events = events

    def wait(self):
        self.events.append(("wait", None))
        self.output[:2].copy_(torch.tensor([[5., 6.], [7., 8.]]))
        self.output[2:].copy_(self.local)


class NcclFp8ProjectionOverlapTest(unittest.TestCase):
    def test_projects_local_rows_before_wait_and_preserves_global_order(self):
        events = []
        local = torch.tensor([[1., 2.], [3., 4.]])

        def gather(output, input, *, group, async_op):
            self.assertTrue(async_op)
            self.assertIs(input, local)
            events.append(("gather", None))
            return _Work(output, input, events)

        with patch("torch.distributed.get_world_size", return_value=2), patch(
            "torch.distributed.get_rank", return_value=1
        ), patch("torch.distributed.all_gather_into_tensor", side_effect=gather):
            result = all_gather_project_local_overlap(local, _Projection(events), object())

        expected = torch.tensor([[5., 6.], [7., 8.], [1., 2.], [3., 4.]]) @ torch.tensor(
            [[1., 2.], [3., 4.]]
        )
        torch.testing.assert_close(result, expected)
        self.assertEqual(events, [("gather", None), ("project", 2), ("wait", None), ("project", 2)])


if __name__ == "__main__":
    unittest.main()
