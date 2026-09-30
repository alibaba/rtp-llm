import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


_source = Path(__file__).resolve().parents[1] / "nccl_fp8_projection_overlap.py"
_spec = importlib.util.spec_from_file_location("nccl_fp8_projection_overlap", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class _Projection:
    N = 4

    def __init__(self, events):
        self.events = events
        self.weight = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=torch.bfloat16,
        )

    def forward_quantized_columns(self, values, scales, start, end):
        self.events.append(("project", start, end))
        return values @ self.weight[:, start:end]


class _Work:
    def __init__(self, events, column):
        self.events = events
        self.column = column

    def wait(self):
        self.events.append(("wait", self.column))


class NcclFp8RsProjectionOverlapTest(unittest.TestCase):
    def test_reduce_scatter_runs_after_each_column_projection(self):
        events = []
        values = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            dtype=torch.bfloat16,
        )
        scales = torch.ones((4, 1), dtype=torch.int32)
        linear = _Projection(events)

        def reduce_scatter(output, source, *, group, async_op):
            self.assertTrue(async_op)
            self.assertEqual(source.dtype, torch.bfloat16)
            column = len([event for event in events if event[0] == "reduce_scatter"])
            events.append(("reduce_scatter", column))
            output.copy_(source[:2])
            return _Work(events, column)

        with patch("torch.distributed.get_world_size", return_value=2), patch(
            "torch.distributed.reduce_scatter_tensor", side_effect=reduce_scatter
        ):
            output = _module.reduce_scatter_project_columns_overlap(
                values, scales, linear, object(), splits=2
            )

        torch.testing.assert_close(output, values[:2] @ linear.weight)
        self.assertEqual(
            events,
            [
                ("project", 0, 2),
                ("reduce_scatter", 0),
                ("project", 2, 4),
                ("reduce_scatter", 1),
                ("wait", 0),
                ("wait", 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
