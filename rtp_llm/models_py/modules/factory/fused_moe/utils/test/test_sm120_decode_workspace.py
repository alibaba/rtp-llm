import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.graph_decode import (
    Sm120DecodeMoeGeometry,
    decode_workspace_geometry,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.sm120.workspace import (
    Sm120MoeWorkspace,
)
from rtp_llm.ops.compute_ops import DecodeExecutionMode


class Sm120DecodeWorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.workspace = Sm120MoeWorkspace()
        self.device = torch.device("cpu")
        self.geometry = Sm120DecodeMoeGeometry(2, 4, 640, 512, 8, 2)
        self.plan = SimpleNamespace(mode=DecodeExecutionMode.GRAPH)

    def test_allocator_can_succeed_when_driver_free_memory_is_low(self):
        with patch.object(torch.cuda, "mem_get_info", return_value=(0, 1 << 30)):
            self.workspace.reserve_decode_graph_workspace(self.geometry, self.device)
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ):
            view = self.workspace.decode_graph_workspace(
                self.plan, self.geometry, self.device
            )
            self.assertEqual(view.payload.device, self.device)
            self.assertEqual(
                view.gathered.shape, (self.geometry.rows, self.geometry.payload_cols)
            )

    def test_allocator_oom_releases_partial_workspace_and_allows_retry(self):
        empty = torch.empty
        partial_allocations = []

        def allocate_until_oom(*args, **kwargs):
            if len(partial_allocations) == 3:
                raise torch.OutOfMemoryError("injected workspace allocator OOM")
            tensor = empty(*args, **kwargs)
            partial_allocations.append(weakref.ref(tensor))
            return tensor

        with patch.object(torch, "empty", side_effect=allocate_until_oom):
            with self.assertRaisesRegex(torch.OutOfMemoryError, "allocator OOM"):
                self.workspace.reserve_decode_graph_workspace(
                    self.geometry, self.device
                )
        self.assertEqual(len(partial_allocations), 3)
        self.assertTrue(all(ref() is None for ref in partial_allocations))
        self.workspace.reserve_decode_graph_workspace(self.geometry, self.device)

    def test_views_share_exactly_one_reservation_and_preserve_strides(self):
        args = (4, 640, 512, 8, 2)
        capacity = decode_workspace_geometry([1, 2, 17], [1, 2], *args)
        self.assertEqual(capacity.expert_capacity, 256)
        self.assertEqual(self.geometry.expert_capacity, 128)
        reserved = self.workspace.reserve_decode_graph_workspace(capacity, self.device)
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(
                torch, "empty", side_effect=AssertionError("unexpected allocation")
            ),
        ):
            small = self.workspace.decode_graph_workspace(
                self.plan, self.geometry, self.device
            )
            large = self.workspace.decode_graph_workspace(
                self.plan, capacity, self.device
            )
        storages = {
            tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
            for view in (small, large)
            for tensor in vars(view).values()
        }
        self.assertEqual(len(storages), len(vars(large)))
        self.assertEqual(sum(storages.values()), reserved)
        self.assertEqual(reserved, capacity.workspace_bytes)
        for name, tensor in vars(small).items():
            self.assertEqual(tensor.data_ptr(), getattr(large, name).data_ptr())
        self.assertEqual(small.expert_x_scale.stride(), (256, 1, 128))
        self.assertEqual(large.expert_x_scale.stride(), (512, 1, 256))
        self.assertEqual(small.expert_x.stride(), (128 * 640, 640, 1))
        self.assertEqual(small.down_in_scale.stride(), (128, 1, 128))

        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.assertIs(
                self.workspace.decode_graph_workspace(
                    self.plan, self.geometry, self.device
                ),
                small,
            )
            with self.assertRaisesRegex(RuntimeError, "warmed before capture"):
                self.workspace.decode_graph_workspace(
                    self.plan, Sm120DecodeMoeGeometry(1, *args), self.device
                )

    def test_uncaptured_fixed_calls_use_per_stream_storage(self):
        self.workspace.reserve_decode_graph_workspace(self.geometry, self.device)
        fixed = SimpleNamespace(mode=DecodeExecutionMode.FIXED)
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "current_stream") as current_stream,
        ):
            graph_view = self.workspace.decode_graph_workspace(
                self.plan, self.geometry, self.device
            )
            current_stream.return_value.cuda_stream = 100
            first = self.workspace.decode_graph_workspace(
                fixed, self.geometry, self.device
            )
            current_stream.return_value.cuda_stream = 200
            second = self.workspace.decode_graph_workspace(
                fixed, self.geometry, self.device
            )
            current_stream.return_value.cuda_stream = 100
            self.assertIs(
                self.workspace.decode_graph_workspace(
                    fixed, self.geometry, self.device
                ),
                first,
            )
        self.assertEqual(
            len(
                {
                    graph_view.payload.data_ptr(),
                    first.payload.data_ptr(),
                    second.payload.data_ptr(),
                }
            ),
            3,
        )


if __name__ == "__main__":
    unittest.main()
