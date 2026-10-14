import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed import tp_collective_workspace as comm
from rtp_llm.models_py.distributed.fp8_collective_projection import Fp8Activation


class TpCollectiveWorkspaceTest(unittest.TestCase):
    def setUp(self):
        comm.clear_tp_collective_workspaces()
        # The registry key uses ProcessGroup identity, not its rank or name.
        self.group = type("Group", (), {"size": lambda _: 2})()
        self.plan = patch.object(
            comm.dist,
            "all_gather_object",
            side_effect=lambda outputs, value, **_: outputs.__setitem__(
                slice(None), [value] * 2
            ),
        )
        self.plan.start()
        self.capture = patch(
            "torch.cuda.is_current_stream_capturing", return_value=False
        )
        self.capture.start()
        self.env = patch.dict(
            os.environ,
            {
                "RTP_LLM_CUSTOM_REDUCE_SCATTER": "auto",
                "RTP_LLM_CUSTOM_ALL_GATHER": "auto",
                "RTP_LLM_FP8_AG_GEMM": "auto",
            },
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.capture.stop()
        self.plan.stop()
        comm.clear_tp_collective_workspaces()

    def test_auto_initialization_reuses_arbitrary_width_workspace(self):
        with patch.object(
            comm, "create_push_reduce_scatter", return_value=object()
        ) as rs, patch.object(
            comm, "create_custom_all_gather", return_value=object()
        ) as ag:
            state = comm.configure_tp_collectives(
                self.group, "cuda:0", max_m=64, width=4096, fp8=True
            )
            reused = comm.configure_tp_collectives(
                self.group, "cuda:0", max_m=32, width=4096, fp8=True
            )
        self.assertIs(state, reused)
        self.assertEqual(rs.call_count, 1)
        self.assertEqual(ag.call_count, 2)
        self.assertEqual(rs.call_args.kwargs["n"], 4096)

    def test_larger_model_preserves_old_graph_storage(self):
        with patch.object(
            comm, "create_push_reduce_scatter", side_effect=lambda *_, **__: object()
        ), patch.object(comm, "create_custom_all_gather", return_value=None):
            old = comm.configure_tp_collectives(
                self.group, "cuda:0", max_m=32, width=1024
            )
            new = comm.configure_tp_collectives(
                self.group, "cuda:0", max_m=64, width=1024
            )
        self.assertIsNot(old.push_rs, new.push_rs)
        self.assertIn(old, comm._RETIRED_WORKSPACES)
        self.assertIs(comm.get_tp_collective_workspace(self.group, "cuda:0", 1024), new)

    def test_rank_plan_disagreement_fails_before_allocation(self):
        def disagree(plans, value, **_):
            plans[:] = [value, (128, *value[1:])]

        with patch.object(
            comm.dist, "all_gather_object", side_effect=disagree
        ), patch.object(comm, "create_push_reduce_scatter") as rs:
            with self.assertRaisesRegex(ValueError, "ranks disagree"):
                comm.configure_tp_collectives(
                    self.group, "cuda:0", max_m=64, width=4096
                )
        rs.assert_not_called()

    def test_explicit_disable_and_unsupported_backend_do_not_rendezvous_later(self):
        with patch.dict(
            os.environ,
            {"RTP_LLM_CUSTOM_REDUCE_SCATTER": "0", "RTP_LLM_CUSTOM_ALL_GATHER": "0"},
        ), patch.object(comm, "create_push_reduce_scatter") as rs, patch.object(
            comm, "create_custom_all_gather"
        ) as ag:
            state = comm.configure_tp_collectives(
                self.group, "cuda:0", max_m=64, width=4096
            )
        rs.assert_not_called()
        ag.assert_not_called()
        self.assertIsNone(state.push_rs)
        self.assertIsNone(comm.try_all_gather(torch.ones(4, 4096), self.group))

    def test_bf16_all_gather_returns_owned_storage(self):
        scratch = torch.zeros(4, 8, dtype=torch.bfloat16)

        def gather(_local, *, staging, out):
            self.assertTrue(staging)
            out.copy_(scratch.fill_(3))
            return out

        ag = SimpleNamespace(all_gather=gather)
        state = comm.TpCollectiveWorkspace(
            self.group, torch.device("cpu"), 8, 4, 2, bf16_ag=ag
        )
        with patch.object(comm, "get_tp_collective_workspace", return_value=state):
            result = comm.try_all_gather(
                torch.ones(2, 8, dtype=torch.bfloat16), self.group
            )
        scratch.zero_()
        torch.testing.assert_close(result, torch.full_like(result, 3), rtol=0, atol=0)

    def test_prequantized_projection_preserves_values_and_scale_wire(self):
        values = torch.full((3, 512), -2, dtype=torch.float8_e4m3fn)
        wire = torch.full((1, 4), 0x7F7F7F7F, dtype=torch.int32)
        activation = Fp8Activation(values, wire)
        captured = []

        def gather(v, w, *, staging):
            self.assertTrue(staging)
            captured.append((v, w))
            return v.repeat(2, 1), w[:, :3].repeat(1, 2).T

        ag = SimpleNamespace(all_gather_fp8=gather)
        state = comm.TpCollectiveWorkspace(
            self.group, torch.device("cpu"), 512, 8, 2, fp8_ag=ag
        )
        projection = SimpleNamespace(
            K=512, scale_ue8m0=True, forward_quantized=lambda v, s: v.float()
        )
        with patch.object(comm, "get_tp_collective_workspace", return_value=state):
            result = comm.try_all_gather_projection(activation, projection, self.group)
        self.assertIs(captured[0][0], values)
        self.assertIs(captured[0][1], wire)
        torch.testing.assert_close(result, torch.full((6, 512), -2.0), rtol=0, atol=0)

    def test_push_rs_cross_host_requires_fabric_allocation(self):
        from rtp_llm.models_py.distributed import push_reduce_scatter as rs
        import torch.distributed._symmetric_memory as symm

        props = SimpleNamespace(
            major=10, minor=3, uuid="local", multi_processor_count=132
        )
        group = type("Group", (), {"size": lambda _: 2, "rank": lambda _: 0})()

        def metadata(peers, value, **_):
            peers[:] = [value, ("remote-host", "remote", *value[2:])]

        def allocate(size, *, dtype, device):
            return torch.empty(size, dtype=dtype)

        zeros = torch.zeros
        handle = SimpleNamespace(
            get_buffer=lambda _, shape, dtype: torch.zeros(shape, dtype=dtype)
        )
        with patch.object(rs, "_load_kernel", return_value=object()), patch.object(
            torch.cuda, "get_device_properties", return_value=props
        ), patch.object(symm, "get_backend", return_value="CUDA"), patch.object(
            rs, "_nvlink_fabric", return_value=(b"fabric", 1)
        ), patch.object(
            rs, "_nvlink_peers", return_value=True
        ), patch.object(
            rs, "_all_ready", side_effect=lambda ready, *_: ready
        ), patch.object(
            rs.dist, "all_gather_object", side_effect=metadata
        ), patch.object(
            symm, "empty", side_effect=allocate
        ), patch.object(
            torch,
            "zeros",
            side_effect=lambda *a, **k: zeros(*a, **{**k, "device": "cpu"}),
        ), patch.object(
            symm, "rendezvous", return_value=handle
        ), patch.object(
            torch.cuda, "synchronize"
        ), patch.object(
            rs.dist, "barrier"
        ), patch.object(
            rs, "_is_fabric_allocation", return_value=False
        ) as fabric:
            self.assertIsNone(
                rs.create_push_reduce_scatter(group, "cuda:0", max_m=8, n=1024)
            )
            fabric.assert_called_once()
            fabric.return_value = True
            self.assertIsNotNone(
                rs.create_push_reduce_scatter(group, "cuda:0", max_m=8, n=1024)
            )


if __name__ == "__main__":
    unittest.main()
