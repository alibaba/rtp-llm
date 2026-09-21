"""K3 standalone RS dispatch; the distributed GPU test checks the real kernel."""

import sys
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.distributed import push_reduce_scatter as push_rs
from rtp_llm.models_py.modules.kimi_k3 import gemm_reduce_scatter as rs


class NvlinkFabricTopologyTest(unittest.TestCase):
    def test_fabric_identity_handles_binary_uuid_across_nvml_bindings(self):
        import ctypes
        from types import SimpleNamespace

        for uuid_type in (ctypes.c_char, ctypes.c_uint8):

            class Info(ctypes.Structure):
                # NVML v2 layout; nvidia-ml-py 12.x uses c_char for clusterUuid.
                _fields_ = [
                    ("version", ctypes.c_uint),
                    ("clusterUuid", uuid_type * 16),
                    ("status", ctypes.c_uint),
                    ("cliqueId", ctypes.c_uint32),
                    ("state", ctypes.c_uint),
                    ("healthMask", ctypes.c_uint32),
                ]

                def __getattribute__(self, name):
                    # Match pynvml._PrintableStructure's automatic text decoding.
                    value = super().__getattribute__(name)
                    return value.decode() if isinstance(value, bytes) else value

            nvml = SimpleNamespace(
                nvmlInit=Mock(),
                nvmlShutdown=Mock(),
                nvmlDeviceGetHandleByUUID=Mock(return_value=object()),
                c_nvmlGpuFabricInfoV_t=Info,
                NVMLError=RuntimeError,
                NVML_GPU_FABRIC_STATE_COMPLETED=3,
                NVML_SUCCESS=0,
            )
            clusters = (
                b"abcdefghijklmnop",
                b"\x00abcdefghijklmno",
                b"abcdefg\x00hijklmno",
                b"\xff\x80abcdefghijklmn",
                # Same prefix before NUL, different identities after it.
                b"prefix\x00AAAAAAAAA",
                b"prefix\x00BBBBBBBBB",
            )
            cases = [(3, 0, cluster, (cluster, 7)) for cluster in clusters]
            cases += [
                (2, 0, clusters[0], None),
                (3, 1, clusters[0], None),
                (3, 0, bytes(16), None),
            ]
            for state, status, cluster, expected in cases:

                def fill(handle, pointer):
                    info = pointer._obj
                    info.state, info.status, info.cliqueId = state, status, 7
                    # Model the driver's binary write, bypassing Python setters.
                    ctypes.memmove(
                        ctypes.addressof(info) + Info.clusterUuid.offset, cluster, 16
                    )

                nvml.nvmlDeviceGetGpuFabricInfoV = fill
                with self.subTest(
                    uuid_type=uuid_type, state=state, status=status, cluster=cluster
                ), patch.dict(sys.modules, {"pynvml": nvml}):
                    self.assertEqual(push_rs._nvlink_fabric("local-uuid"), expected)
            del nvml.nvmlDeviceGetGpuFabricInfoV
            with self.subTest(uuid_type=uuid_type, api="missing"), patch.dict(
                sys.modules, {"pynvml": nvml}
            ):
                self.assertIsNone(push_rs._nvlink_fabric("old-nvml"))

    def test_cross_host_requires_one_nonempty_fabric_clique(self):
        uuids = ["a", "b"]
        hosts = ["node0", "node1"]
        for fabrics, expected in (
            ([(b"cluster", 1), (b"cluster", 1)], True),
            ([(b"cluster", 1), (b"cluster", 2)], False),
            ([(b"cluster", 1), (b"other", 1)], False),
            ([None, None], False),
            ([(b"cluster", 1), None], False),
        ):
            with self.subTest(fabrics=fabrics):
                self.assertEqual(
                    push_rs._nvlink_peers(uuids, hostnames=hosts, fabrics=fabrics),
                    expected,
                )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class K3PushDispatchTest(unittest.TestCase):
    def test_cross_host_rejection_precedes_fused_rs_allocation(self):
        group = Mock(size=Mock(return_value=8))
        backend = Mock()
        backend.GemmRSBuffer.return_value.num_bytes = 1024

        def hosts(out, value, **kwargs):
            out[:] = ["node0"] * 4 + ["node1"] * 4

        with patch.dict(rs._STATES, {}, clear=True), patch.dict(
            sys.modules, {"deep_gemm": backend}
        ), patch.object(
            rs, "create_push_reduce_scatter", return_value=None
        ), patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ), patch.object(
            rs.dist, "all_reduce"
        ), patch.object(
            rs.dist, "all_gather_object", side_effect=hosts
        ):
            with self.assertRaisesRegex(RuntimeError, "cross-host"):
                rs.configure_gemm_reduce_scatter(group, "cuda:0", max_m=512, n=7168)
            backend.GemmRSBuffer.assert_not_called()
            self.assertFalse(rs._STATES)

    def test_same_host_reserves_fused_workspace_before_optional_push(self):
        group = Mock(size=Mock(return_value=8))
        backend = Mock()
        order = []

        def hosts(out, value, **kwargs):
            out[:] = ["node0"] * 8

        def fused(*args, **kwargs):
            order.append("fused")
            return Mock(num_bytes=1024)

        def push(*args, **kwargs):
            order.append("push")
            return None

        backend.GemmRSBuffer.side_effect = fused
        with patch.dict(rs._STATES, {}, clear=True), patch.dict(
            sys.modules, {"deep_gemm": backend}
        ), patch.object(
            rs, "create_push_reduce_scatter", side_effect=push
        ), patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ), patch.object(
            rs.dist, "all_reduce"
        ), patch.object(
            rs.dist, "all_gather_object", side_effect=hosts
        ):
            rs.configure_gemm_reduce_scatter(group, "cuda:0", max_m=512, n=7168)
            self.assertEqual(order, ["fused", "push"])
            self.assertIsNone(rs._STATES[(group, 0)].push)
            self.assertIsNotNone(rs._STATES[(group, 0)].workspace)

    def test_tp16_prefill_keeps_small_push_policy_without_fused_gemm(self):
        device = torch.device("cuda", torch.cuda.current_device())
        group = Mock(size=Mock(return_value=16))
        push = Mock()
        with patch.dict(rs._STATES, {}, clear=True), patch.object(
            rs, "create_push_reduce_scatter", return_value=push
        ):
            rs.configure_gemm_reduce_scatter(group, device, max_m=1024, n=7168)
            rs.configure_gemm_reduce_scatter(
                group, device, max_m=1024, n=7168, fp8=True
            )
            state = rs._STATES[(group, device.index)]
            self.assertIsNone(state.workspace)
            for rows in (16, 496, 512, 1024):
                push.reset_mock()
                with patch.object(rs.dist, "reduce_scatter_tensor") as nccl:
                    result = rs.reduce_scatter(
                        torch.zeros((rows, 7168), device=device, dtype=torch.bfloat16),
                        group,
                    )
                self.assertEqual(result.shape, (rows // 16, 7168))
                self.assertEqual(push.reduce_scatter.call_count, int(rows < 512))
                self.assertEqual(nccl.call_count, int(rows >= 512))

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
