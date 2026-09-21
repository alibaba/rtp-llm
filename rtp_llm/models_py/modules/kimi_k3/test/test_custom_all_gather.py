"""AG/GEMM role and per-rank input-row dispatch; GPU tests cover collectives."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.distributed import custom_all_gather as custom_ag
from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    QuantizedActivation,
)
from rtp_llm.models_py.modules.kimi_k3 import all_gather_gemm as ag


class _Projection:
    scale_ue8m0 = True
    K = 128

    def forward_quantized(self, values, scales):
        # Makes row ownership and the scale transpose observable without a GPU.
        return values[:, :1].to(torch.bfloat16) + scales[:, :1].to(torch.bfloat16)


class CustomAllGatherDispatchTest(unittest.TestCase):
    def test_role_and_per_rank_rows_select_staging_direct_or_overlap(self):
        group = Mock(size=Mock(return_value=8))
        for fp8 in (False, True):
            for prefill in (False, True):
                state = ag._AllGatherGemmState(
                    fp8,
                    group,
                    torch.device("cpu"),
                    8,
                    65536,
                    128,
                    torch.bfloat16,
                    0,
                    use_fused=prefill,
                )
                custom = Mock()
                state.custom = custom
                for m, prefill_backend in (
                    (1, "staging"),
                    (3, "staging"),
                    (127, "staging"),
                    (128, "direct"),
                    (129, "direct"),
                    (512, "direct"),
                    (4095, "direct"),
                    (4096, "overlap"),
                    (4097, "overlap"),
                    (8192, "overlap"),
                ):
                    with self.subTest(fp8=fp8, prefill=prefill, local_m=m):
                        physical_m = m * 8
                        backend = prefill_backend if prefill else "staging"
                        values = torch.ones((m, 128), dtype=torch.bfloat16)
                        logical_m = physical_m - 3
                        # Keep logical M below a boundary while physical M reaches it.
                        wire = torch.arange((m + 3) // 4 * 4, dtype=torch.int32)[None]
                        if fp8:
                            values = values.to(torch.float8_e4m3fn)
                            local = QuantizedActivation(values, wire)
                            weights = [_Projection()]
                            gathered = (
                                values.view(torch.uint8).repeat(8, 1).view(values.dtype)
                            )
                            scales = wire[:, :m].repeat(1, 8).T
                            reference = weights[0].forward_quantized(gathered, scales)
                            custom.all_gather_fp8.return_value = (gathered, scales)
                        else:
                            local = values
                            weights = [torch.ones((128, 1), dtype=torch.bfloat16)]
                            gathered = values.repeat(8, 1)
                            reference = gathered @ weights[0]
                            custom.all_gather.return_value = gathered
                        custom.reset_mock()
                        with patch.dict(
                            ag._STATES, {(group, 0, fp8): state}, clear=True
                        ), patch.object(
                            ag, "get_process_group", return_value=group
                        ), patch.object(
                            ag, "collective_gemm_state_key", return_value=(group, 0)
                        ), patch.object(
                            ag,
                            "fused_all_gather_matmul",
                            return_value=(None, [reference]),
                        ) as bf16_fused, patch.object(
                            ag, "fused_all_gather_fp8_linear", return_value=[reference]
                        ) as fp8_fused, patch.object(
                            ag.dist, "all_gather_into_tensor"
                        ) as nccl:
                            out = ag.all_gather_gemm(
                                local, weights, logical_m=logical_m
                            )
                        overlap = backend == "overlap"
                        fused = fp8_fused if fp8 else bf16_fused
                        self.assertEqual(fused.call_count, int(overlap))
                        method = custom.all_gather_fp8 if fp8 else custom.all_gather
                        self.assertEqual(method.call_count, int(not overlap))
                        if not overlap:
                            self.assertEqual(
                                method.call_args.kwargs,
                                {"staging": backend == "staging"},
                            )
                        nccl.assert_not_called()
                        torch.testing.assert_close(
                            out[0], reference[:logical_m], rtol=0, atol=0
                        )

    def test_initialization_caps_only_prefill_custom_capacity(self):
        group = Mock(size=Mock(return_value=8))
        for fp8 in (False, True):
            for prefill in (False, True):
                with self.subTest(fp8=fp8, prefill=prefill), patch.dict(
                    ag._STATES, {}, clear=True
                ), patch.object(
                    ag, "collective_gemm_state_key", return_value=(group, 0)
                ), patch.object(
                    ag, "create_custom_all_gather", return_value=Mock()
                ) as create, patch.object(
                    ag, "reserve_fused_all_gather_matmul_workspace"
                ):
                    ag.configure_all_gather_gemm(
                        group,
                        torch.device("cuda", 0),
                        max_m=65536,
                        k=7168,
                        dtype=torch.bfloat16,
                        fp8=fp8,
                        use_fused=prefill,
                    )
                    create.assert_called_once_with(
                        group,
                        torch.device("cuda", 0),
                        max_m=32760 if prefill else 65536,
                        k=7168,
                        fp8=fp8,
                        staging_only=not prefill,
                    )
                    state = ag._STATES[(group, 0, fp8)]
                    self.assertEqual(state.max_m, 65536)
                    self.assertEqual(state.overlap_min_local_m, 4096)
                    self.assertIs(state.custom, create.return_value)
                    # Reconfiguration is idempotent and does not rendezvous again.
                    ag.configure_all_gather_gemm(
                        group,
                        torch.device("cuda", 0),
                        max_m=65536,
                        k=7168,
                        dtype=torch.bfloat16,
                        fp8=fp8,
                        use_fused=prefill,
                    )
                    self.assertEqual(create.call_count, 1)


class CustomAllGatherInitializationTest(unittest.TestCase):
    def test_unsupported_shapes_do_not_enter_collectives(self):
        with patch.object(custom_ag, "_all_ready") as collective:
            for size, k in ((1, 7168), (4, 7168), (16, 7168), (8, 4096)):
                result = custom_ag.create_custom_all_gather(
                    Mock(size=Mock(return_value=size)), "cpu", max_m=8, k=k, fp8=False
                )
                self.assertIsNone(result)
            collective.assert_not_called()

    def test_collective_initialization_rejection_stops_before_next_rendezvous(self):
        import torch.distributed._symmetric_memory as symm

        for failure in (
            "kernel",
            "peer_capability",
            "topology",
            "sm_count",
            "allocation",
            "peer_allocation",
            "mapping",
        ):
            with self.subTest(failure=failure), ExitStack() as stack:
                group = Mock(size=Mock(return_value=8), rank=Mock(return_value=0))
                props = SimpleNamespace(
                    major=10, minor=3, uuid="0", multi_processor_count=148
                )
                stack.enter_context(
                    patch.object(
                        torch.cuda, "is_current_stream_capturing", return_value=False
                    )
                )
                stack.enter_context(
                    patch.object(
                        torch.cuda, "get_device_properties", return_value=props
                    )
                )
                stack.enter_context(
                    patch.object(symm, "get_backend", return_value="CUDA")
                )
                load = stack.enter_context(
                    patch.object(
                        custom_ag, "_load_kernels", return_value=(Mock(), Mock())
                    )
                )
                if failure == "kernel":
                    load.side_effect = AttributeError("custom kernel missing")
                ready_calls = []

                def ready(value, *args):
                    ready_calls.append(value)
                    peer_failure = (
                        failure == "peer_capability" and len(ready_calls) == 1
                    ) or (failure == "peer_allocation" and len(ready_calls) == 3)
                    return value and not peer_failure

                stack.enter_context(
                    patch.object(custom_ag, "_all_ready", side_effect=ready)
                )

                def gather_metadata(output, value, **kwargs):
                    for i in range(8):
                        output[i] = (value[0], str(i), *value[2:])
                    if failure == "sm_count":
                        output[1] = (*output[1][:-1], 132)

                stack.enter_context(
                    patch.object(
                        custom_ag.dist, "all_gather_object", side_effect=gather_metadata
                    )
                )
                stack.enter_context(
                    patch.object(
                        custom_ag, "_nvlink_peers", return_value=failure != "topology"
                    )
                )
                allocate = stack.enter_context(
                    patch.object(symm, "empty", side_effect=torch.empty)
                )
                if failure == "allocation":
                    allocate.side_effect = RuntimeError("out of memory")
                mapping = stack.enter_context(
                    patch.object(
                        symm,
                        "rendezvous",
                        side_effect=[
                            Mock(multicast_ptr=16),
                            RuntimeError("mapping failed"),
                        ],
                    )
                )
                self.assertIsNone(
                    custom_ag.create_custom_all_gather(
                        group, "cpu", max_m=8, k=7168, fp8=True
                    )
                )
                self.assertEqual(mapping.call_count, 2 if failure == "mapping" else 0)
                if failure in ("kernel", "peer_capability", "topology", "sm_count"):
                    allocate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
