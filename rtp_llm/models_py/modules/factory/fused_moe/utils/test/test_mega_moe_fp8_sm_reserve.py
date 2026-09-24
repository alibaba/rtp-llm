import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.fp8_sm_reserve import (
    configure_mega_moe_fp8_num_sms,
)


class MegaMoeFp8SmReserveTest(unittest.TestCase):
    def test_disabled_is_exact_noop(self):
        for value in (None, "0"):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["MEGA_MOE_FP8_RESERVE_SM"] = value
                deep_gemm = Mock()
                with patch("torch.cuda.get_device_properties") as properties:
                    with configure_mega_moe_fp8_num_sms(deep_gemm, "cuda:1"):
                        pass
                    properties.assert_not_called()
                self.assertEqual(deep_gemm.mock_calls, [])

    def test_budget_and_restoration(self):
        # Full device, existing outer budget, nested reserve, odd cluster count.
        for physical, current, target in (
            (148, 148, 146),
            (148, 120, 120),
            (148, 146, 146),
            (147, 147, 144),
        ):
            for raises in (False, True):
                with self.subTest(physical=physical, current=current, raises=raises):
                    deep_gemm = Mock()
                    deep_gemm.get_num_sms.return_value = current
                    properties = SimpleNamespace(
                        major=10, multi_processor_count=physical
                    )
                    with patch.dict(
                        os.environ, {"MEGA_MOE_FP8_RESERVE_SM": "1"}
                    ), patch(
                        "torch.cuda.get_device_properties", return_value=properties
                    ) as get_properties:
                        try:
                            with configure_mega_moe_fp8_num_sms(deep_gemm, "cuda:1"):
                                expected = [] if current == target else [call(target)]
                                self.assertEqual(
                                    deep_gemm.set_num_sms.call_args_list, expected
                                )
                                if raises:
                                    raise RuntimeError("launch failed")
                        except RuntimeError as error:
                            self.assertTrue(raises)
                            self.assertEqual(str(error), "launch failed")
                        expected = (
                            [] if current == target else [call(target), call(current)]
                        )
                        self.assertEqual(deep_gemm.set_num_sms.call_args_list, expected)
                        get_properties.assert_called_once_with("cuda:1")

    def test_executor_scopes_override_to_kernel(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
            mega_moe_fp8 as executor_module,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_fp8_se import (
            MegaMoeFp8SEExecutor,
        )

        launch = executor_module.MegaMoeFp8Executor._launch
        self.assertIs(MegaMoeFp8SEExecutor._launch, launch)
        for enabled in ("0", "1"):
            for raises in (False, True):
                with self.subTest(enabled=enabled, raises=raises):
                    state = SimpleNamespace(num_sms=148)
                    observed = []

                    def kernel(*args, **kwargs):
                        observed.append(state.num_sms)
                        if raises:
                            raise RuntimeError("kernel failed")

                    deep_gemm = SimpleNamespace(
                        get_num_sms=lambda: state.num_sms,
                        set_num_sms=lambda value: setattr(state, "num_sms", value),
                        mega_fp8=SimpleNamespace(fp8_fp8_mega_moe=kernel),
                    )
                    executor = SimpleNamespace(
                        config=SimpleNamespace(layer_id=0),
                        l1=object(),
                        l2=object(),
                        _mega_buf=SimpleNamespace(
                            _rtp_fp8_num_sms=146 if enabled == "1" else 148
                        ),
                        _maybe_pre_kernel_barrier=lambda tokens: self.assertEqual(
                            state.num_sms, 148
                        ),
                    )
                    with patch.dict(
                        os.environ, {"MEGA_MOE_FP8_RESERVE_SM": enabled}
                    ), patch.dict(sys.modules, {"deep_gemm": deep_gemm}), patch(
                        "torch.cuda.get_device_properties",
                        return_value=SimpleNamespace(
                            major=10, multi_processor_count=148
                        ),
                    ), patch.object(
                        executor_module, "mega_moe_snapshot_active", return_value=False
                    ), patch.object(
                        executor_module,
                        "sync_cuda_graph_warmup_ranks",
                        side_effect=lambda *args: self.assertEqual(state.num_sms, 148),
                    ):
                        if raises:
                            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                                launch(executor, object(), 8, "cuda:1")
                        else:
                            launch(executor, object(), 8, "cuda:1")
                        executor._mega_buf._rtp_fp8_num_sms -= 2
                        with self.assertRaisesRegex(RuntimeError, "SM budget changed"):
                            launch(executor, object(), 8, "cuda:1")
                    self.assertEqual(observed, [146 if enabled == "1" else 148])
                    self.assertEqual(state.num_sms, 148)

    def test_buffers_use_launch_budget_and_cache_by_sms(self):
        from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe import (
            buffer,
            se_buffer,
        )

        for allocate, cache, extra in (
            (buffer._get_or_create_mega_fp8_buf, buffer._MEGA_FP8_BUF_CACHE, {}),
            (
                se_buffer._get_or_create_mega_fp8_se_buf,
                se_buffer._MEGA_FP8_SE_BUF_CACHE,
                {"num_shared_experts": 1},
            ),
        ):
            with self.subTest(allocate=allocate.__name__), patch.dict(
                cache, {}, clear=True
            ):
                state = SimpleNamespace(num_sms=148)
                allocations = []

                def create(*args, **kwargs):
                    allocations.append(state.num_sms)
                    return SimpleNamespace(shared_l1_acts_sf=object())

                deep_gemm = SimpleNamespace(
                    get_num_sms=lambda: state.num_sms,
                    set_num_sms=lambda value: setattr(state, "num_sms", value),
                    mega_fp8=SimpleNamespace(get_symm_buffer_for_mega_moe_fp8=create),
                )
                group = object()
                results = []
                with patch.dict(sys.modules, {"deep_gemm": deep_gemm}), patch(
                    "torch.cuda.get_device_properties",
                    return_value=SimpleNamespace(major=10, multi_processor_count=148),
                ):
                    for enabled in ("0", "1", "0", "1"):
                        with patch.dict(
                            os.environ, {"MEGA_MOE_FP8_RESERVE_SM": enabled}
                        ):
                            result = allocate(group, 512, 8192, 8, 4096, 1024, **extra)
                            self.assertEqual(
                                result._rtp_fp8_num_sms, 146 if enabled == "1" else 148
                            )
                            self.assertEqual(state.num_sms, 148)
                            results.append(result)
                self.assertEqual(allocations, [148, 146])
                self.assertIs(results[0], results[2])
                self.assertIs(results[1], results[3])
                self.assertIsNot(results[0], results[1])

    def test_pre_blackwell_is_noop(self):
        deep_gemm = Mock()
        with patch.dict(os.environ, {"MEGA_MOE_FP8_RESERVE_SM": "1"}), patch(
            "torch.cuda.get_device_properties",
            return_value=SimpleNamespace(major=9, multi_processor_count=132),
        ):
            with configure_mega_moe_fp8_num_sms(deep_gemm, "cuda:0"):
                pass
        self.assertEqual(deep_gemm.mock_calls, [])


if __name__ == "__main__":
    unittest.main()
