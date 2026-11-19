import sys
import unittest
from contextlib import ExitStack
from types import ModuleType
from unittest.mock import patch

import torch

from rtp_llm.models_py.triton_kernels.common.kda_checkpoint import (
    get_cula_checkpoint_kernel,
)


def checkpoint_kernel(
    *args,
    checkpoint_states,
    checkpoint_interval,
    cu_seqlens_cpu,
    use_beta_sigmoid_in_kernel,
    use_intracard_cp,
    **kwargs,
):
    raise AssertionError("selection must not launch a kernel")


class KdaCheckpointTest(unittest.TestCase):
    def setUp(self):
        get_cula_checkpoint_kernel.cache_clear()
        self.addCleanup(get_cula_checkpoint_kernel.cache_clear)
        patches = ExitStack()
        self.addCleanup(patches.close)
        self.kda = ModuleType("cula.kda")
        self.kda.chunk_kda = checkpoint_kernel
        modules = {"cula": ModuleType("cula"), "cula.kda": self.kda}
        patches.enter_context(patch.dict(sys.modules, modules))
        self.capability = patches.enter_context(
            patch.object(torch.cuda, "get_device_capability", return_value=(10, 3))
        )
        patches.enter_context(patch.object(torch.version, "hip", None))

    def select(self, **updates):
        args = dict(
            device=torch.device("cuda", 1),
            dtype=torch.bfloat16,
            head_dim=128,
            state_dtype=torch.float32,
            lower_bound=-5.0,
            page_size=64,
        )
        return get_cula_checkpoint_kernel(**(args | updates))

    def test_supported_operands_and_hardware(self):
        for sm in ((10, 0), (10, 3)):
            self.capability.return_value = sm
            get_cula_checkpoint_kernel.cache_clear()
            for bound in (-5, -3.5, -0.001):
                for page_size in (64, 128, 192):
                    with self.subTest(sm=sm, bound=bound, page_size=page_size):
                        self.assertIs(
                            self.select(lower_bound=bound, page_size=page_size),
                            checkpoint_kernel,
                        )
            self.capability.assert_called_with(torch.device("cuda", 1))

    def test_unsupported_operands_fall_back_before_cuda_query(self):
        for field, values in {
            "device": [torch.device("cpu")],
            "dtype": [torch.float32, torch.float16],
            "state_dtype": [torch.bfloat16, torch.float16],
            "head_dim": [64, 256],
            "lower_bound": [None, -5.01, 0, 1.0, float("nan")],
            "page_size": [-64, 0, 32, 65],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.assertIsNone(self.select(**{field: value}))
        with patch.object(torch.version, "hip", "6.0"):
            self.assertIsNone(self.select())
        self.capability.assert_not_called()

    def test_unsupported_compute_capability_falls_back(self):
        for sm in ((9, 0), (10, 1), (12, 0)):
            with self.subTest(sm=sm):
                self.capability.return_value = sm
                get_cula_checkpoint_kernel.cache_clear()
                self.assertIsNone(self.select())

    def test_old_abi_accepting_arbitrary_kwargs_falls_back(self):
        def previous_kernel(q, k, v, g, beta, **kwargs):
            raise AssertionError("old cuLA kernel was called")

        self.kda.chunk_kda = previous_kernel
        self.assertIsNone(self.select())

    def test_missing_or_unloadable_package_falls_back(self):
        with patch.dict(sys.modules, {"cula": None, "cula.kda": None}):
            self.assertIsNone(self.select())
        get_cula_checkpoint_kernel.cache_clear()
        with patch("builtins.__import__", side_effect=OSError("missing CUDA library")):
            self.assertIsNone(self.select())

    def test_uninspectable_or_positional_only_kernel_falls_back(self):
        self.kda.chunk_kda = object()
        self.assertIsNone(self.select())
        get_cula_checkpoint_kernel.cache_clear()

        def positional_only(
            checkpoint_states,
            /,
            *,
            checkpoint_interval,
            cu_seqlens_cpu,
            use_beta_sigmoid_in_kernel,
            use_intracard_cp,
        ):
            pass

        self.kda.chunk_kda = positional_only
        self.assertIsNone(self.select())


if __name__ == "__main__":
    unittest.main()
