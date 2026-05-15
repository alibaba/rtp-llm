import os
import sys
import types
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _stub_package(name: str, path: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules.setdefault(name, module)


_stub_package("rtp_llm", os.path.join(_REPO, "rtp_llm"))
_stub_package("rtp_llm.models_py", os.path.join(_REPO, "rtp_llm", "models_py"))
_stub_package(
    "rtp_llm.models_py.modules",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4"),
)

from rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup import (
    _collect_dsv4_branch_kernel_configs,
    _collect_dsv4_dense_gemm_shapes,
    resolve_dense_gemm_warmup_max_m,
)


def _module_type(name, attrs):
    def __init__(self):
        nn.Module.__init__(self)
        for key, value in attrs.items():
            setattr(self, key, value)

    return type(name, (nn.Module,), {"__init__": __init__})


class Dsv4KernelJitWarmupTest(unittest.TestCase):
    def test_dense_warmup_prefill_uses_sequence_bound(self):
        self.assertEqual(
            resolve_dense_gemm_warmup_max_m(
                max_seq_len=1048576,
                max_batch_size=1024,
                role_type_name="PREFILL",
                max_potential_token_num=5120,
                is_speculative=True,
                gen_num_per_cycle=4,
            ),
            1048576,
        )

    def test_dense_warmup_decode_uses_model_token_capacity(self):
        self.assertEqual(
            resolve_dense_gemm_warmup_max_m(
                max_seq_len=1048576,
                max_batch_size=1024,
                role_type_name="DECODE",
                max_potential_token_num=5120,
                is_speculative=True,
                gen_num_per_cycle=4,
            ),
            5120,
        )

    def test_dense_warmup_decode_fallback_accounts_for_speculative_width(self):
        self.assertEqual(
            resolve_dense_gemm_warmup_max_m(
                max_seq_len=1048576,
                max_batch_size=1024,
                role_type_name="RoleType.DECODE",
                max_potential_token_num=0,
                is_speculative=True,
                gen_num_per_cycle=4,
            ),
            5120,
        )

    def test_dense_warmup_decode_fallback_defaults_to_batch_size(self):
        self.assertEqual(
            resolve_dense_gemm_warmup_max_m(
                max_seq_len=1048576,
                max_batch_size=1024,
                role_type_name="DECODE",
            ),
            1024,
        )

    def test_collect_branch_configs_covers_all_ratios(self):
        args = SimpleNamespace(
            compress_ratios=[0, 4, 128, 4],
            window_size=4096,
            rope_head_dim=64,
            index_topk=512,
        )
        configs = _collect_dsv4_branch_kernel_configs(v4=None, v4_args=args)
        self.assertEqual(
            configs["combine"],
            (
                (4096, 1, 0),
                (4096, 4, 512),
                (4096, 128, 512),
            ),
        )
        self.assertEqual(
            configs["compressor"],
            (
                (128, 64, 4, True),
                (512, 64, 4, True),
                (512, 64, 128, False),
            ),
        )

    def test_collect_dense_shapes_dedupes_representatives(self):
        root = nn.Module()

        Fp8Linear = _module_type(
            "CudaFp8DeepGEMMLinear",
            {
                "N": 16,
                "K": 32,
                "weight": torch.empty((16, 32), dtype=torch.bfloat16),
                "weight_scales": torch.empty((16, 1), dtype=torch.int32),
                "scale_ue8m0": True,
            },
        )
        root.add_module("fp8_a", Fp8Linear())
        root.add_module("fp8_b_same_shape", Fp8Linear())

        Fp4Linear = _module_type(
            "QuantizedLinear",
            {
                "storage": "fp4",
                "out_features": 8,
                "in_features": 64,
                "weight": torch.empty((8, 32), dtype=torch.int8),
                "scale_gemm": torch.empty((8, 1), dtype=torch.int32),
            },
        )
        root.add_module("fp4", Fp4Linear())

        Grouped = _module_type(
            "GroupedFP4Strategy",
            {
                "_w13": torch.empty((2, 10, 32), dtype=torch.int8),
                "_s13_dense_t": torch.empty((2, 2, 10), dtype=torch.int32),
                "_w2": torch.empty((2, 64, 5), dtype=torch.int8),
                "_s2_dense_t": torch.empty((2, 1, 64), dtype=torch.int32),
            },
        )
        root.add_module("grouped", Grouped())

        shapes = _collect_dsv4_dense_gemm_shapes(root)
        self.assertEqual(
            sorted(shapes.keys()),
            [
                ("fp8", 16, 32),
                ("fp8_fp4", 8, 64),
                ("fp8_fp4", 10, 64),
                ("fp8_fp4", 64, 10),
            ],
        )
        self.assertEqual(shapes[("fp8", 16, 32)]["name"], "fp8_a")


if __name__ == "__main__":
    unittest.main()
