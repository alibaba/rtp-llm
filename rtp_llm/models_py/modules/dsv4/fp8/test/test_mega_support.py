from __future__ import annotations

import inspect
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
    FLASH_GEOMETRY,
    PRO_GEOMETRY,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_hca_weights import (
    HCA_COMPRESS_RATIO,
    HCA_STATE_WIDTH,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_support import (
    _REQUIRED_DEEP_GEMM_SYMBOLS,
    _REQUIRED_EXTENSION_PARAMETERS,
    _REQUIRED_EXTENSION_SYMBOLS,
    mega_decode_unavailable_reason,
)
from rtp_llm.models_py.modules.dsv4.transformer import V4Args


def _callable_with_parameters(name):
    def function(*_args, **_kwargs):
        return None

    function.__signature__ = inspect.Signature(
        inspect.Parameter(parameter, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in _REQUIRED_EXTENSION_PARAMETERS.get(name, ())
    )
    return function


def _module_with_symbols(names):
    return SimpleNamespace(**{name: _callable_with_parameters(name) for name in names})


def _supported_extension():
    extension = _module_with_symbols(_REQUIRED_EXTENSION_SYMBOLS)
    extension.geometry_csa = lambda: {
        "n_main": PRO_GEOMETRY.n_main,
        "n_index": 64 * 128,
        "n_merged": PRO_GEOMETRY.n_merged,
        "num_main_heads": PRO_GEOMETRY.main_heads,
        "num_index_heads": 64,
        "slot_dtype_bits": 64,
        "n_main_flash": FLASH_GEOMETRY.n_main,
        "n_merged_flash": FLASH_GEOMETRY.n_merged,
        "num_main_heads_flash": FLASH_GEOMETRY.main_heads,
    }
    extension.geometry_hca = lambda: {
        "n_q_pro": PRO_GEOMETRY.n_main,
        "front_n_fp8_pro": PRO_GEOMETRY.front_fp8_rows,
        "compress_ratio": HCA_COMPRESS_RATIO,
        "state_width": HCA_STATE_WIDTH,
        "slot_dtype_bits": 64,
        "n_q_flash": FLASH_GEOMETRY.n_main,
        "front_n_fp8_flash": FLASH_GEOMETRY.front_fp8_rows,
    }
    extension.geometry_moe_front = lambda hidden: {
        "abi_version": 1,
        "kernel_contract_version": 2,
        "hidden": hidden,
        "hc_mult": 4,
        "experts": 384 if hidden == 7168 else 256,
        "topk": 6,
        "max_m": 128,
    }
    return extension


class MegaSupportTest(unittest.TestCase):
    def test_non_blackwell_device_is_rejected_before_extension_import(self) -> None:
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            reason = mega_decode_unavailable_reason(
                V4Args(ep_size=8), torch.device("cuda:0")
            )

        self.assertIn("sm_100a or sm_103a", reason or "")
        self.assertIn("sm_90", reason or "")

    def test_missing_extension_abi_is_reported(self) -> None:
        fake_rtp_kernel = SimpleNamespace(dsv4_mega=SimpleNamespace())
        with patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ), patch.dict(sys.modules, {"rtp_kernel": fake_rtp_kernel}):
            reason = mega_decode_unavailable_reason(
                V4Args(ep_size=8), torch.device("cuda:0")
            )

        self.assertIn("missing DSV4 Mega ABI", reason or "")
        self.assertIn("geometry_moe_front", reason or "")

    def test_sm100_and_sm103_share_the_same_support_path(self) -> None:
        fake_rtp_kernel = SimpleNamespace(dsv4_mega=_supported_extension())
        fake_deep_gemm = _module_with_symbols(_REQUIRED_DEEP_GEMM_SYMBOLS)
        for capability in ((10, 0), (10, 3)):
            with self.subTest(capability=capability), patch.object(
                torch.cuda, "get_device_capability", return_value=capability
            ), patch.dict(
                sys.modules,
                {"rtp_kernel": fake_rtp_kernel, "deep_gemm": fake_deep_gemm},
            ):
                reason = mega_decode_unavailable_reason(
                    V4Args(ep_size=8), torch.device("cuda:0")
                )

            self.assertIsNone(reason)

    def test_model_geometry_is_checked_before_device(self) -> None:
        reason = mega_decode_unavailable_reason(V4Args(dim=3072), torch.device("cpu"))

        self.assertIn("unsupported hidden size 3072", reason or "")

    def test_incompatible_attention_signature_is_reported_at_startup(self) -> None:
        extension = _supported_extension()
        extension.hc_reduce_fuse_out = lambda: None
        fake_rtp_kernel = SimpleNamespace(dsv4_mega=extension)
        with patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 0)
        ), patch.dict(sys.modules, {"rtp_kernel": fake_rtp_kernel}):
            reason = mega_decode_unavailable_reason(
                V4Args(ep_size=8), torch.device("cuda:0")
            )

        self.assertIn("ABI is incompatible", reason or "")
        self.assertIn("hc_reduce_fuse_out missing", reason or "")

    def test_compiled_geometry_mismatch_is_reported_at_startup(self) -> None:
        extension = _supported_extension()
        extension.geometry_hca = lambda: {"n_q_pro": -1}
        fake_rtp_kernel = SimpleNamespace(dsv4_mega=extension)
        fake_deep_gemm = _module_with_symbols(_REQUIRED_DEEP_GEMM_SYMBOLS)
        with patch.object(
            torch.cuda, "get_device_capability", return_value=(10, 3)
        ), patch.dict(
            sys.modules,
            {"rtp_kernel": fake_rtp_kernel, "deep_gemm": fake_deep_gemm},
        ):
            reason = mega_decode_unavailable_reason(
                V4Args(ep_size=8), torch.device("cuda:0")
            )

        self.assertIn("HCA geometry mismatch", reason or "")


if __name__ == "__main__":
    unittest.main()
