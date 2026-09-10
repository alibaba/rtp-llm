from __future__ import annotations

import inspect
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.mega_csa_weights import (
    FLASH_GEOMETRY,
    HC_MIX,
    MAX_BATCH,
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


class _FakeMoeFrontPlan:
    def run_learned_out(
        self,
        *args,
        router_logits=None,
        norm_eps=1.0e-6,
        hc_eps=1.0e-6,
        route_scale=2.5,
        use_pdl=True,
    ):
        return None

    def run_hash_out(
        self,
        *args,
        norm_eps=1.0e-6,
        hc_eps=1.0e-6,
        route_scale=2.5,
        use_pdl=True,
    ):
        return None


def _supported_extension():
    extension = _module_with_symbols(_REQUIRED_EXTENSION_SYMBOLS)
    extension.Dsv4MoeFrontPlan = _FakeMoeFrontPlan
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
        "kernel_contract_version": 3,
        "hidden": hidden,
        "hc_mult": 4,
        "hc_width": HC_MIX,
        "experts": 384 if hidden == 7168 else 256,
        "topk": 6,
        "max_m": MAX_BATCH,
    }
    return extension


def _pro_args(**overrides):
    values = {
        "dim": 7168,
        "q_lora_rank": 1536,
        "n_heads": 128,
        "o_groups": 16,
        "index_topk": 1024,
        "n_routed_experts": 384,
        "ep_size": 8,
    }
    values.update(overrides)
    return V4Args(**values)


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

    def test_official_geometries_support_sm100_and_sm103(self) -> None:
        fake_rtp_kernel = SimpleNamespace(dsv4_mega=_supported_extension())
        fake_deep_gemm = _module_with_symbols(_REQUIRED_DEEP_GEMM_SYMBOLS)
        for args in (V4Args(ep_size=8), _pro_args()):
            for capability in ((10, 0), (10, 3)):
                with self.subTest(dim=args.dim, capability=capability), patch.object(
                    torch.cuda, "get_device_capability", return_value=capability
                ), patch.dict(
                    sys.modules,
                    {"rtp_kernel": fake_rtp_kernel, "deep_gemm": fake_deep_gemm},
                ):
                    reason = mega_decode_unavailable_reason(
                        args, torch.device("cuda:0")
                    )

                self.assertIsNone(reason)

    def test_model_geometry_is_checked_before_device(self) -> None:
        reason = mega_decode_unavailable_reason(V4Args(dim=3072), torch.device("cpu"))

        self.assertIn("unsupported hidden size 3072", reason or "")

    def test_mhc_geometry_requires_four_lanes(self) -> None:
        reason = mega_decode_unavailable_reason(V4Args(hc_mult=2), torch.device("cpu"))

        self.assertIn("hc_mult=2", reason or "")
        self.assertIn("expected 4", reason or "")

    def test_mhc_sinkhorn_iterations_must_match_the_kernel(self) -> None:
        reason = mega_decode_unavailable_reason(
            V4Args(hc_sinkhorn_iters=3), torch.device("cpu")
        )

        self.assertIn("hc_sinkhorn_iters=3", reason or "")
        self.assertIn("expected 20", reason or "")

    def test_supported_kernel_block_sizes_pass_startup_checks(self) -> None:
        fake_rtp_kernel = SimpleNamespace(dsv4_mega=_supported_extension())
        fake_deep_gemm = _module_with_symbols(_REQUIRED_DEEP_GEMM_SYMBOLS)
        for kernel_tokens_per_block in (128, 256, 512):
            with self.subTest(
                kernel_tokens_per_block=kernel_tokens_per_block
            ), patch.object(
                torch.cuda, "get_device_capability", return_value=(10, 3)
            ), patch.dict(
                sys.modules,
                {"rtp_kernel": fake_rtp_kernel, "deep_gemm": fake_deep_gemm},
            ):
                reason = mega_decode_unavailable_reason(
                    V4Args(
                        ep_size=8,
                        kernel_tokens_per_block=kernel_tokens_per_block,
                    ),
                    torch.device("cuda:0"),
                )

            self.assertIsNone(reason)

    def test_unsupported_kernel_block_size_is_rejected_before_device(self) -> None:
        reason = mega_decode_unavailable_reason(
            V4Args(kernel_tokens_per_block=1024), torch.device("cpu")
        )

        self.assertIn("kernel_tokens_per_block=1024", reason or "")
        self.assertIn("[128, 256, 512]", reason or "")

    def test_index_topk_must_match_the_official_geometry(self) -> None:
        for args, expected in (
            (V4Args(index_topk=1024), 512),
            (_pro_args(index_topk=512), 1024),
        ):
            with self.subTest(dim=args.dim):
                reason = mega_decode_unavailable_reason(args, torch.device("cpu"))

            self.assertIn(f"index_topk={args.index_topk}", reason or "")
            self.assertIn(f"expected {expected}", reason or "")

    def test_fp32_gate_requires_the_ordinary_path(self) -> None:
        with patch.dict(os.environ, {"DSV4_GATE_FP32": "1"}):
            reason = mega_decode_unavailable_reason(
                V4Args(ep_size=8), torch.device("cuda:0")
            )

        self.assertEqual(reason, "DSV4_GATE_FP32=1 requires the ordinary DSV4 path")

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

    def test_moe_front_geometry_requires_hc_width(self) -> None:
        extension = _supported_extension()
        extension.geometry_moe_front = lambda hidden: {
            "abi_version": 1,
            "kernel_contract_version": 3,
            "hidden": hidden,
            "hc_mult": 4,
            "experts": 256,
            "topk": 6,
            "max_m": MAX_BATCH,
        }
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

        self.assertIn("MoE-front geometry mismatch", reason or "")
        self.assertIn("hc_width", reason or "")

    def test_moe_front_plan_methods_are_required(self) -> None:
        class MissingHashPlan(_FakeMoeFrontPlan):
            run_hash_out = None

        extension = _supported_extension()
        extension.Dsv4MoeFrontPlan = MissingHashPlan
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

        self.assertIn("Dsv4MoeFrontPlan.run_hash_out", reason or "")

    def test_moe_front_plan_signature_is_checked_when_available(self) -> None:
        class IncompletePlan(_FakeMoeFrontPlan):
            def run_hash_out(self, *args, norm_eps=1.0e-6):
                return None

        extension = _supported_extension()
        extension.Dsv4MoeFrontPlan = IncompletePlan
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

        self.assertIn("Dsv4MoeFrontPlan.run_hash_out missing", reason or "")
        self.assertIn("hc_eps", reason or "")


if __name__ == "__main__":
    unittest.main()
