"""Pinned-source ABI adaptation; raw legacy APIs remain unsupported."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


PATH = Path(__file__).resolve().parents[1] / "mega_moe/prefill_backend.py"
SPEC = importlib.util.spec_from_file_location("prefill_situ_contract", PATH)
BACKEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND)


class PrefillSiTUBackendTest(unittest.TestCase):
    def setUp(self):
        self.source = SimpleNamespace(
            _C=SimpleNamespace(
                get_token_alignment_for_mega_moe=mock.Mock(return_value=1920),
                get_symm_buffer_size_for_mega_moe=mock.Mock(return_value=(123, None)),
            ),
            fp8_fp4_mega_moe=mock.Mock(),
            transform_weights_for_mega_moe=mock.Mock(),
            get_symm_buffer_for_mega_moe=mock.Mock(),
        )
        self.backend = BACKEND._PinnedSiTUBackend(self.source)

    def test_gate_and_up_mapping(self):
        for up, expected in ((25.0, 25.0), (None, 0.0)):
            self.backend.fp8_fp4_mega_moe_k3(
                y="out", l1_weights="l1", l2_weights="l2", sym_buffer="buf",
                activation_beta=4.0, activation_linear_beta=up,
            )
            kw = self.source.fp8_fp4_mega_moe.call_args.kwargs
            self.assertEqual(kw["activation_alpha"], 4.0)
            self.assertEqual(kw["activation_beta"], expected)
            self.assertEqual(kw["activation"], "situ")
            self.assertEqual(kw["recipe"], (1, 1, 32))
            self.assertEqual(kw["sym_buffer"], "buf")

    def test_matching_transform_and_routed_only_allocator(self):
        self.assertIs(self.backend.transform_weights_for_mega_moe_k3,
                      self.source.transform_weights_for_mega_moe)
        self.backend.get_symm_buffer_for_mega_moe_k3(
            group="g", num_experts=896, num_max_tokens_per_rank=8192,
            num_topk=16, hidden=3584, intermediate_hidden=3072,
        )
        kw = self.source.get_symm_buffer_for_mega_moe.call_args.kwargs
        self.assertEqual(kw["num_shared_experts"], 0)
        self.assertEqual(kw["mma_type"], "fp8xfp4")
        self.assertEqual(self.backend._C.get_token_alignment_for_mega_moe_k3(), 1920)
        args = (8, 896, 9600, 16, 3584, 3072, "fp8xfp4", "situ")
        self.backend._C.get_symm_buffer_size_for_mega_moe_k3(*args)
        self.source._C.get_symm_buffer_size_for_mega_moe.assert_called_once_with(*args, 0)

    def test_production_launch_positional_abi(self):
        # MegaMoeExecutor._launch passes its four buffers positionally.
        self.backend.fp8_fp4_mega_moe_k3(
            "out", "l1", "l2", "buf",
            cumulative_local_expert_recv_stats=None,
            recipe=(1, 1, 32), activation="situ", activation_clamp=None,
            activation_beta=4.0, activation_linear_beta=25.0, fast_math=True,
        )
        self.source.fp8_fp4_mega_moe.assert_called_once_with(
            y="out", l1_weights="l1", l2_weights="l2", sym_buffer="buf",
            cumulative_local_expert_recv_stats=None, recipe=(1, 1, 32),
            activation="situ", activation_clamp=None, fast_math=True,
            activation_alpha=4.0, activation_beta=25.0,
        )

    def test_shared_or_other_activation_is_rejected(self):
        with self.assertRaises(ValueError):
            self.backend.fp8_fp4_mega_moe_k3(
                y=None, l1_weights=None, l2_weights=None, sym_buffer=None,
                activation="swiglu",
            )
        with self.assertRaises(TypeError):
            self.backend.fp8_fp4_mega_moe_k3(
                y=None, l1_weights=None, l2_weights=None, sym_buffer=None,
                shared_x=None,
            )

    def test_provenance_and_binary_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / "_C.fixture.so"
            binary.write_bytes(b"test-only-receipt")
            info = dict(commit=BACKEND._COMMIT, patch_sha256=BACKEND._EPOCH_PATCH,
                        isolation_helper_sha256=BACKEND._ISOLATED_HELPER,
                        isolation_compile_flags=sorted(BACKEND._ISOLATION_FLAGS),
                        math_source_changed=False, torch="test-runtime",
                        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
            receipt = root / "BUILD_INFO.json"
            receipt.write_text(json.dumps(info))
            package = SimpleNamespace(__file__=str(root / "__init__.py"))
            self.assertEqual(BACKEND._validate_package(package, "test-runtime"), info)
            with self.assertRaises(RuntimeError):
                BACKEND._validate_package(package, "different-runtime")
            info["isolation_compile_flags"] = []
            receipt.write_text(json.dumps(info))
            with self.assertRaises(RuntimeError):
                BACKEND._validate_package(package, "test-runtime")
            info["isolation_compile_flags"] = sorted(BACKEND._ISOLATION_FLAGS)
            receipt.write_text(json.dumps(info))
            binary.write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                BACKEND._validate_package(package, "test-runtime")


if __name__ == "__main__":
    unittest.main()
