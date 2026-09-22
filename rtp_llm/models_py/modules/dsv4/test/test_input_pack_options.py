"""CPU contracts for standalone MegaMoE pack validation and route masking."""

from __future__ import annotations

import builtins
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from dsv4_source_loader import load_source_module

_TEST_DIR = Path(__file__).resolve().parent
_MOE_DIR = _TEST_DIR.parent / "moe"
_OPTIONS_PATH = _MOE_DIR / "input_pack_options.py"
_KERNEL_PATH = _MOE_DIR / "_mega_input_pack_triton.py"


def _load_options(package_name: str = "_dsv4_input_pack_options_contract"):
    if not _OPTIONS_PATH.is_file():
        raise FileNotFoundError(f"public pack rules module is missing: {_OPTIONS_PATH}")
    return load_source_module(
        package_name,
        "input_pack_options",
        str(_OPTIONS_PATH),
        [str(_MOE_DIR)],
    )


class InputPackOptionsTest(unittest.TestCase):
    def test_count_and_mask_intersect_without_mutating_routes(self):
        options = _load_options()
        indices = torch.tensor([[11, 12], [21, 22], [31, 32], [41, 42]])
        weights = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        original_indices = indices.clone()
        original_weights = weights.clone()

        got_indices, got_weights = options.mask_pack_routes(
            indices,
            weights,
            valid_token_count=3,
            valid_token_mask=torch.tensor([True, False, True, True]),
        )

        self.assertTrue(
            torch.equal(
                got_indices,
                torch.tensor([[11, 12], [0, 0], [31, 32], [0, 0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                got_weights,
                torch.tensor([[0.1, 0.2], [0.0, 0.0], [0.5, 0.6], [0.0, 0.0]]),
            )
        )
        self.assertTrue(torch.equal(indices, original_indices))
        self.assertTrue(torch.equal(weights, original_weights))

    def test_invalid_token_counts_are_rejected(self):
        options = _load_options()
        for value in (-1, 5, True, 2.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "outside the local token shard"
                ):
                    options.validate_pack_options(4, valid_token_count=value)

    def test_invalid_mask_and_shared_options_are_rejected(self):
        options = _load_options()
        with self.assertRaisesRegex(ValueError, "one entry per local token"):
            options.validate_pack_options(3, valid_token_mask=torch.ones((3, 1)))
        shared_input = torch.ones((3, 4), dtype=torch.bfloat16)
        shared_out = torch.empty((3, 4), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            options.validate_pack_options(3, shared_input=shared_input)
        with self.assertRaisesRegex(ValueError, "shape, dtype or device mismatch"):
            options.validate_pack_options(
                3, shared_input=shared_input, shared_out=shared_out
            )

    def test_public_rules_import_without_triton_or_model_backend(self):
        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "triton" or name.startswith(("triton.", "rtp_llm.models_py")):
                raise AssertionError(f"unexpected backend import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        spec = importlib.util.spec_from_file_location(
            "_dsv4_input_pack_options_isolated", _OPTIONS_PATH
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.object(builtins, "__import__", side_effect=guarded_import):
            spec.loader.exec_module(module)
        module.validate_pack_options(2, valid_token_count=2)

    def test_minimal_source_bundle_loads_kernel_and_shared_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            shutil.copy2(_KERNEL_PATH, bundle / _KERNEL_PATH.name)
            shutil.copy2(_OPTIONS_PATH, bundle / _OPTIONS_PATH.name)
            kernel = load_source_module(
                "_dsv4_minimal_pack_bundle",
                "_mega_input_pack_triton",
                str(bundle / _KERNEL_PATH.name),
                [str(bundle)],
            )

            indices = torch.tensor([[1, 2], [3, 4], [5, 6]])
            weights = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            got_indices, got_weights = kernel.mask_pack_routes(
                indices, weights, valid_token_count=2
            )

            self.assertTrue(
                torch.equal(got_indices, torch.tensor([[1, 2], [3, 4], [0, 0]]))
            )
            self.assertTrue(
                torch.equal(
                    got_weights,
                    torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]]),
                )
            )


if __name__ == "__main__":
    unittest.main()
