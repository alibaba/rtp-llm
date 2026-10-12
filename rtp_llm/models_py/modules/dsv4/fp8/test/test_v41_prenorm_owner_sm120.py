"""Checkpoint prenorm selection, composed mHC, and precision-safe admission."""

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_collapse_owner_sm120 as owner_tests,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_mhc_precision import mhc_fp32
from rtp_llm.models_py.modules.dsv4.hc import v41_prenorm


@contextmanager
def matmul_precision(value):
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision(value)
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


class PrenormOwnerSM120Test(owner_tests.CollapseOwnerSM120Test):
    component = "prenorm"
    mode_value = "fp32"
    dispatch_module = v41_prenorm
    reference_context = staticmethod(mhc_fp32)

    @property
    def dispatch_name(self):
        return "prenorm"

    def setUp(self):
        previous_dtype = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, previous_dtype)
        torch.set_default_dtype(torch.float32)
        previous = torch.get_float32_matmul_precision()
        self.addCleanup(torch.set_float32_matmul_precision, previous)
        torch.set_float32_matmul_precision("highest")

    def additional_mutations(self, selected):
        owners = [
            unit
            for layer in selected.v4.layers
            for unit in (layer.attn_hc, layer.ffn_hc)
        ]
        mixes = [unit.pre_mix_out for unit in owners]
        for precision in ("high", "medium"):
            with matmul_precision(precision):
                with self.assertRaisesRegex(ValueError, "highest FP32"):
                    selected.forward(owner_tests.inputs(selected.kv_cache))
                self.assertEqual(torch.get_float32_matmul_precision(), precision)
                self.assertEqual(selected._worker_state, "ready")
                for owner, saved in zip(owners, mixes):
                    self.assertIs(owner.pre_mix_out, saved)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        return 2

    @torch.inference_mode()
    def test_precision_rejected_before_materialization(self):
        owner_tests.Dsv4SharedRuntimeBufferStore._reset_for_test()
        self.addCleanup(owner_tests.Dsv4SharedRuntimeBufferStore._reset_for_test)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "DSV41_ENGRAM_HOST_BYTES": str(2**20),
                "DSV41_ENGRAM_CACHE_DIR": "/dev/shm",
                "DSV4_MOE_STRATEGY": "auto",
                "MOEDBG": "0",
            },
        ):
            config = owner_tests.checkpoint(Path(directory))
            config.update(self.extra_choices)
            config.update(self.policy)
            args, weights = owner_tests.fixture()
            before = [
                {key: value.clone() for key, value in layer.items()}
                for layer in (weights.global_weights, *weights.weights)
            ]
            parallelism = owner_tests.ParallelismConfig()
            parallelism.role_type = owner_tests.RoleType.PREFILL
            model = owner_tests.DeepSeekV41Model(
                owner_tests.model_config(args, config), parallelism, weights, None, 2
            )
            try:
                for precision in ("high", "medium"):
                    with matmul_precision(precision):
                        with self.assertRaisesRegex(ValueError, "highest FP32"):
                            model.initialize(owner_tests.resources())
                        self.assertEqual(
                            torch.get_float32_matmul_precision(), precision
                        )
                        self.assertIs(model.weight, weights)
                        self.assertIsNone(model.v4)
                        self.assertEqual(model._worker_state, "new")
                        self.assertFalse(model._materialized)
                        self.assertFalse(model._engram_tables)
                        for actual, saved in zip(
                            (weights.global_weights, *weights.weights), before
                        ):
                            self.assertEqual(actual.keys(), saved.keys())
                            for key, value in actual.items():
                                owner_tests.exact(value, saved[key])
                    print(
                        "DSV41_PRENORM_OWNER_COLD_PRECISION "
                        + json.dumps(
                            dict(
                                combined=bool(self.extra_choices),
                                precision=precision,
                                weights_exact=True,
                                tables_empty=True,
                                precision_unchanged=True,
                            )
                        ),
                        flush=True,
                    )
                self.assertTrue(model.initialize(owner_tests.resources()))
                self.assertEqual(model._worker_state, "ready")
            finally:
                model.close()


class PrenormCombinedOwnerSM120Test(PrenormOwnerSM120Test):
    extra_choices = {
        "mhc_collapse_mode": "ordered_fp32",
        "mhc_post_mode": "ordered_fp32",
    }


if __name__ == "__main__":
    unittest.main()
