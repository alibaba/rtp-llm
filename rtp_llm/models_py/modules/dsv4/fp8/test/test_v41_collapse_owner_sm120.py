"""Checkpoint-selected mHC readout and mutation-safe native worker admission."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv4 import native_l20_transfer
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_transformer_sm120 import fixture
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    decode_inputs,
    inputs,
    model_config,
    resources,
    tagged_cache,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_mhc_collapse_reference import (
    mhc_collapse_reference,
)
from rtp_llm.models_py.modules.dsv4.hc import delayed
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import V41L20Contract
from rtp_llm.ops import ParallelismConfig, RoleType

MODE = {"mhc_collapse_mode": "ordered_fp32"}


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


class CollapseOwnerSM120Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise unittest.SkipTest("Requires a leased SM120 GPU")
        torch.set_num_threads(1)

    def test_transfer_policy_rejects_before_cache_views(self):
        identity = ("a" * 64, "b" * 64, "c" * 64, 256)
        default = V41L20Contract(*identity)
        selected = V41L20Contract(*identity, **MODE)
        selected.require_attention_config(MODE)
        self.assertNotEqual(selected.fingerprint, default.fingerprint)
        with patch.object(
            native_l20_transfer,
            "native_l20_cache_views",
            side_effect=AssertionError("Policy mismatch reached cache views"),
        ):
            for config, contract in ((MODE, default), ({}, selected)):
                with self.assertRaisesRegex(ValueError, "mHC collapse mode"):
                    native_l20_transfer.export_native_l20_delta(
                        config, None, contract, 0, 0, (), {}
                    )
                with self.assertRaisesRegex(ValueError, "mHC collapse mode"):
                    native_l20_transfer.NativeL20ReplicaReceiver(
                        config, None, contract, 0
                    )

    @torch.inference_mode()
    def test_selected_worker_matches_reference_and_rejects_mutation_before_writes(self):
        Dsv4SharedRuntimeBufferStore._reset_for_test()
        self.addCleanup(Dsv4SharedRuntimeBufferStore._reset_for_test)
        previous_dtype = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, previous_dtype)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(41023)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "DSV41_ENGRAM_HOST_BYTES": str(2**20),
                "DSV41_ENGRAM_CACHE_DIR": "/dev/shm",
                "DSV4_MOE_STRATEGY": "auto",
                "MOEDBG": "0",
            },
        ):
            config = checkpoint(Path(directory))
            args, weights = fixture()
            models = []
            try:
                for selected in (True, False):
                    clone = SimpleNamespace(
                        global_weights={
                            key: value.clone()
                            for key, value in weights.global_weights.items()
                        },
                        weights=[
                            {key: value.clone() for key, value in layer.items()}
                            for layer in weights.weights
                        ],
                    )
                    parallelism = ParallelismConfig()
                    parallelism.role_type = RoleType.PREFILL
                    cfg = dict(config, **MODE) if selected else config
                    model = DeepSeekV41Model(
                        model_config(args, cfg), parallelism, clone, None, 2
                    )
                    models.append(model)
                    self.assertTrue(model.initialize(resources()))
                    self.assertTrue(model.initialize(resources(tagged_cache(model))))
                selected, reference = models
                self.assertEqual(
                    selected._agreement_config()["v41_config"]["mhc_collapse_mode"],
                    "ordered_fp32",
                )
                for model, mode in ((selected, "ordered_fp32"), (reference, None)):
                    self.assertEqual(model._mhc_collapse_mode, mode)
                    for layer in model.v4.layers:
                        self.assertEqual(layer.attn_hc.collapse_mode, mode)
                        self.assertEqual(layer.ffn_hc.collapse_mode, mode)
                    self.assertIs(
                        model.v4.head_hc._last_ffn_ref(), model.v4.layers[-1].ffn_hc
                    )
                for step in range(3):
                    outputs = []
                    for model in models:
                        request = (
                            inputs(model.kv_cache)
                            if step == 0
                            else decode_inputs(
                                model.kv_cache,
                                [8, 9] if step == 1 else [10, 11],
                                [3, 2] if step == 1 else [4, 3],
                                (
                                    [[5, 4, 3], [7, 6, -1]]
                                    if step == 1
                                    else [[8, 5, 4], [9, 7, 6]]
                                ),
                                order=[0, 1] if step == 1 else [1, 0],
                                prompt_lengths=[3, 2],
                            )
                        )
                        if model is reference:
                            with mhc_collapse_reference(True):
                                output = model.forward(request)
                        else:
                            with patch.object(
                                delayed,
                                "collapse_delayed",
                                side_effect=AssertionError(
                                    "Selected worker used default collapse"
                                ),
                            ):
                                output = model.forward(request)
                        outputs.append(output.hidden_states.clone())
                    exact(*outputs)
                    for key, value in selected.kv_cache.storage.items():
                        exact(value, reference.kv_cache.storage[key])
                before = {
                    key: value.clone()
                    for key, value in selected.kv_cache.storage.items()
                }
                with patch.dict(selected._v4_args.v41_config, mhc_collapse_mode=None):
                    with self.assertRaisesRegex(ValueError, "collapse mode changed"):
                        selected.forward(inputs(selected.kv_cache))
                for unit in (
                    selected.v4.layers[0].attn_hc,
                    selected.v4.layers[-1].ffn_hc,
                ):
                    with patch.object(unit, "_collapse_mode", None):
                        with self.assertRaisesRegex(
                            ValueError, "collapse modes disagree"
                        ):
                            selected.forward(inputs(selected.kv_cache))
                for key, value in before.items():
                    exact(value, selected.kv_cache.storage[key])
                selected._check_execution_config()
                selected.forward(inputs(selected.kv_cache))
            finally:
                for model in reversed(models):
                    model.close()
        print(
            "DSV41_COLLAPSE_OWNER "
            + json.dumps(
                dict(
                    layers=6,
                    prefill=1,
                    decode=2,
                    hidden_exact=3,
                    cache_exact=3,
                    mutation_rejections=3,
                    recovered=True,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
