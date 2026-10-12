"""Checkpoint-selected mHC arithmetic and mutation-safe native worker admission."""

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


def exact(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


class CollapseOwnerSM120Test(unittest.TestCase):
    component = "collapse"
    extra_choices = {}
    reference_context = staticmethod(mhc_collapse_reference)

    @property
    def policy(self):
        return {f"mhc_{self.component}_mode": "ordered_fp32"}

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
        selected = V41L20Contract(*identity, **self.policy)
        selected.require_attention_config(self.policy)
        self.assertNotEqual(selected.fingerprint, default.fingerprint)
        with patch.object(
            native_l20_transfer,
            "native_l20_cache_views",
            side_effect=AssertionError("Policy mismatch reached cache views"),
        ):
            for config, contract in ((self.policy, default), ({}, selected)):
                with self.assertRaisesRegex(ValueError, f"mHC {self.component} mode"):
                    native_l20_transfer.export_native_l20_delta(
                        config, None, contract, 0, 0, (), {}
                    )
                with self.assertRaisesRegex(ValueError, f"mHC {self.component} mode"):
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
            config.update(self.extra_choices)
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
                    cfg = dict(config, **self.policy) if selected else config
                    model = DeepSeekV41Model(
                        model_config(args, cfg), parallelism, clone, None, 2
                    )
                    models.append(model)
                    self.assertTrue(model.initialize(resources()))
                    self.assertTrue(model.initialize(resources(tagged_cache(model))))
                selected, reference = models
                self.assertEqual(
                    selected._agreement_config()["v41_config"][
                        f"mhc_{self.component}_mode"
                    ],
                    "ordered_fp32",
                )
                for model, mode in ((selected, "ordered_fp32"), (reference, None)):
                    self.assertEqual(
                        getattr(model, f"_mhc_{self.component}_mode"), mode
                    )
                    for layer in model.v4.layers:
                        self.assertEqual(
                            getattr(layer.attn_hc, f"{self.component}_mode"), mode
                        )
                        self.assertEqual(
                            getattr(layer.ffn_hc, f"{self.component}_mode"), mode
                        )
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
                            with self.reference_context(True):
                                output = model.forward(request)
                        else:
                            with patch.object(
                                delayed,
                                f"{self.component}_delayed",
                                side_effect=AssertionError(
                                    f"Selected worker used default {self.component}"
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
                with patch.dict(
                    selected._v4_args.v41_config, {f"mhc_{self.component}_mode": None}
                ):
                    with self.assertRaisesRegex(
                        ValueError, f"{self.component} mode changed"
                    ):
                        selected.forward(inputs(selected.kv_cache))
                for unit in (
                    selected.v4.layers[0].attn_hc,
                    selected.v4.layers[-1].ffn_hc,
                ):
                    with patch.object(unit, f"_{self.component}_mode", None):
                        with self.assertRaisesRegex(
                            ValueError, f"{self.component} modes disagree"
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
            f"DSV41_{self.component.upper()}_OWNER "
            + json.dumps(
                dict(
                    layers=6,
                    prefill=1,
                    decode=2,
                    hidden_exact=3,
                    cache_exact=3,
                    mutation_rejections=3,
                    recovered=True,
                    **(
                        {"collapse_selected": bool(self.extra_choices)}
                        if self.component == "post"
                        else {}
                    ),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
