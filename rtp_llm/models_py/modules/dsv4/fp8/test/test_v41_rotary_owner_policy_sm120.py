"""Configured V4.1 rotary origins across owner and worker initialization."""

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.deepseek_v4_model import Dsv4SharedRuntimeBufferStore
from rtp_llm.models_py.model_desc.deepseek_v41_model import DeepSeekV41Model
from rtp_llm.models_py.modules.dsv4 import native_l20_transfer, rope
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_rotary_compute_device_sm120 as rotary_api,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
    make_layer,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_transformer_sm120 import fixture
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_worker_sm120 import (
    checkpoint,
    decode_inputs,
    inputs,
    model_config,
    resources,
    tagged_cache,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_output_precision import (
    output_projection_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_tp8_projection import tp8_projection
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import V41L20Contract
from rtp_llm.ops import ParallelismConfig, RoleType
from rtp_llm.tools.dsv41_attention_replay import NASReader, native_attention
from rtp_llm.tools.dsv41_output_projection_replay import immutable_sources
from rtp_llm.tools.dsv41_query_owner_replay import tensor_digest
from rtp_llm.tools.dsv41_reference_probe import model_args


@contextmanager
def fp32_fixture_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class RotaryOwnerPolicySM120Test(rotary_api.RotaryComputeDeviceSM120Test):
    @torch.inference_mode()
    def test_configured_nas_owners_match_held_outputs(self):
        root = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        config = json.loads((root / "config.json").read_text())["text_config"]
        args = model_args(self.official, root, 256)
        reader = NASReader(root)
        owners = {
            layer: native_attention(
                args, dict(config, rotary_compute_device="cuda"), reader, layer
            )
            for layer in (0, 6)
        }
        with tp8_projection(True), output_projection_reference(
            True, kernel=self.kernel
        ):
            for case in self.cases:
                owner = owners[case["layer"]]
                self.assertEqual(owner._rotary_compute_device, "cuda")
                self.exact(case["released_freqs_full"], owner.freqs_cis)
                start = case["identity"]["start_pos"]
                rows = len(case["identity"]["input_ids"])
                # TP8's lazy diagnostic WO_b shards exist after the first call.
                self.exact(
                    case["outputs"]["released"],
                    owner._project_output(
                        case["attention_before"].cuda().clone(),
                        owner.freqs_cis[start : start + rows],
                    ),
                )
                sources = immutable_sources(owner, self.shards[case["layer"]])
                before = {key: tensor_digest(value) for key, value in sources.items()}
                actual = owner._project_output(
                    case["attention_before"].cuda().clone(),
                    owner.freqs_cis[start : start + rows],
                )
                self.exact(case["outputs"]["released"], actual)
                self.assertEqual(
                    before,
                    {key: tensor_digest(value) for key, value in sources.items()},
                )
        print(
            'DSV41_ROPE_OWNER_HELD {"cases": 3, "owners": 2, "outputs_exact": 3}',
            flush=True,
        )

    @torch.inference_mode()
    def test_meta_initialization_memo_sharing_and_pending_rejection(self):
        default = torch.get_default_device()
        for policy in ("cpu", "cuda"):
            for ratio in (0, 1, 2):
                config = {"rotary_compute_device": policy}
                with torch.device("meta"), patch.object(
                    AttentionV41FP8, "reset_rope_cache", return_value=None
                ):
                    owner = make_layer(2 if ratio else 0, ratio, v41_config=config)
                self.assertFalse(owner._rotary_ready)
                lens = torch.tensor([1], dtype=torch.int32)
                with self.assertRaisesRegex(ValueError, "Reset V4.1 rotary"):
                    owner._require_qualified_prefill(lens, lens * 0, cp_on=False)
                owner.reset_rope_cache("cuda")
                selected = owner.attn_sink.device if policy == "cuda" else "cpu"
                expected = rope.precompute_freqs_cis(
                    *rotary_api.parameters(owner),
                    device=owner.attn_sink.device,
                    compute_device=selected,
                )
                self.assertIs(expected, owner.freqs_cis)
                peer = make_layer(2 if ratio else 0, ratio, v41_config=config)
                self.assertIs(peer.freqs_cis, owner.freqs_cis)
                self.assertTrue(owner._rotary_ready)
                self.assertEqual(default, torch.get_default_device())
        print(
            'DSV41_ROPE_OWNER_META {"ratios": 3, "origins": 2, "pending_rejected": 6, "memo_shared": 6}',
            flush=True,
        )

    @torch.inference_mode()
    def test_mixed_policies_mutation_and_conflicting_reset_reject(self):
        legacy = make_layer()
        legacy._shared_attention = {"layers": {0: legacy}}
        legacy._check_rotary_config()
        shared = {}
        owner = make_layer(shared=shared, v41_config={"rotary_compute_device": "cuda"})
        cache = NativeTaggedCache([owner])
        before = {key: value.clone() for key, value in cache.storage.items()}
        original = owner.freqs_cis
        with self.assertRaisesRegex(ValueError, "origins disagree"):
            make_layer(1, shared=shared, v41_config={"rotary_compute_device": "cpu"})
        self.assertEqual(set(shared["layers"]), {0})
        with self.assertRaisesRegex(ValueError, "conflicts"):
            owner.reset_rope_cache("cuda", compute_device="cpu")
        with self.assertRaisesRegex(ValueError, "materialized CUDA owner"):
            owner.reset_rope_cache("cpu")
        owner.v41_config["rotary_compute_device"] = "cpu"
        lens = torch.tensor([1], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "origin changed"):
            owner._require_qualified_prefill(lens, lens * 0, cp_on=False)
        with self.assertRaisesRegex(ValueError, "origin changed"):
            owner.reset_rope_cache("cuda")
        owner.v41_config["rotary_compute_device"] = "cuda"
        self.assertIs(owner.freqs_cis, original)
        for key, value in before.items():
            self.exact(value, cache.storage[key])
        owner.reset_rope_cache("cuda", compute_device="cuda")
        self.assertIs(owner.freqs_cis, original)
        print(
            'DSV41_ROPE_OWNER_REJECTIONS {"mixed": 1, "override": 1, "destination": 1, "mutation": 2, "cache_unchanged": 1, "legacy_rebind": 1}',
            flush=True,
        )

    def test_native_contract_mismatch_rejects_before_cache_views(self):
        contract = V41L20Contract("a" * 64, "b" * 64, "c" * 64, 256)
        selected = {"rotary_compute_device": "cuda"}
        with patch.object(
            native_l20_transfer,
            "native_l20_cache_views",
            side_effect=AssertionError("mismatch reached cache views"),
        ):
            with self.assertRaisesRegex(ValueError, "rotary origin"):
                native_l20_transfer.export_native_l20_delta(
                    selected, None, contract, 0, 0, (), {}
                )
            with self.assertRaisesRegex(ValueError, "rotary origin"):
                native_l20_transfer.NativeL20ReplicaReceiver(
                    selected, None, contract, 0
                )
        print(
            'DSV41_ROPE_OWNER_TRANSFER {"export_rejected": 1, "replica_rejected": 1}',
            flush=True,
        )

    @torch.inference_mode()
    def test_selected_worker_prefill_decode_and_mutation_before_writes(self):
        Dsv4SharedRuntimeBufferStore._reset_for_test()
        self.addCleanup(Dsv4SharedRuntimeBufferStore._reset_for_test)
        with fp32_fixture_default(), tempfile.TemporaryDirectory() as directory, patch.dict(
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
                for policy in ("cuda", None):
                    clone = SimpleNamespace(
                        global_weights={
                            k: v.clone() for k, v in weights.global_weights.items()
                        },
                        weights=[
                            {k: v.clone() for k, v in layer.items()}
                            for layer in weights.weights
                        ],
                    )
                    parallelism = ParallelismConfig()
                    parallelism.role_type = RoleType.PREFILL
                    cfg = (
                        config
                        if policy is None
                        else dict(config, rotary_compute_device=policy)
                    )
                    model = DeepSeekV41Model(
                        model_config(args, cfg), parallelism, clone, None, 2
                    )
                    models.append(model)
                    self.assertTrue(model.initialize(resources()))
                    if policy is None:
                        for layer in model.v4.layers:
                            layer.attn.reset_rope_cache(
                                layer.attn.attn_sink.device,
                                compute_device=layer.attn.attn_sink.device,
                            )
                    self.assertTrue(model.initialize(resources(tagged_cache(model))))
                selected, reference = models
                self.assertEqual(
                    selected._agreement_config()["v41_config"]["rotary_compute_device"],
                    "cuda",
                )
                for left, right in zip(selected.v4.layers, reference.v4.layers):
                    self.assertIs(left.attn.freqs_cis, right.attn.freqs_cis)
                    self.assertTrue(left.attn._rotary_ready)
                for step in range(3):
                    outputs = []
                    for model in models:
                        if step == 0:
                            request = inputs(model.kv_cache)
                        else:
                            request = decode_inputs(
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
                        outputs.append(model.forward(request).hidden_states.clone())
                    self.exact(*outputs)
                    for key, value in selected.kv_cache.storage.items():
                        self.exact(value, reference.kv_cache.storage[key])
                before = {
                    key: value.clone()
                    for key, value in selected.kv_cache.storage.items()
                }
                selected._v4_args.v41_config["rotary_compute_device"] = "cpu"
                with self.assertRaisesRegex(ValueError, "origin changed"):
                    selected.forward(inputs(selected.kv_cache))
                for key, value in before.items():
                    self.exact(value, selected.kv_cache.storage[key])
                selected._v4_args.v41_config["rotary_compute_device"] = "cuda"
                selected._check_rotary_config()
            finally:
                for model in reversed(models):
                    model.close()
        print(
            'DSV41_ROPE_OWNER_WORKER {"layers": 6, "prefill": 1, "decode": 2, "hidden_exact": 3, "cache_exact": 3, "mutation_rejected": 1}',
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
