"""Immutable short sparse selection through native attention and worker paths."""

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
from rtp_llm.models_py.modules.dsv4 import native_l20_transfer
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_short_sparse_sm120 as short
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_short_sparse_api_sm120 as api,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
    make_layer,
    prepare,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_decode_owner_sm120 import (
    decode_metadata,
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
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_short_sparse_plan_reference import (
    short_sparse_plan_reference,
)
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import V41L20Contract
from rtp_llm.ops import ParallelismConfig, RoleType

MODE = {"sparse_attention_mode": "short_preserved"}


@contextmanager
def fp32_fixture_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class ShortSparseOwnerSM120Test(api.ShortSparseAPISM120Test):
    @torch.inference_mode()
    def test_selected_native_owner_uses_released_plans(self):
        checked = 0
        for gen in (0, 1):
            for ratio in (0, 1, 2):
                torch.manual_seed(41661)
                owner = make_layer(2 if ratio else 0, ratio, gen=gen, v41_config=MODE)
                cache = NativeTaggedCache([owner], gen=gen)
                tables = {
                    tag: torch.tensor([[2]], device="cuda", dtype=torch.int32)
                    for tag in cache.layouts
                }
                original_prefill = short.prefill_attention
                provider = owner._short_sparse_decode

                def record_prefill(q, kv, idx, lens, sinks, *, global_rows, out=None):
                    actual = original_prefill(
                        q, kv, idx, lens, sinks, global_rows=global_rows, out=out
                    )
                    rows = len(q)
                    positions = torch.arange(rows, device=q.device)
                    sw = torch.where(
                        positions[None] <= positions[:, None],
                        positions[None] + global_rows,
                        -1,
                    )
                    gl = torch.arange(global_rows, device=q.device)[None].expand(
                        rows, -1
                    )
                    if global_rows:
                        visible = (positions + 1) // ratio
                        gl = torch.where(gl < visible[:, None], gl, -1)
                    plan = torch.cat((sw, gl), -1).int().unsqueeze(1).contiguous()
                    self.exact(self.released(q, kv, sinks, plan), actual)
                    return actual

                def record_decode(q, sinks, swa, slots, **kwargs):
                    actual = provider(q, sinks, swa, slots, **kwargs)
                    ring = torch.cat(
                        (
                            torch.arange(start + 1, 128, device="cuda"),
                            torch.arange(start + 1, device="cuda"),
                        )
                    )
                    ordered = slots[:, ring]
                    workspace = [swa.read(ordered.flatten()).view(1, 128, 512)]
                    live = [ordered >= 0]
                    glob, gp = kwargs.get("global_cache"), kwargs.get("global_slots")
                    if glob is not None:
                        workspace.append(glob.read(gp.flatten()).view(1, 512, 512))
                        live.append(gp >= 0)
                    workspace = torch.cat(workspace, 1)
                    live = torch.cat(live, 1)
                    plan = (
                        torch.where(
                            live, torch.arange(live.shape[1], device="cuda")[None], -1
                        )
                        .int()
                        .unsqueeze(1)
                    )
                    self.exact(self.released(q, workspace, sinks, plan), actual)
                    return actual

                owner._short_sparse_decode = record_decode
                with patch.object(short, "prefill_attention", record_prefill):
                    for start in (0, 3, 4):
                        rows = 3 if start == 0 else 1
                        x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
                        if start == 0:
                            positions, _, _ = prepare(
                                owner, x, cache, [rows], tables=tables
                            )
                            owner(x, positions, cache, tables)
                        else:
                            metadata = decode_metadata(owner, cache, [start], tables)
                            owner.forward_decode(x[None], metadata, cache)
                        checked += 1
        self.assertEqual(checked, 18)
        print(
            'DSV41_SHORT_OWNER_NATIVE {"cases": 18, "generations": 2, "ratios": 3, "released_kernel_oracle": true}',
            flush=True,
        )

    @torch.inference_mode()
    def test_selected_held_prefill_enters_maintained_component(self):
        for case in (self.cases[0], self.cases[2]):
            values = self.inputs(case)
            owner = make_layer(2, 2, v41_config=MODE)
            owner.attn_sink = values["sink"]
            owner.n_heads = values["q"].shape[1]
            rows = len(values["q"])
            common = SimpleNamespace(
                cp_on=False,
                batch_size=1,
                prefix_lengths=torch.zeros(1, dtype=torch.int32),
                freqs_cis=owner.freqs_cis[:rows],
                workspace=None,
            )
            captured = []

            def project(attention, freqs, *, out):
                captured.append(attention.clone())
                out.zero_()

            with patch.object(
                owner, "_project_prefill_q", return_value=values["q"]
            ), patch.object(owner, "_prefill_output_proj_into", side_effect=project):
                owner._prefill_sparse_attention(
                    SimpleNamespace(qr=values["q"][:, 0]),
                    common,
                    kv=values["kv"],
                    indices=values["indices"],
                    topk_length=values["lengths"],
                    profile_name="held-short-owner",
                )
            self.assertEqual(len(captured), 1)
            self.exact(case["outputs"]["released_history"], captured[0])
        print('DSV41_SHORT_OWNER_HELD {"cases": 2, "outputs_exact": 2}', flush=True)

    @torch.inference_mode()
    def test_policy_mutation_geometry_and_contract_admission(self):
        shared = {}
        owner = make_layer(shared=shared, v41_config=MODE)
        cache = NativeTaggedCache([owner])
        before = {key: value.clone() for key, value in cache.storage.items()}
        with self.assertRaisesRegex(ValueError, "modes disagree"):
            make_layer(1, shared=shared)
        self.assertEqual(set(shared["layers"]), {0})
        lens = torch.tensor([3], device="cuda", dtype=torch.int32)
        for target in (owner.v41_config, shared):
            key = (
                "sparse_attention_mode"
                if target is owner.v41_config
                else "_sparse_attention_mode"
            )
            target[key] = None
            with self.assertRaisesRegex(ValueError, "mode changed"):
                owner._require_qualified_prefill(lens, lens * 0, cp_on=False)
            target[key] = "short_preserved"
        for lengths, prefixes, cp_on, decode in (
            (lens, lens * 0, True, False),
            (lens.repeat(2), lens.repeat(2) * 0, False, False),
            (lens * 0 + 129, lens * 0, False, False),
            (lens, lens * 0 + 1, False, False),
            (lens * 0 + 1, lens * 0 + 128, False, True),
        ):
            with self.assertRaisesRegex(NotImplementedError, "short sparse mode"):
                owner._require_qualified_prefill(
                    lengths, prefixes, cp_on=cp_on, is_decode=decode
                )
        sink = owner.attn_sink
        owner.attn_sink = sink.bfloat16()
        with self.assertRaisesRegex(NotImplementedError, "SM120 geometry"):
            owner._require_qualified_prefill(lens, lens * 0, cp_on=False)
        owner.attn_sink = sink
        for key, old in before.items():
            self.exact(old, cache.storage[key])
        identity = ("a" * 64, "b" * 64, "c" * 64, 256)
        default = V41L20Contract(*identity)
        explicit = V41L20Contract(*identity, sparse_attention_mode="short_preserved")
        explicit.require_attention_config(MODE)
        self.assertNotEqual(default.fingerprint, explicit.fingerprint)
        with patch.object(
            native_l20_transfer,
            "native_l20_cache_views",
            side_effect=AssertionError("mismatch reached cache views"),
        ):
            for cfg, contract in ((MODE, default), ({}, explicit)):
                with self.assertRaisesRegex(ValueError, "sparse mode"):
                    native_l20_transfer.export_native_l20_delta(
                        cfg, None, contract, 0, 0, (), {}
                    )
                with self.assertRaisesRegex(ValueError, "sparse mode"):
                    native_l20_transfer.NativeL20ReplicaReceiver(cfg, None, contract, 0)
        print(
            'DSV41_SHORT_OWNER_REJECTIONS {"mixed": 1, "mutation": 2, "geometry": 6, "transfer": 4, "cache_exact": true}',
            flush=True,
        )

    @torch.inference_mode()
    def test_selected_worker_prefill_decode_and_prewrite_rejections(self):
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
                for selected in (True, False):
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
                    cfg = dict(config, **MODE) if selected else config
                    model = DeepSeekV41Model(
                        model_config(args, cfg), parallelism, clone, None, 2
                    )
                    models.append(model)
                    self.assertTrue(model.initialize(resources()))
                    self.assertTrue(model.initialize(resources(tagged_cache(model))))
                selected, reference = models
                self.assertEqual(
                    selected._agreement_config()["v41_config"]["sparse_attention_mode"],
                    "short_preserved",
                )
                for step in range(3):
                    outputs = []
                    for model in models:
                        request = (
                            inputs(model.kv_cache, ids=[3, 4, 5], lengths=[3])
                            if step == 0
                            else decode_inputs(
                                model.kv_cache,
                                [8 if step == 1 else 9],
                                [2 + step],
                                [[5, 4, 3]] if step == 1 else [[8, 5, 4]],
                                prompt_lengths=[3],
                            )
                        )
                        with short_sparse_plan_reference(
                            model is reference,
                            attentions=[layer.attn for layer in model.v4.layers],
                        ):
                            outputs.append(model.forward(request).hidden_states.clone())
                    self.exact(*outputs)
                    for key, value in selected.kv_cache.storage.items():
                        self.exact(value, reference.kv_cache.storage[key])
                before = {
                    key: value.clone()
                    for key, value in selected.kv_cache.storage.items()
                }
                selected._v4_args.v41_config["sparse_attention_mode"] = None
                with self.assertRaisesRegex(ValueError, "mode changed"):
                    selected.forward(inputs(selected.kv_cache, ids=[3], lengths=[1]))
                selected._v4_args.v41_config["sparse_attention_mode"] = (
                    "short_preserved"
                )
                for request in (
                    inputs(selected.kv_cache),
                    inputs(selected.kv_cache, ids=[3], lengths=[1], prefixes=[4]),
                    inputs(selected.kv_cache, ids=[3] * 129, lengths=[129]),
                    decode_inputs(
                        selected.kv_cache, [8], [128], [[5, 4, 3]], prompt_lengths=[3]
                    ),
                ):
                    with self.assertRaisesRegex(NotImplementedError, "short sparse"):
                        selected.forward(request)
                for key, old in before.items():
                    self.exact(old, selected.kv_cache.storage[key])
                self.assertEqual(selected._worker_state, "ready")
                selected._check_execution_config()
            finally:
                for model in reversed(models):
                    model.close()
        print(
            'DSV41_SHORT_OWNER_WORKER {"layers": 6, "prefill": 1, "decode": 2, "hidden_exact": 3, "cache_exact": 3, "mutation_rejected": 1, "geometry_rejected": 4, "ready": true}',
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
