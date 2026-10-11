"""Qualify short sparse masks on held NAS operands and native cache owners."""

import hashlib
import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_attention_sm120 as decode
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_attention_sm120 as prefill
from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
    quantize_and_insert_k_cache_fp4,
)
from rtp_llm.models_py.modules.dsv4.fp8._v41_swa_triton import (
    quantize_and_insert_swa_k_cache,
)
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    NativeTaggedCache,
    make_layer,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_decode_attention_sm120 import pool
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_decode_tilelang_reference import (
    _ReferenceDecodeAttention,
    workspace_decode,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_short_sparse_plan_reference import (
    decode_plans,
    prefill_attention,
    prefill_plan,
    short_sparse_plan_reference,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_prefill_reference import (
    sparse_prefill_reference,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache
from rtp_llm.tools import dsv41_deep_native_history_replay as native_history
from rtp_llm.tools.dsv41_deep_native_history_replay import execute_native


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ShortSparsePlanSM120Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("Short sparse plan qualification requires leased SM120")
        torch.set_num_threads(1)
        cls.artifacts = Path(os.environ["DSV41_TEST_SPARSE_ARTIFACT_ROOT"])
        name = "attention-boundary-probe-3"
        qualification = cls.artifacts / (name + ".qualification.json")
        if (
            digest(qualification)
            != "e0fa34033538e82a24e0d8579919f5d60186a7c39d341ad4bafce5c3f640c014"
        ):
            raise ValueError("Held sparse qualification changed")
        qualified = json.loads(qualification.read_text())
        if not qualified["accepted"] or qualified["changes"]:
            raise ValueError("Held sparse controls are unqualified")
        report = json.loads((cls.artifacts / (name + ".json")).read_text())
        tensors = cls.artifacts / (name + ".pt")
        if digest(tensors) != report["tensor_artifact_sha256"]:
            raise ValueError("Held sparse tensors changed")
        raw = torch.load(tensors, map_location="cpu", weights_only=True)
        cls.cases = raw["sparse"]
        if [(c["prompt"], c["step"], c["layer"]) for c in cls.cases] != [
            ("france", 0, 14),
            ("france", 2, 8),
            ("arithmetic", 0, 12),
        ]:
            raise ValueError("Held sparse coverage differs")
        kernel_path = Path(os.environ["DSV41_TEST_REFERENCE_KERNEL"])
        if digest(kernel_path) != os.environ["DSV41_TEST_REFERENCE_KERNEL_SHA256"]:
            raise ValueError("Released sparse kernel changed")
        sys.path.insert(0, str(kernel_path.parent))
        cls.kernel = importlib.import_module("kernel")
        if Path(cls.kernel.__file__).resolve() != kernel_path.resolve():
            raise ValueError("Released kernel import differs")

    def exact(self, expected, actual):
        expected, actual = expected.cpu().contiguous(), actual.cpu().contiguous()
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.numel(), actual.numel())
        self.assertTrue(
            torch.equal(
                expected.reshape(-1).view(torch.uint8),
                actual.reshape(-1).view(torch.uint8),
            )
        )

    def inputs(self, case):
        return {name: value.cuda().clone() for name, value in case["held"].items()}

    def released(self, query, kv, sinks, plan):
        return torch.cat(
            [
                self.kernel.sparse_attn(
                    query[:, h : h + 8].unsqueeze(0).contiguous(),
                    kv.reshape(1, -1, 512),
                    sinks[h : h + 8].contiguous(),
                    plan.reshape(1, len(query), -1),
                    512**-0.5,
                )[0]
                for h in range(0, query.shape[1], 8)
            ],
            1,
        ).contiguous()

    def decode_fixture(self, case, gen):
        values = self.inputs(case)
        count = len(values["kv"]) - 128
        swa, packed, backing = pool("swa_kv", gen)
        glob, gp, gb = pool("csa_kv", gen)
        base = 2 * swa.layout.entries
        gbase = 2 * glob.layout.entries
        positions = torch.arange(128, device="cuda", dtype=torch.int32)
        gs = torch.arange(count, device="cuda", dtype=torch.int32) + gbase
        quantize_and_insert_swa_k_cache(
            values["kv"][count:, 0].contiguous(), packed, positions + base
        )
        quantize_and_insert_k_cache_fp4(values["kv"][:count, 0].contiguous(), gp, gs)
        self.exact(values["kv"][count:, 0], swa.read(positions + base))
        self.exact(values["kv"][:count, 0], glob.read(gs))
        idx = values["indices"].reshape(1, -1)
        sw = torch.where(idx[:, :128] >= 0, idx[:, :128] - count + base, -1).int()
        gl = torch.where(idx[:, 128:] >= 0, idx[:, 128:] + gbase, -1).int()
        return (
            values,
            {"swa": swa, "global": glob},
            {"swa": sw, "global": gl},
            [backing, gb],
        )

    @torch.inference_mode()
    def test_held_prefill_masks_controls_and_restoration(self):
        checked = 0
        for case in self.cases:
            if case["step"]:
                continue
            values = self.inputs(case)
            before = {name: value.clone() for name, value in values.items()}
            count = len(values["kv"]) - len(values["q"])
            plan = prefill_plan(
                values["indices"], values["lengths"], len(values["kv"]), count
            )
            self.exact(case["reference_plan"]["indices"], plan)
            with sparse_prefill_reference(True, global_rows=count):
                self.exact(
                    case["outputs"]["native_history"],
                    prefill.sparse_prefill_attention(*values.values()),
                )
                self.exact(
                    case["outputs"]["released_history"],
                    prefill_attention(*values.values(), global_rows=count),
                )
                self.exact(
                    case["outputs"]["native_history"],
                    prefill.sparse_prefill_attention(*values.values()),
                )
            for name, value in values.items():
                self.exact(before[name], value)
            checked += 1
        self.assertEqual(checked, 2)
        print(
            'DSV41_SHORT_PLAN_HELD_PREFILL {"cases": 2, "plans": 2, "restorations": 2}',
            flush=True,
        )

    @torch.inference_mode()
    def test_held_decode_both_packed_generations(self):
        case = self.cases[1]
        for gen in (0, 1):
            values, caches, plans, backing = self.decode_fixture(case, gen)
            before = [
                value.clone()
                for value in list(values.values()) + list(plans.values()) + backing
            ]
            starts = torch.tensor(
                [case["identity"]["start_pos"]], device="cuda", dtype=torch.int32
            )
            baseline, _, _ = workspace_decode(
                values["q"], values["sink"], caches, plans, initial_max=-1e30
            )
            self.exact(case["outputs"]["native_history"], baseline)
            ordered = decode_plans(
                plans, starts, swa_entries=caches["swa"].layout.entries
            )
            result, _, _ = workspace_decode(
                values["q"], values["sink"], caches, ordered, initial_max=-1e30
            )
            self.exact(case["outputs"]["released_history"], result)
            self.exact(plans["global"], ordered["global"])
            self.exact(plans["swa"].sort().values, ordered["swa"].sort().values)
            restored, _, _ = workspace_decode(
                values["q"], values["sink"], caches, plans, initial_max=-1e30
            )
            self.exact(baseline, restored)
            for old, current in zip(
                before, list(values.values()) + list(plans.values()) + backing
            ):
                self.exact(old, current)
        print(
            'DSV41_SHORT_PLAN_HELD_DECODE {"generations": 2, "restorations": 2}',
            flush=True,
        )

    @torch.inference_mode()
    def test_explicit_prefill_owners_and_exception_restore(self):
        case = self.cases[0]
        values = self.inputs(case)
        owner, other = make_layer(2, 2), make_layer(2, 2)
        qkv = SimpleNamespace(qr=values["q"][:, 0])
        common = SimpleNamespace(
            cp_on=False, batch_size=1, prefix_lengths=torch.zeros(1, dtype=torch.int32)
        )
        kwargs = dict(
            kv=values["kv"],
            indices=values["indices"],
            topk_length=values["lengths"],
            profile_name="scope",
        )

        def original(layer, qkv, common, **kwargs):
            return prefill.sparse_prefill_attention(*values.values())

        with sparse_prefill_reference(True, global_rows=2), patch.object(
            AttentionV41FP8, "_prefill_sparse_attention", original
        ):
            baseline = original(owner, qkv, common, **kwargs)
            with short_sparse_plan_reference(True, attentions=[owner]):
                self.exact(
                    case["outputs"]["released_history"],
                    owner._prefill_sparse_attention(qkv, common, **kwargs),
                )
                self.exact(
                    baseline, other._prefill_sparse_attention(qkv, common, **kwargs)
                )
                self.exact(baseline, prefill.sparse_prefill_attention(*values.values()))
                with short_sparse_plan_reference(False), short_sparse_plan_reference(
                    True, attentions=[owner]
                ):
                    self.exact(
                        case["outputs"]["released_history"],
                        owner._prefill_sparse_attention(qkv, common, **kwargs),
                    )
                with self.assertRaisesRegex(RuntimeError, "scope sentinel"):
                    with short_sparse_plan_reference(True, attentions=[owner]):
                        raise RuntimeError("scope sentinel")
                self.exact(baseline, prefill.sparse_prefill_attention(*values.values()))
            self.assertIs(AttentionV41FP8._prefill_sparse_attention, original)
            self.exact(baseline, original(owner, qkv, common, **kwargs))

    @torch.inference_mode()
    def test_explicit_decode_owners_and_exception_restore(self):
        case = self.cases[1]
        values, caches, plans, _ = self.decode_fixture(case, 1)
        owner, other = make_layer(2, 2), make_layer(2, 2)
        starts = torch.tensor([6], device="cuda", dtype=torch.int32)
        metadata = SimpleNamespace(start_pos=starts)
        x = torch.empty(1, 1, 512, device="cuda", dtype=torch.bfloat16)
        valid = torch.ones(1, device="cuda", dtype=torch.bool)

        def original(layer, x, metadata, kv_cache=None):
            return decode._attention(
                values["q"],
                values["sink"],
                caches["swa"],
                plans["swa"],
                valid_rows=valid,
                global_cache=caches["global"],
                global_slots=plans["global"],
            )

        native = decode._attention
        with patch.object(AttentionV41FP8, "forward_decode", original), patch.object(
            decode, "_attention", _ReferenceDecodeAttention()
        ):
            baseline = original(owner, x, metadata)
            with short_sparse_plan_reference(True, attentions=[owner]):
                self.exact(
                    case["outputs"]["released_history"],
                    owner.forward_decode(x, metadata),
                )
                self.exact(baseline, other.forward_decode(x, metadata))
                with self.assertRaisesRegex(ValueError, "first ring"):
                    owner.forward_decode(x, SimpleNamespace(start_pos=starts + 128))
                self.exact(baseline, other.forward_decode(x, metadata))
            self.assertIs(AttentionV41FP8.forward_decode, original)
            self.exact(baseline, original(owner, x, metadata))
        self.assertIs(decode._attention, native)

    def test_scope_admission_and_disabled_context(self):
        original = AttentionV41FP8.forward_decode
        with short_sparse_plan_reference(False, attentions=[object()]):
            self.assertIs(original, AttentionV41FP8.forward_decode)
        for enabled, owners in ((1, ()), (True, ()), (True, (object(),))):
            with self.assertRaises(ValueError), short_sparse_plan_reference(
                enabled, attentions=owners
            ):
                pass
        owner = make_layer()
        with self.assertRaisesRegex(
            ValueError, "distinct"
        ), short_sparse_plan_reference(True, attentions=[owner, owner]):
            pass
        with patch.object(owner, "window_size", 256):
            with self.assertRaisesRegex(
                ValueError, "admitted"
            ), short_sparse_plan_reference(True, attentions=[owner]):
                pass

    @torch.inference_mode()
    def test_native_owner_cache_integration(self):
        checked = 0
        for gen in (0, 1):
            for ratio in (0, 1, 2):
                torch.manual_seed(41661)
                owner = make_layer(2 if ratio else 0, ratio, gen=gen)
                cache = NativeTaggedCache([owner], gen=gen)
                tables = {
                    tag: torch.tensor([[2]], device="cuda", dtype=torch.int32)
                    for tag in cache.layouts
                }

                def read_cache(layer, cache, tables, tag, count):
                    view = V41PagedCache(
                        cache.get_layer_cache(layer.cache_layer_id, tag).kv_cache_base,
                        tag,
                        gen,
                    )
                    slots = tables[tag][
                        0, 0
                    ].long() * view.layout.entries + torch.arange(count, device="cuda")
                    return view.read(slots.contiguous())

                with patch.object(
                    native_history, "cache_values", read_cache
                ), short_sparse_plan_reference(True, attentions=[owner]):
                    for start in (0, 3, 4):
                        rows = 3 if start == 0 else 1
                        x = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
                        _, captured = execute_native(owner, cache, tables, x, start, {})
                        count = (start + rows) // ratio if ratio else 0
                        if start:
                            ring = torch.cat(
                                (
                                    torch.arange(start + 1, 128, device="cuda"),
                                    torch.arange(start + 1, device="cuda"),
                                )
                            )
                            sw = torch.where(ring <= start, ring + count, -1).reshape(
                                1, -1
                            )
                        else:
                            p = torch.arange(rows, device="cuda")
                            sw = torch.where(p[None] <= p[:, None], p[None] + count, -1)
                        g = torch.arange(count, device="cuda")[None].expand(rows, -1)
                        if count:
                            visible = (
                                start + torch.arange(rows, device="cuda") + 1
                            ) // ratio
                            g = torch.where(g < visible[:, None], g, -1)
                        plan = torch.cat((sw, g), -1).int().unsqueeze(1).contiguous()
                        expected = self.released(
                            captured["q"].reshape(rows, 4, 512).contiguous(),
                            captured["kv"],
                            owner.attn_sink,
                            plan,
                        )
                        self.exact(expected, captured["attention"])
                        checked += 1
        self.assertEqual(checked, 18)
        print(
            'DSV41_SHORT_PLAN_NATIVE_OWNER {"cases": 18, "generations": 2, "ratios": 3}',
            flush=True,
        )

    @torch.inference_mode()
    def test_changing_prefill_component_graph(self):
        case = self.cases[0]
        values = self.inputs(case)
        originals = {name: value.clone() for name, value in values.items()}
        plan = case["reference_plan"]["indices"].cuda()
        prefill_attention(*values.values(), global_rows=2)
        prefill_attention(*values.values(), global_rows=2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = prefill_attention(*values.values(), global_rows=2)
        for step, factor in enumerate((1.0, 0.5, 1.5)):
            values["q"].copy_(originals["q"] * factor)
            values["indices"].copy_(originals["indices"])
            expected_plan = plan.clone()
            if step == 1:
                removed = values["indices"][2, 0, 0].clone()
                values["indices"][2, 0, 0] = -1
                expected_plan[2] = torch.where(
                    expected_plan[2] == removed, -1, expected_plan[2]
                )
            graph.replay()
            self.exact(
                self.released(values["q"], values["kv"], values["sink"], expected_plan),
                result,
            )
        self.exact(originals["kv"], values["kv"])
        self.exact(originals["sink"], values["sink"])
        print(
            'DSV41_SHORT_PLAN_PREFILL_GRAPH {"graphs": 1, "changed_replays": 3}',
            flush=True,
        )

    @torch.inference_mode()
    def test_changing_decode_component_graphs(self):
        case = self.cases[1]
        for gen in (0, 1):
            values, caches, plans, backing = self.decode_fixture(case, gen)
            before = [value.clone() for value in backing]
            starts = torch.tensor([6], device="cuda", dtype=torch.int32)
            original_q = values["q"].clone()

            def execute():
                ordered = decode_plans(
                    plans, starts, swa_entries=caches["swa"].layout.entries
                )
                return workspace_decode(
                    values["q"], values["sink"], caches, ordered, initial_max=-1e30
                )[0]

            execute()
            execute()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = execute()
            for step, start in enumerate((6, 5, 7)):
                starts.fill_(start)
                values["q"].copy_(original_q * (1 + step * 0.25))
                logical = torch.arange(128, device="cuda")
                base = 2 * caches["swa"].layout.entries
                slots = (
                    torch.where(logical <= start, logical + base, -1)
                    .int()
                    .reshape(1, 128)
                )
                if step == 1:
                    slots[0, 0] = -1
                plans["swa"].copy_(slots)
                ring = torch.cat(
                    (
                        torch.arange(start + 1, 128, device="cuda"),
                        torch.arange(start + 1, device="cuda"),
                    )
                )
                sw = slots[:, ring]
                workspace = torch.cat(
                    (
                        caches["swa"].read(sw.flatten()).reshape(1, 128, 512),
                        caches["global"]
                        .read(plans["global"].flatten())
                        .reshape(1, 512, 512),
                    ),
                    1,
                )
                live = torch.cat((sw, plans["global"]), 1) >= 0
                ref = (
                    torch.where(live, torch.arange(640, device="cuda")[None], -1)
                    .int()
                    .unsqueeze(1)
                )
                graph.replay()
                self.exact(
                    self.released(values["q"], workspace, values["sink"], ref), result
                )
            for old, value in zip(before, backing):
                self.exact(old, value)
        print(
            'DSV41_SHORT_PLAN_DECODE_GRAPHS {"graphs": 2, "changed_replays": 6}',
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
