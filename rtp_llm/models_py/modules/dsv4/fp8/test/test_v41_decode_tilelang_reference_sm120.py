"""Qualify scoped maintained decode on held NAS inputs and native page geometry."""

import hashlib
import importlib
import json
import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4 import tilelang_kernels
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_attention_sm120 as decode
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_attention_sm120 as prefill
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_decode_attention_sm120 import pool
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_decode_tilelang_reference import (
    decode_tilelang_reference,
    workspace_decode,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_sparse_precision import (
    sparse_sink_div_rn,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import V41PagedCache


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def marker(name, **values):
    print("DSV41_TILELANG_SCOPE_" + name + " " + json.dumps(values), flush=True)


class V41DecodeTileLangReferenceSM120Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() != (12, 0):
            raise RuntimeError("TileLang decode scope requires leased SM120 hardware")
        torch.set_num_threads(1)
        cls.cases = []
        for prompt in ("france", "arithmetic"):
            prefix = f"DSV41_TEST_DECODE_{prompt.upper()}"
            raw_path = Path(os.environ[prefix + "_TENSORS"])
            if digest(raw_path) != os.environ[prefix + "_SHA256"]:
                raise ValueError("Held TileLang tensors changed")
            raw = torch.load(raw_path, map_location="cpu", weights_only=True)
            if len(raw["cases"]) != (14 if prompt == "france" else 21):
                raise ValueError("Held TileLang coverage differs")
            cls.cases.extend((prompt, case) for case in raw["cases"])
        cls.kernel_path = Path(os.environ["DSV41_TEST_REFERENCE_KERNEL"])
        if digest(cls.kernel_path) != os.environ["DSV41_TEST_REFERENCE_KERNEL_SHA256"]:
            raise ValueError("Released sparse kernel changed")
        sys.path.insert(0, str(cls.kernel_path.parent))
        cls.kernel = importlib.import_module("kernel")
        if Path(cls.kernel.__file__).resolve() != cls.kernel_path.resolve():
            raise ValueError("Released sparse kernel import differs")

    def setUp(self):
        torch.manual_seed(419910)

    def exact(self, expected, actual):
        expected, actual = expected.cpu().contiguous(), actual.cpu().contiguous()
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertTrue(
            torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        )

    def held(self, case):
        query = case["operands_before"]["q"].cuda().clone()
        sinks = case["operands_before"]["sink"].cuda().clone()
        caches = {
            name: V41PagedCache(
                value.cuda().clone(), "swa_kv" if name == "swa" else "csa_kv", 0
            )
            for name, value in case["pages_before"].items()
        }
        plans = {
            name: value.cuda().clone() for name, value in case["plans_before"].items()
        }
        return query, sinks, caches, plans

    def invoke(self, query, sinks, caches, plans):
        def packed(cache):
            width = (
                cache.layout.payload_bytes_per_entry
                + cache.layout.scale_bytes_per_entry
            )
            return cache.raw.as_strided(
                (cache.raw.shape[0], cache.layout.entries, width),
                (cache.raw.stride(0), width, 1),
            )

        global_cache = caches.get("global")
        return decode.sparse_decode_attention(
            query.unsqueeze(1),
            packed(caches["swa"]),
            plans["swa"].unsqueeze(1),
            sinks,
            None if global_cache is None else packed(global_cache),
            None if global_cache is None else plans["global"].unsqueeze(1),
        ).squeeze(1)

    def typed(self, query, sinks, caches, plans, valid_rows, out=None):
        return decode._attention(
            query,
            sinks,
            caches["swa"],
            plans["swa"],
            valid_rows=valid_rows,
            global_cache=caches.get("global"),
            global_slots=plans.get("global"),
            out=out,
        )

    def released(self, query, sinks, caches, plans, valid_rows):
        # Decode on CPU independently of the GPU workspace adapter.
        parts, slots = [], []
        active = valid_rows.cpu()
        for name in ("swa", "global"):
            if name not in caches:
                continue
            cache = caches[name]
            cpu = V41PagedCache(cache.raw.cpu(), cache.tag, cache.gen_num_per_cycle)
            selected = torch.where(active[:, None], plans[name].cpu(), -1)
            parts.append(
                cpu.read(selected.reshape(-1), backend="eager").view(
                    query.shape[0], -1, 512
                )
            )
            slots.append(selected)
        workspace = torch.cat(parts, dim=1).cuda()
        live = torch.cat(slots, dim=1) >= 0
        indices = (
            torch.where(live, torch.arange(live.shape[1], dtype=torch.int32)[None], -1)
            .unsqueeze(1)
            .cuda()
        )
        q = torch.where(valid_rows[:, None, None], query, 0)
        outputs = [
            self.kernel.sparse_attn(
                q[:, start : start + 8].unsqueeze(1).contiguous(),
                workspace,
                sinks[start : start + 8].contiguous(),
                indices,
                512**-0.5,
            )
            for start in range(0, query.shape[1], 8)
        ]
        return torch.where(
            valid_rows[:, None, None], torch.cat(outputs, dim=2).squeeze(1), 0
        )

    @torch.inference_mode()
    def test_all_held_controls_and_v4_default(self):
        controls = candidate = restored = defaults = unchanged = 0
        for prompt, case in self.cases:
            with self.subTest(prompt=prompt, layer=case["layer"], step=case["step"]):
                query, sinks, caches, plans = self.held(case)
                inputs = [
                    query,
                    sinks,
                    *plans.values(),
                    *(cache.raw for cache in caches.values()),
                ]
                before = [value.clone() for value in inputs]
                with sparse_sink_div_rn(True):
                    self.exact(
                        case["outputs"]["baseline"],
                        self.invoke(query, sinks, caches, plans),
                    )
                    controls += 1
                    with decode_tilelang_reference(True):
                        self.exact(
                            case["outputs"]["released"],
                            self.invoke(query, sinks, caches, plans),
                        )
                        candidate += 1
                    self.exact(
                        case["outputs"]["baseline"],
                        self.invoke(query, sinks, caches, plans),
                    )
                    restored += 1
                output, _, _ = workspace_decode(query, sinks, caches, plans)
                self.exact(case["outputs"]["candidate"], output)
                defaults += 1
                for expected, actual in zip(before, inputs):
                    self.exact(expected, actual)
                    unchanged += 1
        marker(
            "HELD",
            cases=controls,
            candidate=candidate,
            restored=restored,
            v4_defaults=defaults,
            unchanged_inputs=unchanged,
        )

    @torch.inference_mode()
    def test_disabled_nested_sink_orders_and_exception_restore(self):
        native, prefill_original = decode._attention, prefill._sparse_prefill
        checks = 0
        for reference_first in (False, True):
            for _, case in self.cases:
                query, sinks, caches, plans = self.held(case)
                contexts = (decode_tilelang_reference(True), sparse_sink_div_rn(True))
                if not reference_first:
                    contexts = contexts[::-1]
                with ExitStack() as stack:
                    for context in contexts:
                        stack.enter_context(context)
                    selected = decode._attention
                    with decode_tilelang_reference(False):
                        self.assertIs(selected, decode._attention)
                        self.exact(
                            case["outputs"]["released"],
                            self.invoke(query, sinks, caches, plans),
                        )
                    with decode_tilelang_reference(True):
                        self.exact(
                            case["outputs"]["released"],
                            self.invoke(query, sinks, caches, plans),
                        )
                    self.assertIs(selected, decode._attention)
                    with self.assertRaisesRegex(RuntimeError, "scope sentinel"):
                        with decode_tilelang_reference(True):
                            raise RuntimeError("scope sentinel")
                    self.assertIs(selected, decode._attention)
                    checks += 1
                self.assertIs(native, decode._attention)
                self.assertIs(prefill_original, prefill._sparse_prefill)
        with decode_tilelang_reference(False):
            self.assertIs(native, decode._attention)
        marker(
            "NESTING",
            cases=checks,
            sink_orders=2,
            disabled=True,
            exceptions=True,
            restored=True,
        )

    def fixture(self, gen, tag, heads, rows):
        swa, _, backing = pool("swa_kv", gen)
        caches, backing_buffers = {"swa": swa}, [backing]
        plans = {
            "swa": torch.randint(
                swa.layout.entries,
                4 * swa.layout.entries,
                (rows, 128),
                device="cuda",
                dtype=torch.int32,
            )
        }
        if tag is not None:
            global_cache, _, backing = pool(tag, gen)
            caches["global"] = global_cache
            backing_buffers.append(backing)
            plans["global"] = torch.randint(
                global_cache.layout.entries,
                4 * global_cache.layout.entries,
                (rows, 512),
                device="cuda",
                dtype=torch.int32,
            )
        query = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16)
        sinks = torch.linspace(-2, 2, heads, device="cuda", dtype=torch.float32)
        valid = torch.ones(rows, device="cuda", dtype=torch.bool)
        return query, sinks, caches, plans, valid, backing_buffers

    @torch.inference_mode()
    def test_masked_tiles_hca_generations_and_padding(self):
        cases = unchanged = 0
        for gen in (0, 1):
            for tag in (None, "csa_kv", "hca_kv"):
                for heads in (1, 8, 64):
                    with self.subTest(gen=gen, tag=tag, heads=heads):
                        query, sinks, caches, plans, valid, backing = self.fixture(
                            gen, tag, heads, 5
                        )
                        plans["swa"][0, ::3] = -1
                        plans["swa"][1, :64] = -1
                        plans["swa"][2:] = -1
                        if tag is not None:
                            plans["global"][2, :192] = -1
                            plans["global"][3:] = -1
                        query[4] = torch.nan
                        valid[4] = False
                        for slots in plans.values():
                            slots[4] = -99
                        inputs = [query, sinks, valid, *plans.values(), *backing]
                        before = [value.clone() for value in inputs]
                        expected = self.released(query, sinks, caches, plans, valid)
                        with decode_tilelang_reference(True):
                            output = self.typed(query, sinks, caches, plans, valid)
                        self.exact(expected, output)
                        self.assertTrue(torch.isfinite(output).all().item())
                        self.assertEqual(torch.count_nonzero(output[3:]).item(), 0)
                        if tag is None:
                            self.assertEqual(torch.count_nonzero(output[2]).item(), 0)
                        for original, actual in zip(before, inputs):
                            self.exact(original, actual)
                            unchanged += 1
                        cases += 1
        marker(
            "GEOMETRY",
            cases=cases,
            generations=2,
            pools=3,
            head_shapes=3,
            unchanged_inputs=unchanged,
            masked_tiles=True,
            inactive_poison=True,
        )

    @torch.inference_mode()
    def test_changed_input_component_graphs(self):
        graphs = replays = 0
        native = decode._attention
        for gen in (0, 1):
            for tag in (None, "csa_kv", "hca_kv"):
                with self.subTest(gen=gen, tag=tag):
                    query, sinks, caches, plans, valid, _ = self.fixture(gen, tag, 8, 2)
                    with decode_tilelang_reference(True):
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(3):
                                self.typed(query, sinks, caches, plans, valid)
                        torch.cuda.current_stream().wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output = self.typed(query, sinks, caches, plans, valid)
                        for iteration in range(3):
                            query.normal_()
                            sinks.add_(0.25)
                            valid.fill_(True)
                            for name, cache in caches.items():
                                slots = torch.arange(
                                    4 * cache.layout.entries,
                                    device="cuda",
                                    dtype=torch.int64,
                                )
                                values = torch.randn(
                                    slots.numel(),
                                    512,
                                    device="cuda",
                                    dtype=torch.bfloat16,
                                )
                                cache.write(values, slots)
                                plans[name].random_(
                                    cache.layout.entries, 4 * cache.layout.entries
                                )
                                plans[name][0, : 64 * (iteration + 1)] = -1
                                plans[name][1, ::3] = -1
                            if iteration == 2:
                                valid[1] = False
                                query[1] = torch.nan
                                for slots in plans.values():
                                    slots[1] = -99
                            graph.replay()
                            self.exact(
                                self.released(query, sinks, caches, plans, valid),
                                output,
                            )
                            replays += 1
                        graphs += 1
                    self.assertIs(decode._attention, native)
        marker(
            "GRAPHS",
            graphs=graphs,
            replays=replays,
            changed_pages=True,
            changed_plans=True,
            changed_queries=True,
            inactive_poison=True,
            restored=True,
        )

    @torch.inference_mode()
    def test_empty_output_aliasing_and_rejected_contracts(self):
        query, sinks, caches, plans, valid, _ = self.fixture(0, "csa_kv", 8, 2)
        expected = self.released(query, sinks, caches, plans, valid)
        with decode_tilelang_reference(True):
            out = torch.empty_like(query)
            self.assertIs(self.typed(query, sinks, caches, plans, valid, out), out)
            self.exact(expected, out)
            with self.assertRaisesRegex(ValueError, "alias"):
                self.typed(query, sinks, caches, plans, valid, query)
            with self.assertRaisesRegex(ValueError, "together"):
                decode._attention(
                    query,
                    sinks,
                    caches["swa"],
                    plans["swa"],
                    valid_rows=valid,
                    global_cache=caches["global"],
                )
            empty = self.typed(
                query[:0],
                sinks,
                caches,
                {name: slots[:0] for name, slots in plans.items()},
                valid[:0],
            )
            self.assertEqual(empty.shape, query[:0].shape)
        with self.assertRaisesRegex(ValueError, "geometry"):
            workspace_decode(query.float(), sinks, caches, plans)
        with self.assertRaisesRegex(ValueError, "cache/plan"):
            workspace_decode(
                query,
                sinks,
                caches,
                {**plans, "swa": plans["swa"][:, :64].contiguous()},
            )
        with self.assertRaisesRegex(ValueError, "valid_rows"):
            workspace_decode(
                query, sinks, caches, plans, valid_rows=valid.to(torch.int32)
            )
        for initializer in (0.0, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "initializer"):
                tilelang_kernels.sparse_attn(
                    query.unsqueeze(1),
                    query,
                    sinks,
                    plans["swa"].unsqueeze(1),
                    512**-0.5,
                    initial_max=initializer,
                )
        marker(
            "CONTRACT",
            empty_rows=True,
            explicit_output=True,
            alias_rejected=True,
            geometry_rejected=True,
            initializer_rejected=True,
        )


if __name__ == "__main__":
    unittest.main()
