"""Qualify the maintained short sparse API against frozen released operands."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_short_sparse_sm120 as short
from rtp_llm.models_py.modules.dsv4.fp8.test import (
    test_v41_short_sparse_plan_reference_sm120 as reference,
)


class ShortSparseAPISM120Test(reference.ShortSparsePlanSM120Test):
    def execute_decode(self, provider, values, caches, plans, starts, valid, out=None):
        return provider(
            values["q"],
            values["sink"],
            caches["swa"],
            plans["swa"],
            start_positions=starts,
            valid_rows=valid,
            global_cache=caches.get("global"),
            global_slots=plans.get("global"),
            out=out,
        )

    @torch.inference_mode()
    def test_api_held_prefill_and_output_buffers(self):
        for case in (self.cases[0], self.cases[2]):
            values = self.inputs(case)
            before = {name: value.clone() for name, value in values.items()}
            count = len(values["kv"]) - len(values["q"])
            plan = short.prefill_plan(
                values["indices"], values["lengths"], len(values["kv"]), count
            )
            self.exact(case["reference_plan"]["indices"], plan)
            self.exact(
                case["outputs"]["released_history"],
                short.prefill_attention(*values.values(), global_rows=count),
            )
            out = torch.empty_like(values["q"])
            self.assertIs(
                out,
                short.prefill_attention(*values.values(), global_rows=count, out=out),
            )
            self.exact(case["outputs"]["released_history"], out)
            for name, value in values.items():
                self.exact(before[name], value)
        print(
            'DSV41_SHORT_API_HELD_PREFILL {"cases": 2, "plans": 2, "output_buffers": 2, "inputs_exact": true}',
            flush=True,
        )

    @torch.inference_mode()
    def test_api_held_decode_and_output_buffers(self):
        case = self.cases[1]
        for gen in (0, 1):
            values, caches, plans, backing = self.decode_fixture(case, gen)
            starts = torch.tensor(
                [case["identity"]["start_pos"]], device="cuda", dtype=torch.int32
            )
            valid = torch.ones(1, device="cuda", dtype=torch.bool)
            inputs = (
                list(values.values()) + list(plans.values()) + backing + [starts, valid]
            )
            before = [value.clone() for value in inputs]
            ordered = short.decode_plans(
                plans, starts, swa_entries=caches["swa"].layout.entries
            )
            self.exact(plans["global"], ordered["global"])
            self.exact(plans["swa"].sort().values, ordered["swa"].sort().values)
            provider = short.ShortSparseDecodeAttention()
            self.exact(
                case["outputs"]["released_history"],
                self.execute_decode(provider, values, caches, plans, starts, valid),
            )
            out = torch.empty_like(values["q"])
            self.assertIs(
                out,
                self.execute_decode(
                    provider, values, caches, plans, starts, valid, out
                ),
            )
            self.exact(case["outputs"]["released_history"], out)
            for old, current in zip(before, inputs):
                self.exact(old, current)
        print(
            'DSV41_SHORT_API_HELD_DECODE {"generations": 2, "output_buffers": 2, "global_plans_exact": true, "inputs_cache_exact": true}',
            flush=True,
        )

    @torch.inference_mode()
    def test_api_changing_graph_inputs_against_released_kernel(self):
        case = self.cases[0]
        values = self.inputs(case)
        originals = {name: value.clone() for name, value in values.items()}
        plan = case["reference_plan"]["indices"].cuda()
        out = torch.empty_like(values["q"])
        for _ in range(2):
            short.prefill_attention(*values.values(), global_rows=2, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = short.prefill_attention(*values.values(), global_rows=2, out=out)
        self.assertIs(out, result)
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
                out,
            )
        for name in ("kv", "lengths", "sink"):
            self.exact(originals[name], values[name])

        case = self.cases[1]
        for gen in (0, 1):
            values, caches, plans, backing = self.decode_fixture(case, gen)
            before = [value.clone() for value in backing]
            starts = torch.tensor([6], device="cuda", dtype=torch.int32)
            valid = torch.ones(1, device="cuda", dtype=torch.bool)
            original_q, original_global = values["q"].clone(), plans["global"].clone()
            provider = short.ShortSparseDecodeAttention()
            out = torch.empty_like(values["q"])
            for _ in range(2):
                self.execute_decode(provider, values, caches, plans, starts, valid, out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = self.execute_decode(
                    provider, values, caches, plans, starts, valid, out
                )
            self.assertIs(out, result)
            for step, start in enumerate((6, 5, 7)):
                starts.fill_(start)
                values["q"].copy_(original_q * (1 + step * 0.25))
                plans["global"].copy_(original_global)
                logical = torch.arange(128, device="cuda")
                base = 2 * caches["swa"].layout.entries
                slots = (
                    torch.where(logical <= start, logical + base, -1).int().view(1, 128)
                )
                if step == 1:
                    slots[0, 0] = -1
                    plans["global"][0, 0] = -1
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
                expected_plan = (
                    torch.where(live, torch.arange(640, device="cuda")[None], -1)
                    .int()
                    .unsqueeze(1)
                )
                graph.replay()
                self.exact(
                    self.released(
                        values["q"], workspace, values["sink"], expected_plan
                    ),
                    out,
                )
            for old, current in zip(before, backing):
                self.exact(old, current)
        print(
            'DSV41_SHORT_API_GRAPHS {"graphs": 3, "changed_replays": 9, "released_kernel_oracle": true, "cache_exact": true}',
            flush=True,
        )

    @torch.inference_mode()
    def test_api_rejection_capture_and_zero_rows(self):
        values = self.inputs(self.cases[0])
        before = {name: value.clone() for name, value in values.items()}
        with self.assertRaisesRegex(ValueError, "alias"):
            short.prefill_attention(*values.values(), global_rows=2, out=values["q"])
        prefill_variants = (
            dict(indices=values["indices"].long()),
            dict(lengths=values["lengths"].long()),
            dict(q=values["q"][..., :256].contiguous()),
            dict(sink=values["sink"].bfloat16()),
        )
        geometry = 0
        for changed in prefill_variants:
            altered = dict(values, **changed)
            with self.assertRaises(ValueError):
                short.prefill_attention(*altered.values(), global_rows=2)
            geometry += 1
        with self.assertRaises(ValueError):
            short.prefill_attention(*values.values(), global_rows=3)
        geometry += 1
        for name, value in values.items():
            self.exact(before[name], value)

        zero_controls = 0
        for gen in (0, 1):
            values, caches, plans, backing = self.decode_fixture(self.cases[1], gen)
            starts = torch.tensor([6], device="cuda", dtype=torch.int32)
            valid = torch.ones(1, device="cuda", dtype=torch.bool)
            inputs = (
                list(values.values()) + list(plans.values()) + backing + [starts, valid]
            )
            before = [value.clone() for value in inputs]
            provider = short.ShortSparseDecodeAttention()
            raw_alias = caches["swa"].raw[0].view(torch.bfloat16)
            raw_alias = raw_alias[: values["q"].numel()].view_as(values["q"])
            for out in (values["q"], raw_alias):
                with self.assertRaisesRegex(ValueError, "alias"):
                    self.execute_decode(
                        provider, values, caches, plans, starts, valid, out
                    )
            for new_starts, new_valid, new_plans in (
                (starts.long(), valid, plans),
                (starts, valid.int(), plans),
                (starts, valid, dict(plans, swa=plans["swa"][:, :127].contiguous())),
                (
                    starts,
                    valid,
                    dict(plans, **{"global": plans["global"][:, :511].contiguous()}),
                ),
            ):
                with self.assertRaises(ValueError):
                    self.execute_decode(
                        provider, values, caches, new_plans, new_starts, new_valid
                    )
                geometry += 1
            with self.assertRaisesRegex(ValueError, "together"):
                provider(
                    values["q"],
                    values["sink"],
                    caches["swa"],
                    plans["swa"],
                    start_positions=starts,
                    valid_rows=valid,
                    global_cache=caches["global"],
                )
            geometry += 1
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                with self.assertRaisesRegex(RuntimeError, "Warm short sparse"):
                    self.execute_decode(provider, values, caches, plans, starts, valid)
            self.assertFalse(provider._warmed)
            expected = self.execute_decode(
                provider, values, caches, plans, starts, valid
            )
            with patch.object(
                torch.cuda, "is_current_stream_capturing", return_value=True
            ):
                self.exact(
                    expected,
                    self.execute_decode(provider, values, caches, plans, starts, valid),
                )
            self.exact(
                torch.zeros_like(values["q"]),
                self.execute_decode(
                    provider, values, caches, plans, starts, torch.zeros_like(valid)
                ),
            )
            masked = {name: torch.full_like(plan, -1) for name, plan in plans.items()}
            self.exact(
                torch.zeros_like(values["q"]),
                self.execute_decode(provider, values, caches, masked, starts, valid),
            )
            empty_values = dict(values, q=values["q"][:0].contiguous())
            empty_plans = {name: plan[:0].contiguous() for name, plan in plans.items()}
            empty = self.execute_decode(
                provider,
                empty_values,
                caches,
                empty_plans,
                starts[:0].contiguous(),
                valid[:0].contiguous(),
            )
            self.assertEqual(empty.shape, empty_values["q"].shape)
            zero_controls += 3
            for old, current in zip(before, inputs):
                self.exact(old, current)
        self.assertEqual(geometry, 15)
        print(
            'DSV41_SHORT_API_CONTROLS {"alias_rejections": 5, "geometry_rejections": 15, "fresh_capture_rejections": 2, "prepared_reuse": 2, "zero_controls": 6, "inputs_cache_exact": true}',
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
