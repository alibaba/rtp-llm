"""One-use local Engram mask admission and the existing PPU gate contract.

Finite qualification is copied from test_engram_cuda.py: exact >= 0.999,
no element with both ULP > 1 and absolute difference > 1e-5. IEEE mask
semantics are checked separately; no MHC SSE or new tolerance is used.
"""

from contextlib import ExitStack, nullcontext
import json
import math
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.ppu import attention_layer, block, engram, layers
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events


ENV = block.LOCAL_PREFILL_ENGRAM_GATE_ENV


def assert_bits(case, actual, expected):
    case.assertEqual(actual.shape, expected.shape)
    case.assertEqual(actual.dtype, expected.dtype)
    # Dtype views cannot represent deferred neg/conj bits. Resolve only in
    # this observation helper; fallback tests separately freeze raw owners
    # and the original deferred-view metadata.
    actual = actual.resolve_conj().resolve_neg().contiguous()
    expected = expected.resolve_conj().resolve_neg().contiguous()
    case.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))


def view_metadata(tensor):
    return (tensor._cdata, tensor.data_ptr(), tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(), tensor.shape, tensor.stride(), tensor.dtype,
            tensor.device, tensor.is_neg(), tensor.is_conj())


def state():
    cache = SharedNativeCache.__new__(SharedNativeCache)
    cache.sharded, cache._is_prefill = False, True
    model = NS(_is_decode_role=False, _is_speculative=False, _prefill_cp_size=1,
               parallelism_config=NS(tp_size=1))
    return NS(prefill=True, idle=False, cache=cache, context=None,
              attn=NS(is_cuda_graph=False), runtime=NS(model=model))


def inputs(rows, width=5120, device="cpu"):
    torch.manual_seed(171 + rows)
    h = (torch.randn(rows, 4, width, device=device) * 0.25).bfloat16()
    q = (torch.randn(4, width, device=device) * 0.5).bfloat16()
    k = (torch.randn(4, width, device=device) * 0.75).bfloat16()
    value = (torch.randn(rows, 5 * width, device=device) * 0.25).bfloat16()
    mask = torch.arange(rows, device=device) % 5 != 1
    return h, q, k, value, mask


def module_fixture(rows=3, width=32, device="cpu"):
    module = layers.PPUSharedEngram.__new__(layers.PPUSharedEngram)
    nn.Module.__init__(module)
    h, module.q_weight, module.k_weight, value, mask = inputs(rows, width, device)
    module.layer_id, module._prefetched = 1, None
    embedded = torch.randn(rows, 1, 64, device=device).bfloat16()
    module.ops = NS(engram_lookup=Mock(return_value=embedded))
    module.wkv = NS(weight=(torch.randn(5 * width, 64, device=device) / 64).bfloat16())
    hashes = torch.arange(rows * 24, device=device, dtype=torch.int64).reshape(rows, 24)
    return module, h, hashes, value, mask


def cpu_oracle(h, q, k, value, mask):
    """Independent chunked FP32 equations, with one final BF16 landing."""
    h, q, k, value, mask = [x.detach().cpu() for x in (h, q, k, value, mask)]
    rows, copies, width = h.shape
    result = torch.empty_like(h)
    for begin in range(0, rows, 64):
        end = min(begin + 64, rows)
        hidden = h[begin:end].float()
        key = value[begin:end, :copies * width].reshape(-1, copies, width).float()
        shared = value[begin:end, copies * width:].reshape(-1, 1, width).float()
        dot = (((hidden * q.float()) * k.float()) * key).sum(-1)
        dot = dot * torch.rsqrt(hidden.square().mean(-1) + 1e-20)
        dot = dot * (torch.rsqrt(key.square().mean(-1) + 1e-20) / math.sqrt(width))
        root = dot.abs().clamp_min(1e-6).sqrt()
        gate = torch.sigmoid(torch.where(dot < 0, -root, root))
        gate = torch.where(mask[begin:end, None], gate, 0.0)
        result[begin:end] = (hidden + gate[..., None] * shared).bfloat16()
    return result


def ieee_gate_diagnostics(h, q, k, value, mask, actual, legacy, independent):
    """Bounded observations only: no oracle, provider or assertion changes."""
    names = ("finite_nonzero", "nan", "+inf", "-inf", "+zero", "-zero")

    def classify(x):
        code = torch.zeros_like(x, dtype=torch.int8)
        code[torch.isnan(x)] = 1
        code[torch.isposinf(x)] = 2
        code[torch.isneginf(x)] = 3
        code[(x == 0) & ~torch.signbit(x)] = 4
        code[(x == 0) & torch.signbit(x)] = 5
        return code

    def scalar(x):
        x = x.detach().cpu().reshape(())
        number = float(x)
        text = ("nan" if math.isnan(number) else "+inf" if number == math.inf
                else "-inf" if number == -math.inf else "-zero" if number == 0 and math.copysign(1, number) < 0
                else "+zero" if number == 0 else repr(number))
        bits = int(x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32))
        return {"value": text, "bits": hex(bits & (0xffff if x.dtype == torch.bfloat16 else 0xffffffff))}

    def stages(x, qw, kw, kv, valid):
        rows, copies, width = x.shape
        hidden = x.float()
        key = kv[:, :copies * width].reshape(rows, copies, width).float()
        shared = kv[:, copies * width:].reshape(rows, 1, width).float()
        raw = (((hidden * qw.float()) * kw.float()) * key).sum(-1)
        rstd_h = torch.rsqrt(hidden.square().mean(-1) + 1e-20)
        rstd_key = torch.rsqrt(key.square().mean(-1) + 1e-20)
        after_h = raw * rstd_h
        key_factor = rstd_key / math.sqrt(width)
        dot = after_h * key_factor
        root = dot.abs().clamp_min(1e-6).sqrt()
        signed = torch.where(dot < 0, -root, root)
        gate = torch.sigmoid(signed)
        masked = torch.where(valid[:, None], gate, 0.0)
        values = {"dot_raw": raw, "rstd_h": rstd_h, "rstd_key": rstd_key,
                  "dot_after_h": after_h, "key_factor": key_factor, "dot_normalized": dot,
                  "root": root, "signed_root": signed, "gate_before_mask": gate,
                  "gate_masked": masked}
        return {name: tensor.detach().cpu() for name, tensor in values.items()}

    host_inputs = [x.detach().cpu() for x in (h, q, k, value, mask)]
    outputs = {"native_has_mask": actual.detach().cpu(),
               "forced_old_torch_masked": legacy.detach().cpu(),
               "independent_cpu_fp32": independent.detach().cpu()}
    codes = {name: classify(x) for name, x in outputs.items()}
    report = {"scope": "diagnostic-only; original assertions follow unchanged",
              "shape": list(h.shape), "max_cells": 12, "max_features_per_cell": 3,
              "mask": host_inputs[-1].tolist(), "classes": {}, "pairs": {}}
    for name, code in codes.items():
        report["classes"][name] = {label: int((code == i).sum()) for i, label in enumerate(names)}
    cells = set()
    for left, right in (("native_has_mask", "forced_old_torch_masked"),
                        ("native_has_mask", "independent_cpu_fp32"),
                        ("forced_old_torch_masked", "independent_cpu_fp32")):
        difference = codes[left] != codes[right]
        rows_lanes = torch.nonzero(difference.any(-1), as_tuple=False).tolist()
        entry = {"classification_mismatches": int(difference.sum()),
                 "affected_row_copies": len(rows_lanes), "cells": []}
        for row, lane in rows_lanes[:12]:
            features = torch.nonzero(difference[row, lane], as_tuple=False).flatten().tolist()
            entry["cells"].append({"row": row, "copy": lane, "count": len(features),
                                   "first_features": features[:4]})
            cells.add((row, lane))
        for label, predicate in (("nan", torch.isnan), ("positive_inf", torch.isposinf),
                                 ("negative_inf", torch.isneginf), ("finite", torch.isfinite)):
            entry[label + "_mismatches"] = int((predicate(outputs[left]) != predicate(outputs[right])).sum())
        report["pairs"][left + " vs " + right] = entry
    hc, hq, hk, hv, hm = host_inputs
    report["nonfinite_weight_coordinates"] = {
        "q": torch.nonzero(~torch.isfinite(hq), as_tuple=False).tolist(),
        "k": torch.nonzero(~torch.isfinite(hk), as_tuple=False).tolist(),
    }
    # These are re-evaluated Torch equations on each device, NOT observed
    # intermediate registers of the existing native Triton kernel.
    host_stages = stages(*host_inputs)
    device_torch_stages = stages(h, q, k, value, mask)
    report["samples"] = []
    for row, lane in sorted(cells)[:12]:
        entry = {"row": row, "copy": lane, "mask": bool(hm[row]),
                 "cpu_fp32_stages": {}, "ppu_torch_fp32_stages": {}, "features": []}
        for label, values in (("cpu_fp32_stages", host_stages),
                              ("ppu_torch_fp32_stages", device_torch_stages)):
            entry[label] = {name: scalar(tensor[row, lane]) for name, tensor in values.items()}
        for feature in (0, 1, h.shape[-1] - 1):
            scalars = {"h": hc[row, lane, feature], "q": hq[lane, feature],
                       "k": hk[lane, feature], "key": hv[row, lane * h.shape[-1] + feature],
                       "shared": hv[row, h.shape[1] * h.shape[-1] + feature]}
            scalars.update({name: tensor[row, lane, feature] for name, tensor in outputs.items()})
            entry["features"].append({"feature": feature,
                "scalars": {name: scalar(tensor) for name, tensor in scalars.items()},
                "classes": {name: names[int(code[row, lane, feature])] for name, code in codes.items()}})
        report["samples"].append(entry)
    return report


def assert_frozen_gate(case, actual, expected):
    # Same criterion and ULP ordering as the existing generic CUDA gate test.
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    case.assertEqual(actual.shape, expected.shape)
    case.assertEqual(actual.dtype, torch.bfloat16)
    case.assertTrue(bool(torch.isfinite(actual).all()))
    case.assertTrue(bool(torch.isfinite(expected).all()))
    ai = actual.view(torch.int16).to(torch.int32)
    bi = expected.view(torch.int16).to(torch.int32)
    ulp = (torch.where(ai >= 0, ai, -32768 - ai)
           - torch.where(bi >= 0, bi, -32768 - bi)).abs()
    diff = (actual.float() - expected.float()).abs()
    exact = (actual == expected).float().mean().item()
    bad = (ulp > 1) & (diff > 1e-5)
    case.assertGreaterEqual(exact, 0.999, exact)
    case.assertEqual(int(bad.sum()), 0)


class LocalEngramGateCPU(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {ENV: "1"}))
        stack.enter_context(patch.object(block, "is_ppu", return_value=True))
        stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        stack.enter_context(patch.object(attention_layer, "optimized_diagnostics_enabled", return_value=False))

    @torch.no_grad()
    def test_default_and_valid_local_cp1_phase_only(self):
        os.environ.pop(ENV)
        self.assertFalse(block._local_prefill_engram_gate_enabled(state()))
        os.environ[ENV] = "1"
        self.assertTrue(block._local_prefill_engram_gate_enabled(state()))
        current = state()
        current.context = NS(cp_size=1, kv_cache_sharded=False)
        self.assertTrue(block._local_prefill_engram_gate_enabled(current))

    @torch.no_grad()
    def test_phase_and_topology_rejections_preserve_original_paths(self):
        for name in ("idle", "is_graph", "is_graph_commit", "verify", "proposal"):
            current = state()
            setattr(current, name, True)
            self.assertFalse(block._local_prefill_engram_gate_enabled(current), name)
        for name in ("decode", "role", "spec", "cp8", "tp8", "sharded", "cache", "native", "attn_graph"):
            current = state()
            if name == "decode": current.prefill = False
            elif name == "role": current.runtime.model._is_decode_role = True
            elif name == "spec": current.runtime.model._is_speculative = True
            elif name == "cp8": current.runtime.model._prefill_cp_size = 8
            elif name == "tp8": current.runtime.model.parallelism_config.tp_size = 8
            elif name == "sharded": current.cache.sharded = True
            elif name == "cache": current.cache = NS(sharded=False, _is_prefill=True)
            elif name == "native": current.cache._is_prefill = False
            else: current.attn.is_cuda_graph = True
            self.assertFalse(block._local_prefill_engram_gate_enabled(current), name)
        for cp in (NS(cp_size=8, kv_cache_sharded=True), NS(cp_size=1, kv_cache_sharded=True)):
            current = state()
            current.context = cp
            self.assertFalse(block._local_prefill_engram_gate_enabled(current))

    def test_grad_capture_device_and_diagnostics_decline_before_prefetch(self):
        with torch.enable_grad():
            self.assertFalse(block._local_prefill_engram_gate_enabled(state()))
        with torch.no_grad():
            for target in ("capture", "device", "diagnostics"):
                manager = (patch("torch.cuda.is_current_stream_capturing", return_value=True) if target == "capture"
                           else patch.object(block, "is_ppu", return_value=False) if target == "device"
                           else patch.object(attention_layer, "optimized_diagnostics_enabled", return_value=True))
                with manager:
                    self.assertFalse(block._local_prefill_engram_gate_enabled(state()), target)

    def test_one_use_and_round_lookup_options_do_not_mix(self):
        module, h, hashes, value, mask = module_fixture()
        schedule = object()
        with patch.object(layers, "wkv_project", return_value=value), patch.object(
            layers, "engram_gate_inject", return_value=h,
        ) as gate:
            module.prefetch(hashes, uniform_rank_rows=True, round_schedule=schedule, local_fused_mask=True)
            self.assertEqual(len(module._prefetched), 3)
            module.ops.engram_lookup.assert_called_once_with(hashes, layer_id=1,
                uniform_rank_rows=True, round_schedule=schedule)
            with self.assertRaisesRegex(RuntimeError, "unconsumed Engram prefetch"):
                module.prefetch(hashes, local_fused_mask=True)
            module(h, hashes, mask)
            self.assertIsNone(module._prefetched)
            self.assertIs(gate.call_args.kwargs["token_mask"], mask)
            self.assertIs(gate.call_args.kwargs["fused_mask"], True)
            module(h, hashes, mask)
            self.assertNotIn("fused_mask", gate.call_args.kwargs)
            self.assertEqual(module.ops.engram_lookup.call_count, 2)
            self.assertEqual(module.ops.engram_lookup.call_args.kwargs, {"layer_id": 1})

    @torch.no_grad()
    def test_actual_block_prefetch_passes_only_independent_gate_choice(self):
        current = state()
        current.runtime.policy, current.plan = NS(mode="compact"), None
        current.engram_round_schedule = object()
        h, ids, positions, cu = torch.empty(3, 4, 32), torch.arange(3), torch.arange(3), torch.tensor([0, 3])
        current.before_block = Mock(return_value=(h, ids, positions, cu))
        current.publication = lambda layer: nullcontext()
        unit = block.SharedPPUBlock.__new__(block.SharedPPUBlock)
        nn.Module.__init__(unit)
        unit.layer_id, unit.attn = 1, NS(runtime=NS(current=current))
        unit.engram, unit.engram_hashes = Mock(), torch.zeros(3, 24, dtype=torch.int64)
        with patch.object(block, "cp8_yoco_fast_prefill_enabled", return_value=False), patch.object(
            block.Block, "forward", return_value=h,
        ):
            for enabled in (False, True):
                os.environ[ENV] = str(int(enabled))
                unit.engram.prefetch.reset_mock()
                self.assertIs(unit(h, ids, positions, cu), h)
                kwargs = unit.engram.prefetch.call_args.kwargs
                self.assertIs(kwargs["round_schedule"], current.engram_round_schedule)
                self.assertIs(kwargs["uniform_rank_rows"], False)
                self.assertEqual(kwargs.get("local_fused_mask", False), enabled)

    def test_clear_missing_hash_and_hash_identity_remove_new_token(self):
        module, h, hashes, value, mask = module_fixture()
        with patch.object(layers, "wkv_project", return_value=value), patch.object(
            layers, "engram_gate_inject", return_value=h,
        ) as gate:
            module.prefetch(hashes, local_fused_mask=True)
            module.clear()
            module(h, hashes, mask)
            self.assertNotIn("fused_mask", gate.call_args.kwargs)
            for wrong in (None, hashes.clone()):
                module.prefetch(hashes, local_fused_mask=True)
                with self.assertRaisesRegex(RuntimeError, "Engram|token rows"):
                    module(h, wrong, mask)
                self.assertIsNone(module._prefetched)
            module.prefetch(hashes, local_fused_mask=True)
            module(h, hashes, mask)
            self.assertIs(gate.call_args.kwargs["fused_mask"], True)

    def test_lookup_projection_and_gate_exceptions_do_not_leave_sticky_choice(self):
        module, h, hashes, value, mask = module_fixture()
        module.ops.engram_lookup.side_effect = RuntimeError("lookup failed")
        with self.assertRaisesRegex(RuntimeError, "lookup failed"):
            module.prefetch(hashes, local_fused_mask=True)
        self.assertIsNone(module._prefetched)
        module.ops.engram_lookup.side_effect = None
        for stage in ("projection", "gate"):
            module.prefetch(hashes, local_fused_mask=True)
            with patch.object(layers, "wkv_project", **(
                {"side_effect": RuntimeError("projection failed")} if stage == "projection" else {"return_value": value}
            )), patch.object(layers, "engram_gate_inject", side_effect=RuntimeError("gate failed")):
                with self.assertRaisesRegex(RuntimeError, stage + " failed"):
                    module(h, hashes, mask)
            self.assertIsNone(module._prefetched)

    def test_false_direct_and_late_flag_paths_keep_original_selection(self):
        module, h, hashes, value, mask = module_fixture()
        with patch.object(layers, "wkv_project", return_value=value), patch.object(
            layers, "engram_gate_inject", return_value=h,
        ) as gate:
            for choice in (False, None, 1, "1"):
                module.prefetch(hashes, local_fused_mask=choice)
                self.assertEqual(len(module._prefetched), 2)
                module(h, hashes, mask)
                self.assertNotIn("fused_mask", gate.call_args.kwargs)
            module.prefetch(hashes, local_fused_mask=False)
            with self.assertRaisesRegex(RuntimeError, "original-row Engram hashes"):
                module(h, None, mask)
            self.assertEqual(len(module._prefetched), 2, "old false-token error lifecycle is unchanged")
            module.clear()
            module.prefetch(hashes, local_fused_mask=False)
            os.environ[ENV] = "1"  # Later enabling cannot change a prior false receipt.
            module(h, hashes, mask)
            self.assertNotIn("fused_mask", gate.call_args.kwargs)
            module.prefetch(hashes, local_fused_mask=True)
            os.environ[ENV] = "0"
            module.clear()
            module(h, hashes, mask)
            self.assertNotIn("fused_mask", gate.call_args.kwargs)

    def test_deferred_and_unsupported_tensor_metadata_decline_without_resolving(self):
        for target in ("hidden_neg", "q_conj", "value_neg", "q_sparse", "q_object"):
            module, h, hashes, value, mask = module_fixture()
            if target == "hidden_neg": h = torch._neg_view(h)
            elif target == "q_conj": module.q_weight = module.q_weight.to(torch.complex64).conj()
            elif target == "value_neg": value = torch._neg_view(value)
            elif target == "q_sparse": module.q_weight = module.q_weight.to_sparse()
            else: module.q_weight = object()
            with patch.object(layers, "wkv_project", return_value=value), patch.object(
                layers, "engram_gate_inject", return_value=h,
            ) as gate:
                module.prefetch(hashes, local_fused_mask=True)
                module(h, hashes, mask)
                self.assertNotIn("fused_mask", gate.call_args.kwargs, target)
                self.assertIsNone(module._prefetched)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalEngramGateDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(is_ppu(), "gate qualification requires an actual ZW PPU")
        self.device = "cuda:0"

    @torch.inference_mode()
    def test_actual_shapes_keep_frozen_metric_against_independent_cpu_and_torch(self):
        for rows in (1, 33, 257, 4096, 4097, 140):
            with self.subTest(rows=rows):
                original = inputs(rows, device=self.device)
                guards, views = [], []
                for tensor in original:
                    backing = torch.full((tensor.shape[0] + 2, *tensor.shape[1:]),
                                         True if tensor.dtype == torch.bool else 17,
                                         dtype=tensor.dtype, device=tensor.device)
                    backing[1:-1].copy_(tensor)
                    guards.append(backing)
                    views.append(backing[1:-1])
                if rows == 33:
                    # The existing provider explicitly supports MASK_STRIDE.
                    mask_backing = torch.ones(2 * rows + 4, dtype=torch.bool, device=self.device)
                    stepped = mask_backing[2:2 * rows + 2:2]
                    stepped.copy_(original[-1])
                    guards[-1], views[-1] = mask_backing, stepped
                h, q, k, value, mask = views
                frozen = [tensor.clone() for tensor in guards]
                independent = cpu_oracle(h, q, k, value, mask)
                legacy = engram.engram_gate_inject(h, q, k, value, token_mask=mask, fused_mask=False)
                actual = engram.engram_gate_inject(h, q, k, value, token_mask=mask, fused_mask=True)
                assert_frozen_gate(self, actual, independent)
                assert_frozen_gate(self, actual, legacy)
                self.assertFalse(any(torch._C._is_alias_of(actual, tensor) for tensor in views))
                for tensor, before in zip(guards, frozen):
                    assert_bits(self, tensor, before)

    @torch.inference_mode()
    def test_actual_zero_dot_maskfalse_nan_inf_and_signedzero_semantics(self):
        h = torch.zeros(16, 4, 5120, device=self.device, dtype=torch.bfloat16)
        q, k = torch.ones(4, 5120, device=self.device, dtype=torch.bfloat16), torch.ones(4, 5120, device=self.device, dtype=torch.bfloat16)
        value = torch.zeros(16, 25600, device=self.device, dtype=torch.bfloat16)
        value[:, 20480:] = 1
        mask = torch.zeros(16, dtype=torch.bool, device=self.device)
        mask[[0, 1, 9, 10, 11]] = True
        h[:2] = -0.5
        value[0, :20480] = -0.0
        h[2:4] = -0.0
        value[3, 20480:] = -1
        h[4:6] = 0.25
        value[4, 20480:] = float("inf")
        value[5, 20480:] = float("nan")
        h[6] = float("nan")
        h[7], h[8] = float("inf"), -float("inf")
        value[9, :20480], value[10, :20480] = float("nan"), float("inf")
        value[11, 20480:] = float("inf")
        q[2, 0], k[3, 0] = float("nan"), float("inf")
        frozen = [x.clone() for x in (h, q, k, value, mask)]
        expected = engram.engram_gate_inject(h, q, k, value, token_mask=mask, fused_mask=False)
        independent = cpu_oracle(h, q, k, value, mask).to(self.device)
        actual = engram.engram_gate_inject(h, q, k, value, token_mask=mask, fused_mask=True)
        diagnostic = ieee_gate_diagnostics(h, q, k, value, mask, actual, expected, independent)
        print("LOCAL_ENGRAM_GATE_IEEE_DIAG=" + json.dumps(diagnostic, sort_keys=True, allow_nan=False), flush=True)
        for name, oracle in (("forced_old_torch_masked", expected), ("independent_cpu_fp32", independent)):
            self.assertTrue(torch.equal(torch.isnan(actual), torch.isnan(oracle)), name + ": isnan")
            self.assertTrue(torch.equal(torch.isposinf(actual), torch.isposinf(oracle)), name + ": isposinf")
            self.assertTrue(torch.equal(torch.isneginf(actual), torch.isneginf(oracle)), name + ": isneginf")
            self.assertTrue(torch.equal(torch.isfinite(actual), torch.isfinite(oracle)), name + ": isfinite")
            zeros = oracle == 0
            assert_bits(self, actual[zeros], oracle[zeros])
        self.assertTrue(bool((actual[:2, 0] > 0).all()), "both zero-dot signs take the positive root")
        self.assertTrue(bool(torch.isnan(actual[4:6]).all()), "maskFalse still computes zero times nonfinite shared")
        assert_bits(self, actual[2:4], expected[2:4])
        for x, before in zip((h, q, k, value, mask), frozen):
            assert_bits(self, x, before)

    @torch.inference_mode()
    def test_actual_unsupported_views_use_legacy_and_keep_input_bytes(self):
        for target in ("hidden_neg", "hidden_stride", "q_neg"):
            module, h, hashes, _, mask = module_fixture(33, 5120, self.device)
            if target == "hidden_neg": h = torch._neg_view(h)
            elif target == "hidden_stride":
                backing = torch.empty(33, 4, 10240, dtype=h.dtype, device=h.device)
                backing[..., ::2].copy_(h)
                h = backing[..., ::2]
            else: module.q_weight = torch._neg_view(module.q_weight)
            value = layers.wkv_project(module.ops.engram_lookup.return_value.flatten(1), module.wkv.weight)
            expected = engram.engram_gate_inject(h, module.q_weight, module.k_weight, value,
                                                token_mask=mask, fused_mask=False)
            sources = (h, module.q_weight, module.k_weight, module.wkv.weight)
            before = [x.clone() for x in sources]
            metadata = [view_metadata(x) for x in sources]
            owners = [x._base if x._base is not None else x for x in sources]
            raw_before = [x.clone() for x in owners]
            module.prefetch(hashes, local_fused_mask=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                actual = module(h, hashes, mask)
                torch.cuda.synchronize()
            events = _trace_events(profile, f"local-engram-gate-fallback-{target}")
            self.assertFalse(any(event.get("cat") == "kernel" and event["name"] == "_engram_gate_kernel" for event in events))
            assert_bits(self, actual, expected)
            self.assertEqual([view_metadata(x) for x in sources], metadata)
            for x, original in zip(sources, before):
                assert_bits(self, x, original)
            for owner, original in zip(owners, raw_before):
                assert_bits(self, owner, original)

    @torch.inference_mode()
    def test_actual_full_profile_has_one_existing_gate_kernel(self):
        h, q, k, value, mask = inputs(4096, device=self.device)
        engram.engram_gate_inject(h, q, k, value, token_mask=mask, fused_mask=True)
        outputs = []
        for fused in (False, True):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                outputs.append(engram.engram_gate_inject(h, q, k, value, token_mask=mask, fused_mask=fused))
                torch.cuda.synchronize()
            events = _trace_events(profile, f"local-engram-gate-full-{int(fused)}")
            names = [event["name"] for event in events if event.get("cat") == "kernel"]
            self.assertEqual(names.count("_engram_gate_kernel"), int(fused), names)
            self.assertEqual(len(names), 1 if fused else 32, names)
        assert_frozen_gate(self, outputs[1], outputs[0])

    @torch.inference_mode()
    def test_actual_prefetch_forward_uses_provider_once_and_does_not_stick(self):
        module, h, hashes, _, mask = module_fixture(33, 5120, self.device)
        value = layers.wkv_project(module.ops.engram_lookup.return_value.flatten(1), module.wkv.weight)
        legacy = engram.engram_gate_inject(h, module.q_weight, module.k_weight, value,
                                          token_mask=mask, fused_mask=False)
        frozen = [x.clone() for x in (h, hashes, mask, module.q_weight, module.k_weight, module.wkv.weight)]
        # wraps spies on the real entry; it does not replace the gate provider.
        with patch.object(layers, "engram_gate_inject", wraps=engram.engram_gate_inject) as call:
            module.prefetch(hashes, local_fused_mask=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                actual = module(h, hashes, mask)
                torch.cuda.synchronize()
            self.assertIs(call.call_args.kwargs["token_mask"], mask)
            self.assertIs(call.call_args.kwargs["fused_mask"], True)
            self.assertIsNone(module._prefetched)
            events = _trace_events(profile, "local-engram-gate-prefetch-forward")
            self.assertEqual(sum(event.get("cat") == "kernel" and event["name"] == "_engram_gate_kernel" for event in events), 1)
            assert_frozen_gate(self, actual, legacy)
            again = module(h, hashes, mask)
            self.assertNotIn("fused_mask", call.call_args.kwargs)
            assert_bits(self, again, legacy)
            self.assertEqual(module.ops.engram_lookup.call_count, 2)
        for x, before in zip((h, hashes, mask, module.q_weight, module.k_weight, module.wkv.weight), frozen):
            assert_bits(self, x, before)


if __name__ == "__main__":
    unittest.main()
