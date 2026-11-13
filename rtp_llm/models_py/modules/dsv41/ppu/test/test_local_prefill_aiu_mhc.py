"""Local attention-pre AIU admission over the existing provider projection.

Split-K rounding can differ from the legacy projection. Acceptance imports the
frozen per-row/global SSE-energy metric (<5e-5), never byte equality or a new
tolerance. Device tests compare an independent CPU pre oracle and legacy pre,
and require actual AIU kernel evidence rather than just an admission flag.
"""

import json
import os
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.layers import (
    LOCAL_PREFILL_AIU_PRENORM_ENV,
    PPUSharedHCUnit,
    PPUSharedRMSNorm,
    _local_prefill_aiu_prenorm_enabled,
)
from rtp_llm.models_py.modules.dsv41.ppu.mhc import MHCConfig, PPUStepMix
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_mhc import _compact_exchange
from rtp_llm.models_py.modules.dsv41.ppu.test.test_mhc_mega_probe import (
    CHUNK_ROWS,
    _make_case,
    _qualification_metric,
)
from rtp_llm.utils.model_weight import W


ENV = LOCAL_PREFILL_AIU_PRENORM_ENV
CP_ENV = "RTP_V41_PPU_CP8_YOCO_AIU_PRENORM"


def _local_state():
    return NS(prefill=True, idle=False, is_graph=False, context=None,
              cache=NS(sharded=False),
              runtime=NS(model=NS(_is_speculative=False), policy=NS(mode="compact")))


def _metadata(shape=(4096, 4, 5120), *, dtype=torch.bfloat16, device="cuda:0", dense=True):
    return NS(shape=shape, ndim=len(shape), dtype=dtype, device=torch.device(device),
              is_contiguous=lambda: dense)


class LocalPrefillAIURouting(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {ENV: "1", CP_ENV: "0"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for target, value in (
            ("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", True),
            ("torch.cuda.get_device_capability", (8, 9)),
            ("torch.cuda.is_current_stream_capturing", False),
        ):
            mocked = patch(target, return_value=value)
            mocked.start()
            self.addCleanup(mocked.stop)

    @torch.inference_mode()
    def test_default_off_supported_geometry_and_rollback(self):
        with patch.dict(os.environ):
            os.environ.pop(ENV, None)
            self.assertFalse(_local_prefill_aiu_prenorm_enabled(_local_state(), _metadata()))
        for rows in (1024, 4096, 4097, 16384):
            self.assertTrue(_local_prefill_aiu_prenorm_enabled(
                _local_state(), _metadata((rows, 4, 5120)),
            ))
        state = _local_state()
        state.context = NS(cp_size=1, kv_cache_sharded=False)
        self.assertTrue(_local_prefill_aiu_prenorm_enabled(state, _metadata()))
        with patch.dict(os.environ, {ENV: "0"}):
            self.assertFalse(_local_prefill_aiu_prenorm_enabled(state, _metadata()))
        for residual in (
            _metadata((140, 4, 5120)), _metadata((1023, 4, 5120)),
            _metadata((16385, 4, 5120)), _metadata((0, 4, 5120)),
            _metadata((4096, 4, 4096)), _metadata((4096, 1, 4, 5120)),
            _metadata(dtype=torch.float32), _metadata(device="cpu"), _metadata(dense=False),
        ):
            self.assertFalse(_local_prefill_aiu_prenorm_enabled(_local_state(), residual))

    def test_grad_decode_idle_graph_sharding_speculation_and_device_do_not_admit(self):
        self.assertFalse(_local_prefill_aiu_prenorm_enabled(_local_state(), _metadata()))
        with torch.inference_mode():
            self.assertFalse(_local_prefill_aiu_prenorm_enabled(None, _metadata()))
            for flag in ("idle", "is_graph", "is_graph_commit", "verify", "proposal"):
                state = _local_state()
                setattr(state, flag, True)
                self.assertFalse(_local_prefill_aiu_prenorm_enabled(state, _metadata()))
            state = _local_state()
            state.prefill = False
            self.assertFalse(_local_prefill_aiu_prenorm_enabled(state, _metadata()))
            for cp in (NS(cp_size=8, kv_cache_sharded=True), NS(cp_size=1, kv_cache_sharded=True)):
                state = _local_state()
                state.context = cp
                self.assertFalse(_local_prefill_aiu_prenorm_enabled(state, _metadata()))
            for missing in (True, False):
                state = _local_state()
                state.cache = None if missing else NS(sharded=True)
                self.assertFalse(_local_prefill_aiu_prenorm_enabled(state, _metadata()))
            state = _local_state()
            state.runtime.model._is_speculative = True
            self.assertFalse(_local_prefill_aiu_prenorm_enabled(state, _metadata()))
            state.runtime.model = None
            self.assertFalse(_local_prefill_aiu_prenorm_enabled(state, _metadata()))
            for target, value in (
                ("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", False),
                ("torch.cuda.get_device_capability", (8, 0)),
                ("torch.cuda.is_current_stream_capturing", True),
            ):
                with patch(target, return_value=value):
                    self.assertFalse(_local_prefill_aiu_prenorm_enabled(_local_state(), _metadata()))

    @torch.inference_mode()
    def test_attention_dispatch_keeps_previous_state_and_cp8_independence(self):
        rows = 1024
        result = PPUStepMix(hidden=object(), pre_mix=torch.zeros(rows, 4),
                            post_mix=torch.zeros(rows, 4, 1), residual_mix=torch.zeros(rows, 4, 4))
        previous = NS(state=object())
        unit = NS(name="attn", hc_mult=4, state=None, _previous_ref=lambda: previous,
                  runtime=NS(current=_local_state()), parameters_map={"fn": object()},
                  ops=NS(mhc=Mock(return_value=result), _mhc=NS(pre=Mock(return_value=result))))
        residual = _metadata((rows, 4, 5120))
        PPUSharedHCUnit._pre_impl(unit, residual)
        unit.ops._mhc.pre.assert_called_once_with(
            residual, unit.parameters_map, state=previous.state, reduce_block_m=None, aiu_prenorm=True,
        )
        unit.ops.mhc.assert_not_called()
        for flag, name in (("0", "attn"), ("1", "ffn")):
            unit.name = name
            unit.ops.mhc.reset_mock()
            unit.ops._mhc.pre.reset_mock()
            with patch.dict(os.environ, {ENV: flag}):
                PPUSharedHCUnit._pre_impl(unit, residual)
            unit.ops.mhc.assert_called_once_with(
                residual, None, phase="pre", parameters=unit.parameters_map, state=previous.state,
            )
            unit.ops._mhc.pre.assert_not_called()
        unit.name = "attn"
        unit.runtime.current.context = NS(cp_size=8, kv_cache_sharded=True, gather_restore_positions=None)
        unit.runtime.current.cache.sharded = True
        for local_flag, cp_flag, expected_aiu in (("1", "0", False), ("0", "1", True)):
            unit.ops.mhc.reset_mock()
            unit.ops._mhc.pre.reset_mock()
            with patch.dict(os.environ, {
                ENV: local_flag, CP_ENV: cp_flag, "RTP_V41_PPU_CP8_YOCO_FAST_PREFILL": "0",
                "RTP_V41_PPU_CP8_YOCO_SMALL_OPS": "0",
            }):
                PPUSharedHCUnit._pre_impl(unit, residual)
            self.assertEqual(unit.ops._mhc.pre.call_count, int(expected_aiu))
            self.assertEqual(unit.ops.mhc.call_count, int(not expected_aiu))


def _units(ops, inputs, eps):
    geometry = NS(hidden_size=5120, hc_mult=4, hc_sinkhorn_iters=20, norm_eps=eps, hc_eps=1e-6)
    tensors = {}
    for which in ("attn", "ffn"):
        for part, source in (("fn", "fn"), ("base", "mix_bases"), ("scale", "mix_scales")):
            tensors[getattr(W, f"v4_hc_{which}_{part}")] = inputs[source]
    views = NS(geometry=geometry, tensor=lambda layer, key: tensors[key])
    attn = PPUSharedHCUnit(views, 1, "attn", ops)
    ffn = PPUSharedHCUnit(views, 0, "ffn", ops)
    attn.runtime = NS(current=_local_state())
    norm = PPUSharedRMSNorm(inputs["rmsnorm_weight"], eps)
    return attn, ffn, norm


def _cpu_pre_reference(residual, parameters, previous_pre, gamma, eps):
    """Independent FP32 equations and BF16 readout/norm landing points."""
    residual = residual.detach().cpu()
    fn, base, scale = (parameters[key].detach().cpu().float() for key in ("fn", "base", "scale"))
    previous_pre = previous_pre.detach().cpu().float() if previous_pre is not None else None
    gamma = gamma.detach().cpu().float()
    parts = {name: [] for name in ("pre", "post", "comb", "hidden", "normalized")}
    for start in range(0, residual.shape[0], CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, residual.shape[0])
        values = residual[start:end].float()
        flat = values.flatten(1)
        mixes = (flat @ fn.T) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps)
        pre = (mixes[:, :4] * scale[0] + base[:4]).sigmoid() + 1e-6
        post = (mixes[:, 4:8] * scale[1] + base[4:8]).sigmoid() * 2.0
        comb = (mixes[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4)
        comb = comb.softmax(-1) + 1e-6
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
        for _ in range(19):
            comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
            comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
        hidden = residual[start:end, 0].contiguous() if previous_pre is None else (
            values * previous_pre[start:end].unsqueeze(-1)
        ).sum(1).to(torch.bfloat16)
        norm_input = hidden.float()
        normalized = (norm_input * torch.rsqrt(norm_input.square().mean(-1, keepdim=True) + eps)
                      * gamma).to(torch.bfloat16)
        for name, value in (("pre", pre), ("post", post.unsqueeze(-1)), ("comb", comb),
                            ("hidden", hidden), ("normalized", normalized)):
            parts[name].append(value)
    return {name: torch.cat(values) for name, values in parts.items()}


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillAIUDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(is_ppu(), "AIU qualification requires a ZW PPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        self.environment = patch.dict(os.environ, {
            ENV: "1", CP_ENV: "0", "RTP_V41_PPU_QUANT_MODE": "mxfp4_fp8_ptpc",
            "RTP_V41_PPU_PTPC_FUSION": "1",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", tf32)
        self.reports = []

    def _finish_ffn(self, attn, ffn, residual, normalized):
        mix = attn.state
        residual = attn.post((normalized * 0.125).contiguous(), residual, mix.post_mix, mix.residual_mix)
        ffn.set_previous(attn)
        hidden, post, comb = ffn.pre(residual)
        return ffn.post((hidden * 0.125).contiguous(), residual, post, comb)

    def _previous_ffn(self, attn, ffn, residual, norm):
        with patch.dict(os.environ, {ENV: "0"}):
            hidden, _, _ = attn.pre(residual)
            residual = self._finish_ffn(attn, ffn, residual, norm(hidden))
        attn.set_previous(ffn)
        return residual

    def _qualify(self, attn, norm, residual, *, aiu):
        from rtp_llm.models_py.modules.dsv4.ppu.hc import ppu_hc_prenorm

        previous = attn._previous_ref() if attn._previous_ref is not None else None
        carry = previous.state if previous is not None else None
        saved = {name: value.clone() for name, value in attn.parameters_map.items()}
        saved["residual"] = residual.clone()
        saved_carry = {field.name: getattr(carry, field.name).clone() for field in fields(carry)
                       if getattr(carry, field.name) is not None} if carry is not None else {}
        self.assertEqual(_local_prefill_aiu_prenorm_enabled(attn.runtime.current, residual), aiu)
        attn.pre(residual)  # Warm the actual existing provider before recording.
        torch.cuda.synchronize()
        with patch.object(ppu_hc_prenorm, "_launch_partials", wraps=ppu_hc_prenorm._launch_partials) as launch:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA]) as profile:
                hidden, post, comb = attn.pre(residual)
                torch.cuda.synchronize()
        candidate = attn.state
        self.assertIs(candidate.pre_mix, attn.pre_mix_out)
        self.assertIs(post, candidate.post_mix)
        self.assertIs(comb, candidate.residual_mix)
        kernels = sorted({event.name for event in profile.events()
                          if event.device_type == torch.autograd.DeviceType.CUDA})
        native = [name for name in kernels if "hc_prenorm" in name.lower()]
        partial = [name for name in kernels if "_hc_mix_stats_partial_kernel" in name]
        self.assertEqual(launch.call_count, int(aiu))
        self.assertEqual(bool(native), aiu, kernels)
        self.assertEqual(bool(partial), not aiu, kernels)
        outputs = dict(pre=candidate.pre_mix, post=post, comb=comb, hidden=hidden, normalized=norm(hidden))
        independent = _cpu_pre_reference(residual, attn.parameters_map,
                                         carry.pre_mix if carry is not None else None,
                                         norm.weight, norm.variance_epsilon)
        with patch.dict(os.environ, {ENV: "0"}):
            legacy_hidden, legacy_post, legacy_comb = attn.pre(residual)
        legacy = dict(pre=attn.state.pre_mix, post=legacy_post, comb=legacy_comb,
                      hidden=legacy_hidden, normalized=norm(legacy_hidden))
        metrics = {}
        for name, value in outputs.items():
            metrics[name] = {}
            for oracle, expected in (("independent_cpu", independent[name]), ("legacy", legacy[name])):
                self.assertEqual(value.dtype, expected.dtype)
                metric = _qualification_metric(torch, value, expected)
                self.assertEqual(metric["qualified_status"], "PASS", (name, oracle, metric))
                metrics[name][oracle] = metric
        for name, value in attn.parameters_map.items():
            self.assertTrue(torch.equal(value, saved[name]), name)
        self.assertTrue(torch.equal(residual, saved["residual"]))
        if carry is not None:
            self.assertIs(previous.state, carry)
            for name, value in saved_carry.items():
                self.assertTrue(torch.equal(getattr(carry, name), value), name)
        else:
            self.assertTrue(torch.equal(hidden, residual[:, 0, :]))
        # Restore the candidate's live state after the independent legacy run.
        attn.state = candidate
        attn.pre_mix_out = candidate.pre_mix
        self.reports.append(dict(rows=residual.shape[0], eps=norm.variance_epsilon,
                                 previous_ffn=carry is not None, aiu=aiu,
                                 provider_calls=launch.call_count, kernels=kernels, outputs=metrics))
        return outputs["normalized"]

    @torch.inference_mode()
    def test_full4096_and_lookahead4097_first_and_previous_ffn(self):
        for eps in (1e-20, 1e-6):
            for rows in (4096, 4097):
                inputs, _, _, _ = _make_case(torch, rows, "cuda:0", "normal")
                for delayed in (False, True):
                    with self.subTest(rows=rows, eps=eps, previous_ffn=delayed):
                        ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=eps))
                        attn, ffn, norm = _units(ops, inputs, eps)
                        residual = inputs["residual"]
                        if delayed:
                            residual = self._previous_ffn(attn, ffn, residual, norm)
                        self._qualify(attn, norm, residual, aiu=True)
        self._write_report("full-lookahead")

    @torch.inference_mode()
    def test_actual_full4096_to_compact140_rejects_old_carry_and_continues(self):
        for eps in (1e-20, 1e-6):
            inputs, _, _, _ = _make_case(torch, 4096, "cuda:0", "normal")
            ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=eps))
            attn, ffn, norm = _units(ops, inputs, eps)
            residual = self._previous_ffn(attn, ffn, inputs["residual"], norm)
            normalized = self._qualify(attn, norm, residual, aiu=True)
            residual = self._finish_ffn(attn, ffn, residual, normalized)
            full_carry = ffn.state
            query, exchange = _compact_exchange(4096, residual.device)
            self.assertEqual(query.chunk_length, 140)
            self.assertEqual(exchange.send_indices.tolist(), list(range(3956, 4096)))
            residual = exchange.compact(residual)
            with self.assertRaisesRegex(ValueError, "pre_mix shape"):
                attn.pre(residual)
            ffn.state = replace(full_carry, **{
                field.name: exchange.compact(getattr(full_carry, field.name))
                if getattr(full_carry, field.name) is not None else None for field in fields(full_carry)
            })
            ffn.pre_mix_out = ffn.state.pre_mix
            for _ in range(2):
                normalized = self._qualify(attn, norm, residual, aiu=False)
                residual = self._finish_ffn(attn, ffn, residual, normalized)
        self._write_report("compact-carry")

    @torch.inference_mode()
    def test_noncontiguous_full_layout_uses_actual_legacy_projection(self):
        inputs, _, _, _ = _make_case(torch, 4096, "cuda:0", "normal")
        ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=1e-20))
        attn, _, norm = _units(ops, inputs, 1e-20)
        backing = torch.empty((4096, 4, 5121), device="cuda:0", dtype=torch.bfloat16)
        residual = backing[:, :, :5120]
        residual.copy_(inputs["residual"])
        self.assertFalse(residual.is_contiguous())
        self._qualify(attn, norm, residual, aiu=False)
        self._write_report("layout-fallback")

    def _write_report(self, label):
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if directory:
            (Path(directory) / f"local-prefill-aiu-mhc-{label}.json").write_text(json.dumps({
                "contract": "existing each-row/global SSE-energy strictly<5e-5",
                "cases": self.reports,
            }, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
