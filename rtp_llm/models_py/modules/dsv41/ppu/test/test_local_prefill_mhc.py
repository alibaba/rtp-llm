"""Local compact140 Mega-mHC admission and the frozen numeric contract.

The existing kernel is rowwise: no cache, position or transport metadata enters
its transition. Equality is not claimed across its FP32 reduction organization
and the legacy chain. The established per-row/global SSE/energy limit, strictly
below5e-5, is imported unchanged from the qualified adapter probe.
"""

import contextlib
import json
import os
import unittest
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.block import Block
from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.block import (
    SharedPPUBlock,
    _local_prefill_mega_mhc_eligible,
)
from rtp_llm.models_py.modules.dsv41.ppu.ced import _bounded_query_layout
from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedHCUnit, PPUSharedRMSNorm
from rtp_llm.models_py.modules.dsv41.ppu.mhc import MHCConfig, PPUStepMix
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_mhc_mega_probe import (
    OUTPUTS,
    _canaries,
    _make_case,
    _qualification_metric,
    independent_transition_reference,
)
from rtp_llm.models_py.modules.dsv41.prefill.ced import _bounded_replay_selected
from rtp_llm.utils.model_weight import W


ENV = "RTP_V41_PPU_LOCAL_PREFILL_MEGA_MHC"  # Retired selector, ignored by production.


def local_state(*, policy=None):
    return NS(
        prefill=True, idle=False, is_graph=False, cache=NS(sharded=False),
        context=None, runtime=NS(policy=policy or SharedPrefillPolicy(mode="compact"),
                                 model=NS(_is_speculative=False)),
    )


def bare_block(state):
    block = SharedPPUBlock.__new__(SharedPPUBlock)
    nn.Module.__init__(block)
    block.layer_id, block.engram, block._mhc_phase = 0, None, "unbound"
    state.before_block = lambda block, *args: args
    state.publication = lambda layer: contextlib.nullcontext()
    block.attn = NS(runtime=NS(current=state, model=state.runtime.model))
    return block


class LocalPrefillMHCContract(unittest.TestCase):
    @torch.inference_mode()
    def test_forward_phase_automatic_local_and_cp8_eligibility(self):
        for flag, mode, sharded, cp, fast_cp, expected in (
            ("0", "compact", False, None, False, "prefill"),
            ("1", "compact", False, None, False, "prefill"),
            ("1", "compact", False, NS(cp_size=1), False, "prefill"),
            ("1", "exact", False, None, False, "ced"),
            ("1", "full", False, None, False, "prefill"),
            ("1", "compact", True, NS(cp_size=8), False, "ced"),
            ("1", "compact", True, NS(cp_size=8), True, "prefill"),
        ):
            with self.subTest(flag=flag, mode=mode, cp=cp, fast_cp=fast_cp):
                state = local_state(policy=SharedPrefillPolicy(mode=mode))
                state.cache.sharded, state.context = sharded, cp
                block = bare_block(state)
                marker = object()
                with patch.dict(os.environ, {ENV: flag}), patch(
                    "rtp_llm.models_py.modules.dsv41.ppu.block.cp8_yoco_operator_eligible",
                    return_value=fast_cp,
                ), patch(
                    "rtp_llm.models_py.modules.dsv41.ppu.block.cp8_yoco_fast_prefill_enabled",
                    return_value=False,
                ), patch.object(Block, "forward", return_value=marker) as parent:
                    self.assertIs(block.forward(None, None, None, None), marker)
                self.assertEqual(block._mhc_phase, expected)
                parent.assert_called_once()

    def test_grad_idle_graph_decode_and_speculative_do_not_admit(self):
        with patch.dict(os.environ, {ENV: "0"}):
            self.assertFalse(_local_prefill_mega_mhc_eligible(None))
        with patch.dict(os.environ, {ENV: "1"}):
            self.assertFalse(_local_prefill_mega_mhc_eligible(local_state()))
            with torch.inference_mode():
                for field in ("idle", "is_graph", "verify"):
                    state = local_state()
                    setattr(state, field, True)
                    self.assertFalse(_local_prefill_mega_mhc_eligible(state))
                state = local_state()
                state.prefill = False
                self.assertFalse(_local_prefill_mega_mhc_eligible(state))
                state = local_state()
                state.runtime.model._is_speculative = True
                self.assertFalse(_local_prefill_mega_mhc_eligible(state))
                state = local_state()
                state.cache = None
                self.assertFalse(_local_prefill_mega_mhc_eligible(state))
                for policy in (SharedPrefillPolicy(mode="compact", max_rollback_tokens=4),
                               SharedPrefillPolicy(mode="compact", encoder_replay=True)):
                    self.assertFalse(_local_prefill_mega_mhc_eligible(local_state(policy=policy)))
                self.assertFalse(_local_prefill_mega_mhc_eligible(
                    local_state(policy=NS(mode="compact")),
                ))

    @torch.inference_mode()
    def test_decode_phase_binding_and_metadata_identity_guard_stay_unchanged(self):
        state = local_state()
        state.executed_layers, state.v4 = [], NS(capture_aux_hidden_layer_ids=())
        block = bare_block(state)
        with patch.dict(os.environ, {ENV: "1"}), patch.object(Block, "forward_decode", return_value="decode"):
            self.assertEqual(block.forward_decode(None, None, None), "decode")
        self.assertEqual(block._mhc_phase, "decode")
        self.assertEqual(state.executed_layers, [0])
        block.attn_hc = NS(state=NS(post_mix=object(), residual_mix=object()))
        with self.assertRaisesRegex(RuntimeError, "live attention pre state"):
            block._try_mega_mhc(None, None, object(), object())


def _device_block(ops, inputs, attention_state):
    eps = ops._mhc.config.rms_eps
    geometry = NS(hidden_size=5120, hc_mult=4, hc_sinkhorn_iters=20,
                  norm_eps=eps, hc_eps=1e-6)
    tensors = {}
    for which in ("attn", "ffn"):
        for part, name in (("fn", "fn"), ("base", "mix_bases"), ("scale", "mix_scales")):
            tensors[getattr(W, f"v4_hc_{which}_{part}")] = inputs[name]
    views = NS(geometry=geometry, tensor=lambda layer, key: tensors[key])
    block = bare_block(local_state())
    block.ops = ops
    block.attn_hc = PPUSharedHCUnit(views, 0, "attn", ops)
    block.ffn_hc = PPUSharedHCUnit(views, 0, "ffn", ops)
    block.ffn_hc.set_previous(block.attn_hc)
    block.attn_hc.state = attention_state
    block.ffn_norm = PPUSharedRMSNorm(inputs["rmsnorm_weight"], eps)
    block.bind_execution_context(
        "prefill", ced=not _local_prefill_mega_mhc_eligible(block.attn.runtime.current),
    )
    return block


def _mix(inputs):
    return PPUStepMix(hidden=None, pre_mix=inputs["shifted_prev_mix"].squeeze(-1),
                      post_mix=inputs["post_mix"], residual_mix=inputs["comb_res_mix"])


def _compact_exchange(rows, device):
    indices = torch.arange(rows, dtype=torch.int64, device=device)
    valid = torch.ones(rows, dtype=torch.bool, device=device)
    prefix = 126976
    original = CPContext(
        cp_size=1, cp_rank=0, chunk_length=rows, padded_seq_len=rows,
        seq_len_full=rows, relative_positions=indices, prefix_length=prefix,
        global_positions=indices + prefix, local_is_real=valid, unpad_restore=indices,
        seq_len_total=prefix + rows,
        cp_info=NS(prefill_qkv_restore_indice_cpu=indices.cpu(), prefill_qkv_padding_mask_cpu=valid.cpu()),
        req_id_per_token=torch.zeros_like(indices, dtype=torch.int32),
        prefix_lengths=torch.tensor([prefix], device=device),
        input_lengths_global=torch.tensor([rows], device=device),
        cu_seqlens_global=torch.tensor([0, rows], device=device),
        chunk_lengths_per_req=(rows,), input_lengths_global_host=(rows,), prefix_lengths_host=(prefix,),
    )
    selected = _bounded_replay_selected((rows,), (rows - 140,))
    query, exchange, _ = _bounded_query_layout(original, selected)
    return query, exchange


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillMHCDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(is_ppu(), "native qualification requires a ZW PPU")
        self.environment = patch.dict(os.environ, {
            ENV: "0", "RTP_V41_PPU_MHC_BACKEND": "mega",
            "RTP_V41_PPU_QUANT_MODE": "mxfp4_fp8_ptpc", "RTP_V41_PPU_PTPC_FUSION": "1",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.saved_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", self.saved_tf32)
        self.reports = []

    def qualify(self, ops, inputs, state=None, *, cp8=False):
        eps = ops._mhc.config.rms_eps
        state = _mix(inputs) if state is None else state
        original = {name: value.clone() for name, value in inputs.items()}
        block = _device_block(ops, inputs, state)
        if cp8:
            forward = block.attn.runtime.current
            forward.context = NS(cp_size=8, kv_cache_sharded=True, gather_restore_positions=None)
            forward.cache.sharded = True
            with patch.object(Block, "forward", return_value=None):
                block.forward(None, None, None, None)
        self.assertEqual(block._mhc_phase, "prefill")
        actual = block._try_mega_mhc(inputs["x"], inputs["residual"], state.post_mix, state.residual_mix)
        self.assertIsNotNone(actual, ops.mhc_execution_stats())
        residual, normalized, post, comb = actual
        accepted = block.ffn_hc.state
        self.assertIsNone(accepted.hidden)
        self.assertIs(block.ffn_hc.pre_mix_out, accepted.pre_mix)
        self.assertIs(post, accepted.post_mix)
        self.assertIs(comb, accepted.residual_mix)
        outputs = dict(zip(OUTPUTS, (residual, accepted.pre_mix.unsqueeze(-1), post, comb, normalized)))
        scalars = dict(hc_norm_eps=eps, hc_pre_eps=1e-6, hc_post_scale=2.0,
                       sinkhorn_eps=1e-6, num_sinkhorn_iters=20,
                       rmsnorm_eps=eps, rmsnorm_scale=1.0)
        independent_inputs = {name: value.detach().cpu() for name, value in inputs.items()}
        independent = independent_transition_reference(torch, independent_inputs, scalars)
        legacy_post = ops.mhc(inputs["residual"], inputs["x"], phase="post", state=state)
        legacy_pre = ops.mhc(legacy_post.hidden, None, phase="pre",
                             parameters=block.ffn_hc.parameters_map, state=state)
        legacy = dict(zip(OUTPUTS, (legacy_post.hidden, legacy_pre.pre_mix.unsqueeze(-1),
                                    legacy_pre.post_mix, legacy_pre.residual_mix,
                                    block.ffn_norm(legacy_pre.hidden))))
        metrics = {}
        for name, value in outputs.items():
            self.assertEqual(value.shape, independent[name].shape)
            self.assertEqual(value.dtype, independent[name].dtype)
            self.assertTrue(bool(torch.isfinite(value).all()), name)
            metrics[name] = {}
            for oracle, expected in (("independent", independent[name]), ("legacy", legacy[name])):
                metric = _qualification_metric(torch, value, expected)
                self.assertEqual(metric["qualified_status"], "PASS", (name, oracle, metric))
                metrics[name][oracle] = metric
        for name, value in inputs.items():
            self.assertTrue(torch.equal(value, original[name]), name)
        # The FFN consumes normalized exactly once; its post and next readout
        # consume the newly accepted FFN state, not the old attention carry.
        branch = (normalized * 0.125).contiguous()
        final = ops.mhc(residual, branch, phase="post", state=accepted).hidden
        expected_post = (independent["new_post_mix"].to(final.device) * branch.float().unsqueeze(1)
                         + torch.einsum("tij,tih->tjh", independent["new_comb_res_mix"].to(final.device),
                                        independent["new_residual"].to(final.device).float())).to(torch.bfloat16)
        self.assertEqual(_qualification_metric(torch, final, expected_post)["qualified_status"], "PASS")
        readout = ops.mhc(final, None, phase="head", state=accepted).hidden
        expected_head = (final.float() * independent["new_prev_mix"].to(final.device)).sum(1).to(torch.bfloat16)
        self.assertEqual(_qualification_metric(torch, readout, expected_head)["qualified_status"], "PASS")
        self.reports.append({"rows": inputs["x"].shape[0], "eps": eps,
                             "phase": block._mhc_phase, "outputs": metrics})
        return final, accepted

    @torch.inference_mode()
    def test_cp8_compact_transition_ignores_retired_and_framework_flags(self):
        ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=1e-6))
        inputs, _, _, _ = _make_case(torch, 140, "cuda:0", "normal")
        with patch.dict(os.environ, {"RTP_V41_PPU_CP8_YOCO_FAST_PREFILL": "0",
                                     "RTP_V41_PPU_CP8_YOCO_SMALL_OPS": "0"}):
            self.qualify(ops, inputs, cp8=True)
        self.assertEqual(ops.mhc_execution_stats()["calls"], {"prefill": 1})
        self.write_report("cp8-auto-selector")

    @torch.inference_mode()
    def test_actual_full_compact_and_lookahead_rows_meet_existing_contract(self):
        for eps in (1e-20, 1e-6):
            ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=eps))
            for rows in (4097, 140):
                with self.subTest(rows=rows, eps=eps):
                    inputs, _, buffers, _ = _make_case(torch, rows, "cuda:0", "normal")
                    self.qualify(ops, inputs)
                    self.assertTrue(all(all(item.values()) for item in _canaries(torch, buffers).values()))
            self.assertEqual(ops.mhc_execution_stats()["calls"], {"prefill": 2})
        self.write_report("full-compact-lookahead")

    @torch.inference_mode()
    def test_full4096_to_actual_compact140_refreshes_all_carry_fields(self):
        for eps in (1e-20, 1e-6):
            with self.subTest(eps=eps):
                self.compact_chain(eps)
        self.write_report("carry-transition")

    def compact_chain(self, eps):
        ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=eps))
        inputs, _, _, _ = _make_case(torch, 4096, "cuda:0", "normal")
        final, full_carry = self.qualify(ops, inputs)
        query, exchange = _compact_exchange(4096, final.device)
        self.assertEqual(query.chunk_length, 140)
        self.assertEqual(exchange.send_indices.tolist(), list(range(3956, 4096)))
        compact_carry = replace(full_carry, **{
            field.name: exchange.compact(getattr(full_carry, field.name))
            if getattr(full_carry, field.name) is not None else None
            for field in fields(full_carry)
        })
        compact_residual = exchange.compact(final)
        parameters = {"fn": inputs["fn"], "base": inputs["mix_bases"], "scale": inputs["mix_scales"]}
        attention = ops.mhc(compact_residual, None, phase="pre", parameters=parameters, state=compact_carry)
        branch = (attention.hidden * 0.125).contiguous()
        compact_inputs = {**inputs, "x": branch, "residual": compact_residual,
                          "shifted_prev_mix": attention.pre_mix.unsqueeze(-1),
                          "post_mix": attention.post_mix, "comb_res_mix": attention.residual_mix}
        # Old full-sized pre carry cannot be dispatched against retained rows.
        invalid = replace(attention, pre_mix=full_carry.pre_mix)
        self.assertIsNone(ops.try_mhc_ffn_transition(
            branch, compact_residual, invalid, parameters, inputs["rmsnorm_weight"], eps, phase="prefill",
        ))
        self.assertEqual(ops.mhc_execution_stats()["fallbacks"], {"prefill:tensor_contract": 1})
        self.qualify(ops, compact_inputs, attention)
        self.assertEqual(ops.mhc_execution_stats()["calls"], {"prefill": 2})

    @torch.inference_mode()
    def test_unsupported_tensor_geometry_retains_native_fallback(self):
        ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=1e-20))
        inputs, _, _, _ = _make_case(torch, 140, "cuda:0", "normal")
        result = ops.try_mhc_ffn_transition(
            inputs["x"], inputs["residual"][:, :, :4096], _mix(inputs),
            {"fn": inputs["fn"], "base": inputs["mix_bases"], "scale": inputs["mix_scales"]},
            inputs["rmsnorm_weight"], 1e-20, phase="prefill",
        )
        self.assertIsNone(result)
        self.assertEqual(ops.mhc_execution_stats()["fallbacks"], {"prefill:layout": 1})
        self.assertEqual(ops.mhc_execution_stats()["calls"], {})

    def write_report(self, label):
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if directory:
            destination = Path(directory) / f"local-prefill-mhc-{label}.json"
            destination.write_text(json.dumps({"contract": "existing per-row/global SSE-energy strictly<5e-5",
                                               "cases": self.reports}, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
