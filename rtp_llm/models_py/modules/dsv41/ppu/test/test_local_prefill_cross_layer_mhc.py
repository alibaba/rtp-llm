"""Bounded local FFN-post -> next-attention-pre Mega-mHC seam.

The existing adapter probe's independent FP32 authority and frozen per-row /
global SSE-energy strictly<5e-5 qualify algorithmic rounding, not byte identity.
Attention/MoE/Engram below are deterministic branch fixtures; HC units, Block
dispatch, delayed carry, RMSNorm, bounded CED transport and head are real.
"""

from contextlib import ExitStack
from dataclasses import fields
import gc
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.block import Block
from rtp_llm.models_py.modules.dsv41.ppu import layers as layer_module
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.block import SharedPPUBlock
from rtp_llm.models_py.modules.dsv41.ppu.layers import (
    PPUSharedHCHead, PPUSharedHCUnit, PPUSharedRMSNorm,
)
from rtp_llm.models_py.modules.dsv41.ppu.mhc import MHCConfig, PPUMHC, PPUStepMix
from rtp_llm.models_py.modules.dsv41.ppu.mhc_mega import PPUMegaMHCTransition
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import (
    LOCAL_PREFILL_CROSS_LAYER_MHC_ENV,
    SharedForwardState,
    _cross_mix_snapshot,
    _cross_target_snapshot,
    bind_cross_mhc_layers,
)
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_mhc import _compact_exchange
from rtp_llm.models_py.modules.dsv41.ppu.test.test_mhc_mega_probe import (
    OUTPUTS, _canaries, _make_case, _qualification_metric,
    independent_transition_reference,
)
from rtp_llm.models_py.modules.dsv41.prefill.ced import CEDPlan
from rtp_llm.utils.model_weight import W


ENV = LOCAL_PREFILL_CROSS_LAYER_MHC_ENV
PREFIX = 126976
BASE_ENV = {
    ENV: "1", "RTP_V41_PPU_MHC_BACKEND": "mega",
    "RTP_V41_PPU_LOCAL_PREFILL_ROWS": "1",
    "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS": "1",
    "RTP_V41_PPU_LOCAL_PREFILL_SPLIT_L20": "1",
    "RTP_V41_PPU_LOCAL_PREFILL_AIU_PRENORM": "0",
    "RTP_V41_PPU_LOCAL_PREFILL_ENGRAM_GATE": "0",
    "RTP_V41_PPU_LOCAL_PREFILL_SWA_SEGMENTS": "0",
    "RTP_V41_PPU_CP8_YOCO_FAST_PREFILL": "0",
    "MOEDBG": "0", "DSV4_CP_SYNC_AFTER_ATTN_ONCE": "0",
}


class _CPUOps:
    """Pure-Torch fake native boundary, retaining no input Tensor references."""

    def __init__(self, eps=1e-20):
        self._mhc = PPUMHC(MHCConfig(rms_eps=eps, use_triton=False))
        self.native_calls = 0
        self.decline = False
        self.error = None

    def mhc(self, residual, branch, **kwargs):
        return self._mhc(residual, branch, **kwargs)

    def try_mhc_ffn_transition(self, branch, residual, state, parameters, norm, eps, *, phase):
        if phase != "prefill" or self.decline:
            return None
        if self.error is not None:
            raise self.error
        self.native_calls += 1
        updated = self.mhc(residual, branch, phase="post", state=state).hidden
        mix = self.mhc(updated, None, phase="pre", state=state, parameters=parameters)
        x = mix.hidden.float()
        normalized = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
                      * norm.float()).to(mix.hidden.dtype)
        return PPUMegaMHCTransition(updated, normalized, mix.pre_mix, mix.post_mix, mix.residual_mix)


class _Branch(nn.Module):
    supports_flat_prefill = True

    def __init__(self, runtime, layer_id, *, attention=False):
        super().__init__()
        self.runtime, self.layer_id = runtime, layer_id
        self.attention = attention
        self._cp_ctx = None
        self.is_kv_source = self.is_index_source = attention and layer_id == 20
        self.rows = []

    def forward(self, hidden, *args, compact_after_producer=None, **kwargs):
        self.rows.append(hidden.shape[0])
        output = (hidden.float() * (0.03125 if self.attention else 0.0625)
                  + 0.0078125).to(hidden.dtype)
        state = self.runtime.current
        if compact_after_producer is not None:
            compact_after_producer()
            output = state.plan.exchange.compact(output)
        if self.attention and self.layer_id == 20:
            state.shared["topk"][20] = torch.zeros(output.shape[0], 1, dtype=torch.int32, device=output.device)
            state.shared["candidates"] = None
        return output


class _Engram(nn.Module):
    def prefetch(self, *args, **kwargs):
        pass

    def forward(self, hidden, *args):
        return (hidden.float() + 0.00390625).to(hidden.dtype)


def _fixture(inputs, ops, *, count=4, enabled="1", birth=True, prefill=True, idle=False):
    eps = ops._mhc.config.rms_eps
    engram_ids = tuple(i for i in (1, 14) if i < count)
    geometry = NS(num_layers=count, engram_layer_ids=engram_ids, hidden_size=5120, hc_mult=4,
                  hc_sinkhorn_iters=20, norm_eps=eps, hc_eps=1e-6)
    tensors = {}
    for i in range(count):
        for which in ("attn", "ffn"):
            branch = int(which == "ffn")
            for part, name in (("fn", "fn"), ("base", "mix_bases"), ("scale", "mix_scales")):
                if part == "base":
                    value = inputs[name] + (i + 1) * 0.0009765625 + branch * 0.00390625
                else:
                    value = inputs[name] * (1 + (i + 1) * 0.0078125 + branch * 0.03125)
                tensors[i, getattr(W, f"v4_hc_{which}_{part}")] = value
    views = NS(geometry=geometry, tensor=lambda layer, key: tensors[layer, key])
    model = NS(_is_decode_role=False, _is_speculative=False, _prefill_cp_size=1,
               parallelism_config=NS(tp_size=1))
    runtime = NS(model=model, policy=SharedPrefillPolicy(mode="compact"), current=None)
    blocks = []
    for i in range(count):
        block = SharedPPUBlock.__new__(SharedPPUBlock)
        nn.Module.__init__(block)
        block.layer_id, block.ops, block.query_only = i, ops, False
        block._mhc_phase, block._cp_sync_after_attn_done = "unbound", False
        block.engram = _Engram() if i in engram_ids else None
        block.engram_hashes = block.engram_token_mask = None
        block.attn = _Branch(runtime, i, attention=True)
        block.ffn = _Branch(runtime, i)
        for which in ("attn", "ffn"):
            unit = PPUSharedHCUnit(views, i, which, ops)
            unit.runtime = runtime
            setattr(block, f"{which}_hc", unit)
            gamma = (inputs["rmsnorm_weight"].float()
                     * (1 + (i + 1) * 0.015625 + int(which == "ffn") * 0.0078125)).to(torch.bfloat16)
            norm = PPUSharedRMSNorm(gamma, eps)
            norm.runtime = runtime
            if inputs["x"].device.type == "cpu":
                norm._ptpc = False
            setattr(block, f"{which}_norm", norm)
        block.ffn_hc.set_previous(block.attn_hc)
        block.attn_hc.set_previous(blocks[-1].ffn_hc if blocks else None)
        blocks.append(block)
    def propagate(context):
        for block in blocks:
            block.attn._cp_ctx = context
    model.v4 = NS(layers=blocks, _propagate_cp_ctx=propagate, capture_aux_hidden_layer_ids=())
    runtime._cross_mhc_bindings = bind_cross_mhc_layers(blocks, geometry)
    cache = SharedNativeCache.__new__(SharedNativeCache)
    cache.sharded, cache._is_prefill = False, True
    rows, device = inputs["x"].shape[0], inputs["x"].device
    token_inputs = NS(input_ids=torch.arange(rows, device=device),
                      combo_position_ids=torch.arange(PREFIX, PREFIX + rows, device=device))
    attn = NS(cu_seqlens=torch.tensor([0, rows], dtype=torch.int32, device=device),
              input_lengths=torch.tensor([rows], dtype=torch.int32),
              prefix_lengths=torch.tensor([PREFIX], dtype=torch.int32), is_cuda_graph=False)
    with patch.dict(os.environ, {ENV: enabled}):
        state = SharedForwardState(runtime, token_inputs, attn, cache, prefill=prefill, idle=idle,
                                   prefixes=(PREFIX,), lengths=(rows,))
    runtime.current, runtime._cross_mhc_evidence = state, state._cross_mhc_evidence
    if birth:
        state.rows()
    return state, blocks


def _producer(state, blocks, inputs, *, source=1):
    block = blocks[source]
    unit = block.ffn_hc
    unit.state = None
    mix = PPUStepMix(hidden=None, pre_mix=inputs["shifted_prev_mix"].squeeze(-1),
                     post_mix=inputs["post_mix"], residual_mix=inputs["comb_res_mix"])
    unit.pre_mix_out, unit.state = mix.pre_mix, mix
    state.executed_layers = [source]
    state.begin_cross_mhc_block(block)
    return unit, mix


def _publish(state, blocks, inputs, *, source=1):
    unit, mix = _producer(state, blocks, inputs, source=source)
    updated = unit.post(inputs["x"], inputs["residual"], mix.post_mix, mix.residual_mix)
    return updated, mix


def _consume(state, block, updated):
    state.executed_layers.append(block.layer_id)
    state.begin_cross_mhc_block(block)
    return block._try_cached_attn_pre(updated)


def _install_plan(state):
    rows = state.lengths[0]
    original = state.identity_context()
    query, exchange = _compact_exchange(rows, state.inputs.input_ids.device)
    query.swa_replay_start = rows - 140
    # Real retained indices stay on the original device, even though the
    # bounded selector's public host input is deliberately CPU.
    assert exchange.send_indices.device == original.global_positions.device
    assert exchange.receive_positions.device == original.global_positions.device
    assert query.global_positions.device == original.global_positions.device
    plan = CEDPlan(original, query,
                   torch.tensor([0, 140], dtype=torch.int32, device=original.global_positions.device),
                   exchange, (), ())
    def prepare():
        state.original_context, state.plan, state.plan_reason = original, plan, "compact140"
    state.prepare_plan = prepare
    return plan


def _run_chain(state, blocks, residual, *, split=True):
    with patch.dict(os.environ, {"RTP_V41_PPU_LOCAL_PREFILL_SPLIT_L20": str(int(split))}):
        for block in blocks:
            residual = block(residual, state.inputs.input_ids, state.inputs.combo_position_ids,
                             state.attn.cu_seqlens, expert_load=None)
    head = PPUSharedHCHead(blocks[-1].ffn_hc, blocks[-1].ops).head(residual)
    return residual, head


def _mix_fields(blocks):
    return {f"{block.layer_id}.{which}.{field.name}": getattr(mix, field.name)
            for block in blocks for which in ("attn", "ffn")
            for mix in (getattr(block, f"{which}_hc").state,)
            for field in fields(mix) if getattr(mix, field.name) is not None and field.name != "hidden"}


class LocalCrossLayerMHCContract(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, BASE_ENV))
        self.stack.enter_context(patch.object(layer_module, "is_ppu", return_value=True))
        # U1 production norms require a native library. Only this CPU branch
        # fixture supplies its independent normalization boundary.
        self.stack.enter_context(patch.object(layer_module, "rms_norm", side_effect=lambda x, w, eps:
            (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
             * w.float()).to(x.dtype)))
        self.stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch(
            "rtp_llm.models_py.modules.dsv41.ppu.attention_layer.optimized_diagnostics_enabled", return_value=False))

    def fixture(self, **kwargs):
        inputs, _, _, _ = _make_case(torch, kwargs.pop("rows", 2), "cpu", "normal")
        state, blocks = _fixture(inputs, _CPUOps(), **kwargs)
        return inputs, state, blocks

    def test_default_latch_phase_and_real_checkpoint_topology(self):
        inputs, state, blocks = self.fixture(count=40)
        self.assertEqual(sorted(state.runtime._cross_mhc_bindings), [*range(1, 13), *range(14, 39)])
        self.assertEqual(len(state.runtime._cross_mhc_bindings), 37)
        self.assertNotIn(0, state.runtime._cross_mhc_bindings)
        self.assertNotIn(13, state.runtime._cross_mhc_bindings)
        self.assertIsNone(Block._try_cached_attn_pre(blocks[0], inputs["residual"]))
        for enabled in ("0", "", "true"):
            _, disabled, _ = self.fixture(enabled=enabled)
            self.assertFalse(disabled._cross_mhc_phase())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV)
            _, disabled, _ = self.fixture(enabled="0")
            self.assertFalse(disabled._cross_mhc_enabled)
        for flag in ("idle", "is_graph", "is_graph_commit", "verify", "proposal"):
            _, other, _ = self.fixture()
            setattr(other, flag, True)
            self.assertFalse(other._cross_mhc_phase(), flag)
        for mutation in (
            lambda s: setattr(s, "prefill", False),
            lambda s: setattr(s.cache, "sharded", True),
            lambda s: setattr(s, "context", NS(cp_size=8, kv_cache_sharded=True)),
            lambda s: setattr(s.runtime.model, "_is_decode_role", True),
            lambda s: setattr(s.runtime.model, "_is_speculative", True),
            lambda s: setattr(s.runtime.model.parallelism_config, "tp_size", 2),
            lambda s: setattr(s.runtime, "policy", SharedPrefillPolicy(mode="exact")),
        ):
            _, other, _ = self.fixture()
            mutation(other)
            self.assertFalse(other._cross_mhc_phase())
        with torch.enable_grad():
            self.assertFalse(state._cross_mhc_phase())

    def test_one_use_real_post_entry_state_and_storage_guard(self):
        inputs, state, blocks = self.fixture()
        updated, previous = _publish(state, blocks, inputs)
        self.assertIsNotNone(state._cross_mhc_ticket)
        accepted = _consume(state, blocks[2], updated.view_as(updated))
        self.assertIsNotNone(accepted)
        self.assertIsNone(blocks[2].attn_hc.state.hidden)
        self.assertIs(blocks[2].attn_hc.pre_mix_out, blocks[2].attn_hc.state.pre_mix)
        self.assertIs(blocks[1].ffn_hc.state, previous)
        self.assertIsNone(blocks[2]._try_cached_attn_pre(updated))
        self.assertEqual(state._cross_mhc_evidence["published"], 1)
        self.assertEqual(state._cross_mhc_evidence["consumed"], 1)
        updated, _ = _publish(state, blocks, inputs)
        self.assertIsNone(_consume(state, blocks[2], updated.clone()))
        self.assertEqual(state._cross_mhc_evidence["fallbacks"], {"consumer_invalid": 1})
        updated, _ = _publish(state, blocks, inputs)
        state.begin_cross_mhc_block(blocks[2], allowed=False)
        self.assertIsNone(state._cross_mhc_ticket)

    def test_captured_tensor_owners_survive_rebinding_then_release(self):
        inputs, state, blocks = self.fixture()
        target = blocks[2].attn_hc
        for name in ("fn", "base", "scale"):
            setattr(target, name, getattr(target, name).clone())
        blocks[2].attn_norm.weight = blocks[2].attn_norm.weight.clone()
        unit, mix = _producer(state, blocks, inputs)
        mutable = NS(**{field.name: getattr(mix, field.name).clone()
                        if isinstance(getattr(mix, field.name), torch.Tensor) else None for field in fields(mix)})
        mutable.hidden = inputs["residual"][:, 0].clone()
        retained = {f"source.{name}": weakref.ref(value) for name, value in vars(mutable).items()}
        retained.update({f"target.{name}": weakref.ref(value) for name, value in target.parameters_map.items()})
        retained["target.gamma"] = weakref.ref(blocks[2].attn_norm.weight)
        unit.state = None
        unit.pre_mix_out = mutable.pre_mix
        unit.state = mutable
        updated = unit.post(inputs["x"], inputs["residual"], mutable.post_mix, mutable.residual_mix)
        for name in ("fn", "base", "scale"):
            setattr(target, name, getattr(target, name).clone())
        blocks[2].attn_norm.weight = blocks[2].attn_norm.weight.clone()
        for name in vars(mutable):
            setattr(mutable, name, getattr(mutable, name).clone())
        unit._ppu_pre_mix_out = mutable.pre_mix
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in retained.values()))
        self.assertIsNone(_consume(state, blocks[2], updated))
        gc.collect()
        self.assertTrue(all(reference() is None for reference in retained.values()))

    def test_raw_and_derived_epoch_mutation_never_recertifies(self):
        for source in ("input_ids", "combo_position_ids", "cu_seqlens", "input_lengths", "prefix_lengths", "derived"):
            with self.subTest(source=source):
                inputs, state, blocks = self.fixture()
                updated, _ = _publish(state, blocks, inputs)
                value = (state.rows()[0] if source == "derived" else
                         getattr(state.inputs if hasattr(state.inputs, source) else state.attn, source))
                value.add_(1)
                value.sub_(1)
                self.assertIsNone(_consume(state, blocks[2], updated))
                self.assertIsNone(state._cross_mhc_epoch(2))
        inputs, state, blocks = self.fixture(birth=False)
        with patch.dict(os.environ, {ENV: "0"}):
            state.rows()  # derived cache was constructed while the flag was off
        state.rows()  # a subsequent hit must not mint a missing birth stamp
        self.assertIsNone(state._cross_mhc_full_stamp)
        self.assertIsNone(state._cross_mhc_epoch(2))
        inputs, state, blocks = self.fixture()
        updated, _ = _publish(state, blocks, inputs)
        state.plan = object()  # future L19 plan creation does not change full rows
        self.assertIsNotNone(_consume(state, blocks[2], updated))
        inputs, state, blocks = self.fixture()
        _publish(state, blocks, inputs)
        state._local_prefill_rows.clear()
        state.rows()
        self.assertIsNone(state._cross_mhc_epoch(2))

    def test_unsupported_metadata_declines_before_tensor_keys(self):
        inputs, state, blocks = self.fixture()
        sparse = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([1.0]), (2, 4))
        neg = torch._neg_view(torch.ones(2, 4))
        conj = torch.ones(2, 4, dtype=torch.complex64).conj()
        for value in (sparse, neg, conj):
            with self.subTest(layout=str(value.layout), neg=value.is_neg(), conj=value.is_conj()):
                mix = NS(hidden=value, pre_mix=None, post_mix=None, residual_mix=None)
                self.assertIsNone(_cross_mix_snapshot(mix))
                original = blocks[2].attn_hc.fn
                blocks[2].attn_hc.fn = value
                self.assertIsNone(_cross_target_snapshot(blocks[2]))
                blocks[2].attn_hc.fn = original
                old = state.inputs.combo_position_ids
                state.inputs.combo_position_ids = value
                self.assertIsNone(state._cross_mhc_original())
                state.inputs.combo_position_ids = old
        updated, _ = _publish(state, blocks, inputs)
        self.assertIsNone(_consume(state, blocks[2], torch._neg_view(updated)))
        self.assertEqual(state._cross_mhc_evidence["fallbacks"], {"consumer_layout": 1})

    def test_compact_birth_no_transport_split_off_and_changed_proof(self):
        inputs, state, blocks = self.fixture(count=22, rows=257)
        plan = _install_plan(state)
        state.prepare_plan()
        full, _ = _publish(state, blocks, inputs, source=19)
        self.assertIsNotNone(state._cross_mhc_ticket)
        compact = plan.exchange.compact(full)
        state.compact_l20_queries(compact, plan.exchange.compact(state.inputs.input_ids))
        self.assertIsNone(state._cross_mhc_ticket)
        self.assertIsNotNone(state._cross_mhc_epoch(140))
        self.assertEqual(state._cross_mhc_evidence["fallbacks"], {"compaction": 1})
        plan.exchange.send_indices.add_(1)
        plan.exchange.send_indices.sub_(1)
        self.assertIsNone(state._cross_mhc_epoch(140))
        inputs, state, blocks = self.fixture(count=22, rows=257)
        _producer(state, blocks, inputs, source=20)
        self.assertIsNone(state.try_ffn_to_next_attn(blocks[20].ffn_hc, inputs["x"], inputs["residual"],
                                                    inputs["post_mix"], inputs["comb_res_mix"]))
        self.assertEqual(state._cross_mhc_evidence["published"], 0)

    def test_actual_block_chain_and_final_head_match_legacy_bytes(self):
        inputs, state, blocks = self.fixture(rows=3)
        with patch.dict(os.environ, {ENV: "0"}):
            old, old_head = _run_chain(state, blocks, inputs["residual"])
        expected = {name: value.clone() for name, value in _mix_fields(blocks).items()}
        candidate, new_blocks = _fixture(inputs, _CPUOps())
        actual, head = _run_chain(candidate, new_blocks, inputs["residual"])
        self.assertTrue(torch.equal(actual, old))
        self.assertTrue(torch.equal(head, old_head))
        for name, value in _mix_fields(new_blocks).items():
            self.assertTrue(torch.equal(value, expected[name]), name)
        self.assertEqual(candidate._cross_mhc_evidence["published"], 2)
        self.assertEqual(candidate._cross_mhc_evidence["consumed"], 2)
        self.assertEqual([block.attn_norm.calls for block in new_blocks], [1, 1, 0, 0])
        self.assertEqual([block.ffn_norm.calls for block in new_blocks], [0] * 4)
        self.assertEqual(state.runtime.model.v4.layers[0].ops.native_calls, 4)
        self.assertEqual(new_blocks[0].ops.native_calls, 6)
        self.assertIsNone(new_blocks[0].attn_hc._previous_ref)
        self.assertIsNotNone(new_blocks[-1].ffn_hc.state.pre_mix)

    def test_native_decline_error_and_fresh_dispatch_evidence_lifetime(self):
        inputs, state, blocks = self.fixture()
        blocks[1].ops.decline = True
        updated, mix = _publish(state, blocks, inputs)
        expected = blocks[1].ops.mhc(inputs["residual"], inputs["x"], phase="post", state=mix).hidden
        self.assertTrue(torch.equal(updated, expected))
        self.assertIsNone(state._cross_mhc_ticket)
        blocks[1].ops.decline = False
        blocks[1].ops.error = RuntimeError("native launch failed")
        with self.assertRaisesRegex(RuntimeError, "native launch failed"):
            _publish(state, blocks, inputs)
        self.assertIsNone(state._cross_mhc_ticket)
        blocks[1].ops.error = None
        _publish(state, blocks, inputs)
        retained = state.runtime._cross_mhc_evidence
        state.discard_next_attn_mhc("forward_exit")
        retained["complete"] = False
        _, next_state, _ = self.fixture()
        self.assertIsNot(retained, next_state.runtime._cross_mhc_evidence)
        self.assertEqual(retained["published"], 1)
        self.assertEqual(retained["fallbacks"], {"forward_exit": 1})
        self.assertFalse(retained["complete"])


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalCrossLayerMHCDevice(unittest.TestCase):
    def setUp(self):
        self.assertTrue(is_ppu(), "native qualification requires a ZW PPU")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.inference_mode())
        self.stack.enter_context(patch.dict(os.environ, dict(BASE_ENV,
            RTP_V41_PPU_QUANT_MODE="mxfp4_fp8_ptpc", RTP_V41_PPU_PTPC_FUSION="1")))
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", old_tf32)
        self.reports = []

    def assertQualified(self, actual, expected, label):
        self.assertEqual(actual.shape, expected.shape, label)
        self.assertEqual(actual.dtype, expected.dtype, label)
        metric = _qualification_metric(torch, actual, expected)
        self.assertEqual(metric["qualified_status"], "PASS", (label, metric))
        self.reports.append({"label": label, "metric": metric})

    def component(self, rows, eps):
        inputs, _, buffers, _ = _make_case(torch, rows, "cuda:0", "normal")
        original = {name: value.clone() for name, value in inputs.items()}
        ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=eps))
        state, blocks = _fixture(inputs, ops)
        target = blocks[2]
        target_parameters = {name: value.clone() for name, value in target.attn_hc.parameters_map.items()}
        target_gamma = target.attn_norm.weight.clone()
        unit, _ = _producer(state, blocks, inputs)
        # Produce a genuine FFN pre carry through the real HC unit. The
        # source FFN weights differ from next-attention weights above.
        attention = ops.mhc(inputs["residual"], None, phase="pre",
                            parameters=blocks[1].attn_hc.parameters_map)
        blocks[1].attn_hc.pre_mix_out = attention.pre_mix
        blocks[1].attn_hc.state = attention
        unit.pre(inputs["residual"])
        previous = unit.state
        updated = unit.post(inputs["x"], inputs["residual"], previous.post_mix, previous.residual_mix)
        self.assertIsNotNone(state._cross_mhc_ticket, ops.mhc_execution_stats())
        transition = state._cross_mhc_ticket.transition
        self.assertIsNotNone(_consume(state, blocks[2], updated))
        actual = dict(zip(OUTPUTS, (updated, transition.pre_mix.unsqueeze(-1),
                                    transition.post_mix, transition.residual_mix, transition.normalized)))
        scalars = dict(hc_norm_eps=eps, hc_pre_eps=1e-6, hc_post_scale=2.0,
                       sinkhorn_eps=1e-6, num_sinkhorn_iters=20, rmsnorm_eps=eps, rmsnorm_scale=1.0)
        independent_inputs = {name: value.cpu() for name, value in inputs.items()}
        independent_inputs.update(shifted_prev_mix=previous.pre_mix.unsqueeze(-1).cpu(),
                                  post_mix=previous.post_mix.cpu(), comb_res_mix=previous.residual_mix.cpu(),
                                  fn=target.attn_hc.fn.cpu(), mix_bases=target.attn_hc.base.cpu(),
                                  mix_scales=target.attn_hc.scale.cpu(), rmsnorm_weight=target.attn_norm.weight.cpu())
        reference = independent_transition_reference(torch, independent_inputs, scalars)
        legacy_post = ops.mhc(inputs["residual"], inputs["x"], phase="post", state=previous).hidden
        legacy = ops.mhc(legacy_post, None, phase="pre", state=previous,
                         parameters=blocks[2].attn_hc.parameters_map)
        legacy_outputs = dict(zip(OUTPUTS, (legacy_post, legacy.pre_mix.unsqueeze(-1), legacy.post_mix,
                                            legacy.residual_mix, blocks[2].attn_norm(legacy.hidden))))
        for name, value in actual.items():
            self.assertFalse(any(torch._C._is_alias_of(value, source) for source in inputs.values()), name)
            for oracle, expected in (("CPU", reference[name]), ("legacy", legacy_outputs[name])):
                self.assertQualified(value, expected, f"M{rows}.eps{eps}.{name}.{oracle}")
        self.assertIsNone(blocks[2].attn_hc.state.hidden)
        self.assertIs(blocks[1].ffn_hc.state, previous)
        self.assertEqual(state._cross_mhc_evidence["consumed"], 1)
        self.assertEqual(ops.mhc_execution_stats()["calls"], {"prefill": 1})
        for name, value in inputs.items():
            self.assertTrue(torch.equal(value, original[name]), name)
        for name, value in target.attn_hc.parameters_map.items():
            self.assertTrue(torch.equal(value, target_parameters[name]), name)
            self.assertFalse(torch.equal(value, blocks[1].ffn_hc.parameters_map[name]), name)
            self.assertFalse(torch.equal(value, blocks[3].attn_hc.parameters_map[name]), name)
        self.assertTrue(torch.equal(target.attn_norm.weight, target_gamma))
        self.assertFalse(torch.equal(target_gamma, blocks[1].ffn_norm.weight))
        self.assertFalse(torch.equal(target_gamma, blocks[3].attn_norm.weight))
        self.assertTrue(all(all(item.values()) for item in _canaries(torch, buffers).values()))

    def test_full_lookahead_compact_and_epsilon_existing_numeric_contract(self):
        for eps in (1e-20, 1e-6):
            for rows in (4096, 4097, 140):
                with self.subTest(rows=rows, eps=eps):
                    self.component(rows, eps)
        self.report("component")

    def test_real_block_hc_chain_engram_barrier_and_head(self):
        inputs, _, _, _ = _make_case(torch, 140, "cuda:0", "normal")
        old_ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=1e-20))
        with patch.dict(os.environ, {ENV: "0"}):
            old_state, old_blocks = _fixture(inputs, old_ops, enabled="0")
            old, old_head = _run_chain(old_state, old_blocks, inputs["residual"])
        old_fields = _mix_fields(old_blocks)
        ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=1e-20))
        state, blocks = _fixture(inputs, ops)
        actual, head = _run_chain(state, blocks, inputs["residual"])
        self.assertQualified(actual, old, "actual Block residual")
        self.assertQualified(head, old_head, "actual delayed head")
        for name, value in _mix_fields(blocks).items():
            self.assertQualified(value, old_fields[name], name)
        self.assertEqual(state._cross_mhc_evidence["edges"], [
            {"src": 1, "dst": 2, "rows": 140, "compacted": False},
            {"src": 2, "dst": 3, "rows": 140, "compacted": False},
        ])
        self.assertEqual([block.attn_norm.calls for block in blocks], [1, 1, 0, 0])
        self.assertEqual(old_ops.mhc_execution_stats()["calls"], {"prefill": 4})
        self.assertEqual(ops.mhc_execution_stats()["calls"], {"prefill": 6})
        self.assertEqual([block.ffn_norm.calls for block in blocks], [0] * 4)
        self.report("block-head")

    def test_actual_split_l20_has_no_pending_full_ticket_transport(self):
        for rows, split in ((4096, True), (4097, True), (4096, False)):
            with self.subTest(rows=rows, split=split):
                inputs, _, _, _ = _make_case(torch, rows, "cuda:0", "normal")
                old_ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=1e-20))
                with patch.dict(os.environ, {ENV: "0"}):
                    old_state, old_blocks = _fixture(inputs, old_ops, count=22, enabled="0")
                    _install_plan(old_state)
                    old, old_head = _run_chain(old_state, old_blocks, inputs["residual"], split=split)
                ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=1e-20))
                state, blocks = _fixture(inputs, ops, count=22)
                plan = _install_plan(state)
                actual, head = _run_chain(state, blocks, inputs["residual"], split=split)
                self.assertEqual(actual.shape, (140, 4, 5120))
                self.assertQualified(actual, old, f"M{rows}.split{split}.residual")
                self.assertQualified(head, old_head, f"M{rows}.split{split}.head")
                for which in ("attn", "ffn"):
                    mix = getattr(blocks[21], f"{which}_hc").state
                    expected = getattr(old_blocks[21], f"{which}_hc").state
                    for field in ("pre_mix", "post_mix", "residual_mix"):
                        self.assertQualified(getattr(mix, field), getattr(expected, field), f"L21.{which}.{field}")
                evidence = state._cross_mhc_evidence
                expected_count = 19 if split else 18
                self.assertEqual(evidence["published"], expected_count)
                self.assertEqual(evidence["consumed"], expected_count)
                self.assertEqual(evidence["fallbacks"], {})
                self.assertIsNone(state._cross_mhc_ticket)
                self.assertEqual(sum(not edge["compacted"] for edge in evidence["edges"]), 18)
                self.assertEqual(sum(edge["compacted"] for edge in evidence["edges"]), int(split))
                self.assertEqual(blocks[20].attn.rows, [rows])
                self.assertEqual(blocks[20].ffn.rows, [140 if split else rows])
                self.assertEqual(blocks[21].attn.rows, [140])
                self.assertTrue(torch.equal(state.compact_ids, plan.exchange.compact(state.inputs.input_ids)))
                self.assertEqual(old_ops.mhc_execution_stats()["calls"], {"prefill": 22})
                self.assertEqual(ops.mhc_execution_stats()["calls"], {"prefill": 22 + expected_count})
                self.assertEqual(sum(block.attn_norm.calls for block in blocks), 3 if split else 4)
        self.report("split-l20")

    def test_actual_native_profile_markers_and_unsupported_geometry_fallback(self):
        inputs, _, _, _ = _make_case(torch, 4096, "cuda:0", "normal")
        ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=1e-20))
        state, blocks = _fixture(inputs, ops)
        # Warm the unchanged provider, then profile precisely one producer/consumer.
        warm, _ = _publish(state, blocks, inputs)
        self.assertIsNotNone(_consume(state, blocks[2], warm))
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            actual, _ = _publish(state, blocks, inputs)
            self.assertIsNotNone(_consume(state, blocks[2], actual))
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as temporary:
            directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", temporary)
            path = Path(directory) / "local-prefill-cross-mhc.trace.json"
            profile.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
        self.assertEqual(sum("ppu_mega_mhc_single" in name for name in kernels), 1, kernels)
        self.assertFalse(any(any(part in name for part in (
            "_hc_mix_stats_partial", "_hc_mix_stats_reduce", "_mhc_post", "_mhc_combine", "sinkhorn",
        )) for name in kernels), kernels)
        markers = [event["name"] for event in events if event.get("cat") == "user_annotation"]
        self.assertEqual(markers.count("dsv41.ppu.cross_mhc.L01_to_L02.produce"), 1)
        self.assertEqual(markers.count("dsv41.ppu.cross_mhc.L02.consume"), 1)
        state, blocks = _fixture(inputs, ops)
        blocks[2].attn_hc.fn = blocks[2].attn_hc.fn[:, :4096].contiguous()
        actual, mix = _publish(state, blocks, inputs)
        self.assertIsNone(state._cross_mhc_ticket)
        expected = ops.mhc(inputs["residual"], inputs["x"], phase="post", state=mix).hidden
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(state._cross_mhc_evidence["published"], 0)
        self.report("profile-fallback")

    def report(self, label):
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if directory:
            (Path(directory) / f"local-prefill-cross-mhc-{label}.json").write_text(json.dumps({
                "contract": "existing per-row/global SSE-energy strictly<5e-5",
                "cases": self.reports,
            }, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
