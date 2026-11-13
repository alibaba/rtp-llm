"""Local L20 full KV producer followed by retained-query/FFN execution.

Transport/cache codecs, projections and attention scoring below are explicit
fixtures. Production _body, global ratio-2 producer, workspace construction,
block callback, HC and state compaction execute unchanged. Device cases also
exercise the qualified native HC transition; they are not a model benchmark.
"""

from contextlib import ExitStack, nullcontext
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.block import Block
from rtp_llm.models_py.modules.dsv41.ppu import attention_layer
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.block import (
    LOCAL_PREFILL_SPLIT_L20_ENV,
    SharedPPUBlock,
    _local_prefill_split_l20_enabled,
)
from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedHCUnit, PPUSharedRMSNorm
from rtp_llm.models_py.modules.dsv41.ppu.mhc import MHCConfig
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_mhc import _compact_exchange
from rtp_llm.models_py.modules.dsv41.ppu.test.test_mhc_mega_probe import _make_case, _qualification_metric
from rtp_llm.utils.model_weight import W


ENV = LOCAL_PREFILL_SPLIT_L20_ENV
PREFIX = 126976


def _state(device="cpu", rows=4096):
    v4 = NS(layers=[], _propagate_cp_ctx=lambda context: None)
    runtime = NS(model=NS(v4=v4, _is_speculative=False), policy=SharedPrefillPolicy(mode="compact"))
    inputs = NS(input_ids=torch.arange(rows, device=device, dtype=torch.int64))
    state = SharedForwardState(runtime, inputs, NS(cu_seqlens=torch.tensor([0, rows], dtype=torch.int32)),
                               NS(sharded=False), prefill=True, idle=False, prefixes=(PREFIX,), lengths=(rows,))
    state.original_context = state.identity_context()
    query, exchange = _compact_exchange(rows, torch.device(device))
    query.swa_replay_start = rows - 140
    state.plan = NS(context=query, exchange=exchange,
                    cu_seqlens=torch.tensor([0, 140], device=device, dtype=torch.int32),
                    project_indexer_weights=torch.nn.functional.linear,
                    candidate_publication=lambda shared: nullcontext())
    return state


def _bare_block(state):
    block = SharedPPUBlock.__new__(SharedPPUBlock)
    nn.Module.__init__(block)
    block.layer_id, block.engram, block._mhc_phase = 20, None, "unbound"
    block.attn = NS(runtime=NS(current=state, model=state.runtime.model), is_kv_source=True,
                    is_index_source=True)
    state.before_block = lambda block, *args: args
    state.publication = lambda layer: nullcontext()
    return block


class LocalSplitL20Routing(unittest.TestCase):
    @torch.inference_mode()
    def test_default_off_local_admission_and_cp8_switch_remain_independent(self):
        for local, cp8, sharded, cp8_switch, expected in (
            ("0", False, False, "1", False), ("1", False, False, "0", True),
            ("0", True, True, "1", True), ("1", True, True, "0", False),
        ):
            state = _state()
            state.cache.sharded = sharded
            if sharded:
                state.context = NS(cp_size=8, kv_cache_sharded=True, gather_restore_positions=None)
            block = _bare_block(state)
            block._forward_compact_l20 = Mock(return_value="split")
            with patch.dict(os.environ, {ENV: local, "RTP_V41_PPU_CP8_YOCO_SPLIT_L20": cp8_switch}), patch(
                "rtp_llm.models_py.modules.dsv41.ppu.block.cp8_yoco_fast_prefill_enabled", return_value=cp8,
            ), patch.object(attention_layer, "optimized_diagnostics_enabled", return_value=False), patch.object(
                Block, "forward", return_value="ordinary",
            ):
                actual = block.forward(None, torch.ones(4096, dtype=torch.int64), None, None)
            self.assertEqual(actual, "split" if expected else "ordinary")
            self.assertEqual(block._forward_compact_l20.call_count, int(expected))
        with patch.dict(os.environ):
            os.environ.pop(ENV, None)
            self.assertFalse(_local_prefill_split_l20_enabled(_state()))

    def test_ineligible_phase_policy_layout_and_model_keep_fallback(self):
        with patch.dict(os.environ, {ENV: "1"}):
            self.assertFalse(_local_prefill_split_l20_enabled(_state()))  # grad enabled
            with torch.inference_mode():
                self.assertFalse(_local_prefill_split_l20_enabled(None))
                for flag in ("idle", "compacted", "is_graph", "is_graph_commit", "verify", "proposal"):
                    state = _state()
                    setattr(state, flag, True)
                    self.assertFalse(_local_prefill_split_l20_enabled(state), flag)
                for field, value in (("prefill", False), ("plan", None), ("cache", NS(sharded=True))):
                    state = _state()
                    setattr(state, field, value)
                    self.assertFalse(_local_prefill_split_l20_enabled(state), field)
                for context in (NS(cp_size=8, kv_cache_sharded=False),
                                NS(cp_size=1, kv_cache_sharded=True),
                                NS(cp_size=1, kv_cache_sharded=False, gather_restore_positions=object())):
                    state = _state()
                    state.context = context
                    self.assertFalse(_local_prefill_split_l20_enabled(state))
                for policy in (SharedPrefillPolicy(mode="full"), SharedPrefillPolicy(mode="exact"),
                               SharedPrefillPolicy(mode="compact", max_rollback_tokens=4),
                               SharedPrefillPolicy(mode="compact", encoder_replay=True)):
                    state = _state()
                    state.runtime.policy = policy
                    self.assertFalse(_local_prefill_split_l20_enabled(state))
                state = _state()
                state.runtime.model._is_speculative = True
                self.assertFalse(_local_prefill_split_l20_enabled(state))

    @torch.inference_mode()
    def test_missing_producer_routing_diagnostics_and_non_l20_do_not_split(self):
        for change in ("kv", "index", "ids", "routing", "diagnostic", "layer"):
            state = _state()
            block = _bare_block(state)
            block._forward_compact_l20 = Mock()
            if change == "kv":
                block.attn.is_kv_source = False
            if change == "index":
                block.attn.is_index_source = False
            if change == "layer":
                block.layer_id = 19
            with patch.dict(os.environ, {ENV: "1"}), patch.object(
                attention_layer, "optimized_diagnostics_enabled", return_value=change == "diagnostic",
            ), patch.object(Block, "forward", return_value="ordinary"):
                output = block.forward(None, None if change == "ids" else torch.ones(4096), None, None,
                                       **({"routing_metadata": object()} if change == "routing" else {}))
            self.assertEqual(output, "ordinary")
            block._forward_compact_l20.assert_not_called()


class _Cache:
    """Absolute-position synthetic banks retain actual producer tensor writes."""
    sharded = False

    def __init__(self, geometry, rows, device, receipt):
        self.geometry, self.lengths, self.receipt = geometry, (PREFIX + rows,), receipt
        self.swa = torch.zeros((PREFIX + rows, 8), device=device, dtype=torch.bfloat16)
        self.global_rows = torch.zeros(((PREFIX + rows) // 2, 8), device=device, dtype=torch.bfloat16)
        self.index = torch.zeros_like(self.global_rows)
        self.csa = torch.zeros((PREFIX + rows, 16), device=device, dtype=torch.float32)
        self.pools = {(20, "global_kv_2"): NS(tensor=self.global_rows), (20, "indexer_kv"): NS(tensor=self.index)}
        # The ordinary producer uses precomputed physical entry slots, not
        # the request-local write optimization reserved for SharedNativeCache.
        # These synthetic banks have identity physical entry addressing.
        entries = torch.arange(self.global_rows.shape[0], device=device, dtype=torch.int64)
        self.slots = {0: NS(global_slots={20: entries}, indexer_slots={20: entries})}

    def stage_swa(self, layer, positions, requests, values, prefixes, **kwargs):
        self.receipt.events.append(("swa", values.shape[0]))
        self.receipt.swa_stage = kwargs
        self.swa[positions.long()] = values

    def insert_request_rows(self, layer, tag, request, positions, values, **kwargs):
        self.receipt.events.append((tag, values.shape[0]))
        self.receipt.writes[tag] = (positions.clone(), values.clone())
        if tag == "csa_state":
            self.csa[positions.long()] = values
        else:
            (self.global_rows if tag == "global_kv_2" else self.index)[positions.long() // 2] = values

    def slots_at_positions(self, layer, tag, request, positions):
        if layer != 20 or request != 0 or tag not in ("global_kv_2", "indexer_kv", "csa_state"):
            raise AssertionError("fixture slot request is outside its producer banks")
        positions = positions.to(device=self.swa.device, dtype=torch.int64)
        return positions if tag == "csa_state" else positions // 2

    def insert_rows(self, layer, tag, slots, values):
        if layer != 20 or tag not in ("global_kv_2", "indexer_kv", "csa_state"):
            raise AssertionError("fixture write is outside its producer banks")
        if slots.ndim != 1 or slots.numel() != values.shape[0]:
            raise AssertionError("producer slots and written rows disagree")
        # Preserve the raw absolute-position receipts used by the byte and
        # full-producer-order assertions; compressed banks address pairs.
        positions = slots if tag == "csa_state" else slots * 2
        self.insert_request_rows(layer, tag, 0, positions, values)

    def read_csa(self, layer, request, positions):
        slots = self.slots_at_positions(layer, "csa_state", request, positions)
        return self.csa[slots].contiguous()

    def prepare_source(self, source):
        self.receipt.events.append(("prepare", source))

    def request_length(self, request):
        return self.lengths[request]

    def descriptors(self, layer):
        return {"swa_kv": None, "global_kv_2": None}

    def read_source(self, source, request, *, global_entries):
        return (self.global_rows[:global_entries], self.index[:global_entries])

    def read_swa(self, layer, request, positions):
        self.receipt.reads.append(positions.clone())
        return self.swa[positions.long()].contiguous()

    def release_swa(self, layer):
        self.receipt.events.append(("release", layer))


def _component(device, rows, *, native=False, eps=1e-6):
    state = _state(device, rows)
    width = 5120 if native else 32
    receipt = NS(events=[], writes={}, reads=[], ffn_rows=[], carry=None, workspace=None)
    g = NS(hidden_size=width, hc_mult=4, hc_sinkhorn_iters=20, norm_eps=eps, hc_eps=1e-6,
           n_heads=1, head_dim=8, index_n_heads=1, index_head_dim=8, index_topk=1,
           window=128, o_groups=1, o_lora_rank=8, ratio=lambda layer: 2,
           kv_source_for=lambda layer: 20, sm_scale=lambda: 8 ** -0.5)
    cache = state.cache = _Cache(g, rows, device, receipt)
    attn = attention_layer.SharedPPUAttention.__new__(attention_layer.SharedPPUAttention)
    nn.Module.__init__(attn)
    attn.geometry, attn.layer_id, attn.kv_source_layer_id = g, 20, 20
    attn.compress_ratio, attn.is_kv_source, attn.is_index_source = 2, True, True
    attn.index_weights = torch.zeros((1, width), device=device, dtype=torch.bfloat16)
    attn.index_wq = nn.Identity()
    attn.index_wk = nn.Linear(8, 8, bias=False, device=device, dtype=torch.bfloat16)
    attn.index_wk.weight.data.copy_(torch.eye(8, device=device, dtype=torch.bfloat16))
    attn.index_k_norm = attn.global_norm = torch.ones(8, device=device, dtype=torch.float32)
    attn.global_wkv = lambda value: value[:, :8].contiguous()
    attn.global_wgate = lambda value: torch.zeros((value.shape[0], 8), device=device)
    attn.project_query_kv = lambda value, **kwargs: (
        value[:, :8].contiguous(), value[:, :8].contiguous(), value[:, :8].contiguous(), None,
    )
    attn.wo_a = nn.Linear(8, 8, bias=False, device=device, dtype=torch.bfloat16)
    attn.wo_a.weight.data.copy_(torch.eye(8, device=device, dtype=torch.bfloat16))
    attn.wo_b = lambda value: value.repeat(1, width // 8).contiguous()
    attn.attn_sink = torch.zeros(1, device=device)
    attn.ops = NS(encode_kv=lambda value, **kwargs: value,
                  decode_kv=lambda value, *, out, **kwargs: out.copy_(value))
    freqs = torch.ones((PREFIX + rows, 1), device=device, dtype=torch.complex64)
    attn.runtime = NS(current=state, policy=state.runtime.policy, freqs_for=lambda layer: freqs, qkv_streams=None)
    attn._cp_ctx = None
    state.v4._propagate_cp_ctx = lambda context: setattr(attn, "_cp_ctx", context)
    block = SharedPPUBlock.__new__(SharedPPUBlock)
    nn.Module.__init__(block)
    block.layer_id, block.engram, block._mhc_phase, block.attn = 20, None, "prefill", attn
    ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=eps))
    block.ops = ops
    if native:
        inputs, _, _, _ = _make_case(torch, rows, device, "normal")
        residual = inputs["residual"]
        parameters = {"fn": inputs["fn"], "base": inputs["mix_bases"], "scale": inputs["mix_scales"]}
    else:
        generator = torch.Generator().manual_seed(117)
        residual = torch.randn((rows, 4, width), generator=generator).to(device=device, dtype=torch.bfloat16)
        parameters = {"fn": torch.randn((24, width * 4), generator=generator).to(device) * 0.01,
                      "base": torch.randn(24, generator=generator).to(device) * 0.1,
                      "scale": torch.tensor([0.1, 0.1, 0.1], device=device)}
    tensors = {getattr(W, f"v4_hc_{which}_{part}"): value
               for which in ("attn", "ffn") for part, value in parameters.items()}
    views = NS(geometry=g, tensor=lambda layer, key: tensors[key])
    block.attn_hc = PPUSharedHCUnit(views, 20, "attn", ops)
    block.ffn_hc = PPUSharedHCUnit(views, 20, "ffn", ops)
    block.ffn_hc.set_previous(block.attn_hc)
    gamma = torch.ones(width, device=device, dtype=torch.bfloat16)
    block.attn_norm, block.ffn_norm = PPUSharedRMSNorm(gamma, eps), PPUSharedRMSNorm(gamma, eps)
    # A real preceding delayed FFN state, not a new attention mix.
    previous = PPUSharedHCUnit(views, 19, "ffn", ops)
    previous.pre(residual)
    block.attn_hc.set_previous(previous)

    def ffn(value, ids, **kwargs):
        receipt.ffn_rows.append((value.shape[0], ids.clone()))
        return (value * 0.125).contiguous()

    block.ffn = ffn
    original_select = lambda value, qr, context, cache, shared, **kwargs: torch.zeros(
        (value.shape[0], 1), device=device, dtype=torch.int32,
    )

    def select(*args, **kwargs):
        value = args[0]
        receipt.events.append(("select", value.shape[0]))
        required = {"global_kv_2": rows // 2, "indexer_kv": rows // 2, "csa_state": rows}
        for tag, count in required.items():
            if receipt.writes[tag][1].shape[0] != count:
                raise AssertionError(f"incomplete {tag} producer before selection")
        return original_select(*args, **kwargs)

    attn._select = select
    original_compact = state.compact_l20_queries

    def compact(hidden, ids):
        receipt.events.append(("callback", hidden.shape[0]))
        receipt.carry = block.attn_hc.state
        return original_compact(hidden, ids)

    state.compact_l20_queries = compact
    return block, state, residual, previous, receipt


class _SplitCases:
    def _run(self, rows=4096, eps=1e-6, *, backend="mega"):
        environment = {ENV: "1", "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE": "1",
                       "RTP_V41_PPU_LOCAL_PREFILL_AIU_PRENORM": "0", "RTP_V41_PPU_MHC_BACKEND": backend,
                       "RTP_V41_PPU_QUANT_MODE": "mxfp4_fp8_ptpc", "RTP_V41_PPU_PTPC_FUSION": "1",
                       attention_layer.FUSION_ENV: "0", attention_layer.STREAMS_ENV: "0"}
        results = []
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, environment))
            stack.enter_context(patch.object(attention_layer, "optimized_diagnostics_enabled", return_value=False))
            stack.enter_context(patch.object(attention_layer, "prepare_packed_attention", return_value=(None, "disabled")))
            stack.enter_context(patch.object(attention_layer, "flash_mla_sparse_fwd_ppu",
                                            side_effect=lambda q, *args, **kwargs: (q.clone(), None, None)))
            for split in (False, True):
                block, state, residual, previous, receipt = _component(self.device, rows, native=self.native, eps=eps)
                original = residual.clone()
                if split:
                    output = block._forward_compact_l20(residual, state.inputs.input_ids, state, None)
                    self.assertTrue(state.compacted)
                    self.assertEqual(receipt.ffn_rows[0][0], 140)
                    self.assertTrue(torch.equal(receipt.ffn_rows[0][1], state.inputs.input_ids[-140:]))
                    self.assertIsNone(receipt.carry.hidden)
                    self.assertEqual(receipt.carry.pre_mix.shape, (140, 4))
                    self.assertEqual(receipt.carry.post_mix.shape, (140, 4, 1))
                    self.assertEqual(receipt.carry.residual_mix.shape, (140, 4, 4))
                    self.assertTrue(bool(state.valid_output_rows[-1]))
                    self.assertEqual(int(state.valid_output_rows.sum()), 140)
                    carry = block.ffn_hc.state
                    next_values = state.before_block(NS(layer_id=21), output, state.inputs.input_ids, None, None)
                    self.assertEqual(next_values[0].shape[0], 140)
                    self.assertIs(block.ffn_hc.state, carry, "L21 must not compact L20 carry twice")
                else:
                    hidden, post, comb = block.attn_hc.pre(residual)
                    attn_out = block.attn(block.attn_norm(hidden))
                    transition = block._try_mega_mhc(attn_out, residual, post, comb)
                    if transition is None:
                        updated = block.attn_hc.post(attn_out, residual, post, comb)
                        hidden, post, comb = block.ffn_hc.pre(updated)
                        hidden = block.ffn_norm(hidden)
                    else:
                        updated, hidden, post, comb = transition
                    output = block.ffn_hc.post(block.ffn(hidden, state.inputs.input_ids), updated, post, comb)
                self.assertTrue(torch.equal(residual, original))
                self.assertEqual(receipt.events[0], ("swa", rows))
                self.assertTrue(receipt.swa_stage["retain_prefix_tail"])
                names = [name for name, _ in receipt.events]
                if split:
                    for tag in ("global_kv_2", "indexer_kv", "csa_state"):
                        self.assertLess(names.index(tag), names.index("callback"))
                    self.assertLess(names.index("callback"), names.index("select"))
                    self.assertEqual(receipt.events[names.index("select")], ("select", 140))
                    # L20 still reads the complete current SWA lookback, before
                    # later layers adopt the retained-only replay start.
                    self.assertEqual(int(receipt.reads[0][0]), PREFIX + rows - 140 - 127)
                    self.assertEqual(int(receipt.reads[0][-1]), PREFIX + rows - 1)
                    workspace = state.shared["prefill_kv_workspace"][2]
                    indices, lengths = workspace.index_plan
                    self.assertTrue(bool((lengths == 129).all()), "one global +128 current SWA keys")
                    sbase = int(workspace.row_meta[0, 2])
                    self.assertTrue(torch.equal(indices[0, 0, 1:129], torch.arange(
                        sbase, sbase + 128, device=indices.device, dtype=indices.dtype,
                    )))
                result = {"output": output, "attn_pre": block.attn_hc.state.pre_mix,
                          "attn_post": block.attn_hc.state.post_mix, "attn_comb": block.attn_hc.state.residual_mix,
                          "ffn_pre": block.ffn_hc.state.pre_mix, "ffn_post": block.ffn_hc.state.post_mix,
                          "ffn_comb": block.ffn_hc.state.residual_mix,
                          "ffn_hidden": block.ffn_hc.state.hidden,
                          "writes": receipt.writes, "events": receipt.events}
                self.assertEqual(previous.state.pre_mix.shape[0], rows)
                if self.native:
                    self.assertEqual(block.ops.mhc_execution_stats()["calls"],
                                     {"prefill": 1} if backend == "mega" else {})
                results.append(result)
        control, candidate = results
        metrics = {}
        for name in ("output", "attn_pre", "attn_post", "attn_comb", "ffn_pre", "ffn_post", "ffn_comb", "ffn_hidden"):
            if control[name] is None:
                self.assertIsNone(candidate[name])
                continue
            expected = control[name][-140:].contiguous()
            metric = _qualification_metric(torch, candidate[name], expected)
            self.assertEqual(metric["qualified_status"], "PASS", (name, metric))
            metrics[name] = metric
        for tag in ("global_kv_2", "indexer_kv", "csa_state"):
            for actual, expected in zip(candidate["writes"][tag], control["writes"][tag]):
                self.assertTrue(torch.equal(actual, expected), tag)
        self.reports.append(dict(rows=rows, eps=eps, backend=backend,
                                 events=candidate["events"], outputs=metrics))

    @torch.inference_mode()
    def test_full_producer_then_compact_query_ffn_carry_and_swa_bounds(self):
        for rows in (4096, 4097):
            for eps in (1e-20, 1e-6):
                with self.subTest(rows=rows, eps=eps):
                    self._run(rows, eps)
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if directory:
            (Path(directory) / f"local-prefill-split-l20-{self.device}.json").write_text(json.dumps(
                {"contract": "unchanged per-row/global SSE-energy strictly<5e-5", "cases": self.reports}, indent=2,
            ) + "\n")


class LocalSplitL20CPU(_SplitCases, unittest.TestCase):
    def setUp(self):
        self.device, self.native, self.reports = "cpu", False, []


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalSplitL20Device(_SplitCases, unittest.TestCase):
    def setUp(self):
        self.assertTrue(is_ppu(), "native carry qualification requires a ZW PPU")
        self.device, self.native, self.reports = "cuda:0", True, []

    @torch.inference_mode()
    def test_explicit_legacy_native_fallback_keeps_selected_carry(self):
        self._run(backend="legacy")


if __name__ == "__main__":
    unittest.main()
