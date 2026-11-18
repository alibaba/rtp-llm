"""CP8 full/compact row epochs for the existing rowwise Mega-mHC seam.

CPU branch fixtures retain real Block/HC/carry dispatch and use a declared
synthetic row transport. Native component cases qualify the actual4096/144
row transition against the existing independent authority. The explicit
WORLD8 case uses the actual CP transport and native HC; its branches are
deterministic fixtures, so its37-edge receipt is not a model benchmark.
"""

from contextlib import ExitStack, nullcontext
from datetime import timedelta
from itertools import accumulate
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import layers as layer_module
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.local_prefill_qkv import CP8QueryPlanBirth
from rtp_llm.models_py.modules.dsv41.ppu.mhc import MHCConfig
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_l20_query_projection import _context
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_cross_layer_mhc import (
    BASE_ENV, ENV, _CPUOps, _consume, _fixture, _mix_fields, _producer, _publish, _run_chain,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_mhc_mega_probe import (
    OUTPUTS, _make_case, _qualification_metric, independent_transition_reference,
)
from rtp_llm.models_py.modules.dsv41.prefill.ced import (
    CEDPlan, _bounded_replay_selected, _host_query_layout,
)


CP_ENV = dict(BASE_ENV, RTP_V41_PPU_CP8_YOCO_FAST_PREFILL="1",
              RTP_V41_PPU_CP8_YOCO_SPLIT_L20="1", RTP_V41_PPU_SPLIT_L20="1",
              RTP_V41_PPU_AIU_PRENORM="0", RTP_V41_PPU_CP8_REUSE_LOCAL_ROWS="0",
              RTP_V41_PPU_CP8_FORWARD_MEGA_CAPACITY="0")


def _report(name, report):
    directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if directory:
        (Path(directory) / name).write_text(json.dumps(report, indent=2) + "\n")
    # Full per-row metrics belong to the test artifact, not several MB of
    # console output. Qualification above still checks every individual row.
    summary = dict(report)
    summary["cases"] = [{**case, "metric": {key: value for key, value in case["metric"].items()
                                           if key != "row_metrics"}} for case in report["cases"]]
    print(json.dumps(summary, sort_keys=True))


def _state(original, ids, *, runtime=None, enabled="1"):
    lengths, prefixes = original.input_lengths_global_host, original.prefix_lengths_host
    if runtime is None:
        model = NS(_is_decode_role=False, _is_speculative=False, _prefill_cp_size=8,
                   parallelism_config=NS(tp_size=8),
                   v4=NS(layers=[], _propagate_cp_ctx=lambda context: None, capture_aux_hidden_layer_ids=()))
        runtime = NS(model=model, policy=SharedPrefillPolicy(mode="compact"), current=None)
    runtime.model._prefill_cp_size = runtime.model.parallelism_config.tp_size = 8
    cache = SharedNativeCache.__new__(SharedNativeCache)
    cache.sharded, cache._is_prefill = True, True
    cache.lengths = tuple(p + n for p, n in zip(prefixes, lengths))
    # The branch fixture has no KV payload. Keep the actual bind_context seam
    # without preparing unrelated SWA/MoE workspaces.
    cache.bind_swa_fresh_lengths = lambda *args, **kwargs: None
    cache.shared_prefix_stripes = object()
    inputs = NS(input_ids=ids, combo_position_ids=original.global_positions)
    attn = NS(cu_seqlens=torch.tensor((0, *accumulate(original.chunk_lengths_per_req)),
                                    dtype=torch.int32, device=ids.device),
              input_lengths=torch.tensor(original.chunk_lengths_per_req, dtype=torch.int32),
              prefix_lengths=torch.tensor(prefixes, dtype=torch.int32), is_cuda_graph=False)
    with (nullcontext() if os.environ.get(ENV) == enabled
          else patch.dict(os.environ, {ENV: enabled})):
        state = SharedForwardState(runtime, inputs, attn, cache, prefill=True, idle=False,
                                   prefixes=prefixes, lengths=lengths)
    runtime.current = state
    runtime._cross_mhc_evidence = state._cross_mhc_evidence
    # bind_context's native cache workspaces are outside this HC fixture.
    with patch("rtp_llm.models_py.modules.dsv41.ppu.layers.cp8_yoco_fast_prefill_enabled", return_value=False):
        state.bind_context(original)
    runtime.model.v4._propagate_cp_ctx(original)
    return state


def _plan(state, *, group=None, synthetic_transport=False):
    with torch.inference_mode(False), torch.no_grad():
        starts = tuple(max(n - 140, 0) for n in state.lengths)
        selected = _bounded_replay_selected(state.lengths, starts)
        compact, exchange, groups = _host_query_layout(state.original_context, selected, group,
                                                       keep_candidate_rows=True)
        compact.swa_replay_starts_host = starts
        plan = CEDPlan(state.original_context, compact,
                       torch.tensor((0, *accumulate(compact.chunk_lengths_per_req)), dtype=torch.int32,
                                    device=state.inputs.input_ids.device), exchange, groups, ())
        plan._ppu_cp8_query_birth = CP8QueryPlanBirth(plan)
    if synthetic_transport:
        # This CPU seam fixture tests state/owner/order only. The separately
        # gated WORLD8 test retains the real all_to_all_single implementation.
        def compact_rows(value):
            indices = torch.arange(compact.chunk_length, device=value.device) % value.shape[0]
            selected_rows = value.index_select(0, indices)
            shape = (compact.chunk_length,) + (1,) * (selected_rows.ndim - 1)
            return selected_rows.masked_fill(~compact.local_is_real.reshape(shape), 0)
        exchange.compact = compact_rows
    return plan


def _install_plan(state, *, group=None, synthetic_transport=False):
    plan = _plan(state, group=group, synthetic_transport=synthetic_transport)

    def prepare():
        state.plan, state.plan_reason = plan, "compact140"

    state.prepare_plan = prepare
    return plan


def _activate(state, plan):
    state.plan, state.plan_reason = plan, "compact140"
    state.compact_ids = torch.zeros(plan.context.chunk_length, dtype=state.inputs.input_ids.dtype,
                                   device=state.inputs.input_ids.device)
    state._activate_compact(torch.empty(plan.context.chunk_length, 4, 5120,
                                       dtype=torch.bfloat16, device=state.inputs.input_ids.device))


def _component(inputs, ops, *, rank=0, lengths=(144,), prefixes=(126976,), count=4, enabled="1"):
    old, blocks = _fixture(inputs, ops, count=count, birth=False, enabled=enabled)
    original, ids = _context(rank, lengths, prefixes, inputs["x"].device)
    if original.chunk_length != inputs["x"].shape[0]:
        raise AssertionError("CP fixture requires the actual rank-local full row count")
    state = _state(original, ids, runtime=old.runtime, enabled=enabled)
    return state, blocks


class CP8CrossLayerMHCContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, CP_ENV))
        self.stack.enter_context(patch.object(layer_module, "is_ppu", return_value=True))
        self.stack.enter_context(patch("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", return_value=True))
        self.stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch(
            "rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata._upload",
            side_effect=lambda values, device, dtype: torch.tensor(values, device=device, dtype=dtype)))
        self.stack.enter_context(patch(
            "rtp_llm.models_py.modules.dsv41.ppu.attention_layer.optimized_diagnostics_enabled", return_value=False))
        self.stack.enter_context(patch.object(layer_module, "rms_norm", side_effect=lambda x, w, eps:
            (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
             * w.float()).to(x.dtype)))

    def test_full_birth_is_cp_specific_and_preserves_original_readonly_contract(self):
        for inference in (False, True):
            with torch.inference_mode(inference):
                original, ids = _context(lengths=(256, 272), prefixes=(17, 49))
            state = _state(original, ids)
            self.assertTrue(state._cross_mhc_phase())
            self.assertFalse(layer_module._local_prefill_business_phase(state))
            self.assertFalse(state._mega_all_live_phase(), "DP/local mask guard must remain local")
            epoch = state._cross_mhc_epoch(original.chunk_length)
            self.assertIsNotNone(epoch)
            state.plan = object()  # L19 future plan must not change the full-row epoch.
            self.assertEqual(state._cross_mhc_epoch(original.chunk_length)[0], epoch[0])
            original.global_positions = original.global_positions.clone()
            self.assertIsNone(state._cross_mhc_epoch(original.chunk_length))
            state.bind_context(original)
            self.assertTrue(state._cp_full_rows_invalid, "consumers cannot recertify a changed birth")

    def test_full_and_compact_masks_follow_actual_rank_padding_without_device_scan(self):
        for rank in range(8):
            original, ids = _context(rank, (4096,) * 8, (126976,) * 8)
            state = _state(original, ids)
            self.assertIsNone(state.mega_attention_valid_mask(original, original.local_is_real, 4096))
            plan = _plan(state)
            _activate(state, plan)
            context, mask = state.context, state.context.local_is_real
            result = state.mega_attention_valid_mask(context, mask, 144)
            real = sum(plan.exchange.receive_sizes)
            self.assertEqual(real, 112 if rank == 0 else 144)
            if rank == 0:
                self.assertIs(result, mask)
            else:
                self.assertIsNone(result)
            proof = state._cross_mhc_evidence["cp_row_proof"]
            self.assertEqual(proof["compact_real_rows_local"], real)
            self.assertEqual(proof["compact_all_live"], rank != 0)
            mask.logical_not_()
            self.assertIs(state.mega_attention_valid_mask(context, mask, 144), mask)
            self.assertIsNone(state._cross_mhc_epoch(144))
        original, ids = _context(0, (257,), (0,))
        state = _state(original, ids)
        self.assertIs(state.mega_attention_valid_mask(original, original.local_is_real, original.chunk_length),
                      original.local_is_real, "ragged original padding must retain its native mask")

    def test_phase_wrong_mask_mutation_and_unborn_context_fail_closed(self):
        for change in ("graph", "decode", "spec", "tp", "mask_clone", "source_write"):
            original, ids = _context(lengths=(256,), prefixes=(0,))
            state = _state(original, ids)
            mask = original.local_is_real
            if change == "graph":
                state.is_graph = True
            elif change == "decode":
                state.prefill = False
            elif change == "spec":
                state.runtime.model._is_speculative = True
            elif change == "tp":
                state.runtime.model.parallelism_config.tp_size = 1
            elif change == "mask_clone":
                mask = mask.clone()
            else:
                original.global_positions.add_(0)
            self.assertIs(state.mega_attention_valid_mask(original, mask, original.chunk_length), mask)
            if change != "mask_clone":
                self.assertFalse(state._cross_mhc_phase())
        original, ids = _context(lengths=(256,), prefixes=(0,))
        with patch("rtp_llm.models_py.modules.dsv41.ppu._quant.is_ppu", return_value=False):
            state = _state(original, ids)
        self.assertIsNone(state._cp_full_rows_birth)
        state.bind_context(original)
        self.assertIsNone(state._cp_full_rows_birth, "later bindings cannot mint an unborn native context proof")
        self.assertIsNone(state._cross_mhc_epoch(original.chunk_length))

    def test_full_ticket_epoch_fence_and_compact_owner_mutations(self):
        inputs, _, _, _ = _make_case(torch, 18, "cpu", "normal")
        for change in ("send", "positions", "clone", "ids"):
            state, blocks = _component(inputs, _CPUOps(), count=22)
            plan = _install_plan(state, synthetic_transport=True)
            full, _ = _publish(state, blocks, inputs, source=19)
            self.assertIsNotNone(state._cross_mhc_ticket)
            old_epoch = state._cross_mhc_epoch(18)[0]
            state.prepare_plan()
            self.assertEqual(state._cross_mhc_epoch(18)[0], old_epoch)
            state.compact_l20_queries(plan.exchange.compact(full), plan.exchange.compact(state.inputs.input_ids))
            self.assertIsNone(state._cross_mhc_ticket)
            self.assertNotEqual(state._cross_mhc_epoch(18)[0], old_epoch,
                                "equal capacities do not make full and compact epochs interchangeable")
            if change == "send":
                plan.exchange.send_indices.add_(0)
            elif change == "positions":
                plan.context.relative_positions.add_(0)
            elif change == "clone":
                plan.context.local_is_real = plan.context.local_is_real.clone()
            else:
                state.compact_ids.add_(0)
            self.assertIsNone(state._cross_mhc_epoch(18))

    def test_actual_block_chain_has37_edges_and_preserves_engram_l20_and_head(self):
        inputs, _, _, _ = _make_case(torch, 18, "cpu", "normal")
        old_state, old_blocks = _component(inputs, _CPUOps(), count=40, enabled="0")
        _install_plan(old_state, synthetic_transport=True)
        with patch.dict(os.environ, {ENV: "0"}):
            expected, expected_head = _run_chain(old_state, old_blocks, inputs["residual"])
        expected_mix = {name: value.clone() for name, value in _mix_fields(old_blocks).items()}
        state, blocks = _component(inputs, _CPUOps(), count=40)
        _install_plan(state, synthetic_transport=True)
        actual, head = _run_chain(state, blocks, inputs["residual"])
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(head, expected_head))
        for name, value in _mix_fields(blocks).items():
            self.assertTrue(torch.equal(value, expected_mix[name]), name)
        evidence = state._cross_mhc_evidence
        self.assertEqual((evidence["published"], evidence["consumed"]), (37, 37))
        self.assertEqual(evidence["fallbacks"], {})
        self.assertEqual(sum(not edge["compacted"] for edge in evidence["edges"]), 18)
        self.assertEqual(sum(edge["compacted"] for edge in evidence["edges"]), 19)
        self.assertEqual(sum(block.attn_norm.calls for block in blocks), 3)
        self.assertEqual(sum(block.ffn_norm.calls for block in blocks), 0)
        self.assertFalse(any(edge["dst"] in (1, 14) for edge in evidence["edges"]))
        self.assertIsNone(state._cross_mhc_ticket)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class CP8CrossLayerMHCNativeRows(unittest.TestCase):
    @torch.inference_mode()
    def test_full4096_and_compact144_use_existing_native_numeric_contract(self):
        self.assertTrue(is_ppu())
        reports = []
        with patch.dict(os.environ, dict(CP_ENV, RTP_V41_PPU_QUANT_MODE="mxfp4_fp8_ptpc",
                                         RTP_V41_PPU_PTPC_FUSION="1")):
            for compact in (False, True):
                full_inputs, _, _, _ = _make_case(torch, 4096, "cuda:0", "normal")
                ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=1e-20))
                state, blocks = _component(full_inputs, ops, rank=0, lengths=(4096,) * 8,
                                           prefixes=(126976,) * 8, count=22)
                if compact:
                    _activate(state, _plan(state))
                    inputs, _, _, _ = _make_case(torch, 144, "cuda:0", "normal")
                    source = 20
                else:
                    inputs, source = full_inputs, 1
                original = {name: value.clone() for name, value in inputs.items()}
                updated, previous = _publish(state, blocks, inputs, source=source)
                self.assertIsNotNone(state._cross_mhc_ticket, ops.mhc_execution_stats())
                transition = state._cross_mhc_ticket.transition
                target = blocks[source + 1]
                self.assertIsNotNone(_consume(state, target, updated))
                authority = {name: value.cpu() for name, value in inputs.items()}
                authority.update(shifted_prev_mix=previous.pre_mix.unsqueeze(-1).cpu(),
                                 post_mix=previous.post_mix.cpu(), comb_res_mix=previous.residual_mix.cpu(),
                                 fn=target.attn_hc.fn.cpu(), mix_bases=target.attn_hc.base.cpu(),
                                 mix_scales=target.attn_hc.scale.cpu(), rmsnorm_weight=target.attn_norm.weight.cpu())
                scalars = dict(hc_norm_eps=1e-20, hc_pre_eps=1e-6, hc_post_scale=2.0,
                               sinkhorn_eps=1e-6, num_sinkhorn_iters=20, rmsnorm_eps=1e-20, rmsnorm_scale=1.0)
                reference = independent_transition_reference(torch, authority, scalars)
                actual = dict(zip(OUTPUTS, (updated, transition.pre_mix.unsqueeze(-1), transition.post_mix,
                                            transition.residual_mix, transition.normalized)))
                for name, value in actual.items():
                    metric = _qualification_metric(torch, value, reference[name])
                    self.assertEqual(metric["qualified_status"], "PASS", (compact, name, metric))
                    reports.append({"compacted": compact, "rows": value.shape[0], "output": name, "metric": metric})
                self.assertTrue(all(torch.equal(value, original[name]) for name, value in inputs.items()))
                self.assertEqual(state._cross_mhc_evidence["consumed"], 1)
                if compact:
                    mask = state.context.local_is_real
                    self.assertIs(state.mega_attention_valid_mask(state.context, mask, 144), mask)
                    self.assertEqual(state._cross_mhc_evidence["cp_row_proof"]["compact_real_rows_local"], 112)
        _report("cp8-cross-mhc-native-row-contract.json",
                {"schema": "cp8-cross-mhc-native-row-contract-v1", "cases": reports})


@patch.dict(os.environ, dict(CP_ENV, RTP_V41_PPU_QUANT_MODE="mxfp4_fp8_ptpc",
                            RTP_V41_PPU_PTPC_FUSION="1", RTP_V41_PPU_LOCAL_PREFILL_SPLIT_L20="1"))
def _world8_worker(rank, store):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    try:
        dist.init_process_group("nccl", init_method="file://" + store, rank=rank, world_size=8,
                                timeout=timedelta(seconds=300))
        # Keep libc environ stable for NCCL/PCCL heartbeat getenv readers.
        with torch.inference_mode():
            inputs, _, _, _ = _make_case(torch, 4096, f"cuda:{rank}", "normal")
            ops = PPUOps(num_local_experts=384, mhc_config=MHCConfig(rms_eps=1e-20))
            state, blocks = _component(inputs, ops, rank=rank, lengths=(4096,) * 8,
                                       prefixes=(126976,) * 8, count=40)
            _install_plan(state, group=dist.group.WORLD)
            full_mask = state.original_context.local_is_real
            if state.mega_attention_valid_mask(state.context, full_mask, 4096) is not None:
                raise AssertionError("native all-live full CP context retained an unnecessary mask")
            result, head = _run_chain(state, blocks, inputs["residual"])
            evidence = state._cross_mhc_evidence
            if (result.shape != (144, 4, 5120) or head.shape != (144, 5120)
                    or (evidence["published"], evidence["consumed"]) != (37, 37)
                    or evidence["fallbacks"] or state._cross_mhc_ticket is not None
                    or sum(block.attn_norm.calls for block in blocks) != 3
                    or not bool(torch.isfinite(head).all())):
                raise AssertionError((rank, "CP37-edge native chain contract", evidence))
            mask = state.context.local_is_real
            result_mask = state.mega_attention_valid_mask(state.context, mask, 144)
            if (rank == 0 and result_mask is not mask) or (rank != 0 and result_mask is not None):
                raise AssertionError((rank, "compact rank padding proof", evidence))
            print(json.dumps({"schema": "cp8-cross-mhc-world8-contract-v1", "rank": rank,
                              "evidence": evidence, "ops": ops.mhc_execution_stats()}, sort_keys=True))
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_CP8_WORLD8") == "1", "explicit eight-rank CP collective test")
class CP8CrossLayerMHCWorld8(unittest.TestCase):
    def test_actual_cp_transport_native_hc37_edges_and_rank_masks(self):
        self.assertTrue(is_ppu())
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory(prefix="cp8-cross-mhc-") as directory:
            mp.spawn(_world8_worker, args=(str(Path(directory) / "store"),), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
