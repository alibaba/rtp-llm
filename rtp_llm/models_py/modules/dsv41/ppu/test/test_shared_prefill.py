"""One PPU device fixture for public CED transport and PPU live carry."""

import json
import os
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.ppu.mhc import PPUStepMix
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.shared_ced import _bounded_query_layout
from rtp_llm.models_py.modules.dsv41.ppu.shared_modules import PPUSharedHCUnit
from rtp_llm.models_py.modules.dsv41.ppu.shared_prefill import SharedForwardState, encoder_replay_inputs
from rtp_llm.models_py.modules.dsv41.prefill.ced import CEDPlan, _query_layout, checkpoint_positions
from rtp_llm.utils.model_weight import W


def context(length, device):
    # CP1 identity framework layout is sufficient for the local primitive
    # fixture. Actual CP8 row ownership is checked by the native main case.
    indices = torch.arange(length, dtype=torch.long, device=device)
    valid = torch.ones(length, dtype=torch.bool, device=device)
    return CPContext(1, 0, length, length, length, indices, 0, indices, valid, indices, length,
                     SimpleNamespace(prefill_qkv_restore_indice_cpu=indices.cpu(), prefill_qkv_padding_mask_cpu=valid.cpu()),
                     req_id_per_token=torch.zeros(length, dtype=torch.int32, device=device),
                     prefix_lengths=torch.zeros(1, dtype=torch.long, device=device),
                     input_lengths_global=torch.tensor([length], device=device), cu_seqlens_global=torch.tensor([0, length], device=device),
                     chunk_lengths_per_req=(length,), input_lengths_global_host=(length,), prefix_lengths_host=(0,))


class SharedPrefillDevice(unittest.TestCase):
    def test_local_encoder_replay_has_no_padding(self):
        history = torch.arange(10)
        fresh = torch.cat((history[2:5], history[7:10]))
        attn = SimpleNamespace()
        inputs = SimpleNamespace(input_ids=fresh, prefill_replay_token_ids=history,
                                 attention_inputs={"swa_kv": attn})
        replay, totals, restore = encoder_replay_inputs(inputs, attn, (3, 3), (2, 2), 1, 0)
        self.assertEqual(totals, (5, 5))
        self.assertTrue(torch.equal(replay.input_ids, history))
        self.assertEqual(replay.combo_position_ids.tolist(), [0, 1, 2, 3, 4] * 2)
        self.assertEqual(replay.attention_inputs["swa_kv"].cu_seqlens.tolist(), [0, 5, 10])
        self.assertEqual(replay.engram_token_windows.shape[0], 10)
        self.assertTrue(torch.equal(restore(history, None), fresh))

    @torch.inference_mode()
    def test_compact_restore_carry_and_exact_primitives(self):
        self.assertTrue(torch.cuda.is_available(), "this target requires a real PPU")
        self.assertFalse(dist.is_initialized(), "fixture owns its process group")
        device = torch.device("cuda", 0)
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group("nccl", rank=0, world_size=1, init_method="file://" + directory + "/rendezvous")
            try:
                original = context(257, device)
                selected = torch.arange(117, 257)
                query, exchange, groups = _bounded_query_layout(original, selected)
                self.assertIsNone(exchange.group)
                query.swa_replay_start = 117
                plan = CEDPlan(original, query, torch.tensor([0, query.chunk_length], dtype=torch.int32, device=device), exchange, groups, (37, 38, 39))
                policy = SharedPrefillPolicy(mode="compact")
                g = SimpleNamespace(hidden_size=5120, hc_mult=4, hc_sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-6)
                tensors = {W.v4_hc_ffn_fn: torch.zeros((24, 20480), dtype=torch.float32, device=device),
                           W.v4_hc_ffn_base: torch.zeros(24, dtype=torch.float32, device=device),
                           W.v4_hc_ffn_scale: torch.ones(3, dtype=torch.float32, device=device)}
                ops = PPUOps(
                    ep_rank=0,
                    ep_size=1,
                    num_routed_experts=384,
                    num_local_experts=384,
                )
                hc = PPUSharedHCUnit(SimpleNamespace(geometry=g, tensor=lambda layer, key: tensors[key]), 20, "ffn", ops)
                marker = torch.arange(257, device=device, dtype=torch.float32)
                carry = PPUStepMix(hidden=marker[:, None].expand(-1, 5120).bfloat16().contiguous(),
                                   pre_mix=marker[:, None].expand(-1, 4).contiguous(),
                                   post_mix=marker[:, None, None].expand(-1, 4, 1).contiguous(),
                                   residual_mix=marker[:, None, None].expand(-1, 4, 4).contiguous())
                hc.state, hc.pre_mix_out = carry, carry.pre_mix
                v4 = SimpleNamespace(layers=[SimpleNamespace(attn=SimpleNamespace(_prefill_meta_shared=None), ffn_hc=hc)] * 40,
                                     capture_aux_hidden_layer_ids=(37, 38, 39), _mtp_hidden_buffer=torch.empty((257, 6), device=device),
                                     _propagate_cp_ctx=lambda value: None, _note_aux_hidden_rows=lambda rows, is_cuda_graph: None)
                runtime = SimpleNamespace(model=SimpleNamespace(v4=v4), policy=policy)
                inputs = SimpleNamespace(input_ids=marker.long(), need_all_logits=False, need_all_hidden_states=False)
                state = SharedForwardState(runtime, inputs, None, None, prefill=True, idle=False, prefixes=(0,), lengths=(257,))
                state.plan, state.original_context = plan, original
                state.shared.update(topk={20: marker[:, None].int()}, candidates=marker[:, None].int())
                hidden = marker[:, None, None].expand(-1, 4, 5120).bfloat16().contiguous()
                saved = hidden.clone()
                compact, ids, positions, cu = state.before_block(SimpleNamespace(layer_id=21), hidden, marker.long(), marker.long(), torch.tensor([0, 257], device=device))
                mapping = query.global_positions
                self.assertEqual(query.unpad_restore.numel(), 140)
                self.assertTrue(torch.equal(ids[query.local_is_real], mapping[query.local_is_real]))
                self.assertTrue(torch.equal(hidden, saved), "input rows were mutated")
                for field in fields(carry):
                    self.assertTrue(torch.equal(getattr(hc.state, field.name), getattr(carry, field.name).index_select(0, mapping)))
                self.assertTrue(torch.equal(hc.pre_mix_out, hc.state.pre_mix))
                self.assertTrue(torch.equal(state.shared["topk"][20].flatten(), mapping.int()))
                aux = mapping.float()[:, None].expand(-1, 6)
                v4._mtp_hidden_buffer[:query.chunk_length].copy_(aux)
                restored = state.restore_head(compact[:, 0])
                self.assertTrue(torch.equal(restored[117:], saved[117:, 0]))
                self.assertEqual(int(torch.count_nonzero(restored[:117])), 0)
                self.assertTrue(torch.equal(v4._mtp_hidden_buffer[117:], marker[117:, None].expand(-1, 6)))
                self.assertEqual(int(torch.count_nonzero(v4._mtp_hidden_buffer[:117])), 0)
                self.assertIsNone(ops.try_mhc_ffn_transition(torch.zeros((257, 5120), dtype=torch.bfloat16, device=device), saved, carry,
                                                          hc.parameters_map, torch.ones(5120, dtype=torch.bfloat16, device=device), 1e-6, phase="ced"))
                exact_original = context(32769, device)
                exact_selected = checkpoint_positions(0, 32769, [1, 0], 512, 136)
                exact_query, exact_exchange, exact_groups = _query_layout(exact_original, exact_selected, dist.group.WORLD, keep_candidate_rows=False)
                exact = CEDPlan(exact_original, exact_query, torch.tensor([0, exact_query.chunk_length], device=device), exact_exchange, exact_groups, ())
                value = torch.arange(32769, device=device, dtype=torch.float32)[:, None]
                exact_restored = exact.restore_rows(exact.exchange.compact(value))
                self.assertTrue(torch.equal(exact_restored[exact_selected.to(device)], value[exact_selected.to(device)]))
                omitted = torch.ones(32769, dtype=torch.bool, device=device)
                omitted[exact_selected.to(device)] = False
                self.assertEqual(int(torch.count_nonzero(exact_restored[omitted])), 0)
                torch.cuda.synchronize()
                output = os.environ.get("RTP_V41_SHARED_ADAPTER_REPORT")
                if output:
                    Path(output).write_text(json.dumps({"device_fixture": "PASS", "logical_rows": 140, "carry_fields": [f.name for f in fields(carry)],
                                                       "exact_primitive_rows": int(exact_selected.numel()), "native_exact_E2E": "NOT_RUN",
                                                       "CP8": "native_main_case_only", "threshold": "exact equality; no numeric tolerance"}, indent=2) + "\n")
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
