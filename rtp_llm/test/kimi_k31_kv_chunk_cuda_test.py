"""Constructed CUDA contracts for KV-only chunking; no checkpoint golden."""

import json
import unittest
from types import SimpleNamespace

import torch
from rtp_llm.ops import ParallelismConfig

from rtp_llm.models_py.kernels.cuda.dsa_kv_chunk import (
    map_dsa_chunk_slots,
    normalize_sparse_partial,
)
from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score
from rtp_llm.models_py.modules.dsv4.indexer_topk import TorchIndexerTopKBackend
from rtp_llm.models_py.triton_kernels.mla.lse_merge import (
    merge_attention_states_in_place,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA component test")
class DsaKVChunkCudaTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3101)
        self.device = "cuda"

    def test_model_attn_res_capacity_is_charged_to_each_phase(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import (
            DsaWorkspaceOwner,
        )

        tokens, hidden, tp = 65536, 7168, 8
        plain = DsaWorkspaceOwner(hidden_dim=hidden, tp_size=tp, device=self.device)
        four = DsaWorkspaceOwner(
            hidden_dim=hidden, tp_size=tp, attn_res_blocks=1, device=self.device
        )
        full = DsaWorkspaceOwner(
            hidden_dim=hidden, tp_size=tp, attn_res_blocks=8, device=self.device
        )
        for indexer in (False, True):
            reference = plain.external_projection_bytes(tokens, indexer=indexer)
            one = tokens // tp * hidden * 2
            self.assertEqual(
                four.external_projection_bytes(tokens, indexer=indexer) - reference, one
            )
            self.assertEqual(
                full.external_projection_bytes(tokens, indexer=indexer) - reference,
                8 * one,
            )
        self.assertEqual(7 * one, 784 * 1024 * 1024)
        cfg = SimpleNamespace(
            num_layers=93,
            hidden_size=hidden,
            k3_runtime_config=SimpleNamespace(
                attn_res_block_size=12,
                dsa=SimpleNamespace(
                    index_num_heads=32,
                    index_head_dim=128,
                    index_topk=2048,
                    indexer_use_fp8=True,
                ),
            ),
            attn_config=SimpleNamespace(
                head_num=96,
                kv_lora_rank=512,
                rope_head_dim=64,
                v_head_dim=128,
                q_lora_rank=1536,
                nope_head_dim=128,
            ),
        )
        parallel = ParallelismConfig()
        parallel.tp_size = tp
        self.assertEqual(DsaWorkspaceOwner.from_model(cfg, parallel).attn_res_blocks, 8)
        cfg.num_layers = 4
        self.assertEqual(DsaWorkspaceOwner.from_model(cfg, parallel).attn_res_blocks, 1)
        cfg.k3_runtime_config.attn_res_block_size = 0
        self.assertEqual(DsaWorkspaceOwner.from_model(cfg, parallel).attn_res_blocks, 0)
        with self.assertRaises(ValueError):
            DsaWorkspaceOwner(attn_res_blocks=-1, device=self.device)

    def test_global_causal_positions_and_request_mask(self):
        q = (torch.randn(1, 7, 32, 128, device=self.device) * 0.05).bfloat16()
        k = (torch.randn(1, 13, 128, device=self.device) * 0.05).bfloat16()
        weights = torch.rand(1, 7, 32, device=self.device)
        qp = torch.tensor(
            [[1000, 7, 2050, -1, 1, 600, 14]], dtype=torch.int32, device=self.device
        )
        kp = torch.tensor(
            [[0, 1, 8, 512, 1024, 2048, 3000, 0, 3, 13, 17, 18, 600]],
            dtype=torch.int32,
            device=self.device,
        )
        qr = torch.tensor(
            [[0, 0, 0, 0, 1, 1, 1]], dtype=torch.int32, device=self.device
        )
        kr = torch.tensor([[0] * 7 + [1] * 6], dtype=torch.int32, device=self.device)
        expected = (
            torch.einsum("bshd,btd->bsht", q.float(), k.float()).relu()
            * weights[..., None]
        ).sum(2)
        valid = (kp[:, None] <= qp[:, :, None]) & (qr[:, :, None] == kr[:, None])
        expected.masked_fill_(~valid, -float("inf"))
        out = torch.empty_like(expected)
        actual = v4_indexer_score(
            q,
            k,
            weights,
            qp,
            kv_positions=kp,
            query_requests=qr,
            key_requests=kr,
            out=out,
        )
        self.assertEqual(actual.data_ptr(), out.data_ptr())
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        selector = TorchIndexerTopKBackend()
        full = selector.select_candidates(actual[0], kp[0], 9)
        running = (
            torch.full((7, 9), -float("inf"), device=self.device),
            torch.full((7, 9), -1, device=self.device, dtype=torch.int32),
        )
        for start in (0, 5, 10):
            end = min(start + 5, 13)
            score = v4_indexer_score(
                q,
                k[:, start:end].contiguous(),
                weights,
                qp,
                kv_positions=kp[:, start:end].contiguous(),
                query_requests=qr,
                key_requests=kr[:, start:end].contiguous(),
            )
            selector.merge_candidates(
                running, selector.select_candidates(score[0], kp[0, start:end], 9)
            )
        # Selection promises paired members, not their visit order. Compare
        # both scores and positions in global-position order; never sort the
        # production result solely to satisfy a test.
        running_order = running[1].argsort(dim=-1)
        full_order = full[1].argsort(dim=-1)
        for current, reference in zip(running, full):
            torch.testing.assert_close(
                current.gather(1, running_order),
                reference.gather(1, full_order),
                rtol=0,
                atol=0,
            )

    def test_final_topk_is_consumed_once_across_ragged_kv_blocks(self):
        indices = torch.tensor(
            [[0, 7, 8, 16, -1], [0, 3, 4, 9, -1]], dtype=torch.int32, device=self.device
        )
        params = SimpleNamespace(
            is_prefill=True,
            query_requests=torch.tensor([0, 1], dtype=torch.int32, device=self.device),
            page_size=4,
            shard_size=2,
            shard_rank=0,
        )
        result = torch.empty_like(indices)
        consumed = torch.zeros_like(indices)
        for slices in ([[0, 8, 0], [0, 4, 8]], [[8, 17, 0], [4, 10, 9]]):
            block = SimpleNamespace(
                slices=torch.tensor(slices, dtype=torch.int32, device=self.device),
                start=0,
                width=0,
            )
            map_dsa_chunk_slots(indices, params, block, result)
            consumed += (result >= 0).int()
        torch.testing.assert_close(consumed, (indices >= 0).int())
        # Owner-local Decode chunks use global PageRR positions before offsets.
        params.is_prefill = False
        params.batch_indice_d = params.query_requests
        consumed.zero_()
        for start in (0, 4, 8):
            block = SimpleNamespace(slices=None, start=start, width=4)
            map_dsa_chunk_slots(indices, params, block, result)
            consumed += (result >= 0).int()
        expected = ((indices >= 0) & ((indices // 4) % 2 == 0)).int()
        torch.testing.assert_close(consumed, expected)
        params.kernel_page_size = 2
        table = torch.tensor(
            [[9, 2, 7, 4, 8, 6], [3, 11, 5, 13, 0, 0]],
            dtype=torch.int64,
            device=self.device,
        )
        consumed.zero_()
        for start in (0, 4, 8):
            block = SimpleNamespace(slices=None, start=start, width=4)
            map_dsa_chunk_slots(indices, params, block, result, block_table=table)
            ref = torch.full_like(indices, -1)
            for row in range(2):
                for column, position in enumerate(indices[row].tolist()):
                    if position >= 0 and position // 4 % 2 == 0:
                        local = (position // 8) * 4 + position % 4
                        cache_block = table[row, local // 2].item()
                        if start <= local < start + 4 and cache_block > 0:
                            ref[row, column] = cache_block * 2 + local % 2
            torch.testing.assert_close(result, ref)
            consumed += (result >= 0).int()
        torch.testing.assert_close(consumed, expected)

    def test_latent_merge_then_one_value_up(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashmla_sparse_impl import (
            flash_mla_sparse_fwd,
        )

        rows, heads, latent, suffix, value, topk = 17, 96, 512, 64, 128, 2048
        q = (
            torch.randn(rows, 128, latent + suffix, device=self.device) * 0.05
        ).bfloat16()
        kv = (torch.randn(31, 1, latent + suffix, device=self.device) * 0.05).bfloat16()
        slots = torch.full((rows, topk), -1, dtype=torch.int32, device=self.device)
        slots[1:, :31] = torch.arange(31, dtype=torch.int32, device=self.device)
        weights = (
            torch.randn(heads, latent, value, device=self.device) * 0.02
        ).bfloat16()
        partial_lse = torch.empty(rows, heads, device=self.device)
        full, _, lse = flash_mla_sparse_fwd(q, kv, slots[:, None], 0.1, d_v=latent)
        normalize_sparse_partial(full, lse, heads, partial_lse)
        expected_lse = partial_lse.clone()
        expected = torch.empty(
            rows, heads, value, dtype=torch.bfloat16, device=self.device
        )
        torch.bmm(
            full[:, :heads].transpose(0, 1), weights, out=expected.transpose(0, 1)
        )
        running = torch.zeros(rows, heads, value, device=self.device)
        running_lse = torch.full((rows, heads), -float("inf"), device=self.device)
        projected = torch.empty_like(expected)
        running_latent = torch.zeros(rows, heads, latent, device=self.device)
        latent_lse = torch.full_like(running_lse, -float("inf"))
        for start, end in ((0, 8), (8, 16), (16, 31)):
            chosen = torch.where((slots >= start) & (slots < end), slots - start, -1)
            part, _, lse = flash_mla_sparse_fwd(
                q, kv[start:end].contiguous(), chosen[:, None], 0.1, d_v=latent
            )
            normalize_sparse_partial(part, lse, heads, partial_lse)
            merge_attention_states_in_place(
                running_latent, latent_lse, part[:, :heads], partial_lse
            )
            # Former block V-up path remains a rounding reference only.
            torch.bmm(
                part[:, :heads].transpose(0, 1), weights, out=projected.transpose(0, 1)
            )
            merge_attention_states_in_place(running, running_lse, projected, partial_lse)
        actual = torch.empty_like(expected)
        torch.bmm(
            running_latent.bfloat16().transpose(0, 1),
            weights,
            out=actual.transpose(0, 1),
        )
        torch.testing.assert_close(
            running_latent.bfloat16(), full[:, :heads], rtol=0.03, atol=0.0005
        )
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.0005)
        torch.testing.assert_close(latent_lse, expected_lse, rtol=2e-5, atol=2e-5)
        self.assertTrue(torch.equal(actual[0], torch.zeros_like(actual[0])))
        self.assertTrue(torch.isneginf(latent_lse[0]).all())
        self.assertTrue(torch.equal(running[0], torch.zeros_like(running[0])))
        self.assertTrue(torch.isneginf(running_lse[0]).all())
        # Constructed BF16 roundoff, rather than real-model precision alignment.
        torch.testing.assert_close(
            running[1:].bfloat16(), expected[1:], rtol=0.03, atol=0.0005
        )
        difference = actual[1:].float() - expected[1:].float()
        print(
            json.dumps(
                {
                    "kind": "constructed_bf16_chunk_roundoff",
                    "max_absolute_error": difference.abs().max().item(),
                    "relative_l2_error": (
                        difference.norm() / expected[1:].float().norm()
                    ).item(),
                    "latent_max_absolute_error": (
                        running_latent - full[:, :heads].float()
                    )
                    .abs()
                    .max()
                    .item(),
                    "lse_max_absolute_error": (latent_lse[1:] - expected_lse[1:])
                    .abs()
                    .max()
                    .item(),
                    "block_v_up_reference_max_error": (running - expected.float())
                    .abs()
                    .max()
                    .item(),
                    "chunk_v_up_calls": 1,
                    "block_v_up_reference_calls": 3,
                    "reference_v_up_calls": 1,
                    "real_model_precision_alignment": False,
                }
            ),
            flush=True,
        )

    def test_merge_padded_head_strides_and_empty_states(self):
        partial = torch.randn(5, 128, 512, device=self.device).bfloat16()
        lse = torch.randn(5, 128, device=self.device)
        output = torch.randn(5, 96, 512, device=self.device)
        state = torch.randn(5, 96, device=self.device)
        state[0].fill_(-float("inf"))
        lse[0].fill_(-float("inf"))
        partial[0].fill_(float("nan"))
        state[1].fill_(-float("inf"))
        lse[2].fill_(-float("inf"))
        previous, previous_lse = output.clone(), state.clone()
        part, part_lse = partial[:, :96], lse[:, :96]
        self.assertFalse(part.is_contiguous())
        self.assertFalse(part_lse.is_contiguous())
        total_lse = torch.logaddexp(previous_lse, part_lse)
        a = torch.nan_to_num(torch.exp(previous_lse - total_lse))
        b = torch.nan_to_num(torch.exp(part_lse - total_lse))
        expected = (
            previous * a[..., None] + torch.nan_to_num(part.float()) * b[..., None]
        )
        merge_attention_states_in_place(output, state, part, part_lse)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(state, total_lse, rtol=1e-5, atol=2e-6)

    def test_stage_arena_reuses_kv_for_final_latent_and_preserves_running(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import (
            DsaPhaseWorkspace,
            staged_workspace_bytes,
        )

        fixed = {
            "running": ((7, 96, 512), torch.float32),
            "running_lse": ((7, 96), torch.float32),
        }
        loop = {
            **fixed,
            "cache.pack": ((4096, 576), torch.bfloat16),
            "cache.gather": ((8192, 576), torch.bfloat16),
            "slots": ((7, 2048), torch.int32),
        }
        final = {
            **fixed,
            "latent": ((7, 96, 512), torch.bfloat16),
            "output": ((7, 96, 128), torch.bfloat16),
        }
        stages = (loop, final)
        workspace = DsaPhaseWorkspace(self.device)
        self.assertEqual(
            workspace.required_staged_bytes(stages), staged_workspace_bytes(stages)
        )
        workspace.reserve_stages(stages)
        self.assertEqual(workspace.resident_bytes, staged_workspace_bytes(stages))
        running = workspace("running", *fixed["running"])
        running.fill_(0.5)
        packed = workspace("cache.pack", *loop["cache.pack"])
        latent = workspace("latent", *final["latent"])
        self.assertEqual(packed.data_ptr(), latent.data_ptr())
        latent.copy_(running)
        torch.testing.assert_close(
            running, torch.full_like(running, 0.5), rtol=0, atol=0
        )
        addresses = [x.data_ptr() for x in workspace.arenas.values()]
        workspace.reserve_stages(stages, forbid_realloc=True)
        self.assertEqual(addresses, [x.data_ptr() for x in workspace.arenas.values()])
        with self.assertRaisesRegex(ValueError, "cannot grow"):
            workspace.reserve_stages(
                ({"large": ((10000000,), torch.float32)},), forbid_realloc=True
            )
        workspace.release()
        self.assertEqual(workspace.resident_bytes, 0)

    def test_prepared_plan_reuses_geometry_and_refreshes_physical_ids(self):
        from types import MethodType

        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import (
            DsaWorkspaceOwner,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_page_rr_params import (
            DsaPageRRParams,
        )

        owner = DsaWorkspaceOwner(device=self.device, tp_size=2)
        params = SimpleNamespace(
            workspace_owner=owner,
            is_prefill=False,
            is_cuda_graph=False,
            positions_d=torch.tensor([12, 31], device=self.device, dtype=torch.int32),
            decode=SimpleNamespace(
                local_causal_lens=torch.ones(2, 1, device=self.device)
            ),
            shard_size=2,
            shard_rank=0,
            page_size=4,
            kernel_page_size=2,
            block_table=torch.ones(2, 8, device=self.device, dtype=torch.int32),
            forbid_realloc=False,
            indexer_execution=None,
        )
        params.prepare_execution = MethodType(DsaPageRRParams.prepare_execution, params)
        first = params.prepare_execution("indexer")
        params.positions_d.add_(1)
        params.block_table.fill_(7)
        reused = params.prepare_execution("indexer")
        self.assertIs(first, reused)
        torch.testing.assert_close(reused.block_ids, params.block_table.to(torch.int64))
        params.block_table = torch.ones(2, 16, device=self.device, dtype=torch.int32)
        resized = params.prepare_execution("indexer")
        self.assertIsNot(resized, reused)
        owner.budgets["indexer"] -= 1
        rebudgeted = params.prepare_execution("indexer")
        self.assertIsNot(rebudgeted, resized)
        owner.phase("sfa").reserve({"resident": ((1024,), torch.float32)})
        self.assertIsNot(params.prepare_execution("indexer"), rebudgeted)

    def test_prefill_block_metadata_preserves_ragged_global_positions(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_chunk_execution import (
            _prefill_block_layout,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_chunk_plan import (
            DsaKVBlock,
        )

        block = DsaKVBlock((2, 5), (4096, 0), (17, 9))
        positions, requests, slices, host = _prefill_block_layout(block, 6, self.device)
        self.assertTrue(host.is_pinned())
        torch.testing.assert_close(
            positions,
            torch.tensor(
                list(range(4096, 4113)) + list(range(9)),
                device=self.device,
                dtype=torch.int32,
            ),
        )
        torch.testing.assert_close(
            requests,
            torch.tensor([2] * 17 + [5] * 9, device=self.device, dtype=torch.int32),
        )
        self.assertEqual(
            slices.tolist(),
            [[0, 0, 0], [0, 0, 0], [4096, 4113, 0], [0, 0, 0], [0, 0, 0], [0, 9, 17]],
        )
        one = DsaKVBlock((1,), (2048,), (13,))
        positions, requests, slices, _ = _prefill_block_layout(one, 3, self.device)
        torch.testing.assert_close(
            positions, torch.arange(2048, 2061, device=self.device, dtype=torch.int32)
        )
        self.assertTrue((requests == 1).all())
        self.assertEqual(slices.tolist(), [[0, 0, 0], [2048, 2061, 0], [0, 0, 0]])

    def test_prefill_prepared_metadata_crosses_cuda_streams(self):
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_chunk_execution import (
            DsaKVExecution,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.dsa_kv_workspace import (
            DsaWorkspaceOwner,
        )
        from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.mla_page_rr_cache_gather import (
            MlaPageRRCacheGather,
        )

        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(producer):
            owner = DsaWorkspaceOwner(device=self.device, tp_size=2)
            params = SimpleNamespace(
                workspace_owner=owner,
                is_prefill=True,
                is_cuda_graph=False,
                query_positions=torch.arange(4, device=self.device),
                positions_d=torch.arange(8, device=self.device),
                kv_lens_host=[4, 4],
                shard_size=2,
                shard_rank=0,
                page_size=4,
                kernel_page_size=2,
                block_table=torch.ones(2, 4, device=self.device, dtype=torch.int32),
                cache_adapter=MlaPageRRCacheGather(4, 2, 0, 2),
                forbid_realloc=False,
            )
            execution = DsaKVExecution(params, "indexer")
        with torch.cuda.stream(consumer):
            execution.begin()
            block = execution.blocks[0]
            positions = block.key_positions.clone()
            requests = block.key_requests.clone()
            slices = block.slices.clone()
            execution.finish()
        consumer.synchronize()
        self.assertEqual(positions.tolist(), [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(requests.tolist(), [0] * 4 + [1] * 4)
        self.assertEqual(slices.tolist(), [[0, 4, 0], [0, 4, 4]])

    def test_lse_merge_cuda_graph_replay_uses_new_state(self):
        output = torch.zeros(7, 96, 128, device=self.device)
        state = torch.full((7, 96), -float("inf"), device=self.device)
        partial = torch.ones_like(output, dtype=torch.bfloat16)
        lse = torch.zeros_like(state)
        for _ in range(2):
            merge_attention_states_in_place(output, state, partial, lse)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            merge_attention_states_in_place(output, state, partial, lse)
        for value in (2.0, -3.0):
            output.zero_()
            state.fill_(-float("inf"))
            partial.fill_(value)
            graph.replay()
            torch.testing.assert_close(
                output, torch.full_like(output, value), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
