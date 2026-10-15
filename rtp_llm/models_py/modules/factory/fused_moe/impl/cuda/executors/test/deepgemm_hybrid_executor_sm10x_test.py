"""SM10x (SM100/SM103) device-side contiguous-layout UTs.

The SM103 lane moved ``DeepGemmHybridExecutor.execute_contiguous`` from the
legacy CPU-side expert-count padding (per-layer D2H ``num_recv_tokens_per_expert``
+ H2D padded counts) to the SM120-style device-side layout:
``ep_scatter`` aligns per-expert regions on device, ``m_indices`` keeps ``-1``
padding rows (the SM100 DeepGEMM scheduler clamps them to group 0 and the
results are never gathered), and no tensor crosses to the host.

These tests pin that contract on real SM10x hardware:

* layout semantics of the device-side scatter vs a Python reference
  (identical invocation to the SM120 eager path),
* DeepGEMM contiguous grouped GEMM with ``-1`` padded tiles (the ue8m0
  zero-value concern: valid rows must not depend on padding-row scales),
* the executor keeps the bf16 reference numerics (inherited base tests),
* the eager forward runs with ``torch.cuda.set_sync_debug_mode("error")``
  (no D2H roundtrip left),
* new-path output is bitwise identical to the legacy CPU-padded path.
"""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_mk_alignment,
    get_theoretical_mk_alignment_for_contiguous_layout,
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    quant_weight_ue8m0_packed,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoe,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.fp8_per_block import (
    CudaFp8PerBlockNoDPMaskedStrategy,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.deepep_normal_executor_test import (
    DeepGemmHybridExecutorTestBase,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_scatter,
    ep_scatter_v2,
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.utils.arch import is_sm10x
from rtp_llm.models_py.utils.math import ceil_div


def _clone_payload(payload: ExpertForwardPayload) -> ExpertForwardPayload:
    """Deep copy of an eager contiguous payload (execute disposes expert_x)."""

    return ExpertForwardPayload(
        expert_x=payload.expert_x.clone(),
        expert_x_origin_dtype=payload.expert_x_origin_dtype,
        expert_x_scale=payload.expert_x_scale.clone(),
        expert_tokens_meta=ExpertTokensMetadata(
            expert_num_tokens=payload.expert_tokens_meta.expert_num_tokens.clone(),
        ),
        expert_topk_ids=payload.expert_topk_ids.clone(),
        expert_topk_weights=payload.expert_topk_weights.clone(),
    )


class DeepGemmHybridExecutorSM10xTest(
    DeepGemmHybridExecutorTestBase, unittest.TestCase
):
    def test_sm10x_device(self):
        self.assertTrue(is_sm10x())
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())

    def test_contiguous_sm10x_is_sync_free(self):
        """The eager contiguous forward must not cross to the CPU.

        The legacy SM103 path read ``num_recv_tokens_per_expert`` back to the
        host (a blocking D2H per layer) and re-uploaded padded counts. The
        production PureTpRouter provides only the GPU count tensor, so drop
        the CPU list, warm every JIT path, then run the measured forward under
        ``torch.cuda.set_sync_debug_mode("error")``.
        """

        def execute_and_check(executor, payload, enable_cuda_graph):
            self.assertFalse(enable_cuda_graph)
            payload.expert_tokens_meta.expert_num_tokens_cpu = None
            warmup_payload = _clone_payload(payload)
            executor.execute(warmup_payload, "silu", None, None, False, None)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                result = executor.execute(payload, "silu", None, None, False, None)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            return result

        original_execute = self._execute
        try:
            with patch.object(self, "_execute", side_effect=execute_and_check):
                self._run_deepep_normal_executor()
        finally:
            self._execute = original_execute

    def test_empty_local_experts_eager(self):
        """Empty-expert batch flows through the full device-side pipeline.

        The legacy path early-returned ``torch.zeros`` before touching the
        input, so ``payload.expert_x`` kept its storage. Without a host read
        there is no zero-count shortcut on SM10x, so the batch runs the normal
        pipeline: the output stays all zeros (ep_gather skips every ``-1``
        route) and the eager input storage is released — the same dispose
        contract as the non-empty path and the SM120 eager path.
        """

        released = {}

        def execute_and_check(executor, payload, enable_cuda_graph):
            result = DeepGemmHybridExecutorTestBase._execute(
                self, executor, payload, enable_cuda_graph
            )
            released["expert_x_storage"] = payload.expert_x.untyped_storage().nbytes()
            return result

        with patch.object(self, "_execute", side_effect=execute_and_check):
            self._run_deepep_normal_executor(empty_local_experts=True)
        self.assertEqual(released["expert_x_storage"], 0)

    def test_contiguous_output_matches_legacy_cpu_padded_layout(self):
        """New device-side layout vs forced legacy CPU-pad path (bitwise)."""

        def execute_and_compare(executor, payload, enable_cuda_graph):
            self.assertFalse(enable_cuda_graph)
            legacy_payload = _clone_payload(payload)
            legacy_payload.expert_tokens_meta.expert_num_tokens_cpu = None
            device_side = executor.execute(payload, "silu", None, None, False, None)
            executor.is_sm10x = False
            try:
                legacy = executor.execute(
                    legacy_payload, "silu", None, None, False, None
                )
            finally:
                executor.is_sm10x = True
            self.assertTrue(
                torch.equal(
                    device_side.fused_expert_output, legacy.fused_expert_output
                ),
                "SM10x device-side contiguous layout changed gathered outputs",
            )
            return device_side

        with patch.object(self, "_execute", side_effect=execute_and_compare):
            self._run_deepep_normal_executor()

    def test_device_side_scatter_layout_matches_sm12_semantics(self):
        """``ep_scatter(align_m=128, raw counts)`` vs a Python reference.

        Uses exactly the invocation the SM10x/SM120 eager contiguous path now
        performs: raw GPU counts, per-expert alignment inside the kernel.
        """

        counts = [200, 0, 64, 129, 1]
        alignment = 128
        padded = [(c + alignment - 1) // alignment * alignment for c in counts]
        starts = []
        acc = 0
        for p in padded:
            starts.append(acc)
            acc += p
        all_tokens = acc
        self.assertEqual(all_tokens, 768)

        num_experts = len(counts)
        token_num, topk, hidden = 100, 8, 2048
        scale_cols = hidden // 128 // 4

        device = "cuda"
        recv_x = (
            (torch.arange(token_num * hidden, dtype=torch.int32) % 120)
            .to(torch.uint8)
            .view(torch.float8_e4m3fn)
            .view(token_num, hidden)
        )
        recv_x_scale = (
            (torch.arange(token_num * scale_cols, dtype=torch.int64) * 7 + 3)
            .to(torch.int32)
            .view(token_num, scale_cols)
        )

        slots = [e for e, c in enumerate(counts) for _ in range(c)]
        recv_topk = torch.full((token_num, topk), -1, dtype=torch.int32, device=device)
        flat = recv_topk.view(-1)
        flat[: len(slots)] = torch.tensor(slots, dtype=torch.int32)

        num_recv = torch.tensor(counts, dtype=torch.int32, device=device)
        output_tensor = torch.empty(
            (all_tokens, hidden), dtype=torch.float8_e4m3fn, device=device
        )
        output_tensor_scale = torch.zeros(
            (scale_cols, all_tokens), dtype=torch.int32, device=device
        ).transpose(0, 1)
        m_indices = torch.full((all_tokens,), -1, dtype=torch.int32, device=device)
        output_index = torch.full_like(recv_topk, -1)
        expert_start_loc = torch.empty_like(num_recv)

        ep_scatter(
            recv_x.to(device),
            recv_x_scale.to(device),
            recv_topk,
            num_recv,
            expert_start_loc,
            output_tensor,
            output_tensor_scale,
            m_indices,
            output_index,
            scale_ue8m0=True,
            align_m=alignment,
        )
        torch.cuda.synchronize()

        # 1. m_indices region map: raw-count rows carry the expert id,
        #    alignment padding rows keep the -1 sentinel (SM12 semantics).
        expected_m_indices = torch.full((all_tokens,), -1, dtype=torch.int32)
        for e, c in enumerate(counts):
            expected_m_indices[starts[e] : starts[e] + c] = e
        self.assertTrue(torch.equal(m_indices.cpu(), expected_m_indices))

        # 2. output_index points each valid slot into its expert's raw-count
        #    region; every row of the region is used exactly once.
        flat_index = output_index.view(-1).cpu()
        valid = flat.view(-1).cpu() >= 0
        self.assertTrue((flat_index[~valid] == -1).all())
        for e, c in enumerate(counts):
            slot_rows = flat_index[valid][torch.tensor(slots) == e]
            self.assertEqual(slot_rows.numel(), c)
            self.assertTrue(
                torch.equal(
                    slot_rows.sort().values,
                    torch.arange(starts[e], starts[e] + c, dtype=torch.int32),
                )
            )

        # 3. Scattered data and packed scales land row-faithfully.
        rows = flat_index[valid].to(torch.long)
        src_tokens = torch.arange(token_num * topk)[valid] // topk
        self.assertTrue(
            torch.equal(
                output_tensor.cpu().view(torch.uint8)[rows],
                recv_x.cpu().view(torch.uint8)[src_tokens],
            )
        )
        self.assertTrue(
            torch.equal(output_tensor_scale.cpu()[rows], recv_x_scale.cpu()[src_tokens])
        )

    def test_deepgemm_negative_padding_sm10x(self):
        """Contiguous grouped GEMM with -1 padded tiles on the SM100 kernel.

        The scheduler clamps ``-1`` tiles to group 0 (no skip, unlike SM120),
        so the padding rows are computed with group-0 weights and garbage
        data. Valid-row outputs must still match the bf16 reference: this is
        the evidence that gathered results never depend on padding-row packed
        scales (the ue8m0 zero-value concern).
        """

        groups, n, k = 3, 256, 512
        alignment = min(
            128, get_theoretical_mk_alignment_for_contiguous_layout(19, groups)
        )
        self.assertEqual(alignment, 128)
        m = 5 * alignment
        inputs = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        weights = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)
        a = sgl_per_token_group_quant_fp8(
            inputs,
            128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        b_parts = [quant_weight_ue8m0_packed(w) for w in weights]
        b = (
            torch.stack([part[0] for part in b_parts]),
            torch.stack([part[1].T for part in b_parts]).transpose(-1, -2),
        )
        indices = torch.full((m,), -1, device="cuda", dtype=torch.int32)
        indices[:7] = 0
        indices[alignment : alignment + 12] = 2
        original_indices = indices.clone()
        valid = indices >= 0
        reference = torch.cat(
            [
                inputs[:7].float() @ weights[0].float().T,
                inputs[alignment : alignment + 12].float() @ weights[2].float().T,
            ]
        )
        output = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)

        def run():
            with configure_deep_gemm_mk_alignment(alignment):
                m_grouped_fp8_gemm_nt_contiguous(
                    a, b, output, indices, disable_ue8m0_cast=False
                )

        run()
        torch.cuda.synchronize()
        actual = output[valid].float()
        self.assertTrue(torch.isfinite(actual).all())
        self.assertLess((actual - reference).norm() / reference.norm(), 0.06)
        self.assertTrue(torch.equal(indices, original_indices))

    def test_masked_ue8m0_pad_scale_garbage_is_gather_invisible(self):
        """Masked grouped GEMM: valid rows never read padding-row packed scales.

        ``execute_masked`` (the DeepGemmMaskedExecutorV2 decode path) now
        allocates its ue8m0 scale workspace with ``torch.empty`` instead of
        ``torch.zeros``: ep_scatter_v2 writes only the routed rows and the
        DeepGEMM masked scheduler never schedules fully padded tiles, so
        padding-row scale bytes can only reach padding-row outputs, which
        ep_gather never reads.  This test reproduces the production data
        flow (scatter into a [E, G_packed, T] workspace, masked grouped
        GEMM) and proves it on hardware: identical valid-row outputs with a
        zeroed, an all-ones and a mixed-pattern padding region.
        """

        num_experts, alignment, hidden, out_dim = 4, 128, 1024, 256
        group_cols = ceil_div(hidden // 128, 4)
        counts = [7, 0, 100, 128]
        masked_m = torch.tensor(counts, dtype=torch.int32, device="cuda")
        num_slots = sum(counts)
        self.assertEqual(num_slots, 235)
        self.assertTrue(all(c <= alignment for c in counts))
        expected_m = min(alignment, ceil_div(num_slots, num_experts))

        torch.manual_seed(11)
        slots = [
            e for e, c in enumerate(counts) for _ in range(c)
        ]  # rank-order slots like a real router
        topk_ids = torch.tensor(slots, dtype=torch.int32, device="cuda").view(-1, 1)
        hidden_states = torch.randn(
            (num_slots, hidden), device="cuda", dtype=torch.bfloat16
        )
        recv_x, recv_x_scale = sgl_per_token_group_quant_fp8(
            hidden_states,
            128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        self.assertEqual(recv_x_scale.shape[1], group_cols)

        weights_bf16 = torch.randn(
            (num_experts, out_dim, hidden), device="cuda", dtype=torch.bfloat16
        )
        # Weight scales must be the MN-major packed layout the masked GEMM
        # asserts (sf.stride(-2) == 1, sf.stride(-1) == tma_align(mn)); the
        # same assembly as the contiguous test below.
        w_parts = [quant_weight_ue8m0_packed(w) for w in weights_bf16]
        b = (
            torch.stack([part[0] for part in w_parts]),
            torch.stack([part[1].T for part in w_parts]).transpose(-1, -2),
        )

        def run(pad_fill: int) -> torch.Tensor:
            output_tensor = torch.full(
                (num_experts, alignment, hidden),
                0x3C,
                device="cuda",
                dtype=torch.uint8,
            ).view(torch.float8_e4m3fn)
            output_tensor_scale = torch.full(
                (num_experts, group_cols, alignment),
                pad_fill,
                device="cuda",
                dtype=torch.int,
            ).transpose(1, 2)
            expert_start_loc = torch.empty_like(masked_m)
            output_index = torch.empty_like(topk_ids)
            # ep_scatter_v2 indexes rows with a global dest token id and
            # takes the data tensor as the 2D [E * alignment, hidden] view
            # (the production invocation); the grouped GEMM below gets the
            # 3D view. Counts must fit each expert's fixed `alignment`-row
            # region.
            ep_scatter_v2(
                recv_x,
                recv_x_scale,
                topk_ids,
                alignment,
                expert_start_loc,
                output_tensor.view(num_experts * alignment, hidden),
                output_tensor_scale,
                output_index,
                scale_ue8m0=True,
            )
            d = torch.full(
                (num_experts, alignment, out_dim),
                float("nan"),
                device="cuda",
                dtype=torch.bfloat16,
            )
            m_grouped_fp8_gemm_nt_masked(
                (output_tensor, output_tensor_scale),
                b,
                d,
                masked_m,
                expected_m,
                disable_ue8m0_cast=False,
            )
            torch.cuda.synchronize()
            return d, output_index

        clean, clean_index = run(0)
        poison_ff, ff_index = run(-1)  # all bits set
        poison_mix, mix_index = run(0x5A5A_3C7F)

        # ep_scatter_v2 hands out rows with atomic_add, so the slot -> row
        # permutation differs run to run; every comparison below is
        # therefore slot-indexed through each run's own output_index.
        def slot_values(d, output_index):
            rows = output_index.view(-1).to(torch.long)
            return d.view(num_experts * alignment, out_dim)[rows]

        # 1. Slot outputs are bitwise identical across clean/poisoned
        #    paddings: valid rows never depend on padding-row packed scales.
        clean_slots = slot_values(clean, clean_index)
        for poisoned, poisoned_index in (
            (poison_ff, ff_index),
            (poison_mix, mix_index),
        ):
            self.assertTrue(
                torch.equal(clean_slots, slot_values(poisoned, poisoned_index)),
                "valid rows must not depend on padding-row packed scales",
            )
        self.assertTrue(torch.isfinite(clean_slots).all())
        # Every valid row was actually written (no NaN left from the D init).
        valid_mask = torch.zeros(
            (num_experts, alignment), dtype=torch.bool, device="cuda"
        )
        for e, c in enumerate(counts):
            valid_mask[e, :c] = True
        self.assertFalse(torch.isnan(clean[valid_mask]).any())

        # 2. Non-vacuity: the poison must reach the scheduled partial tiles.
        #    Rows in [count, ceil(count/BLOCK_M)*BLOCK_M) are padding rows of
        #    the last scheduled tile; with an all-ones padding region the
        #    computed padding rows must differ from the zeroed ones (or the
        #    store is masked and they stay NaN in both runs), while rows in
        #    never-scheduled tiles must stay untouched in every run.
        scheduled_pad = torch.zeros(
            (num_experts, alignment), dtype=torch.bool, device="cuda"
        )
        unscheduled = torch.zeros(
            (num_experts, alignment), dtype=torch.bool, device="cuda"
        )
        for e, c in enumerate(counts):
            tile_end = ceil_div(c, 128) * 128
            scheduled_pad[e, c:tile_end] = True
            unscheduled[e, tile_end:] = True
        if scheduled_pad.any():
            if torch.isnan(clean[scheduled_pad]).all():
                self.assertTrue(
                    torch.isnan(poison_ff[scheduled_pad]).all(),
                    "masked store must skip padding rows consistently",
                )
            else:
                self.assertFalse(
                    torch.equal(clean[scheduled_pad], poison_ff[scheduled_pad]),
                    "poisoned padding must show up in padding rows only",
                )
        self.assertTrue(torch.isnan(clean[unscheduled]).all())
        self.assertTrue(torch.isnan(poison_ff[unscheduled]).all())

        # 3. The gathered rows still match the bf16 reference.
        rows = clean_index.view(-1).to(torch.long)
        flat_ids = topk_ids.view(-1).to(torch.long)
        reference = torch.stack(
            [
                hidden_states[s].float() @ weights_bf16[flat_ids[s]].float().T
                for s in range(num_slots)
            ]
        )
        self.assertLess(
            (clean_slots.float() - reference).norm() / reference.norm(), 0.06
        )

    def test_masked_executor_v2_sm10x_e2e(self):
        """End-to-end masked strategy on SM10x: self-zeroed counts + empty scales.

        Runs the exact Xing4 decode stack — ``fp8_per_block_no_dp_masked``
        (PureTpRouterFp8PerBlock router with the in-kernel self-zeroed
        expert_count + DeepGemmMaskedExecutorV2 with the uninitialized ue8m0
        scale workspace) — under CUDA graph capture/replay and eagerly with
        poisoned allocator memory.  num_total (64 tokens x topk 2 = 128)
        launches a single recompute program, so the router exercises the
        self-zero path on every call.  All three runs (replay with the
        capture-time workspaces, two eager runs backed by different poison
        patterns) must agree bitwise: any padding-row garbage leaking into
        a gathered row, or any stale count surviving a replay, would break
        the equality.
        """

        self.assertTrue(is_sm10x())
        self.assertTrue(has_deep_gemm())
        num_experts, top_k, num_tokens = 4, 2, 64
        hidden, inter = 1024, 512
        torch.manual_seed(20261015)
        torch.cuda.manual_seed(20261015)

        model_config = ModelConfig()
        model_config.quant_config = Fp8BlockWiseQuantConfig()
        model_config.data_type = "bf16"
        model_config.expert_num = num_experts
        model_config.moe_k = top_k
        model_config.hidden_size = hidden
        model_config.moe_inter_size = inter

        parallelism_config = ParallelismConfig()
        parallelism_config.world_size = 1
        parallelism_config.local_world_size = 1

        moe_config = MoeConfig()
        moe_config.moe_strategy = "fp8_per_block_no_dp_masked"
        moe_config.use_all_gather = True
        moe_config.use_deepep_moe = False
        moe_config.use_deepep_internode = False
        moe_config.use_deepep_low_latency = False
        config = MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            enable_cuda_graph=True,
        )

        gate_up = torch.randn(
            (num_experts, inter * 2, hidden), device="cuda", dtype=torch.bfloat16
        )
        down = torch.randn(
            (num_experts, hidden, inter), device="cuda", dtype=torch.bfloat16
        )
        # The masked executor hands moe_s1/moe_s2 straight to the masked
        # grouped GEMM, so they need the MN-major packed ue8m0 layout
        # (sf.stride(-2) == 1, sf.stride(-1) == tma_align(mn)), the same
        # assembly the loader produces and the contiguous test uses.
        weights = {}
        for w_name, s_name, src in (
            (W.moe_w1, W.moe_s1, gate_up),
            (W.moe_w2, W.moe_s2, down),
        ):
            parts = [quant_weight_ue8m0_packed(w) for w in src]
            weights[w_name] = torch.stack([part[0] for part in parts])
            weights[s_name] = torch.stack(
                [part[1].T for part in parts]
            ).transpose(-1, -2)

        strategy = CudaFp8PerBlockNoDPMaskedStrategy()
        fused_moe = FusedMoe(
            strategy.create_router(config),
            strategy.create_executor(config, weights),
            expert_num=num_experts,
        )

        def make_inputs(offset: int):
            hidden_states = torch.randn(
                (num_tokens, hidden), device="cuda", dtype=torch.bfloat16
            )
            token_ids = torch.arange(num_tokens, device="cuda")
            topk_ids = torch.stack(
                (
                    (token_ids + offset) % num_experts,
                    (token_ids + offset + 1) % num_experts,
                ),
                dim=1,
            ).to(torch.int64)
            topk_weights = torch.rand(
                (num_tokens, top_k), device="cuda", dtype=torch.float32
            )
            topk_weights /= topk_weights.sum(dim=1, keepdim=True)
            return hidden_states, topk_weights, topk_ids

        # Warm every JIT path outside capture.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            fused_moe(*make_inputs(offset=0))
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        static_hidden, static_weights, static_ids = make_inputs(offset=0)
        graph_output = None
        try:
            with torch.cuda.graph(graph):
                graph_output = fused_moe(static_hidden, static_weights, static_ids)

            replay_hidden, replay_weights, replay_ids = make_inputs(offset=2)
            static_hidden.copy_(replay_hidden)
            static_weights.copy_(replay_weights)
            static_ids.copy_(replay_ids)
            graph.replay()
            captured = graph_output.clone()

            # Poison the allocator so each eager run's torch.empty
            # workspaces (the masked scale buffer and the self-zeroed
            # counters included) come back filled with a distinct garbage
            # pattern; the two eager runs then only agree bitwise if no
            # padding garbage reaches a gathered row and no stale counter
            # survives a fresh launch.
            poison = torch.full((1 << 20,), 0x7F7F7F7F, dtype=torch.int32,
                                device="cuda")
            del poison
            eager = fused_moe(replay_hidden, replay_weights, replay_ids)
            torch.cuda.synchronize()

            poison = torch.full((1 << 20,), 0x3C3C3C3C, dtype=torch.int32,
                                device="cuda")
            del poison
            eager_again = fused_moe(replay_hidden, replay_weights, replay_ids)
            torch.cuda.synchronize()

            self.assertTrue(
                torch.equal(captured, eager),
                "graph replay and the poisoned eager run disagree",
            )
            self.assertTrue(
                torch.equal(eager, eager_again),
                "two eager runs with different workspace garbage disagree",
            )
        finally:
            torch.cuda.synchronize()
            del graph_output
            graph.reset()
            del graph
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
