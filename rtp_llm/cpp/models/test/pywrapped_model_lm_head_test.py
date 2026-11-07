import unittest
from unittest.mock import Mock

import torch
from torch import nn

from rtp_llm.cpp.models.test.libth_pywrapped_model_lm_head_test import run_lm_head
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops.compute_ops import clear_comm_ops, register_comm_ops


class ProjectionModel(GptModelBase):
    def __init__(self):
        nn.Module.__init__(self)
        self.nvfp4_lm_head = nn.Linear(
            8, 16, bias=False, device="cuda", dtype=torch.bfloat16
        )
        with torch.no_grad():
            self.nvfp4_lm_head.weight.copy_(
                torch.arange(128, device="cuda").reshape(16, 8) / 128
            )
        self.projected_rows = []

    def forward(self, inputs, fmha_impl=None):
        raise AssertionError("This fixture exercises C++ output processing only")

    def compute_quantized_logits(self, hidden):
        self.projected_rows.append(hidden.shape[0])
        self.assert_outside_graph()
        return super().compute_quantized_logits(hidden)

    @staticmethod
    def assert_outside_graph():
        if torch.cuda.is_current_stream_capturing():
            raise AssertionError("LM head must run after the model graph replay")


class PyWrappedModelLmHeadTest(unittest.TestCase):
    def setUp(self):
        self.model = ProjectionModel()
        self.hidden = (
            torch.arange(48, device="cuda").reshape(6, 8).to(torch.bfloat16) / 16
        )
        self.indexes = torch.tensor([5, 1], dtype=torch.int32, pin_memory=True)
        # Dispatch sentinel; the callback's projection is a deterministic BF16
        # reference. Packed codes/scales/GEMM have a separate SM120 test.
        self.packed = torch.empty((16, 4), dtype=torch.uint8, device="cuda")

    def test_prefill_all_logits_and_decode_share_cpp_processing(self):
        for packed in (False, True):
            for need_all in (False, True):
                for has_context in (False, True):
                    with (
                        self.subTest(
                            packed=packed, need_all=need_all, has_context=has_context
                        ),
                        torch.no_grad(),
                    ):
                        kernel = (
                            self.packed if packed else self.model.nvfp4_lm_head.weight
                        )
                        result = run_lm_head(
                            self.model,
                            kernel,
                            self.hidden,
                            self.indexes,
                            has_context_request=has_context,
                            need_all_logits=need_all,
                            output_vocab_size=13,
                        )
                        # The dense C++ GEMM accumulates into FP32. The packed
                        # projection returns BF16 before the shared output path.
                        all_logits = (
                            self.model.nvfp4_lm_head(self.hidden).float()
                            if packed
                            else self.hidden.float()
                            @ self.model.nvfp4_lm_head.weight.float().t()
                        )[:, :13]
                        selected = has_context and not need_all
                        expected = (
                            all_logits[[5, 1]] if selected or need_all else all_logits
                        )
                        torch.testing.assert_close(result["logits"], expected)
                        self.assertEqual(result["logits"].dtype, torch.float32)
                        torch.testing.assert_close(
                            result["hidden_states"],
                            self.hidden[[5, 1]] if selected else self.hidden,
                        )
                        torch.testing.assert_close(
                            result["all_hidden_states"], self.hidden
                        )
                        if need_all:
                            torch.testing.assert_close(result["all_logits"], all_logits)
                        else:
                            self.assertIsNone(result["all_logits"])
                        if packed:
                            self.assertEqual(
                                self.model.projected_rows[-1], 2 if selected else 6
                            )
                        else:
                            self.assertEqual(self.model.projected_rows, [])

    def test_cp_selected_rows_use_same_projection_and_vocab_crop(self):
        selected = self.hidden[[5, 1]]
        with torch.no_grad():
            result = run_lm_head(
                self.model,
                self.packed,
                selected,
                self.indexes,
                already_selected=True,
                output_vocab_size=13,
            )
            expected = self.model.nvfp4_lm_head(selected).float()[:, :13]
        torch.testing.assert_close(result["logits"], expected)
        torch.testing.assert_close(result["all_hidden_states"], selected)
        self.assertEqual(self.model.projected_rows, [2])

    def test_tp_joins_bf16_vocabulary_shards_before_fp32(self):
        communicated = []

        def gather(receivers, mode, senders, inplace):
            # Emulate the peer shard while exercising the actual C++ gather
            # adapter, rank-major layout conversion, crop and FP32 conversion.
            buffer = receivers[0]
            local = buffer.reshape(2, 2, 16)[0].clone()
            communicated.append(local)
            buffer.reshape(2, 2, 16).copy_(torch.stack((local, local + 16)))

        register_comm_ops(lambda *args: None, lambda *args: None, gather)
        try:
            with torch.no_grad():
                result = run_lm_head(
                    self.model,
                    self.packed,
                    self.hidden,
                    self.indexes,
                    tp_size=2,
                    output_vocab_size=29,
                )
                local = self.model.nvfp4_lm_head(self.hidden[[5, 1]])
            self.assertEqual(len(communicated), 1)
            self.assertEqual(communicated[0].dtype, torch.bfloat16)
            expected = torch.cat((local, local + 16), dim=1)[:, :29].float()
            torch.testing.assert_close(result["logits"], expected)
        finally:
            clear_comm_ops()

    def test_row_indexes_ignored_at_tp1_keep_legacy_all_logits(self):
        # need_all_logits with explicit row indexes at tp==1: the local lm head still
        # runs the full GEMM, but with no TP gather to shrink the row metadata is
        # ignored and the legacy full all_logits comes back.
        scoring = torch.tensor([0, 2, 3], dtype=torch.int32, pin_memory=True)
        with torch.no_grad():
            result = run_lm_head(
                self.model,
                self.packed,
                self.hidden,
                self.indexes,
                need_all_logits=True,
                all_logits_row_indexes=scoring,
                output_vocab_size=13,
            )
        full = self.model.nvfp4_lm_head(self.hidden).float()[:, :13]
        # tp==1 keeps the legacy contract: full-window all_logits, token-offset
        # dispatch (all_logits_lm_rows == -1) and no sharded scoring fields. Row
        # selection only engages under TP where it shrinks the gather payload.
        torch.testing.assert_close(result["all_logits"], full, rtol=0, atol=0)
        torch.testing.assert_close(result["logits"], full[[5, 1]], rtol=0, atol=0)
        self.assertEqual(result["all_logits_lm_rows"], -1)
        self.assertIsNone(result["all_logits_lse"])
        self.assertIsNone(result["all_logits_topk_ids"])
        # hidden_states/all_hidden_states keep the legacy full-sequence contract.
        torch.testing.assert_close(result["hidden_states"], self.hidden, rtol=0, atol=0)
        torch.testing.assert_close(
            result["all_hidden_states"], self.hidden, rtol=0, atol=0
        )
        # The projection ran on all 6 token rows (full GEMM, select-after-mm).
        self.assertEqual(self.model.projected_rows[-1], 6)

    def test_row_selected_empty_indexes_keep_only_lm_rows(self):
        # tp==1: a defined-but-empty row set still takes the legacy path — the full
        # all_logits window is returned unchanged.
        scoring = torch.tensor([], dtype=torch.int32, pin_memory=True)
        with torch.no_grad():
            result = run_lm_head(
                self.model,
                self.packed,
                self.hidden,
                self.indexes,
                need_all_logits=True,
                all_logits_row_indexes=scoring,
                output_vocab_size=13,
            )
        full = self.model.nvfp4_lm_head(self.hidden).float()[:, :13]
        torch.testing.assert_close(result["all_logits"], full, rtol=0, atol=0)
        torch.testing.assert_close(result["logits"], full[[5, 1]], rtol=0, atol=0)
        self.assertEqual(result["all_logits_lm_rows"], -1)

    def test_row_selected_tp_gather_sends_only_needed_rows(self):
        # tp_size>1 row selection never gathers [rows, vocab]: only the 2 lm rows ship
        # whole, and the scoring rows ride one compact partial-reduction payload
        # [lse | target | topk_values | topk_ids-as-f32]. The mock emulates a peer whose
        # shard equals this shard's columns + 16 (its global ids are +16), so every merged
        # output is computable from rank 0's local logits alone.
        gathered_shapes = []
        # Partials are produced for the scoring rows only (1 here); the lm prefix rows
        # of the merged outputs are placeholders never consumed downstream.
        R, kl = 1, 3

        def gather(receivers, mode, senders, inplace):
            # tpSyncEmbeddingOrLogits uses the inplace form: rank 0's data already sits in
            # the first slot of the recv buffer and `senders` is empty.
            buffer = receivers[0]
            gathered_shapes.append(tuple(buffer.shape))
            local = buffer.view(2, buffer.size(0), -1)[0].clone()
            if buffer.size(0) > 1:
                # lm-row gather: [rows, 2*16], rank1 = rank0 + 16.
                buffer.view(2, buffer.size(0), -1).copy_(
                    torch.stack((local, local + 16))
                )
            else:
                # Scoring payload [1, 2P]: synthesize rank1's partials from rank0's
                # layout [lse R | tgt R | tkv R*kl | tki R*kl].
                p = local.reshape(-1)
                l0, t0, v0, i0 = p.split([R, R, R * kl, R * kl])
                # Label 7 lives on rank0's shard, so rank1's target column is -inf.
                # Rank1's shard only owns global ids [16, 29): a real partial can never
                # emit padded ids 29-31, so the emulation wraps rank0's local top-k ids
                # into the peer's valid 13-column range instead of a raw +16 shift.
                peer_ids = i0.view(torch.int32).remainder(13) + 16
                r1 = torch.cat(
                    (
                        l0 + 16,
                        torch.full_like(t0, float("-inf")),
                        v0 + 16,
                        peer_ids.view(torch.float32),
                    )
                )
                buffer.view(2, -1).copy_(torch.stack((p, r1)))

        register_comm_ops(lambda *args: None, lambda *args: None, gather)
        try:
            scoring = torch.tensor([4], dtype=torch.int32, pin_memory=True)
            labels = torch.tensor([7], dtype=torch.int32, pin_memory=True)
            with torch.no_grad():
                result = run_lm_head(
                    self.model,
                    self.packed,
                    self.hidden,
                    self.indexes,
                    need_all_logits=True,
                    all_logits_row_indexes=scoring,
                    all_logits_row_labels=labels,
                    all_logits_top_k=3,
                    tp_size=2,
                    output_vocab_size=29,
                )
            # One 2x16 lm-row gather + one 16-wide payload for the single scoring row
            # (P = 1*(2+2*3) = 8 floats -> [1, 2*8]); never a [rows, vocab] gather.
            self.assertEqual(gathered_shapes, [(2, 32), (1, 16)])
            local = self.model.nvfp4_lm_head(self.hidden)[[5, 1, 4]]
            # The lm-row gather communicates bf16 (peer adds 16 on the bf16 grid);
            # the fp32 scoring payload is exact.
            expected = torch.cat((local[:2], local[:2] + 16), dim=1)[:, :29].float()
            torch.testing.assert_close(result["logits"], expected, rtol=0, atol=0)
            self.assertIsNone(result["all_logits"])
            self.assertEqual(result["all_logits_lm_rows"], 2)

            # Merged layout [lm(2) placeholder zeros ++ scoring(1)]: only row 2 is
            # reduced; the lm prefix rows are never consumed downstream.
            local_f = local.float()
            scoring_row_lse = local_f[2].logsumexp(-1)
            merged_lse = torch.logsumexp(
                torch.stack((scoring_row_lse, scoring_row_lse + 16), -1), -1
            )
            expected_lse = torch.cat((torch.zeros(2, device="cuda"), merged_lse[None]))
            torch.testing.assert_close(
                result["all_logits_lse"], expected_lse, rtol=1e-6, atol=1e-6
            )
            expected_target = torch.tensor(
                [0.0, 0.0, local_f[2][7].item()], device="cuda"
            )
            torch.testing.assert_close(
                result["all_logits_target_logits"], expected_target, rtol=0, atol=0
            )
            # Union's top-3 are rank1's entries: rank0's local top-3 values shifted +16,
            # ids wrapped into rank1's valid [16, 29) global range.
            v0, i0 = local_f[2].topk(3, -1)
            expected_topv = torch.cat(
                (torch.zeros(2, 3, device="cuda"), (v0 + 16)[None])
            )
            expected_topi = torch.cat(
                (
                    torch.zeros(2, 3, dtype=torch.int32, device="cuda"),
                    (i0.to(torch.int32).remainder(13) + 16)[None],
                )
            )
            torch.testing.assert_close(
                result["all_logits_topk_values"], expected_topv, rtol=0, atol=0
            )
            torch.testing.assert_close(
                result["all_logits_topk_ids"], expected_topi, rtol=0, atol=0
            )
        finally:
            clear_comm_ops()

    def test_scoring_partial_multi_chunk_forward_matches_single_chunk(self):
        # scoring_partial_chunk_rows reaches the forward loop as a synchronized field
        # (the value the shape-hint broadcast would deliver); chunk=2 over the 3
        # scoring rows issues 2 payload all-gathers plus the lm-row gather, and the
        # merged scoring outputs must equal the single-chunk run bitwise.
        gathered_shapes = []
        kl = 3

        def gather(receivers, mode, senders, inplace):
            buffer = receivers[0]
            gathered_shapes.append(tuple(buffer.shape))
            local = buffer.view(2, buffer.size(0), -1)[0].clone()
            if buffer.size(0) > 1:
                # lm-row gather [2, 2*16]: peer shard = local + 16.
                buffer.view(2, buffer.size(0), -1).copy_(
                    torch.stack((local, local + 16))
                )
            else:
                # payload [1, 2*P], P = rows*(2+2*kl): peer = +16 values, valid ids.
                rows = (buffer.size(1) // 2) // (2 + 2 * kl)
                p = local.reshape(-1)
                l0, t0, v0, i0 = p.split([rows, rows, rows * kl, rows * kl])
                r1 = torch.cat(
                    (
                        l0 + 16,
                        torch.full_like(t0, float("-inf")),
                        v0 + 16,
                        (i0.view(torch.int32).remainder(13) + 16).view(torch.float32),
                    )
                )
                buffer.view(2, -1).copy_(torch.stack((p, r1)))

        register_comm_ops(lambda *args: None, lambda *args: None, gather)
        try:
            scoring = torch.tensor([0, 2, 3], dtype=torch.int32, pin_memory=True)
            labels = torch.tensor([7, -1, 4], dtype=torch.int32, pin_memory=True)
            kwargs = dict(
                need_all_logits=True,
                all_logits_row_indexes=scoring,
                all_logits_row_labels=labels,
                all_logits_top_k=3,
                tp_size=2,
                output_vocab_size=29,
            )
            with torch.no_grad():
                whole = run_lm_head(
                    self.model, self.packed, self.hidden, self.indexes, **kwargs
                )
                gathered_shapes.clear()
                chunked = run_lm_head(
                    self.model,
                    self.packed,
                    self.hidden,
                    self.indexes,
                    scoring_partial_chunk_rows=2,
                    **kwargs,
                )
            # 3 scoring rows with chunk=2: the lm-row gather [2, 2*16] ships first,
            # then chunks of 2+1 payload widths P=rows*(2+2*3) -> 16, 8.
            self.assertEqual(gathered_shapes, [(2, 32), (1, 32), (1, 16)])
            for key in (
                "all_logits_lse",
                "all_logits_target_logits",
                "all_logits_topk_values",
                "all_logits_topk_ids",
                "logits",
            ):
                torch.testing.assert_close(chunked[key], whole[key], rtol=0, atol=0)
            self.assertEqual(chunked["all_logits_lm_rows"], 2)
        finally:
            clear_comm_ops()

    def test_int8_check_tp2_sharded_scoring_then_decode(self):
        # tp>1 + INT8 GEMV + CHECK on a dense bf16 head: a scoring prefill (5 selected
        # rows, M=6 <= 8 so the GEMV engages) runs the sharded check arm — int8 and ref
        # payload gathers per chunk plus the ref lm-row gather — then an ordinary decode
        # forward keeps the check stats accumulating. Served values must be the bf16
        # reference merge, not the int8 output.
        import os

        old_env = {
            k: os.environ.get(k)
            for k in ("RTP_LLM_INT8_LM_HEAD", "RTP_LLM_INT8_LM_HEAD_CHECK")
        }
        os.environ["RTP_LLM_INT8_LM_HEAD"] = "1"
        os.environ["RTP_LLM_INT8_LM_HEAD_CHECK"] = "1"
        try:
            hidden = (
                torch.arange(6 * 256, device="cuda").reshape(6, 256).to(torch.bfloat16)
                / 512
            )
            kernel = (
                torch.arange(16 * 256, device="cuda")
                .reshape(16, 256)
                .to(torch.bfloat16)
                / 512
            )
            gathered_shapes = []
            kl = 2  # shard_k = min(k_max, vocab_l) = min(2, 16)

            def gather(receivers, mode, senders, inplace):
                buffer = receivers[0]
                gathered_shapes.append(tuple(buffer.shape))
                local = buffer.view(2, buffer.size(0), -1)[0].clone()
                if buffer.size(0) > 1:
                    # fp32 lm-row gather [rows, 2*16]: peer shard = local + 16.
                    buffer.view(2, buffer.size(0), -1).copy_(
                        torch.stack((local, local + 16))
                    )
                else:
                    rows = (buffer.size(1) // 2) // (2 + 2 * kl)
                    p = local.reshape(-1)
                    l0, t0, v0, i0 = p.split([rows, rows, rows * kl, rows * kl])
                    r1 = torch.cat(
                        (
                            l0 + 16,
                            torch.full_like(t0, float("-inf")),
                            v0 + 16,
                            (i0.view(torch.int32).remainder(13) + 16).view(
                                torch.float32
                            ),
                        )
                    )
                    buffer.view(2, -1).copy_(torch.stack((p, r1)))

            register_comm_ops(lambda *args: None, lambda *args: None, gather)
            try:
                scoring = torch.tensor([0, 2, 3], dtype=torch.int32, pin_memory=True)
                labels = torch.tensor([7, -1, 4], dtype=torch.int32, pin_memory=True)
                with torch.no_grad():
                    result = run_lm_head(
                        self.model,
                        kernel,
                        hidden,
                        self.indexes,
                        need_all_logits=True,
                        all_logits_row_indexes=scoring,
                        all_logits_row_labels=labels,
                        all_logits_top_k=2,
                        tp_size=2,
                        output_vocab_size=29,
                        second_forward=True,
                    )
            finally:
                clear_comm_ops()

            # Check arm, one chunk of 3 scoring rows: int8 payload gather then ref
            # payload gather (P = 3*(2+2*2) = 18 floats -> [1, 36]), then the ref
            # lm-row gather [2, 32]. The decode forward adds its int8 logits gather
            # and its ref gather.
            self.assertEqual(
                gathered_shapes,
                [(1, 36), (1, 36), (2, 32), (2, 32), (2, 32)],
            )
            stats = result.get("int8_lm_head_check_stats")
            self.assertIsNotNone(stats)
            self.assertTrue(stats.is_cuda)
            # Scoring forward checked the 3 scoring rows; the decode forward checked
            # its 2 lm rows.
            self.assertEqual(int(stats[0].item()), 5)
            self.assertGreaterEqual(int(stats[1].item()), 0)
            self.assertLessEqual(int(stats[1].item()), int(stats[0].item()))

            # Check mode serves the bf16 reference merge: lm rows = gathered ref, and
            # the merged top-k over the 3 scoring rows is the peer-dominated ref top-k.
            # (rtol/atol 1e-4: the test-side fp32 GEMM and the served bf16-accumulate
            # ref can differ only by accumulation order, both over identical bf16
            # inputs.)
            ref = hidden.float() @ kernel.float().t()
            ref_lm = torch.cat((ref[[5, 1]], ref[[5, 1]] + 16), dim=1)[:, :29]
            torch.testing.assert_close(result["logits"], ref_lm, rtol=1e-4, atol=1e-4)
            self.assertIsNone(result["all_logits"])
            self.assertEqual(result["all_logits_lm_rows"], 2)

            # Merged layout [lm(2) placeholder zeros ++ scoring(3)]: the lm prefix
            # rows are never consumed downstream.
            local_f = ref[[0, 2, 3]]
            v0, i0 = local_f.topk(2, -1)
            # Merged top-2 re-sorts ALL candidates (value desc, id asc): rank0's own
            # top-1 can legitimately beat the peer's top-2 when the local top-2 gap
            # exceeds 16, so the expected merge cannot assume the peer sweeps both
            # slots.
            cand_v = torch.cat((v0, v0 + 16), dim=1)
            cand_i = torch.cat(
                (i0.to(torch.int32), i0.to(torch.int32).remainder(13) + 16), dim=1
            )
            idx = torch.argsort(cand_i, dim=1, stable=True)
            cand_v = cand_v.gather(1, idx)
            cand_i = cand_i.gather(1, idx)
            idx = torch.argsort(-cand_v, dim=1, stable=True)
            cand_v = cand_v.gather(1, idx)
            cand_i = cand_i.gather(1, idx)
            expected_topv = torch.cat((torch.zeros(2, 2, device="cuda"), cand_v[:, :2]))
            expected_topi = torch.cat(
                (torch.zeros(2, 2, dtype=torch.int32, device="cuda"), cand_i[:, :2])
            )
            torch.testing.assert_close(
                result["all_logits_topk_values"], expected_topv, rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                result["all_logits_topk_ids"], expected_topi, rtol=0, atol=0
            )
            # Label 7 is on rank0's shard -> its target logit is rank0's local value;
            # the -1 label row merges -inf.
            l0 = local_f.logsumexp(-1)
            merged_lse = torch.logsumexp(torch.stack((l0, l0 + 16), -1), -1)
            expected_lse = torch.cat((torch.zeros(2, device="cuda"), merged_lse))
            torch.testing.assert_close(
                result["all_logits_lse"], expected_lse, rtol=1e-4, atol=1e-4
            )
            expected_target = torch.tensor(
                [0.0, 0.0, local_f[0][7].item(), float("-inf"), local_f[2][4].item()],
                device="cuda",
            )
            torch.testing.assert_close(
                result["all_logits_target_logits"],
                expected_target,
                rtol=1e-4,
                atol=1e-4,
            )
            # The decode forward served its own reference logits.
            torch.testing.assert_close(result["logits2"], ref_lm, rtol=1e-4, atol=1e-4)
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_invalid_python_projection_is_rejected_at_boundary(self):
        self.model.compute_quantized_logits = lambda hidden: hidden
        with self.assertRaisesRegex(RuntimeError, "BF16 local logits"):
            run_lm_head(self.model, self.packed, self.hidden, self.indexes)

    def test_non_bf16_projection_is_rejected_before_tp_gather(self):
        gather = Mock()
        register_comm_ops(lambda *args: None, lambda *args: None, gather)
        try:
            for dtype in (torch.float16, torch.float32):
                for already_selected in (False, True):
                    with self.subTest(dtype=dtype, already_selected=already_selected):
                        self.model.compute_quantized_logits = (
                            lambda hidden: torch.zeros(
                                (hidden.shape[0], 16), device=hidden.device, dtype=dtype
                            )
                        )
                        with self.assertRaisesRegex(RuntimeError, "BF16 local logits"):
                            run_lm_head(
                                self.model,
                                self.packed,
                                self.hidden,
                                self.indexes,
                                already_selected=already_selected,
                                tp_size=2,
                            )
            gather.assert_not_called()
        finally:
            clear_comm_ops()

    def test_public_bf16_projection_callback_is_supported(self):
        class PublicProjection:
            def initialize(inner, resources):
                return True

            def forward(inner, inputs, fmha_impl=None):
                raise AssertionError("Only projection should be called")

            def compute_quantized_logits(inner, hidden):
                return self.model.compute_quantized_logits(hidden)

        with torch.no_grad():
            result = run_lm_head(
                PublicProjection(), self.packed, self.hidden, self.indexes
            )
            expected = self.model.compute_quantized_logits(self.hidden[[5, 1]]).float()
        torch.testing.assert_close(result["logits"], expected)

    def test_int8_check_stats_survive_scoring_then_decode(self):
        # INT8 GEMV + CHECK: a short scoring (need_all_logits) prefill must initialize the
        # check stats on the same CUDA device as the served reference, and a following
        # ordinary decode forward keeps accumulating without a device-mismatch crash.
        # K must be a multiple of 256 and M*K*2 bytes fit in per-block shared memory for
        # the GEMV to engage.
        import os

        old_env = {
            k: os.environ.get(k)
            for k in ("RTP_LLM_INT8_LM_HEAD", "RTP_LLM_INT8_LM_HEAD_CHECK")
        }
        os.environ["RTP_LLM_INT8_LM_HEAD"] = "1"
        os.environ["RTP_LLM_INT8_LM_HEAD_CHECK"] = "1"
        try:
            hidden = (
                torch.arange(6 * 256, device="cuda").reshape(6, 256).to(torch.bfloat16)
                / 512
            )
            kernel = (
                torch.arange(16 * 256, device="cuda")
                .reshape(16, 256)
                .to(torch.bfloat16)
                / 512
            )
            scoring = torch.tensor([0, 2, 3], dtype=torch.int32, pin_memory=True)
            labels = torch.tensor([7, -1, 4], dtype=torch.int32, pin_memory=True)
            with torch.no_grad():
                result = run_lm_head(
                    self.model,
                    kernel,
                    hidden,
                    self.indexes,
                    need_all_logits=True,
                    all_logits_row_indexes=scoring,
                    all_logits_row_labels=labels,
                    all_logits_top_k=2,
                    output_vocab_size=13,
                    second_forward=True,
                )
            stats = result.get("int8_lm_head_check_stats")
            self.assertIsNotNone(stats)
            self.assertTrue(stats.is_cuda)
            # First forward checked all 6 served (legacy tp==1) rows; the decode
            # forward checked the 2 lm_output_indexes rows.
            self.assertEqual(int(stats[0].item()), 8)
            self.assertGreaterEqual(int(stats[1].item()), 0)
            # Mismatch count is bounded by the checked row count.
            self.assertLessEqual(int(stats[1].item()), int(stats[0].item()))
            # Check mode serves the bf16 reference logits.
            ref = hidden.float() @ kernel.float().t()
            torch.testing.assert_close(
                result["logits"], ref[[5, 1]][:, :13], rtol=1e-2, atol=1e-2
            )
            torch.testing.assert_close(
                result["all_logits"], ref[:, :13], rtol=1e-2, atol=1e-2
            )
            torch.testing.assert_close(
                result["logits2"], ref[[5, 1]][:, :13], rtol=1e-2, atol=1e-2
            )
            self.assertIsNone(result["all_logits2"])
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


if __name__ == "__main__":
    unittest.main()
