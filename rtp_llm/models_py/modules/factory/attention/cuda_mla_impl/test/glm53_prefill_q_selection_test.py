"""CP Q preselection: numerical kernels, output layout, and allocation lifetime.

Only cross-rank transport/cache storage are substituted. GPU tests run the real
RoPE, input/output BMM, FlashMLA sparse attention, and Triton scatter kernels.
The CP2 regression uses the C++ input splitter and SparseMlaParams on each rank's
layout, but runs ranks sequentially on one GPU, not with distributed transport.
"""

import contextlib
import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.cp_utils import (
    generate_q_indices,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl import (
    flashmla_sparse_cp_impl as cp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.glm53_prefill_workspace import (
    Glm53PrefillWorkspace,
)
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.rope_emb_new import (
    NewMlaRotaryEmbeddingOp,
)
from rtp_llm.ops.compute_ops import rtp_llm_ops
from rtp_llm.utils.model_weight import W


def _minimal_op(ids, identity=False):
    op = cp.SparseMlaFp8CPOp.__new__(cp.SparseMlaFp8CPOp)
    op.total_local_ids = ids
    op.total_local_ids_is_identity = identity
    op.scatter_reverse_indices = None
    op.use_cuda_graph = False
    op.kv_restore_unpad_indices = torch.arange(3, device=ids.device)
    op.kv_cache_write_op = SimpleNamespace(forward=Mock())
    op.write_cache_store_impl = None
    op.attn_inputs = None
    op.mla_params = None
    op.kv_cache_sharded = False
    op._gather = None
    return op


@contextlib.contextmanager
def _local_transport():
    with patch.object(cp, "all_gather", side_effect=lambda x, **_: x) as gather:
        with patch.object(cp.common, "apply_write_cache_store"):
            yield gather


class QPreselectionControlTest(unittest.TestCase):
    """CPU tests make the API/fallback/lifetime contracts independently visible."""

    def _wrapper(self, ids, rows, *, identity=False, workspace=True, graph=False):
        impl = cp.SparseMlaCpImpl.__new__(cp.SparseMlaCpImpl)
        impl.num_heads, impl.nope_head_dim = 2, 3
        impl.rope_head_dim, impl.kv_lora_rank = 1, 5
        impl._glm53_prefill_workspace = Glm53PrefillWorkspace() if workspace else None
        impl.fmha_params = SimpleNamespace(batch_indice_d=torch.zeros(rows))
        impl.rope_params = None
        impl.fmha_impl = SimpleNamespace(
            full_rope_pos_ids=None,
            total_local_ids=ids,
            total_local_ids_is_identity=identity,
            use_cuda_graph=graph,
        )
        return impl

    def test_guard_and_small_q_lifetime(self):
        cases = [
            ("permuted", [0, 2, 1, 3], False, True, False, "valid", True),
            ("padding", [0, 2, 1], False, True, False, "valid", True),
            ("identity", [0, 1, 2, 3], True, True, False, "valid", False),
            # An inconsistent identity flag conservatively retains the legacy path.
            ("short_identity", [0, 1, 2], True, True, False, "valid", False),
            ("disabled", [0, 2, 1, 3], False, False, False, "valid", False),
            ("graph", [0, 2, 1, 3], False, True, True, "valid", False),
            ("no_ids", None, False, True, False, "valid", False),
            ("empty_ids", [], False, True, False, "valid", False),
            ("no_topk", [0, 2, 1, 3], False, True, False, "none", False),
            ("empty_topk", [0, 2, 1, 3], False, True, False, "empty", False),
        ]
        for name, index, identity, workspace, graph, kind, selected in cases:
            with self.subTest(name=name):
                ids = None if index is None else torch.tensor(index, dtype=torch.long)
                impl = self._wrapper(
                    ids, 4, identity=identity, workspace=workspace, graph=graph
                )
                q = torch.arange(32, dtype=torch.float32).view(4, 2, 4)
                expected = q[ids].clone() if selected else q.clone()
                topk = (
                    None
                    if kind == "none"
                    else torch.ones((0 if kind == "empty" else 4, 1), dtype=torch.int32)
                )
                observed = {}

                def project(q_arg, layer_id, out=None):
                    self.assertEqual(layer_id, 2)
                    torch.testing.assert_close(q_arg, expected)
                    observed["selected_ref"] = weakref.ref(q_arg)
                    if out is None:
                        out = torch.empty(q_arg.size(0), 2, 6)
                    self.assertEqual(out.shape, (q_arg.size(0), 2, 6))
                    out.fill_(1)
                    return out

                def attend(
                    q_arg, *args, original_q_len=None, scatter_out=None, **kwargs
                ):
                    self.assertEqual(original_q_len, 4 if selected else None)
                    self.assertEqual(q_arg.shape[0], len(index) if selected else 4)
                    if selected:
                        # No hidden Python reference may retain the 256-wide copy
                        # until the KV gather/attention peak.
                        self.assertIsNone(observed["selected_ref"]())
                    if workspace:
                        self.assertEqual(q_arg.data_ptr(), scatter_out.data_ptr())
                        self.assertEqual(scatter_out.shape, (4, 2, 5))
                    return None

                impl._apply_input_bmm = project
                impl.fmha_impl.forward = attend
                self.assertIsNone(
                    impl.forward(
                        q, torch.empty(4, 5), torch.empty(4, 1), object(), 2, topk
                    )
                )
                if workspace:
                    self.assertIsNone(impl._glm53_prefill_workspace._storage)

    def test_workspace_released_on_each_exception_boundary(self):
        for failure in ("rope", "input_bmm", "attention", "output_bmm"):
            with self.subTest(failure=failure):
                impl = self._wrapper(torch.tensor([0, 2, 1]), 4)
                if failure == "rope":
                    impl.fmha_impl.full_rope_pos_ids = torch.arange(4)
                    impl.rope_impl = SimpleNamespace(
                        forward=Mock(side_effect=RuntimeError(failure))
                    )
                impl._apply_input_bmm = (
                    Mock(side_effect=RuntimeError(failure))
                    if failure == "input_bmm"
                    else lambda q, layer_id, out=None: out
                )
                impl.fmha_impl.forward = (
                    Mock(side_effect=RuntimeError(failure))
                    if failure == "attention"
                    else lambda *args, **kwargs: torch.empty(4, 2, 5)
                )
                impl._apply_output_bmm = Mock(side_effect=RuntimeError(failure))
                with self.assertRaisesRegex(RuntimeError, failure):
                    impl.forward(
                        torch.empty(4, 2, 4),
                        torch.empty(4, 5),
                        torch.empty(4, 1),
                        object(),
                        0,
                        torch.ones(3, 1),
                    )
                self.assertIsNone(impl._glm53_prefill_workspace._storage)

    def test_no_q_work_still_runs_kv_collectives_and_sharded_gather(self):
        for ids, topk in (
            (torch.empty(0, dtype=torch.long), torch.empty(0, 128, dtype=torch.int32)),
            (torch.arange(3), None),
            (torch.arange(3), torch.empty(0, 128, dtype=torch.int32)),
        ):
            for sharded in (False, True):
                with self.subTest(
                    rows=ids.numel(), topk_none=topk is None, sharded=sharded
                ):
                    op = _minimal_op(ids)
                    op.kv_cache_sharded = sharded
                    op.sharded_slot_mapping = None
                    op._gather = object() if sharded else None
                    op._allocate_fused_kv = Mock(return_value=torch.empty(3, 6))
                    op._gather_sharded_kv_cache = Mock()
                    with _local_transport() as gather:
                        result = op.forward(
                            torch.empty(ids.numel(), 2, 6),
                            torch.empty(3, 5),
                            torch.empty(3, 1),
                            topk,
                            torch.zeros(3),
                            object(),
                        )
                    self.assertIsNone(result)
                    self.assertEqual(gather.call_count, 2)
                    op.kv_cache_write_op.forward.assert_called_once()
                    self.assertEqual(
                        op._gather_sharded_kv_cache.call_count, int(sharded)
                    )

    def test_preselected_input_validation(self):
        for index, selected_rows, original_rows in (
            ([0, 2], 3, 4),  # Q rows do not match selected ids.
            ([0, 2, 1], 3, 2),  # Destination is smaller than selected Q.
            ([0, 2], 2, -1),
        ):
            with self.subTest(
                index=index, selected_rows=selected_rows, original_rows=original_rows
            ):
                op = _minimal_op(torch.tensor(index))
                with _local_transport(), self.assertRaises(ValueError):
                    op.forward(
                        torch.empty(selected_rows, 2, 6),
                        torch.empty(3, 5),
                        torch.empty(3, 1),
                        torch.ones(len(index), 1),
                        torch.zeros(3),
                        object(),
                        original_q_len=original_rows,
                    )


@unittest.skipUnless(
    torch.cuda.is_available(), "real BMM/RoPE/FlashMLA/scatter coverage requires CUDA"
)
class QPreselectionCudaTest(unittest.TestCase):
    HEADS, NOPE, ROPE, LATENT, VALUE, TOPK = 64, 192, 64, 512, 256, 128

    def setUp(self):
        torch.manual_seed(1729)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.weights = [
            {
                W.mla_kc: torch.randn(
                    self.HEADS,
                    self.NOPE,
                    self.LATENT,
                    device=self.device,
                    dtype=self.dtype,
                )
                * 0.04,
                W.mla_vc: torch.randn(
                    self.HEADS,
                    self.LATENT,
                    self.VALUE,
                    device=self.device,
                    dtype=self.dtype,
                )
                * 0.03,
            }
        ]

    def _fixture(
        self, chunks, removed=(), noncontiguous=False, neox=True, *, input_lengths=None
    ):
        rows = sum(chunks)
        first, second = generate_q_indices(chunks)
        selected = [i for i in first + second if i not in removed]
        ids = torch.tensor(selected, dtype=torch.long, device=self.device)
        req_list = [r for r, length in enumerate(chunks) for _ in range(length)]
        req_ids = torch.tensor(req_list, dtype=torch.long, device=self.device)[ids]
        # Each request has a different prefix length and distinct KV contents.
        lengths = [
            19 + 7 * r + length
            for r, length in enumerate(
                chunks if input_lengths is None else input_lengths
            )
        ]
        starts = [sum(lengths[:r]) for r in range(len(chunks))]
        fused = (
            torch.randn(
                sum(lengths),
                self.LATENT + self.ROPE,
                device=self.device,
                dtype=self.dtype,
            )
            * 0.1
        )
        for r, (start, length) in enumerate(zip(starts, lengths)):
            fused[start : start + length, : self.LATENT] += r * 0.75
        topk = torch.full(
            (len(selected), self.TOPK), -1, dtype=torch.int32, device=self.device
        )
        for n, original in enumerate(selected):
            r = req_list[original]
            # Different row-specific ranges detect Q/top-k misalignment.
            count = 3 + original % 13
            valid = (torch.arange(count, device=self.device) * 3 + original) % lengths[
                r
            ]
            topk[n, :count] = valid
        q_shape = (rows, self.HEADS, self.NOPE + self.ROPE)
        q = torch.randn(q_shape, device=self.device, dtype=self.dtype) * 0.2
        if noncontiguous:
            backing = torch.empty(
                rows,
                self.HEADS * 2,
                self.NOPE + self.ROPE,
                device=self.device,
                dtype=self.dtype,
            )
            backing[:, ::2].copy_(q)
            q = backing[:, ::2]
            self.assertFalse(q.is_contiguous())
        ckv = torch.randn(rows, self.LATENT, device=self.device, dtype=self.dtype)
        kpe = torch.randn(rows, self.ROPE, device=self.device, dtype=self.dtype)
        positions = torch.arange(rows, device=self.device, dtype=torch.int32) * 3 + 11
        angles = (
            torch.arange(
                max(rows * 3 + 12, 12), device=self.device, dtype=torch.float32
            )[:, None]
            * torch.arange(1, self.ROPE // 2 + 1, device=self.device)[None, :]
            * 0.003
        )
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).contiguous()
        return SimpleNamespace(
            rows=rows,
            ids=ids,
            req_ids=req_ids,
            starts=starts,
            lengths=lengths,
            fused=fused,
            topk=topk,
            q=q,
            ckv=ckv,
            kpe=kpe,
            positions=positions,
            cos_sin=cos_sin,
            neox=neox,
        )

    def _cp2_fixture(self, input_lengths, rank):
        from rtp_llm.cpp.models.context_parallel.test import (
            libth_context_parallel_py_wrapper_test as cp_test,
        )

        actual_lengths = torch.tensor(input_lengths, dtype=torch.int32)
        # Nonzero, globally unique token IDs distinguish padding and requests.
        tokens = torch.arange(1, sum(input_lengths) + 1, dtype=torch.int32)
        local_tokens, chunks, _, _, _ = cp_test.handle_inputs_with_hidden(
            tokens,
            actual_lengths,
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=self.dtype),
            rank,
            2,
        )
        padding = chunks * 2 - actual_lengths
        params = rtp_llm_ops.SparseMlaParams()
        params.fill_cp_plan_params(
            cp_test.generate_qkv_padding_mask(chunks, padding, 2),
            cp_test.generate_qkv_restore_indices(chunks, 2),
            *generate_q_indices(chunks.tolist()),
            rank,
            local_tokens.numel(),
            actual_lengths,
            torch.tensor(
                [19 + 7 * r for r in range(len(input_lengths))], dtype=torch.int32
            ),
        )
        removed = torch.nonzero(local_tokens == 0).flatten().tolist()
        data = self._fixture(chunks.tolist(), removed, input_lengths=input_lengths)
        # Check the existing numerical fixture's Q order against the production
        # metadata, rather than assuming a rank-local arange is the selection.
        torch.testing.assert_close(data.ids, params.cp_total_local_ids, rtol=0, atol=0)
        data.ids = params.cp_total_local_ids.clone()
        local_tokens = local_tokens.to(self.device)
        torch.testing.assert_close(
            data.ids.sort().values,
            torch.nonzero(local_tokens != 0).flatten(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            local_tokens[data.ids].long(),
            params.cp_total_global_ids + 1,
            rtol=0,
            atol=0,
        )
        global_requests = torch.repeat_interleave(
            torch.arange(len(input_lengths), device=self.device),
            actual_lengths.to(device=self.device, dtype=torch.long),
        )
        torch.testing.assert_close(
            data.req_ids, global_requests[params.cp_total_global_ids], rtol=0, atol=0
        )
        global_positions = torch.cat(
            [
                torch.arange(length, device=self.device, dtype=torch.int32) + 19 + 7 * r
                for r, length in enumerate(input_lengths)
            ]
        )
        data.positions.zero_()
        data.positions[data.ids] = global_positions[params.cp_total_global_ids]
        self.assertLess(int(data.positions.max()), data.cos_sin.size(0))
        # Keep row-specific TopK and random values, and make both global token
        # identity and request-local KV position visible in the numerical oracle.
        data.q[..., 0] += local_tokens[:, None].to(self.dtype) / tokens.numel()
        data.q.mul_(20)
        for start, length in zip(data.starts, data.lengths):
            data.fused[start : start + length, : self.LATENT] += (
                torch.arange(length, device=self.device, dtype=self.dtype)[:, None]
                / length
            )
        data.fused.mul_(4)
        return data, removed

    def _impl(self, data, *, workspace, graph=False):
        op = _minimal_op(
            data.ids,
            identity=data.ids.numel() == data.rows
            and torch.equal(data.ids, torch.arange(data.rows, device=self.device)),
        )
        op.num_heads, op.kv_lora_rank, op.qk_rope_head_dim = (
            self.HEADS,
            self.LATENT,
            self.ROPE,
        )
        op.scale = (self.NOPE + self.ROPE) ** -0.5
        op.use_cuda_graph = graph
        op.kv_restore_unpad_indices = torch.arange(data.rows, device=self.device)
        op.full_rope_pos_ids = data.positions if data.rows else None
        op.kv_cache_sharded = True
        op.sharded_slot_mapping = None
        op.precomputed_req_ids = data.req_ids
        op.block_table = torch.zeros(
            len(data.lengths), 1, dtype=torch.int32, device=self.device
        )
        op._gather = SimpleNamespace(
            total_kv_len=sum(data.lengths),
            batch_size=len(data.lengths),
            workspace_starts=torch.tensor(
                data.starts, dtype=torch.int32, device=self.device
            ),
            seq_lens=torch.tensor(data.lengths, dtype=torch.int32, device=self.device),
        )
        op.precomputed_kv_offsets = op._gather.workspace_starts[op.precomputed_req_ids]
        op.sharded_kv_restore_indices = (
            torch.arange(data.fused.shape[0] - 1, -1, -1, device=self.device) + 3
        )

        def gather_sharded(cache, dst, return_rank_major=False):
            if return_rank_major:
                raw = torch.full(
                    (data.fused.shape[0] + 5, data.fused.shape[1]),
                    float("nan"),
                    dtype=data.fused.dtype,
                    device=self.device,
                )
                raw.index_copy_(0, op.sharded_kv_restore_indices, data.fused)
                return raw
            dst.copy_(data.fused)
            return None

        op._gather_sharded_kv_cache = gather_sharded
        impl = cp.SparseMlaCpImpl.__new__(cp.SparseMlaCpImpl)
        impl.num_heads, impl.nope_head_dim = self.HEADS, self.NOPE
        impl.rope_head_dim, impl.kv_lora_rank = self.ROPE, self.LATENT
        impl.weights, impl.fmha_impl = self.weights, op
        impl.fmha_params = SimpleNamespace(
            batch_indice_d=torch.zeros(data.rows, device=self.device)
        )
        impl.rope_params = None
        impl.rope_impl = NewMlaRotaryEmbeddingOp(data.cos_sin, data.neox)
        impl._glm53_prefill_workspace = Glm53PrefillWorkspace() if workspace else None
        return impl

    @staticmethod
    def _copy_preserving_strides(tensor):
        copied = torch.empty_strided(
            tensor.shape, tensor.stride(), device=tensor.device, dtype=tensor.dtype
        )
        return copied.copy_(tensor)

    def _run(self, impl, data):
        with _local_transport():
            return impl.forward(
                self._copy_preserving_strides(data.q),
                data.ckv.clone(),
                data.kpe.clone(),
                object(),
                0,
                data.topk,
            )

    def _reference(self, impl, data):
        # Independent FP32 attention over each request's own selected KV rows.
        q = data.q.clone()
        impl.rope_impl.forward(
            q[..., self.NOPE :],
            data.kpe.clone(),
            None,
            precomputed_pos_ids=data.positions,
        )
        projected = torch.cat(
            (
                torch.einsum(
                    "thd,hdk->thk",
                    q[..., : self.NOPE].float(),
                    self.weights[0][W.mla_kc].float(),
                ).to(self.dtype),
                q[..., self.NOPE :],
            ),
            dim=-1,
        )[data.ids]
        result = torch.zeros(
            data.rows, self.HEADS, self.LATENT, device=self.device, dtype=self.dtype
        )
        for row, original in enumerate(data.ids.tolist()):
            request = data.req_ids[row].item()
            index = data.topk[row]
            kv = data.fused[index[index >= 0].long() + data.starts[request]].float()
            score = projected[row].float() @ kv.T * impl.fmha_impl.scale
            result[original] = (score.softmax(-1) @ kv[:, : self.LATENT]).to(self.dtype)
        return torch.einsum(
            "thk,hkv->thv", result.float(), self.weights[0][W.mla_vc].float()
        ).to(self.dtype)

    def test_real_kernels_match_legacy_and_independent_attention(self):
        cases = [
            ("single_identity", [16], (), False, True),
            ("multi_request", [8, 12, 4], (), False, True),
            ("odd_chunks", [3, 5, 7], (), False, False),
            ("padding", [8, 6], (3, 7, 10, 13), False, True),
            ("strided_raw_q", [8, 6], (), True, False),
            ("one_selected_row", [4, 2], (0, 1, 2, 3, 4), False, True),
            ("larger_batch", [67, 71, 83], (66, 137, 220), False, True),
        ]
        for name, chunks, removed, noncontiguous, neox in cases:
            with self.subTest(case=name):
                data = self._fixture(chunks, removed, noncontiguous, neox)
                legacy = self._impl(data, workspace=False)
                optimized = self._impl(data, workspace=True)
                expected = self._run(legacy, data)
                actual = self._run(optimized, data)
                reference = self._reference(legacy, data)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.012)
                torch.testing.assert_close(actual, reference, rtol=0.025, atol=0.02)
                self.assertTrue(torch.isfinite(actual).all().item())
                self.assertEqual(actual.shape, (data.rows, self.HEADS, self.VALUE))
                if removed:
                    self.assertEqual(
                        torch.count_nonzero(actual[list(removed)]).item(), 0
                    )
                self.assertIsNone(optimized._glm53_prefill_workspace._storage)

    def test_cp2_multi_request_delayed_scatter_matches_reference(self):
        for lengths, rows, valid_rows in (
            ([256, 512], 384, (384, 384)),
            ([257, 513], 388, (382, 388)),
        ):
            for rank in range(2):
                with self.subTest(lengths=lengths, rank=rank):
                    data, removed = self._cp2_fixture(lengths, rank)
                    self.assertEqual(data.rows, rows)
                    self.assertEqual(data.ids.numel(), valid_rows[rank])
                    self.assertGreaterEqual(rows, 256)
                    self.assertFalse(
                        torch.equal(data.ids, torch.arange(rows, device=self.device))
                    )
                    legacy = self._impl(data, workspace=False)
                    with patch.dict(os.environ, {"RTP_LLM_CP_KV_INDEX_OPT": "0"}):
                        expected = self._run(legacy, data)
                    reference = self._reference(legacy, data)
                    for index_opt in ("0", "1"):
                        with self.subTest(index_opt=index_opt):
                            impl = self._impl(data, workspace=True)
                            # A globally padded batch can still give rank 1 a
                            # full local permutation; only rank 0 loses Q rows.
                            delayed = index_opt == "1" and not removed
                            with (
                                patch.dict(
                                    os.environ, {"RTP_LLM_CP_KV_INDEX_OPT": index_opt}
                                ),
                                patch.object(
                                    impl.fmha_impl,
                                    "forward",
                                    wraps=impl.fmha_impl.forward,
                                ) as attend,
                                patch.object(
                                    cp, "triton_kv_scatter", wraps=cp.triton_kv_scatter
                                ) as scatter,
                            ):
                                actual = self._run(impl, data)
                            attend.assert_called_once()
                            self.assertEqual(
                                attend.call_args.kwargs["original_q_len"], rows
                            )
                            self.assertEqual(
                                attend.call_args.kwargs["defer_output_scatter"], delayed
                            )
                            self.assertEqual(
                                attend.call_args.args[0].size(0), valid_rows[rank]
                            )
                            scatter.assert_called_once()
                            # Observe the actual real-kernel boundary: delayed
                            # scatter sees V BMM's VALUE width, not LATENT width.
                            self.assertEqual(
                                scatter.call_args.args[0].shape,
                                (
                                    valid_rows[rank],
                                    self.HEADS,
                                    self.VALUE if delayed else self.LATENT,
                                ),
                            )
                            torch.testing.assert_close(
                                scatter.call_args.args[1], data.ids, rtol=0, atol=0
                            )
                            self.assertEqual(scatter.call_args.args[2], rows)
                            torch.testing.assert_close(
                                actual, expected, rtol=0.02, atol=0.012
                            )
                            torch.testing.assert_close(
                                actual, reference, rtol=0.025, atol=0.02
                            )
                            self.assertTrue(torch.isfinite(actual).all().item())
                            self.assertEqual(
                                actual.shape, (rows, self.HEADS, self.VALUE)
                            )
                            if removed:
                                self.assertEqual(
                                    torch.count_nonzero(actual[removed]).item(), 0
                                )
                            self.assertIsNone(impl._glm53_prefill_workspace._storage)
                            # The same oracle must reject a row-order error,
                            # including transitions between requests' Q halves.
                            wrong_order = actual.clone()
                            wrong_order[data.ids] = actual[data.ids.roll(1)]
                            with self.assertRaises(AssertionError):
                                torch.testing.assert_close(
                                    wrong_order, reference, rtol=0.025, atol=0.02
                                )

    def test_preselected_projection_reaches_attention_without_second_copy(self):
        data = self._fixture([7, 9], removed=(6, 15))
        impl = self._impl(data, workspace=True)
        original_project = impl._apply_input_bmm
        original_attend = impl.fmha_impl._attend_gather
        original_forward = impl.fmha_impl.forward
        # Compute the old order independently: RoPE -> full projection -> gather.
        reference_q = data.q.clone()
        impl.rope_impl.forward(
            reference_q[..., self.NOPE :],
            data.kpe.clone(),
            None,
            precomputed_pos_ids=data.positions,
        )
        expected_q = original_project(reference_q, 0)[data.ids]
        observed = {}

        def project(q, layer_id, out=None):
            self.assertEqual(q.shape, (data.ids.numel(), self.HEADS, 256))
            observed["selected"] = weakref.ref(q)
            observed["workspace_ptr"] = out.data_ptr()
            return original_project(q, layer_id, out)

        def forward(q, *args, original_q_len=None, scatter_out=None, **kwargs):
            self.assertIsNone(observed["selected"]())
            self.assertEqual(original_q_len, data.rows)
            self.assertEqual(q.data_ptr(), observed["workspace_ptr"])
            self.assertEqual(scatter_out.data_ptr(), observed["workspace_ptr"])
            return original_forward(
                q,
                *args,
                original_q_len=original_q_len,
                scatter_out=scatter_out,
                **kwargs,
            )

        def attend(q, *args):
            self.assertEqual(q.data_ptr(), observed["workspace_ptr"])
            # A nearly uniform attention distribution must not hide reordered Q.
            torch.testing.assert_close(q, expected_q, rtol=0.008, atol=0.002)
            return original_attend(q, *args)

        impl._apply_input_bmm = project
        impl.fmha_impl.forward = forward
        impl.fmha_impl._attend_gather = attend
        actual = self._run(impl, data)
        torch.testing.assert_close(
            actual,
            self._run(self._impl(data, workspace=False), data),
            rtol=0.02,
            atol=0.012,
        )

    def test_nonuniform_attention_exposes_incorrect_q_order(self):
        data = self._fixture([9, 13], removed=(8, 21))
        # Keep Q*K large enough for nonuniform softmax, without amplifying the
        # shared KV values so much that BF16 output quantization dominates the
        # independent FP32 oracle (MLA uses the same latent tensor as K and V).
        data.q.mul_(20)
        data.fused.mul_(4)
        legacy = self._impl(data, workspace=False)
        optimized = self._impl(data, workspace=True)
        expected = self._run(legacy, data)
        actual = self._run(optimized, data)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.012)
        torch.testing.assert_close(
            actual, self._reference(legacy, data), rtol=0.025, atol=0.02
        )

        # Explicit sensitivity check: mismatched Q/top-k rows must fail the
        # exact same tolerance used by the regression assertion above.
        wrong = self._impl(data, workspace=True)
        original_project = wrong._apply_input_bmm
        wrong._apply_input_bmm = lambda q, layer_id, out=None: original_project(
            q.roll(1, dims=0), layer_id, out
        )
        wrong_output = self._run(wrong, data)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(wrong_output, expected, rtol=0.02, atol=0.012)

    def test_workspace_reuse_across_layers_and_graph_fallback(self):
        data = self._fixture([5, 9], removed=(4, 13))
        for graph in (False, True):
            with self.subTest(graph=graph):
                impl = self._impl(data, workspace=True, graph=graph)
                expected = self._run(self._impl(data, workspace=False), data)
                for _ in range(3):
                    torch.testing.assert_close(
                        self._run(impl, data), expected, rtol=0.02, atol=0.012
                    )
                    self.assertIsNone(impl._glm53_prefill_workspace._storage)

    def test_preselected_api_identity_and_noncontiguous_input(self):
        for chunks in ([8], [4, 6]):
            with self.subTest(chunks=chunks):
                data = self._fixture(chunks)
                impl = self._impl(data, workspace=False)
                projected = impl._apply_input_bmm(data.q, 0)
                with _local_transport():
                    expected = impl.fmha_impl.forward(
                        projected,
                        data.ckv,
                        data.kpe,
                        data.topk,
                        impl.fmha_params.batch_indice_d,
                        object(),
                    )
                    selected = projected[data.ids]
                    backing = torch.empty(
                        selected.size(0),
                        self.HEADS * 2,
                        576,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    backing[:, ::2].copy_(selected)
                    actual = impl.fmha_impl.forward(
                        backing[:, ::2],
                        data.ckv,
                        data.kpe,
                        data.topk,
                        impl.fmha_params.batch_indice_d,
                        object(),
                        original_q_len=data.rows,
                    )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_zero_rows_keeps_kv_participation(self):
        data = self._fixture([0])
        impl = self._impl(data, workspace=True)
        with _local_transport() as gather:
            self.assertIsNone(
                impl.forward(data.q, data.ckv, data.kpe, object(), 0, data.topk)
            )
        self.assertEqual(gather.call_count, 2)
        self.assertIsNone(impl._glm53_prefill_workspace._storage)


if __name__ == "__main__":
    unittest.main()
