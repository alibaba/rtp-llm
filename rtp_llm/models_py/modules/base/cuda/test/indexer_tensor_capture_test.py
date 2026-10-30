"""CPU orchestration tests; mocked kernels do not validate CUDA numerics.

Import production modules and bindings through Bazel runfiles. GPU kernels and
collectives are mocked; capture serialization uses the real writer.
"""

import json
import os
import sys
import tempfile
import traceback
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from rtp_llm.models_py.modules.base.cuda import indexer_launch_fusion, indexer_op
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.models_py.modules.hybrid import indexer, mla_attention
from rtp_llm.models_py.utils import fuse_config
from rtp_llm.ops import compute_ops
from rtp_llm.utils import tensor_capture, tensor_capture_indexer
from rtp_llm.utils.model_weight import W


def _cpu_topk(logits, starts, ends, output, *args):
    output.fill_(-1)
    for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
        count = min(end - start, output.shape[1])
        if count:
            output[row, :count] = torch.topk(logits[row, start:end], count).indices
    # The real op writes output and returns None.


@contextmanager
def _cpu_runtime():
    test_env = {
        key: os.environ[key] for key in ("TMPDIR", "TEST_TMPDIR") if key in os.environ
    }
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, test_env, clear=True))
        stack.enter_context(
            patch.object(indexer_op._prefill_optimization, "ENABLED", False)
        )
        ops = indexer_op.rtp_llm_ops
        for name in (
            "topk_v3",
            "dsv4_persistent_topk",
            "cp_gather_indexer_k_quant_cache",
        ):
            stack.enter_context(patch.object(ops, name))
        for name in ("topk_v3_tie_break", "dsv4_top_k_per_row_prefill"):
            stack.enter_context(patch.object(ops, name, side_effect=_cpu_topk))
        for name in (
            "cuda_graph_warmup_forward_enabled",
            "cuda_graph_capture_forward_enabled",
        ):
            stack.enter_context(patch.object(compute_ops, name, return_value=False))
        for name in ("fp8_mqa_logits", "fp8_paged_mqa_logits"):
            stack.enter_context(patch.object(indexer_op.deep_gemm, name))
        for name in ("barrier", "all_gather", "sgl_per_token_group_quant_fp8"):
            stack.enter_context(patch.object(indexer_op, name))
        stack.enter_context(patch.object(mla_attention, "all_reduce"))
        # Keep the real Torch fallback; the Triton path cannot run on CPU tensors.
        stack.enter_context(
            patch.object(indexer_launch_fusion, "try_write_topk", return_value=False)
        )
        stack.enter_context(
            patch.object(
                LinearFactory,
                "create_linear_from_weights",
                side_effect=lambda *a, **kw: torch.nn.Identity(),
            )
        )
        stack.enter_context(
            patch.object(
                indexer, "LayerNorm", side_effect=lambda *a, **kw: torch.nn.Identity()
            )
        )
        stack.enter_context(
            patch.object(fuse_config, "fuse_kernels_enabled", return_value=False)
        )
        yield SimpleNamespace(
            base=indexer_op,
            indexer=indexer,
            attention=mla_attention,
            ops=ops,
            deep=indexer_op.deep_gemm,
            compute=compute_ops,
        )


class _Token:
    def __init__(self):
        self.output = None
        self.cancel = Mock()
        self.finish = Mock(side_effect=self._finish)
        self.prepare_topk = Mock(side_effect=self._prepare_topk)

    def _prepare_topk(self, logits, *, backend, parameters):
        self.logits = logits.clone()
        self.backend = backend
        self.parameters = parameters.copy()
        return True

    def _finish(self, output):
        self.output = output.clone()


class _Context(tensor_capture_indexer.IndexerCaptureContext):
    # Mock the adapter boundary, not op-specific methods on the generic context.
    mqa_enabled = False

    def __init__(self, mqa_enabled=False):
        self.mqa_enabled = mqa_enabled
        self.context = SimpleNamespace(
            identity={
                "requests": [
                    {"q_len": 3, "kv_len": 2},
                    {"q_len": 2, "kv_len": 1},
                ]
            },
            selected=Mock(return_value=True),
        )
        self.calls = []
        self.tokens = []
        self.note_skip = Mock()
        self.begin_topk = Mock(side_effect=self._begin)
        self.begin_mqa = Mock(side_effect=self._begin)
        self.begin_paged_topk = Mock(side_effect=self._begin)
        self.begin_paged_mqa = Mock(side_effect=self._begin)

    def _begin(self, **kwargs):
        self.calls.append(
            {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
        )
        token = _Token()
        self.tokens.append(token)
        return token


class IndexerTensorCaptureTest(unittest.TestCase):
    def setUp(self):
        runtime = _cpu_runtime()
        self.s = runtime.__enter__()
        self.addCleanup(runtime.__exit__, None, None, None)
        self.op = self.s.base.IndexerOp(2, 128, 3, 64, layer_idx=7)
        self.context = _Context()
        self.logits = torch.arange(32, dtype=torch.float32).view(2, 16)[:, ::2]
        self.starts = torch.tensor([1, 3], dtype=torch.int32)
        self.ends = torch.tensor([5, 5], dtype=torch.int32)
        self.output = torch.empty((2, 3), dtype=torch.int32)

    def run_topk(self, context=None, max_seq_len=100, capture_token=None):
        return self.op._run_prefill_topk(
            self.logits,
            self.starts,
            self.ends,
            self.output,
            max_seq_len,
            context=context,
            query_row_start=2,
            total_query_rows=5,
            capture_token=capture_token,
        )

    def ragged(self, context, rows=5, mqa_effect=None, **input_flags):
        # Two requests: query lengths [3, 2], prefixes [2, 1]. Chunk [2, 4)
        # crosses their boundary. KV length is 8, not query length 5.
        q = torch.zeros((rows, 2, 128))
        q[:, 0, 0] = torch.arange(rows)
        q = q.to(torch.float8_e4m3fn)
        scores = torch.arange(80, dtype=torch.float32).view(5, 16)[:, ::2]

        def gather(cache, key, scale, *args):
            key.copy_(torch.ones(key.shape).to(torch.float8_e4m3fn))
            scale.view(torch.float32).fill_(1)

        def mqa(q, kv, weights, starts, ends, **kwargs):
            logits = scores[int(q[0, 0, 0]) : int(q[-1, 0, 0]) + 1]
            if mqa_effect is not None:
                mqa_effect(q, kv, weights, starts, ends, logits)
            return logits

        self.s.ops.cp_gather_indexer_k_quant_cache.side_effect = gather
        self.s.deep.fp8_mqa_logits.side_effect = mqa
        params = SimpleNamespace(
            prefill_total_kv_tokens=8,
            prefill_max_kv_len=5,
            ks=torch.tensor([0, 0, 0, 5, 5], dtype=torch.int32)[:rows],
            ke=torch.tensor([3, 4, 5, 7, 8], dtype=torch.int32)[:rows],
            topk_indices_offset=torch.tensor([0, 0, 0, 5, 5], dtype=torch.int32)[:rows],
        )
        inputs = SimpleNamespace(
            tensor_capture_context=context,
            is_prefill=True,
            kv_cache_kernel_block_id_device=torch.zeros((2, 1), dtype=torch.int32),
            cu_kv_seqlens=torch.tensor([0, 5, 8], dtype=torch.int32),
        )
        inputs.__dict__.update(input_flags)
        cache = SimpleNamespace(
            kv_scale_base=torch.empty((2, 64, 132), dtype=torch.uint8)
        )
        with patch.dict(os.environ, {"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "2"}):
            result = self.op._get_topk_ragged(
                q, torch.ones((rows, 2, 1)), cache, params, inputs
            )
        return result, scores

    def test_per_row_dispatch_records_exact_arguments_and_none_return(self):
        for value, force in (("1", True), ("0", False), ("false", True)):
            with (
                self.subTest(radix=value),
                patch.dict(os.environ, {"DSV4_PREFILL_TOPK_FORCE_RADIX": value}),
            ):
                context = _Context()
                self.assertIsNone(self.run_topk(context))
                args = self.s.ops.dsv4_top_k_per_row_prefill.call_args.args
                self.assertIs(args[3], self.output)
                self.assertEqual(args[4:], (2, 16, 2, 3, force))
                self.assertEqual(
                    context.calls[0]["parameters"],
                    {
                        "num_rows": 2,
                        "stride0": 16,
                        "stride1": 2,
                        "force_radix_sort": force,
                    },
                )
                self.assertEqual(context.calls[0]["layer"], 7)
                self.assertEqual(context.calls[0]["backend"], "dsv4_per_row")
                torch.testing.assert_close(context.tokens[0].output, self.output)
        self.assertEqual(self.s.ops.dsv4_top_k_per_row_prefill.call_count, 3)
        self.s.ops.topk_v3_tie_break.assert_not_called()

    def test_v3_dispatch_clamps_actual_max_length(self):
        self.op._prefill_topk_backend = "topk_v3_tie_break"
        for supplied, expected in ((100, 8), (0, 1), (-5, 1), (5, 5)):
            with self.subTest(max_seq_len=supplied):
                context = _Context()
                self.run_topk(context, supplied)
                self.assertEqual(
                    self.s.ops.topk_v3_tie_break.call_args.args[4:], (3, expected)
                )
                self.assertEqual(
                    context.calls[0]["parameters"],
                    {"max_seq_len": expected, "stride0": 16, "stride1": 2},
                )
                torch.testing.assert_close(context.tokens[0].output, self.output)
        self.s.ops.dsv4_top_k_per_row_prefill.assert_not_called()

    def test_empty_short_exact_long_and_tied_valid_domains(self):
        self.logits = torch.tensor(
            [
                [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0],
                [99.0, -2.0, 99.0, 99.0, 99.0, 99.0],
                [99.0, -3.0, -1.0, -2.0, 99.0, 99.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 99.0],
                [99.0, 1.0, 1.0, 1.0, 1.0, 99.0],
            ]
        )
        self.starts = torch.tensor([0, 1, 1, 0, 1], dtype=torch.int32)
        self.ends = torch.tensor([0, 2, 4, 5, 5], dtype=torch.int32)
        self.output = torch.empty((5, 3), dtype=torch.int32)
        self.op._run_prefill_topk(
            self.logits,
            self.starts,
            self.ends,
            self.output,
            6,
            context=self.context,
            total_query_rows=5,
        )
        self.assertEqual(self.output[0].tolist(), [-1, -1, -1])
        self.assertEqual(self.output[1].tolist(), [0, -1, -1])
        self.assertEqual(sorted(self.output[2].tolist()), [0, 1, 2])
        for row, length in enumerate((0, 1, 3, 5, 4)):
            valid = self.output[row][self.output[row] >= 0]
            self.assertEqual(valid.numel(), min(length, 3))
            self.assertTrue(torch.all(valid < length))
        torch.testing.assert_close(self.context.tokens[0].output, self.output)

    def test_radix_is_resolved_once_before_begin(self):
        def begin(**kw):
            os.environ["DSV4_PREFILL_TOPK_FORCE_RADIX"] = "0"
            return self.context._begin(**kw)

        self.context.begin_topk.side_effect = begin
        self.run_topk(self.context)
        self.assertTrue(self.s.ops.dsv4_top_k_per_row_prefill.call_args.args[-1])
        self.assertTrue(self.context.calls[0]["parameters"]["force_radix_sort"])

    def test_disabled_does_not_enter_capture_or_query_cuda(self):
        with (
            patch.object(
                self.op,
                "_begin_topk_capture",
                side_effect=AssertionError("capture entered"),
            ),
            patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                side_effect=AssertionError("CUDA queried"),
            ),
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")),
            patch.object(torch.Tensor, "clone", side_effect=AssertionError("snapshot")),
        ):
            self.run_topk()
        self.s.compute.cuda_graph_capture_forward_enabled.assert_not_called()
        self.s.compute.cuda_graph_warmup_forward_enabled.assert_not_called()
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()

    def test_unselected_context_runs_kernel_once(self):
        self.context.begin_topk.return_value = None
        self.context.begin_topk.side_effect = None
        self.run_topk(self.context)
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()
        self.assertEqual(self.context.tokens, [])

    def test_kernel_exception_identity_preserved_even_if_cancel_fails(self):
        error = RuntimeError("original kernel failure")
        token = _Token()
        token.cancel.side_effect = RuntimeError("cancel failure")
        self.context.begin_topk.side_effect = None
        self.context.begin_topk.return_value = token
        self.s.ops.dsv4_top_k_per_row_prefill.side_effect = error
        with (
            self.assertLogs(self.s.base.__name__, level="WARNING"),
            self.assertRaises(RuntimeError) as raised,
        ):
            self.run_topk(self.context)
        self.assertIs(raised.exception, error)
        token.cancel.assert_called_once_with(reason="kernel_error")
        token.finish.assert_not_called()
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()

    def test_begin_and_skip_failure_are_fail_open_and_observable(self):
        self.context.begin_topk.side_effect = RuntimeError("begin failure")
        self.context.note_skip.side_effect = RuntimeError("report failure")
        with self.assertLogs(self.s.base.__name__, level="WARNING") as logs:
            self.run_topk(self.context)
        self.assertEqual(len(logs.output), 2)
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()
        self.context.note_skip.assert_called_once_with("capture_error", layer=7)

    def test_finish_failure_cancels_and_does_not_repeat_kernel(self):
        token = _Token()
        token.finish.side_effect = RuntimeError("finish failure")
        self.context.begin_topk.side_effect = None
        self.context.begin_topk.return_value = token
        with self.assertLogs(self.s.base.__name__, level="WARNING"):
            self.run_topk(self.context)
        token.cancel.assert_called_once_with(reason="capture_error")
        self.context.note_skip.assert_called_once_with("capture_error", layer=7)
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()
        self.assertTrue(torch.any(self.output >= 0))

    def test_missing_layer_is_not_layer_zero(self):
        self.op.layer_idx = None
        self.run_topk(self.context)
        self.context.begin_topk.assert_not_called()
        self.context.note_skip.assert_called_once_with("missing_layer", layer=None)
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()

    def test_missing_row_mapping_skips(self):
        self.op._run_prefill_topk(
            self.logits, self.starts, self.ends, self.output, 8, context=self.context
        )
        self.context.begin_topk.assert_not_called()
        self.context.note_skip.assert_called_once_with("missing_row_mapping", layer=7)

    def test_ops_warmup_and_capture_flags_reject_without_torch_capture(self):
        for name, reason in (
            ("cuda_graph_capture_forward_enabled", "graph"),
            ("cuda_graph_warmup_forward_enabled", "warmup"),
        ):
            with (
                self.subTest(flag=name),
                patch.object(self.s.compute, name, return_value=True),
                patch.object(
                    torch.cuda, "is_current_stream_capturing", return_value=False
                ),
            ):
                context = _Context()
                self.run_topk(context)
                context.begin_topk.assert_not_called()
                context.note_skip.assert_called_once_with(reason, layer=7)

    def test_chunks_cross_requests_preserve_prefix_and_raw_snapshot(self):
        with patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "1"}):
            result, scores = self.ragged(self.context)
        self.assertEqual([c["query_row_start"] for c in self.context.calls], [0, 2, 4])
        self.assertEqual([c["total_query_rows"] for c in self.context.calls], [5, 5, 5])
        self.assertEqual(self.s.ops.dsv4_top_k_per_row_prefill.call_count, 3)
        self.assertEqual(
            self.s.ops.cp_gather_indexer_k_quant_cache.call_args.args[1].shape[0], 8
        )
        middle = self.context.calls[1]
        self.assertEqual(middle["row_starts"].tolist(), [0, 5])
        self.assertEqual(middle["row_ends"].tolist(), [5, 7])
        self.assertEqual(middle["logits"].shape, (2, 8))
        raw = self.context.tokens[1].output
        self.assertEqual(raw.tolist(), [[4, 3, 2], [1, 0, -1]])
        self.assertEqual(result[2:4].tolist(), [[2, 3, 4], [5, 6, -1]])
        scores.fill_(-100)
        result.fill_(-9)
        self.assertEqual(raw.tolist(), [[4, 3, 2], [1, 0, -1]])
        self.assertNotEqual(middle["logits"][0, 0].item(), -100)

    def test_prefill_optimization_preserves_raw_kernel_capture(self):
        for joint in (False, True):
            with (
                self.subTest(joint=joint),
                patch.object(self.s.base._prefill_optimization, "ENABLED", True),
                patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "0"}),
            ):
                context = _Context(mqa_enabled=joint)
                kernel = self.s.ops.dsv4_top_k_per_row_prefill
                kernel.reset_mock()
                result, _ = self.ragged(context)
                self.assertEqual(kernel.call_count, 3)
                topk = kernel.call_args_list[1]
                token = context.tokens[1]
                if joint:
                    captured_logits = token.logits
                    self.assertIs(token.prepare_topk.call_args.args[0], topk.args[0])
                else:
                    captured_logits = context.calls[1]["logits"]
                    self.assertIs(
                        context.begin_topk.call_args_list[1].kwargs["logits"],
                        topk.args[0],
                    )
                torch.testing.assert_close(captured_logits, topk.args[0])
                torch.testing.assert_close(token.output, topk.args[3])
                self.assertIs(token.finish.call_args.args[0], topk.args[3])
                self.assertEqual(token.output.tolist(), [[4, 3, 2], [1, 0, -1]])
                self.assertEqual(result[2:4].tolist(), [[2, 3, 4], [5, 6, -1]])
                token.finish.assert_called_once()
                token.cancel.assert_not_called()

    def test_global_chunk_override(self):
        with patch.dict(os.environ, {"DSV4_CHUNK_TOKENS": "4"}):
            self.ragged(self.context)
        self.assertEqual([c["query_row_start"] for c in self.context.calls], [0, 4])

    def test_empty_query_skips_topk_but_preserves_existing_gather(self):
        result, _ = self.ragged(self.context, rows=0)
        self.assertEqual(result.shape, (0, 3))
        self.context.note_skip.assert_called_once_with("empty", layer=7)
        self.context.begin_topk.assert_not_called()
        self.s.deep.fp8_mqa_logits.assert_not_called()
        self.s.ops.cp_gather_indexer_k_quant_cache.assert_called_once()

    def test_input_graph_cp_decode_flags_skip_capture_not_inference(self):
        for flags, reason in (
            ({"is_cuda_graph": True}, "graph"),
            ({"prefill_cuda_graph_copy_params": object()}, "graph"),
            ({"context_parallel_info": object()}, "cp"),
            ({"is_prefill": False}, "decode"),
        ):
            with self.subTest(flags=flags):
                context = _Context()
                result, _ = self.ragged(context, **flags)
                self.assertEqual(result.shape, (5, 3))
                context.begin_topk.assert_not_called()
                context.note_skip.assert_called_once_with(reason, layer=7)

    def test_cp_empty_entry_reports_empty_and_does_not_capture(self):
        empty = torch.empty(0, dtype=torch.int32)
        result = self.op._get_topk_ragged_cp(
            torch.empty((0, 2, 128)),
            torch.empty((0, 2, 1)),
            None,
            None,
            SimpleNamespace(tensor_capture_context=self.context),
            empty,
            empty,
            0,
            empty,
            empty,
            empty,
            empty,
        )
        self.assertEqual(result.shape, (0, 3))
        self.context.note_skip.assert_called_once_with("empty", layer=7)
        self.context.begin_topk.assert_not_called()

    def test_paged_entry_without_explicit_decode_phase_skips_capture(self):
        self.s.deep.fp8_paged_mqa_logits.return_value = self.logits
        result = self.op._get_topk_paged(
            torch.zeros((2, 2, 128)),
            torch.ones((2, 2, 1)),
            SimpleNamespace(kv_scale_base=torch.empty((2, 64, 132), dtype=torch.uint8)),
            SimpleNamespace(
                expanded_seq_lens=torch.tensor([4, 5]),
                kvlen_d=torch.tensor([4, 5]),
                schedule_metadata=torch.ones(1),
            ),
            SimpleNamespace(
                tensor_capture_context=self.context,
                decode_cu_seqlens_d=torch.arange(3),
                kv_cache_kernel_block_id_device=torch.zeros((2, 1), dtype=torch.int32),
            ),
        )
        self.assertEqual(result.shape, (2, 3))
        self.context.note_skip.assert_called_once_with("phase", layer=7)
        self.context.begin_topk.assert_not_called()
        self.context.begin_paged_topk.assert_not_called()
        self.context.begin_paged_mqa.assert_not_called()
        self.s.ops.dsv4_persistent_topk.assert_called_once()

    def test_verify_and_draft_prefill_shape_still_route_to_paged(self):
        indexer = object.__new__(self.s.indexer.Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.parallelism_config = None
        indexer.indexer_op = SimpleNamespace(
            _get_topk_paged=Mock(return_value=self.output),
            _get_topk_ragged=Mock(),
            _get_topk_ragged_cp=Mock(),
        )
        for flags in (
            {"is_prefill": False},
            {"is_prefill": True, "is_target_verify": True},
            {"is_prefill": True, "is_draft_extend": True},
        ):
            result = indexer._compute_topk(
                None, None, None, None, SimpleNamespace(**flags), None
            )
            self.assertIs(result, self.output)
        self.assertEqual(indexer.indexer_op._get_topk_paged.call_count, 3)
        indexer.indexer_op._get_topk_ragged.assert_not_called()
        indexer.indexer_op._get_topk_ragged_cp.assert_not_called()

    def test_hybrid_skip_reporting_failure_is_fail_open(self):
        self.context.note_skip.side_effect = RuntimeError("skip report failure")
        with self.assertLogs(self.s.indexer.__name__, level="WARNING"):
            self.s.indexer._note_indexer_capture_skip(
                SimpleNamespace(tensor_capture_context=self.context),
                "dense_fast_path",
                7,
            )

    def test_layer_propagates_through_real_indexer_constructor(self):
        config = SimpleNamespace(
            indexer_head_num=2,
            indexer_head_dim=128,
            indexer_topk=3,
            rope_head_dim=64,
            kernel_tokens_per_block=64,
            rope_config=SimpleNamespace(indexer_is_neox_style=True),
        )
        indexer = self.s.indexer.Indexer(
            config,
            {
                W.mla_indexer_k_norm_w: torch.ones(128),
                W.mla_indexer_k_norm_b: torch.zeros(128),
            },
            {W.rope_cos_sin_cache: None},
            11,
            1e-5,
            None,
        )
        self.assertEqual(indexer.indexer_op.layer_idx, 11)
        self.assertIsNone(self.s.base.IndexerOp(2, 128, 3, 64).layer_idx)

    def test_dense_fast_path_keeps_cache_update_without_topk(self):
        indexer = object.__new__(self.s.indexer.Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.layer_idx = 7
        indexer._get_k_bf16 = Mock(return_value=torch.zeros((1, 128)))
        indexer.indexer_op = SimpleNamespace(quant_k_only=Mock())
        result = indexer.forward(
            torch.zeros((1, 128)),
            None,
            None,
            SimpleNamespace(slot_mapping=None),
            SimpleNamespace(tensor_capture_context=self.context),
            use_fast_path=True,
        )
        self.assertIsNone(result)
        indexer.indexer_op.quant_k_only.assert_called_once()
        self.context.note_skip.assert_called_once_with("dense_fast_path", layer=7)
        self.context.begin_topk.assert_not_called()

    def test_shared_no_indexer_and_dense_reuse_paths(self):
        for reuse, forced, sparse, reason in (
            (True, False, True, "reused"),
            (False, True, True, "reused"),
            (True, False, False, "dense_fast_path"),
            (False, False, True, "no_indexer"),
        ):
            with self.subTest(reason=reason, forced=forced):
                context = _Context()
                attention = object.__new__(self.s.attention.MlaAttention)
                torch.nn.Module.__init__(attention)
                attention.layer_idx = 7
                attention.reuse_topk_indices = reuse
                attention.indexer = None
                shared = torch.ones((1, 3), dtype=torch.int32)
                result = attention._run_sparse_indexer(
                    None,
                    None,
                    None,
                    None,
                    SimpleNamespace(
                        is_sparse=lambda sparse=sparse: sparse,
                        attn_inputs=SimpleNamespace(tensor_capture_context=context),
                    ),
                    prev_topk_indices=shared,
                    force_reuse_topk_indices=forced,
                )
                self.assertIs(result, shared if reason == "reused" else None)
                context.note_skip.assert_called_once_with(reason, layer=7)
                context.begin_topk.assert_not_called()

    @contextmanager
    def recorder(
        self, tp_rank=0, config_overrides=None, identity_overrides=None, mqa=False
    ):
        with tempfile.TemporaryDirectory(
            prefix="indexer-capture-",
            dir=os.environ.get("TEST_TMPDIR") or os.environ.get("TMPDIR"),
        ) as directory:
            root = Path(directory).resolve()
            config = {
                "enabled": True,
                "authorized_capture": True,
                "layers": [7],
                "ops": (
                    ["indexer.mqa_logits", "indexer.topk"] if mqa else ["indexer.topk"]
                ),
                "phases": ["prefill"],
                "rank_filter": "record_owner",
                "max_calls": 16,
                "max_bundle_bytes": 1 << 20,
                "max_total_bytes": 1 << 24,
                "max_inflight_bytes": 1 << 24,
            }
            config.update(config_overrides or {})
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            identity = {
                "schema_version": 1,
                "session_id": "test-session",
                "owner_id": "owner0",
                "replica_id": "replica0",
                "world_rank": 0,
                "dp_rank": 0,
                "exec_id": "42",
                "model_role": "target",
                "requests": [
                    {"phase": "prefill", "request_id": "r1", "q_len": 3, "kv_len": 2},
                    {"phase": "prefill", "request_id": "r2", "q_len": 2, "kv_len": 1},
                ],
            }
            identity.update(identity_overrides or {})
            with patch.dict(
                os.environ,
                {
                    "RTP_LLM_RECORD_SESSION": "test-session",
                    "RTP_LLM_RECORD_DIR": str(root),
                },
            ):
                recorder = tensor_capture.TensorCaptureRecorder(
                    str(config_path), tp_rank=tp_rank, world_rank=0
                )
                try:
                    context = tensor_capture_indexer.for_context(
                        recorder.for_execution(json.dumps(identity))
                    )
                    yield tensor_capture, recorder, context, root
                finally:
                    recorder.close()

    def test_generic_execution_context_is_adapted_at_real_indexer_callsite(self):
        for joint in (False, True):
            with (
                self.subTest(joint=joint),
                self.recorder(mqa=joint) as (capture, recorder, adapter, root),
            ):
                context = adapter.context
                self.assertIsInstance(context, capture.TensorCaptureContext)
                self.assertFalse(hasattr(context, "begin_topk"))
                self.assertFalse(hasattr(context, "begin_mqa"))
                self.ragged(context)
                if not recorder.close():
                    thread = recorder._thread
                    frame = sys._current_frames().get(thread.ident) if thread else None
                    stack = (
                        "".join(traceback.format_stack(frame))
                        if frame
                        else "<no writer frame>"
                    )
                    self.fail(
                        "capture close timed out: stats=%r\nwriter stack:\n%s"
                        % (recorder.stats(), stack)
                    )
                self.assertEqual(recorder.stats()["completed"], 3)
                manifests = list(root.rglob("manifest.json"))
                self.assertEqual(len(manifests), 3)
                for path in manifests:
                    manifest, tensors = capture.load_capture(path)
                    self.assertEqual(manifest["schema_version"], 2 if joint else 1)
                    self.assertIn("indices", tensors)
                    self.assertEqual("q_fp8" in tensors, joint)
                self.assertEqual(recorder.stats()["inflight_bytes"], 0)

    def test_real_capture_file_roundtrip_and_cpu_kernel_replay(self):
        for backend in ("dsv4_per_row", "topk_v3_tie_break"):
            with (
                self.subTest(backend=backend),
                self.recorder() as (capture, recorder, context, root),
            ):
                self.assertIsNotNone(context)
                self.op._prefill_topk_backend = backend
                with patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "1"}):
                    result, scores = self.ragged(context)
                torch.testing.assert_close(
                    scores, torch.arange(80, dtype=torch.float32).view(5, 16)[:, ::2]
                )
                original_scores = scores.clone()
                scores.fill_(-100)
                result.fill_(-9)
                self.assertTrue(recorder.close())
                self.assertEqual(recorder.stats()["completed"], 3)
                self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                manifests = list(root.rglob("manifest.json"))
                self.assertEqual(len(manifests), 3)
                bundles = sorted(
                    (capture.load_capture(path) for path in manifests),
                    key=lambda item: item[0]["query_row_start"],
                )
                self.assertEqual([m["query_row_start"] for m, _ in bundles], [0, 2, 4])
                for manifest, tensors in bundles:
                    self.assertTrue(manifest["complete"])
                    self.assertEqual(manifest["layer"], 7)
                    self.assertEqual(manifest["execution_id"], "42")
                    self.assertEqual(manifest["backend"], backend)
                    self.assertEqual(manifest["total_query_rows"], 5)
                    self.assertEqual(
                        manifest["parameters"]["index_space"], "relative_to_row_start"
                    )
                    desc = manifest["tensors"]["logits"]
                    self.assertEqual(desc["original_stride"], [16, 2])
                    row_start = manifest["query_row_start"]
                    for row, (start, stop) in enumerate(
                        zip(
                            tensors["row_starts"].tolist(), tensors["row_ends"].tolist()
                        )
                    ):
                        saved = tensors["logits"][row]
                        torch.testing.assert_close(
                            saved[start:stop],
                            original_scores[row_start + row, start:stop],
                            atol=0,
                            rtol=0,
                        )
                        self.assertTrue(torch.all(saved[:start] == 0))
                        self.assertTrue(torch.all(saved[stop:] == 0))
                    # Reconstruct the source strides for replay; the file itself
                    # is contiguous. This is still a CPU mock-kernel replay.
                    logits = torch.empty_strided(desc["shape"], desc["original_stride"])
                    logits.copy_(tensors["logits"])
                    output = torch.empty_like(tensors["indices"])
                    self.op._run_prefill_topk(
                        logits,
                        tensors["row_starts"],
                        tensors["row_ends"],
                        output,
                        manifest["parameters"].get("max_seq_len", 5),
                    )
                    torch.testing.assert_close(
                        output, tensors["indices"], atol=0, rtol=0
                    )
                self.assertEqual(
                    bundles[1][1]["indices"].tolist(), [[4, 3, 2], [1, 0, -1]]
                )

    def test_real_oversize_and_layer_filter_leave_inference_unchanged(self):
        for overrides, reason in (
            ({"max_bundle_bytes": 1}, "oversize"),
            ({"layers": [8]}, "layer_filter"),
        ):
            with (
                self.subTest(reason=reason),
                self.recorder(config_overrides=overrides) as (
                    _,
                    recorder,
                    context,
                    root,
                ),
            ):
                result, _ = self.ragged(context)
                self.assertEqual(result.shape, (5, 3))
                self.assertTrue(recorder.close())
                self.assertEqual(recorder.stats()["completed"], 0)
                self.assertEqual(recorder.stats()["skip_reasons"][reason], 3)
                self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                self.assertEqual(list(root.rglob("manifest.json")), [])

    def test_real_mixed_decode_or_propose_context_is_rejected(self):
        for overrides in (
            {"model_role": "draft"},
            {
                "requests": [
                    {"phase": "decode", "request_id": "r1", "q_len": 1, "kv_len": 2},
                    {"phase": "prefill", "request_id": "r2", "q_len": 2, "kv_len": 1},
                ]
            },
        ):
            with (
                self.subTest(overrides=overrides),
                self.recorder(identity_overrides=overrides) as (
                    _,
                    recorder,
                    context,
                    root,
                ),
            ):
                self.assertIsNone(context)
                self.ragged(context)
                self.assertTrue(recorder.close())
                self.assertEqual(recorder.stats()["skip_reasons"]["invalid_context"], 1)
                self.assertEqual(list(root.rglob("manifest.json")), [])

    def test_real_non_owner_is_not_given_context(self):
        with self.recorder(tp_rank=1) as (_, recorder, context, root):
            self.assertIsNone(context)
            self.ragged(context)
            self.assertTrue(recorder.close())
            self.assertEqual(recorder.stats()["skip_reasons"]["non_owner_rank"], 1)
            self.assertEqual(list(root.rglob("manifest.json")), [])

            self.assertEqual(self.s.ops.dsv4_top_k_per_row_prefill.call_count, 3)

    def test_real_microbatch_row_mismatch_does_not_publish(self):
        with self.recorder() as (_, recorder, context, root):
            # A leaked context for the full five-row execution must not accept
            # a three-row microbatch as a complete execution.
            result, _ = self.ragged(context, rows=3)
            self.assertEqual(result.shape, (3, 3))
            self.assertTrue(recorder.close())
            self.assertEqual(recorder.stats()["completed"], 0)
            self.assertEqual(
                recorder.stats()["skip_reasons"]["invalid_topk_or_snapshot_error"], 2
            )
            self.assertEqual(list(root.rglob("manifest.json")), [])
            self.assertEqual(self.s.ops.dsv4_top_k_per_row_prefill.call_count, 2)

    def test_real_kernel_failure_releases_reservation_without_manifest(self):
        with self.recorder() as (_, recorder, context, root):
            error = RuntimeError("kernel failure")
            self.s.ops.dsv4_top_k_per_row_prefill.side_effect = error
            with self.assertRaises(RuntimeError) as raised:
                self.ragged(context)
            self.assertIs(raised.exception, error)
            self.assertTrue(recorder.close())
            self.assertEqual(recorder.stats()["skip_reasons"]["kernel_error"], 1)
            self.assertEqual(recorder.stats()["inflight_bytes"], 0)
            self.assertEqual(recorder.stats()["reserved_bytes"], 0)
            self.assertEqual(list(root.rglob("manifest.json")), [])
            self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()

    def test_before_snapshot_survives_kernel_mutating_input(self):
        before = self.logits.clone()

        def kernel(*args):
            _cpu_topk(*args)
            args[0].fill_(-100)

        self.s.ops.dsv4_top_k_per_row_prefill.side_effect = kernel
        self.run_topk(self.context)
        torch.testing.assert_close(self.context.calls[0]["logits"], before)
        self.assertTrue(torch.all(self.logits == -100))

    def test_mqa_group_chunks_share_actual_inputs_logits_and_raw_indices(self):
        for backend in ("dsv4_per_row", "topk_v3_tie_break"):
            with self.subTest(backend=backend):
                self.op._prefill_topk_backend = backend
                context = _Context(mqa_enabled=True)
                self.s.deep.fp8_mqa_logits.reset_mock()
                kernel = getattr(
                    self.s.ops,
                    (
                        backend
                        if backend == "topk_v3_tie_break"
                        else "dsv4_top_k_per_row_prefill"
                    ),
                )
                kernel.reset_mock()
                with patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "1"}):
                    result, _ = self.ragged(context)
                self.assertEqual(self.s.deep.fp8_mqa_logits.call_count, 3)
                self.assertEqual(kernel.call_count, 3)
                context.begin_topk.assert_not_called()
                self.assertEqual(
                    [c["query_row_start"] for c in context.calls], [0, 2, 4]
                )
                self.assertEqual(
                    [c["total_query_rows"] for c in context.calls], [5, 5, 5]
                )
                self.assertEqual(context.calls[1]["row_starts"].tolist(), [0, 5])
                self.assertEqual(context.calls[1]["row_ends"].tolist(), [5, 7])
                for begin, mqa, topk, token in zip(
                    context.begin_mqa.call_args_list,
                    self.s.deep.fp8_mqa_logits.call_args_list,
                    kernel.call_args_list,
                    context.tokens,
                ):
                    self.assertEqual(begin.kwargs["layer"], 7)
                    self.assertEqual(begin.kwargs["k"], 3)
                    for name, actual in zip(
                        (
                            "q_fp8",
                            "k_fp8",
                            "k_scale",
                            "weights",
                            "row_starts",
                            "row_ends",
                        ),
                        (mqa.args[0], *mqa.args[1], *mqa.args[2:]),
                    ):
                        self.assertIs(begin.kwargs[name], actual)
                    self.assertEqual(mqa.kwargs, {"clean_logits": False})
                    self.assertEqual(begin.kwargs["q_fp8"].dtype, torch.float8_e4m3fn)
                    self.assertEqual(begin.kwargs["q_fp8"].shape[1:], (2, 128))
                    self.assertEqual(begin.kwargs["k_fp8"].shape, (8, 128))
                    self.assertEqual(begin.kwargs["k_scale"].shape, (8,))
                    self.assertEqual(begin.kwargs["k_scale"].dtype, torch.float32)
                    self.assertIs(token.prepare_topk.call_args.args[0], topk.args[0])
                    self.assertEqual(token.backend, backend)
                    if backend == "dsv4_per_row":
                        self.assertEqual(
                            token.parameters,
                            dict(
                                num_rows=topk.args[4],
                                stride0=topk.args[5],
                                stride1=topk.args[6],
                                force_radix_sort=topk.args[8],
                            ),
                        )
                    else:
                        self.assertEqual(
                            token.parameters,
                            dict(
                                max_seq_len=topk.args[5],
                                stride0=16,
                                stride1=2,
                            ),
                        )
                    token.finish.assert_called_once()
                    token.cancel.assert_not_called()
                self.assertEqual(
                    context.tokens[1].output.tolist(), [[4, 3, 2], [1, 0, -1]]
                )
                self.assertEqual(result[2:4].tolist(), [[2, 3, 4], [5, 6, -1]])

    def test_mqa_snapshot_precedes_kernel_input_mutation(self):
        context = _Context(mqa_enabled=True)

        def mutate(q, kv, weights, starts, ends, logits):
            q.copy_(torch.zeros(q.shape).to(q.dtype))
            kv[0].copy_(torch.zeros(kv[0].shape).to(kv[0].dtype))
            kv[1].fill_(9)
            weights.fill_(9)

        result, scores = self.ragged(context, mqa_effect=mutate)
        first = context.calls[0]
        self.assertEqual(first["q_fp8"][:, 0, 0].float().tolist(), [0, 1])
        self.assertTrue(torch.all(first["k_fp8"].float() == 1))
        self.assertTrue(torch.all(first["k_scale"] == 1))
        self.assertTrue(torch.all(first["weights"] == 1))
        scores.fill_(-100)
        result.fill_(-9)
        self.assertEqual(context.tokens[1].output.tolist(), [[4, 3, 2], [1, 0, -1]])
        self.assertNotEqual(context.tokens[1].logits[0, 0].item(), -100)

    def test_mqa_rejection_and_begin_failure_never_fall_back_to_topk(self):
        expected, _ = self.ragged(None)
        for failure in (None, RuntimeError("snapshot failure")):
            with self.subTest(failure=failure):
                context = _Context(mqa_enabled=True)
                context.begin_mqa.side_effect = failure
                context.begin_mqa.return_value = None
                self.s.deep.fp8_mqa_logits.reset_mock()
                self.s.ops.dsv4_top_k_per_row_prefill.reset_mock()
                result, _ = self.ragged(context)
                torch.testing.assert_close(result, expected)
                self.assertEqual(context.begin_mqa.call_count, 3)
                context.begin_topk.assert_not_called()
                self.assertEqual(self.s.deep.fp8_mqa_logits.call_count, 3)
                self.assertEqual(self.s.ops.dsv4_top_k_per_row_prefill.call_count, 3)
                if failure is not None:
                    self.assertEqual(context.note_skip.call_count, 3)
                    context.note_skip.assert_called_with("capture_error", layer=7)

    def test_mqa_prepare_failure_cancels_once_and_preserves_topk(self):
        for failure in (False, RuntimeError("prepare failure")):
            for kernel_fails in (False, True):
                with self.subTest(failure=failure, kernel_fails=kernel_fails):
                    context = _Context(mqa_enabled=True)
                    token = _Token()
                    token.prepare_topk.side_effect = (
                        failure if isinstance(failure, Exception) else None
                    )
                    token.prepare_topk.return_value = False
                    error = RuntimeError("original topk failure")
                    self.s.ops.dsv4_top_k_per_row_prefill.reset_mock()
                    self.s.ops.dsv4_top_k_per_row_prefill.side_effect = (
                        error if kernel_fails else _cpu_topk
                    )
                    if kernel_fails:
                        with self.assertRaises(RuntimeError) as raised:
                            self.run_topk(context, capture_token=token)
                        self.assertIs(raised.exception, error)
                    else:
                        self.run_topk(context, capture_token=token)
                        self.assertTrue(torch.any(self.output >= 0))
                    token.cancel.assert_called_once_with(reason="capture_error")
                    token.finish.assert_not_called()
                    context.begin_topk.assert_not_called()
                    context.begin_mqa.assert_not_called()
                    self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()

    def test_mqa_topk_interrupt_cancels_entire_token_lifetime(self):
        for backend in ("dsv4_per_row", "topk_v3_tie_break"):
            for stage in ("parameters", "prepare", "kernel", "finish"):
                with self.subTest(backend=backend, stage=stage), ExitStack() as stack:
                    self.op._prefill_topk_backend = backend
                    context = _Context(mqa_enabled=True)
                    token = _Token()
                    error = KeyboardInterrupt(stage)
                    max_seq_len = 100
                    kernel = getattr(
                        self.s.ops,
                        (
                            "topk_v3_tie_break"
                            if backend == "topk_v3_tie_break"
                            else "dsv4_top_k_per_row_prefill"
                        ),
                    )
                    kernel.reset_mock()
                    kernel.side_effect = error if stage == "kernel" else _cpu_topk
                    if stage == "parameters":
                        if backend == "topk_v3_tie_break":

                            class InterruptedLength:
                                def __int__(self):
                                    raise error

                            max_seq_len = InterruptedLength()
                        else:
                            stack.enter_context(
                                patch.object(
                                    self.s.base,
                                    "_fp8_prefill_topk_force_radix_sort",
                                    side_effect=error,
                                )
                            )
                    elif stage == "prepare":
                        token.prepare_topk.side_effect = error
                    elif stage == "finish":
                        token.finish.side_effect = error
                    with self.assertRaises(KeyboardInterrupt) as raised:
                        self.run_topk(
                            context, max_seq_len=max_seq_len, capture_token=token
                        )
                    self.assertIs(raised.exception, error)
                    token.cancel.assert_called_once_with(
                        reason="kernel_error" if stage == "kernel" else "capture_error"
                    )
                    self.assertEqual(
                        token.prepare_topk.call_count, 0 if stage == "parameters" else 1
                    )
                    self.assertEqual(
                        kernel.call_count, 1 if stage in ("kernel", "finish") else 0
                    )
                    self.assertEqual(
                        token.finish.call_count, 1 if stage == "finish" else 0
                    )
                    context.begin_topk.assert_not_called()
                    context.begin_mqa.assert_not_called()

    def test_real_mqa_interrupt_releases_budget_before_close_or_gc(self):
        for stage in ("parameters", "prepare", "kernel", "finish"):
            with (
                self.subTest(stage=stage),
                self.recorder(mqa=True) as (_, recorder, context, root),
                ExitStack() as stack,
            ):
                error = KeyboardInterrupt(stage)
                tokens = []
                begin_mqa = context.begin_mqa
                release = stack.enter_context(
                    patch.object(recorder, "_release", wraps=recorder._release)
                )

                def begin(**kwargs):
                    token = begin_mqa(**kwargs)
                    self.assertIsNotNone(token)
                    self.assertGreater(recorder.stats()["inflight_bytes"], 0)
                    tokens.append(token)  # Keep ownership alive across the exception.
                    stack.enter_context(
                        patch.object(token, "cancel", wraps=token.cancel)
                    )
                    for method, phase in (
                        ("prepare_topk", "prepare"),
                        ("finish", "finish"),
                    ):
                        stack.enter_context(
                            patch.object(
                                token,
                                method,
                                wraps=getattr(token, method),
                                side_effect=error if stage == phase else None,
                            )
                        )
                    return token

                stack.enter_context(
                    patch.object(context, "begin_mqa", side_effect=begin)
                )
                if stage == "parameters":
                    stack.enter_context(
                        patch.object(
                            self.s.base,
                            "_fp8_prefill_topk_force_radix_sort",
                            side_effect=error,
                        )
                    )
                self.s.deep.fp8_mqa_logits.reset_mock()
                kernel = self.s.ops.dsv4_top_k_per_row_prefill
                kernel.reset_mock()
                kernel.side_effect = error if stage == "kernel" else _cpu_topk
                with self.assertRaises(KeyboardInterrupt) as raised:
                    self.ragged(context)
                self.assertIs(raised.exception, error)
                self.assertEqual(len(tokens), 1)
                tokens[0].cancel.assert_called_once_with(
                    reason="kernel_error" if stage == "kernel" else "capture_error"
                )
                release.assert_called_once()
                self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                self.assertEqual(recorder.stats()["reserved_bytes"], 0)
                self.assertEqual(recorder.stats()["completed"], 0)
                self.s.deep.fp8_mqa_logits.assert_called_once()
                self.assertEqual(
                    kernel.call_count, 1 if stage in ("kernel", "finish") else 0
                )
                self.assertEqual(list(root.rglob("manifest.json")), [])
                self.assertTrue(recorder.close())
                release.assert_called_once()

    def test_mqa_kernel_exception_is_original_even_when_cancel_fails(self):
        for cancel_fails in (False, True):
            with self.subTest(cancel_fails=cancel_fails), ExitStack() as stack:
                context = _Context(mqa_enabled=True)
                token = _Token()
                context.begin_mqa.side_effect = None
                context.begin_mqa.return_value = token
                error = RuntimeError("original mqa failure")
                if cancel_fails:
                    token.cancel.side_effect = RuntimeError("cancel failure")
                    stack.enter_context(
                        self.assertLogs(self.s.base.__name__, level="WARNING")
                    )
                self.s.deep.fp8_mqa_logits.reset_mock()
                self.s.ops.dsv4_top_k_per_row_prefill.reset_mock()

                def fail(*args):
                    raise error

                with self.assertRaises(RuntimeError) as raised:
                    self.ragged(context, mqa_effect=fail)
                self.assertIs(raised.exception, error)
                token.cancel.assert_called_once_with(reason="kernel_error")
                token.prepare_topk.assert_not_called()
                token.finish.assert_not_called()
                context.begin_topk.assert_not_called()
                self.s.deep.fp8_mqa_logits.assert_called_once()
                self.s.ops.dsv4_top_k_per_row_prefill.assert_not_called()

    def test_mqa_topk_failure_cancels_group_after_one_mqa(self):
        context = _Context(mqa_enabled=True)
        error = RuntimeError("original topk failure")
        self.s.ops.dsv4_top_k_per_row_prefill.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            self.ragged(context)
        self.assertIs(raised.exception, error)
        self.s.deep.fp8_mqa_logits.assert_called_once()
        self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()
        token = context.tokens[0]
        token.prepare_topk.assert_called_once()
        token.cancel.assert_called_once_with(reason="kernel_error")
        token.finish.assert_not_called()
        context.begin_topk.assert_not_called()

    def test_mqa_guards_run_before_input_snapshot(self):
        for flag, reason in (
            ("cuda_graph_warmup_forward_enabled", "warmup"),
            ("cuda_graph_capture_forward_enabled", "graph"),
            (None, "missing_layer"),
        ):
            with self.subTest(reason=reason), ExitStack() as stack:
                if flag:
                    stack.enter_context(
                        patch.object(self.s.compute, flag, return_value=True)
                    )
                else:
                    stack.enter_context(patch.object(self.op, "layer_idx", None))
                context = _Context(mqa_enabled=True)
                result, _ = self.ragged(context)
                self.assertEqual(result.shape, (5, 3))
                context.begin_mqa.assert_not_called()
                context.begin_topk.assert_not_called()
                context.note_skip.assert_called_with(reason, layer=self.op.layer_idx)
        for flags, reason in (
            ({"is_cuda_graph": True}, "graph"),
            ({"prefill_cuda_graph_copy_params": object()}, "graph"),
            ({"context_parallel_info": object()}, "cp"),
            ({"is_prefill": False}, "decode"),
        ):
            with self.subTest(reason=reason):
                context = _Context(mqa_enabled=True)
                result, _ = self.ragged(context, **flags)
                self.assertEqual(result.shape, (5, 3))
                context.begin_mqa.assert_not_called()
                context.begin_topk.assert_not_called()
                context.note_skip.assert_called_once_with(reason, layer=7)

    def test_mqa_disabled_has_no_capture_metadata_snapshot_or_cuda_probe(self):
        with (
            patch.object(
                self.op, "_begin_mqa_capture", side_effect=AssertionError("begin")
            ),
            patch.object(
                self.op, "_capture_skip_reason", side_effect=AssertionError("guard")
            ),
            patch.object(torch.Tensor, "clone", side_effect=AssertionError("snapshot")),
            patch.object(torch.Tensor, "cpu", side_effect=AssertionError("D2H")),
            patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                side_effect=AssertionError("CUDA"),
            ),
        ):
            result, _ = self.ragged(None)
        self.assertEqual(result.shape, (5, 3))
        self.s.compute.cuda_graph_capture_forward_enabled.assert_not_called()
        self.s.compute.cuda_graph_warmup_forward_enabled.assert_not_called()
        self.assertEqual(self.s.deep.fp8_mqa_logits.call_count, 3)
        self.assertEqual(self.s.ops.dsv4_top_k_per_row_prefill.call_count, 3)

    def test_real_mqa_group_roundtrip_has_one_shared_logits_snapshot(self):
        for backend in ("dsv4_per_row", "topk_v3_tie_break"):
            with (
                self.subTest(backend=backend),
                self.recorder(mqa=True) as (capture, recorder, context, root),
            ):
                self.assertIsNotNone(context)
                self.assertTrue(context.mqa_enabled)
                self.op._prefill_topk_backend = backend
                with (
                    patch.object(
                        tensor_capture_indexer,
                        "_snapshot_logits",
                        wraps=tensor_capture_indexer._snapshot_logits,
                    ) as snapshot,
                    patch.object(
                        context,
                        "begin_topk",
                        side_effect=AssertionError("group fallback"),
                    ),
                    patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "1"}),
                ):
                    result, scores = self.ragged(context)
                self.assertEqual(snapshot.call_count, 3)
                self.assertEqual(result[2:4].tolist(), [[2, 3, 4], [5, 6, -1]])
                scores.fill_(-100)
                result.fill_(-9)
                self.assertTrue(recorder.close())
                self.assertEqual(recorder.stats()["completed"], 3)
                self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                paths = list(root.rglob("manifest.json"))
                self.assertEqual(len(paths), 3)
                bundles = sorted(
                    (capture.load_capture(path) for path in paths),
                    key=lambda item: item[0]["query_row_start"],
                )
                self.assertEqual([m["query_row_start"] for m, _ in bundles], [0, 2, 4])
                for manifest, tensors in bundles:
                    self.assertEqual(manifest["schema_version"], 2)
                    self.assertEqual(manifest["op"], "indexer.mqa_topk")
                    self.assertTrue(manifest["complete"])
                    self.assertEqual(manifest["layer"], 7)
                    self.assertEqual(manifest["execution_id"], "42")
                    self.assertEqual(manifest["total_query_rows"], 5)
                    self.assertEqual(manifest["backend"], backend)
                    self.assertEqual(manifest["parameters"]["k"], 3)
                    self.assertEqual(
                        manifest["parameters"]["index_space"], "relative_to_row_start"
                    )
                    self.assertEqual(
                        manifest["mqa_parameters"],
                        dict(
                            backend="deep_gemm.fp8_mqa_logits",
                            clean_logits=False,
                            weights_include_query_scale=True,
                            key_scale_format="float32_per_128_channels",
                        ),
                    )
                    self.assertEqual(
                        set(tensors),
                        {
                            "q_fp8",
                            "k_fp8",
                            "k_scale",
                            "weights",
                            "row_starts",
                            "row_ends",
                            "logits",
                            "indices",
                        },
                    )
                    self.assertEqual(manifest["operations"][0]["outputs"], ["logits"])
                    self.assertEqual(
                        manifest["operations"][1]["inputs"],
                        ["logits", "row_starts", "row_ends"],
                    )
                    self.assertEqual(
                        manifest["tensors"]["logits"]["file"], "logits.bin"
                    )
                    self.assertEqual(
                        manifest["tensors"]["logits"]["original_stride"], [16, 2]
                    )
                    start = manifest["query_row_start"]
                    rows = manifest["query_row_count"]
                    self.assertEqual(tensors["q_fp8"].dtype, torch.float8_e4m3fn)
                    self.assertEqual(tensors["q_fp8"].shape, (rows, 2, 128))
                    self.assertEqual(
                        tensors["q_fp8"][:, 0, 0].float().tolist(),
                        list(range(start, start + rows)),
                    )
                    self.assertEqual(tensors["k_fp8"].shape, (8, 128))
                    self.assertTrue(torch.all(tensors["k_fp8"].float() == 1))
                    self.assertTrue(torch.all(tensors["k_scale"] == 1))
                    self.assertTrue(torch.all(tensors["weights"] == 1))
                    for row, (begin, end) in enumerate(
                        zip(tensors["row_starts"], tensors["row_ends"])
                    ):
                        expected = (
                            torch.arange(16, dtype=torch.float32)[::2]
                            + (start + row) * 16
                        )
                        saved = tensors["logits"][row]
                        torch.testing.assert_close(
                            saved[begin:end], expected[begin:end], atol=0, rtol=0
                        )
                        self.assertTrue(torch.all(saved[:begin] == 0))
                        self.assertTrue(torch.all(saved[end:] == 0))
                self.assertEqual(
                    bundles[1][1]["indices"].tolist(), [[4, 3, 2], [1, 0, -1]]
                )
                for path in paths:
                    self.assertEqual(len(list(path.parent.glob("*.bin"))), 8)

    def test_real_mqa_inputs_snapshot_survives_mqa_mutation(self):
        expected, _ = self.ragged(None)
        with self.recorder(mqa=True, config_overrides={"max_calls": 1}) as (
            capture,
            recorder,
            context,
            root,
        ):

            def mutate(q, kv, weights, starts, ends, logits):
                q.copy_(torch.zeros(q.shape).to(q.dtype))
                kv[0].copy_(torch.zeros(kv[0].shape).to(kv[0].dtype))
                kv[1].fill_(9)
                weights.fill_(9)

            result, _ = self.ragged(context, mqa_effect=mutate)
            torch.testing.assert_close(result, expected)
            self.assertTrue(recorder.close())
            paths = list(root.rglob("manifest.json"))
            self.assertEqual(len(paths), 1)
            _, tensors = capture.load_capture(paths[0])
            self.assertEqual(tensors["q_fp8"][:, 0, 0].float().tolist(), [0, 1])
            self.assertTrue(torch.all(tensors["k_fp8"].float() == 1))
            self.assertTrue(torch.all(tensors["k_scale"] == 1))
            self.assertTrue(torch.all(tensors["weights"] == 1))
            self.assertEqual(recorder.stats()["completed"], 1)
            self.assertEqual(recorder.stats()["skip_reasons"]["max_calls"], 2)
            self.assertEqual(recorder.stats()["inflight_bytes"], 0)

    def test_real_mqa_failures_release_once_without_publishing(self):
        for stage in ("mqa", "topk", "prepare"):
            with (
                self.subTest(stage=stage),
                self.recorder(mqa=True) as (capture, recorder, context, root),
                ExitStack() as stack,
            ):
                release = stack.enter_context(
                    patch.object(recorder, "_release", wraps=recorder._release)
                )
                stack.enter_context(
                    patch.object(
                        context,
                        "begin_topk",
                        side_effect=AssertionError("group fallback"),
                    )
                )
                error = RuntimeError("original " + stage + " failure")
                self.s.deep.fp8_mqa_logits.reset_mock()
                self.s.ops.dsv4_top_k_per_row_prefill.reset_mock()
                self.s.ops.dsv4_top_k_per_row_prefill.side_effect = (
                    error if stage == "topk" else _cpu_topk
                )
                if stage == "prepare":
                    stack.enter_context(
                        patch.object(
                            tensor_capture_indexer,
                            "_snapshot_logits",
                            side_effect=error,
                        )
                    )
                    result, _ = self.ragged(context)
                    self.assertEqual(result.shape, (5, 3))
                else:

                    def fail(*args):
                        raise error

                    with self.assertRaises(RuntimeError) as raised:
                        self.ragged(
                            context, mqa_effect=fail if stage == "mqa" else None
                        )
                    self.assertIs(raised.exception, error)
                self.assertTrue(recorder.close())
                self.assertEqual(list(root.rglob("manifest.json")), [])
                self.assertEqual(recorder.stats()["completed"], 0)
                self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                self.assertEqual(recorder.stats()["reserved_bytes"], 0)
                count = 3 if stage == "prepare" else 1
                self.assertEqual(release.call_count, count)
                self.assertEqual(self.s.deep.fp8_mqa_logits.call_count, count)
                self.assertEqual(
                    self.s.ops.dsv4_top_k_per_row_prefill.call_count,
                    0 if stage == "mqa" else count,
                )
                reason = (
                    "logits_snapshot_error" if stage == "prepare" else "kernel_error"
                )
                self.assertEqual(recorder.stats()["skip_reasons"][reason], count)

    def test_real_mqa_quota_and_mapping_rejection_preserve_inference(self):
        for overrides, rows, reason in (
            ({"max_bundle_bytes": 1}, 5, "oversize"),
            ({"max_total_bytes": 1}, 5, "total_budget"),
            ({"max_inflight_bytes": 1}, 5, "inflight_budget"),
            ({"layers": [8]}, 5, "layer_filter"),
            ({}, 3, "invalid_mqa_contract"),
        ):
            with (
                self.subTest(reason=reason),
                self.recorder(mqa=True, config_overrides=overrides) as (
                    _,
                    recorder,
                    context,
                    root,
                ),
            ):
                expected, _ = self.ragged(None, rows=rows)
                with patch.object(
                    context, "begin_topk", side_effect=AssertionError("group fallback")
                ):
                    result, _ = self.ragged(context, rows=rows)
                torch.testing.assert_close(result, expected)
                self.assertTrue(recorder.close())
                self.assertEqual(recorder.stats()["completed"], 0)
                self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                self.assertEqual(recorder.stats()["reserved_bytes"], 0)
                self.assertEqual(
                    recorder.stats()["skip_reasons"][reason], 2 if rows == 3 else 3
                )
                self.assertEqual(list(root.rglob("manifest.json")), [])

    def cp_ragged(
        self,
        context,
        *,
        identity=False,
        sharded=True,
        empty=False,
        local_k_rows=1,
        mapping=True,
        mqa_error=None,
        **input_flags,
    ):
        from rtp_llm.models_py.triton_kernels.sparse_mla import (
            fused_indexer_cp_assemble as assemble,
        )

        # Global requests have Q lengths [3, 2], prefixes [2, 1]. The first
        # score chunk crosses requests; neither CP mapping is contiguous.
        gids = torch.tensor([] if empty else [0, 3, 4], dtype=torch.long)
        lids = torch.tensor(
            [] if empty else ([0, 1, 2] if identity else [4, 0, 2]),
            dtype=torch.long,
        )
        q = torch.zeros((5, 2, 128))
        q[lids, 0, 0] = gids.float()
        weights = torch.ones((5, 2, 1))
        weights[lids, :, 0] = gids.float().unsqueeze(-1) + 1
        q = q.to(torch.float8_e4m3fn)
        starts = torch.tensor([0, 0, 0, 5, 5], dtype=torch.int32)[gids]
        ends = torch.tensor([3, 4, 5, 7, 8], dtype=torch.int32)[gids]
        scores = torch.arange(80, dtype=torch.float32).view(5, 16)[:, ::2]
        events = []

        def fill_key(key, scale):
            key.copy_(torch.ones(key.shape).to(key.dtype))
            scale.view(torch.float32).fill_(1)

        def gather(cache, key, scale, *args):
            events.append("cache")
            fill_key(key, scale)

        def collective(tensor, group):
            self.assertEqual(group, self.s.base.Group.TP)
            events.append(
                "indexer_k_quant"
                if tensor.dtype == torch.float8_e4m3fn
                else "indexer_k_scale"
            )
            return tensor.repeat(8, 1)

        def restore(gathered_key, gathered_scale, key, scale, indices, *args):
            events.append("restore")
            key.copy_(gathered_key[indices])
            scale.copy_(gathered_scale[indices])

        def mqa(query, kv, weight, ks, ke, **kwargs):
            events.append("mqa")
            if mqa_error is not None:
                raise mqa_error
            return scores[query[:, 0, 0].float().long()]

        plan = SimpleNamespace(
            total_actual_local_T=local_k_rows,
            total_local_T=1,
            restore_indices=torch.arange(8),
        )
        inputs = SimpleNamespace(
            tensor_capture_context=context,
            is_prefill=True,
            kv_cache_kernel_block_id_device=torch.zeros((2, 1), dtype=torch.int32),
        )
        inputs.__dict__.update(input_flags)
        params = SimpleNamespace(prefill_max_kv_len=5)
        cache = SimpleNamespace(
            kv_scale_base=torch.empty((2, 64, 132), dtype=torch.uint8)
        )
        restore_kernel = MagicMock()
        restore_kernel.__getitem__.return_value.side_effect = restore
        with (
            patch.dict(os.environ, {"DSV4_FP8_INDEXER_SCORE_CHUNK_ROWS": "2"}),
            patch.object(assemble, "all_gather", side_effect=collective) as all_gather,
            patch.object(assemble, "_gather_restore_separate_kernel", restore_kernel),
            patch.object(
                self.s.ops, "cp_gather_indexer_k_quant_cache", side_effect=gather
            ),
            patch.object(self.s.deep, "fp8_mqa_logits", side_effect=mqa) as logits_op,
        ):
            try:
                result = self.op._get_topk_ragged_cp(
                    q,
                    weights,
                    cache,
                    params,
                    inputs,
                    lids,
                    torch.tensor([0, 5, 8], dtype=torch.int32),
                    8,
                    starts,
                    ends,
                    ends - starts,
                    starts,
                    kv_cache_sharded=sharded,
                    cp_size=8,
                    cp_rank=3,
                    indexer_cp_plan=plan,
                    indexer_cp_local_cu=torch.tensor(
                        [0, local_k_rows], dtype=torch.int32
                    ),
                    total_local_ids_is_identity=identity,
                    total_global_ids=(
                        gids
                        if mapping is True
                        else (None if mapping is False else mapping)
                    ),
                )
            finally:
                self.assertEqual(
                    [c.args[0].dtype for c in all_gather.call_args_list],
                    [torch.float8_e4m3fn, torch.uint8] if sharded else [],
                )
                self.s.base.all_gather.assert_not_called()
                self.s.base.barrier.assert_not_called()
                self.s.attention.all_reduce.assert_not_called()
        return SimpleNamespace(
            result=result,
            gids=gids,
            events=events,
            mqa_calls=logits_op.call_args_list,
            scores=scores,
        )

    def test_cp8_chunks_capture_existing_mapping_and_actual_kernel_inputs(self):
        for identity in (False, True):
            for sharded in (False, True):
                for joint in (False, True):
                    for backend in ("dsv4_per_row", "topk_v3_tie_break"):
                        with self.subTest(
                            identity=identity,
                            sharded=sharded,
                            joint=joint,
                            backend=backend,
                        ):
                            self.op._prefill_topk_backend = backend
                            context = _Context(mqa_enabled=joint)
                            with patch.dict(
                                os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "1"}
                            ):
                                baseline = self.cp_ragged(
                                    None, identity=identity, sharded=sharded
                                )
                                run = self.cp_ragged(
                                    context, identity=identity, sharded=sharded
                                )
                            torch.testing.assert_close(run.result, baseline.result)
                            self.assertEqual(run.events, baseline.events)
                            self.assertEqual(len(context.calls), 2)
                            for i, (call, token, mqa) in enumerate(
                                zip(context.calls, context.tokens, run.mqa_calls)
                            ):
                                self.assertIsNone(call["query_row_start"])
                                self.assertEqual(call["total_query_rows"], 5)
                                self.assertEqual(call["local_query_row_start"], i * 2)
                                self.assertEqual(call["local_query_rows"], 3)
                                self.assertEqual(call["cp_rank"], 3)
                                self.assertEqual(call["cp_size"], 8)
                                self.assertEqual(call["kv_cache_sharded"], sharded)
                                torch.testing.assert_close(
                                    call["query_global_ids"],
                                    run.gids[i * 2 : i * 2 + 2],
                                )
                                torch.testing.assert_close(
                                    call["row_starts"], mqa.args[3]
                                )
                                torch.testing.assert_close(
                                    call["row_ends"], mqa.args[4]
                                )
                                if joint:
                                    begin = context.begin_mqa.call_args_list[i].kwargs
                                    for name, actual in zip(
                                        (
                                            "q_fp8",
                                            "k_fp8",
                                            "k_scale",
                                            "weights",
                                            "row_starts",
                                            "row_ends",
                                        ),
                                        (mqa.args[0], *mqa.args[1], *mqa.args[2:]),
                                    ):
                                        self.assertIs(begin[name], actual)
                                    self.assertEqual(
                                        call["q_fp8"][:, 0, 0].float().tolist(),
                                        call["query_global_ids"].tolist(),
                                    )
                                    self.assertEqual(call["k_fp8"].shape, (8, 128))
                                    self.assertTrue(torch.all(call["k_scale"] == 1))
                                    self.assertEqual(
                                        call["weights"][:, 0].tolist(),
                                        (call["query_global_ids"] + 1).tolist(),
                                    )
                                token.finish.assert_called_once()
                                token.cancel.assert_not_called()
                            if joint:
                                context.begin_topk.assert_not_called()
                            else:
                                context.begin_mqa.assert_not_called()
                            self.assertEqual(
                                context.tokens[0].output.tolist(),
                                [[2, 1, 0], [1, 0, -1]],
                            )
                            self.assertEqual(
                                run.result.tolist(), [[0, 1, 2], [5, 6, -1], [5, 6, 7]]
                            )

    def test_cp8_collectives_preserved_for_empty_and_capture_rejection(self):
        for empty in (False, True):
            for local_k_rows in (0, 1):
                for mode in ("disabled", "capture", "budget", "missing_mapping"):
                    with self.subTest(
                        empty=empty,
                        local_k_rows=local_k_rows,
                        mode=mode,
                    ):
                        context = (
                            None if mode == "disabled" else _Context(mqa_enabled=True)
                        )
                        if mode == "budget":
                            context.begin_mqa.side_effect = None
                            context.begin_mqa.return_value = None
                        run = self.cp_ragged(
                            context,
                            empty=empty,
                            local_k_rows=local_k_rows,
                            mapping=mode != "missing_mapping",
                        )
                        expected = (
                            (["cache"] if local_k_rows else [])
                            + ["indexer_k_quant", "indexer_k_scale", "restore"]
                            + ([] if empty else ["mqa", "mqa"])
                        )
                        self.assertEqual(run.events, expected)
                        self.assertEqual(run.result.shape, (0 if empty else 3, 3))
                        if context is not None:
                            context.begin_topk.assert_not_called()
                            if empty or mode == "missing_mapping":
                                context.begin_mqa.assert_not_called()
                                context.note_skip.assert_called_once_with(
                                    "empty" if empty else "missing_row_mapping",
                                    layer=7,
                                )

    def test_cp_mapping_and_identity_errors_fail_open_without_contiguous_fallback(self):
        baseline = self.cp_ragged(None)
        bad_slice = Mock(shape=(3,))
        # An object with valid shape but no tensor slicing exercises the chunk guard.
        for mapping in (False, torch.tensor([0]), bad_slice):
            with self.subTest(mapping=mapping):
                context = _Context(mqa_enabled=True)
                run = self.cp_ragged(context, mapping=mapping)
                torch.testing.assert_close(run.result, baseline.result)
                self.assertEqual(run.events, baseline.events)
                context.begin_mqa.assert_not_called()
                context.begin_topk.assert_not_called()
                self.assertTrue(context.note_skip.called)
        context = _Context(mqa_enabled=True)
        context.context.identity = {"requests": [{"wrong_key": 3}]}
        run = self.cp_ragged(context)
        torch.testing.assert_close(run.result, baseline.result)
        context.note_skip.assert_called_once_with("capture_error", layer=7)
        context.begin_mqa.assert_not_called()
        context.begin_topk.assert_not_called()

    def test_cp_disabled_does_not_inspect_mapping_or_enter_capture_helpers(self):
        with (
            patch.object(
                self.op,
                "_begin_mqa_capture",
                side_effect=AssertionError("capture entered"),
            ),
            patch.object(
                self.op,
                "_begin_topk_capture",
                side_effect=AssertionError("capture entered"),
            ),
            patch.object(
                self.op,
                "_capture_skip_reason",
                side_effect=AssertionError("capture guard"),
            ),
        ):
            self.cp_ragged(None, mapping=object())
        self.s.compute.cuda_graph_capture_forward_enabled.assert_not_called()
        self.s.compute.cuda_graph_warmup_forward_enabled.assert_not_called()

    def test_cp_graph_decode_and_warmup_guards_preserve_computation(self):
        for flags, reason in (
            ({"is_cuda_graph": True}, "graph"),
            ({"prefill_cuda_graph_copy_params": object()}, "graph"),
            ({"is_prefill": False}, "decode"),
            ({"is_target_verify": True}, "paged"),
            ({"is_draft_extend": True}, "paged"),
        ):
            with self.subTest(flags=flags):
                context = _Context(mqa_enabled=True)
                run = self.cp_ragged(context, **flags)
                self.assertEqual(len(run.mqa_calls), 2)
                context.note_skip.assert_called_once_with(reason, layer=7)
                context.begin_mqa.assert_not_called()
                context.begin_topk.assert_not_called()
        for flag, reason in (
            ("cuda_graph_warmup_forward_enabled", "warmup"),
            ("cuda_graph_capture_forward_enabled", "graph"),
        ):
            with (
                self.subTest(flag=flag),
                patch.object(self.s.compute, flag, return_value=True),
            ):
                context = _Context(mqa_enabled=True)
                self.cp_ragged(context)
                context.note_skip.assert_called_with(reason, layer=7)
                context.begin_mqa.assert_not_called()
                context.begin_topk.assert_not_called()

    def test_cp_joint_rejection_and_capture_errors_do_not_change_inference(self):
        baseline = self.cp_ragged(None)
        for stage in ("budget", "begin", "prepare", "finish"):
            with self.subTest(stage=stage):
                context = _Context(mqa_enabled=True)
                tokens = []

                def begin(**kwargs):
                    if stage == "begin":
                        raise TypeError("capture mapping parameter rejected")
                    if stage == "budget":
                        return None
                    token = _Token()
                    getattr(
                        token, "prepare_topk" if stage == "prepare" else "finish"
                    ).side_effect = RuntimeError(stage)
                    tokens.append(token)
                    return token

                context.begin_mqa.side_effect = begin
                run = self.cp_ragged(context)
                torch.testing.assert_close(run.result, baseline.result)
                self.assertEqual(run.events, baseline.events)
                context.begin_topk.assert_not_called()
                for token in tokens:
                    token.cancel.assert_called_once_with(reason="capture_error")
                self.assertEqual(context.begin_mqa.call_count, 2)

    def test_cp_kernel_errors_cancel_joint_once_and_preserve_exception(self):
        for stage in ("mqa", "topk"):
            for error in (RuntimeError("kernel failed"), KeyboardInterrupt()):
                with self.subTest(stage=stage, error=type(error)):
                    context = _Context(mqa_enabled=True)
                    self.s.ops.dsv4_top_k_per_row_prefill.reset_mock()
                    self.s.ops.dsv4_top_k_per_row_prefill.side_effect = (
                        error if stage == "topk" else _cpu_topk
                    )
                    with self.assertRaises(type(error)) as raised:
                        self.cp_ragged(
                            context, mqa_error=error if stage == "mqa" else None
                        )
                    self.assertIs(raised.exception, error)
                    self.assertEqual(len(context.tokens), 1)
                    context.tokens[0].cancel.assert_called_once_with(
                        reason="kernel_error"
                    )
                    context.tokens[0].finish.assert_not_called()
                    context.begin_topk.assert_not_called()
                    self.assertEqual(
                        self.s.ops.dsv4_top_k_per_row_prefill.call_count,
                        int(stage == "topk"),
                    )

    def test_cp_helpers_reject_missing_ids_and_bad_metadata_without_kernel_changes(
        self,
    ):
        for cp_size, reason in (
            (8, "missing_row_mapping"),
            (torch.tensor([8, 8]), "capture_error"),
        ):
            with self.subTest(reason=reason):
                context = _Context()
                self.s.ops.dsv4_top_k_per_row_prefill.reset_mock()
                self.op._run_prefill_topk(
                    self.logits,
                    self.starts,
                    self.ends,
                    self.output,
                    8,
                    context=context,
                    total_query_rows=5,
                    cp_size=cp_size,
                )
                context.begin_topk.assert_not_called()
                context.note_skip.assert_called_once_with(reason, layer=7)
                self.s.ops.dsv4_top_k_per_row_prefill.assert_called_once()
                self.assertTrue(torch.any(self.output >= 0))
                context.note_skip.reset_mock()
                self.assertIsNone(
                    self.op._begin_mqa_capture(
                        context,
                        self.logits,
                        (None, None),
                        None,
                        self.starts,
                        self.ends,
                        0,
                        5,
                        cp_size=cp_size,
                    )
                )
                context.begin_mqa.assert_not_called()
                context.note_skip.assert_called_once_with(reason, layer=7)

    def test_cp_topk_capture_api_errors_preserve_raw_kernel_output(self):
        baseline = self.cp_ragged(None)
        context = _Context()
        context.begin_topk.side_effect = TypeError("mapping API rejected")
        with self.assertLogs(self.s.base._logger, level="WARNING"):
            run = self.cp_ragged(context)
        torch.testing.assert_close(run.result, baseline.result)
        self.assertEqual(run.events, baseline.events)
        self.assertEqual(context.begin_topk.call_count, 2)
        context.begin_mqa.assert_not_called()
        context.note_skip.assert_called_with("capture_error", layer=7)

    def test_cp_real_capture_roundtrip_preserves_global_ids_and_raw_indices(self):
        for joint in (False, True):
            with (
                self.subTest(joint=joint),
                self.recorder(mqa=joint) as (capture, recorder, context, root),
            ):
                with patch.dict(os.environ, {"DSV4_INDEXER_TOPK_CANONICALIZE": "1"}):
                    run = self.cp_ragged(context)
                self.assertTrue(recorder.close())
                self.assertEqual(recorder.stats()["completed"], 2)
                self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                paths = list(root.rglob("manifest.json"))
                self.assertEqual(len(paths), 2)
                bundles = sorted(
                    (capture.load_capture(path) for path in paths),
                    key=lambda item: int(item[1]["query_global_ids"][0]),
                )
                self.assertEqual(bundles[0][1]["query_global_ids"].tolist(), [0, 3])
                self.assertEqual(bundles[1][1]["query_global_ids"].tolist(), [4])
                self.assertEqual(
                    bundles[0][1]["indices"].tolist(), [[2, 1, 0], [1, 0, -1]]
                )
                self.assertEqual(run.result[:2].tolist(), [[0, 1, 2], [5, 6, -1]])
                if joint:
                    for _, tensors in bundles:
                        self.assertEqual(
                            tensors["q_fp8"][:, 0, 0].float().tolist(),
                            tensors["query_global_ids"].tolist(),
                        )
                        self.assertEqual(tensors["k_fp8"].shape, (8, 128))

    def test_cp_real_quota_rejection_never_falls_back_to_topk(self):
        for joint in (False, True):
            for overrides, reason in (
                ({"max_bundle_bytes": 1}, "oversize"),
                ({"max_total_bytes": 1}, "total_budget"),
                ({"max_inflight_bytes": 1}, "inflight_budget"),
            ):
                with (
                    self.subTest(joint=joint, reason=reason),
                    self.recorder(mqa=joint, config_overrides=overrides) as (
                        _,
                        recorder,
                        context,
                        root,
                    ),
                ):
                    baseline = self.cp_ragged(None)
                    run = self.cp_ragged(context)
                    torch.testing.assert_close(run.result, baseline.result)
                    self.assertTrue(recorder.close())
                    self.assertEqual(recorder.stats()["completed"], 0)
                    self.assertEqual(recorder.stats()["reserved_bytes"], 0)
                    self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                    self.assertEqual(recorder.stats()["skip_reasons"][reason], 2)
                    self.assertEqual(list(root.rglob("manifest.json")), [])

    def test_hybrid_forward_passes_existing_cp_global_ids(self):
        module = object.__new__(self.s.indexer.Indexer)
        torch.nn.Module.__init__(module)
        module.parallelism_config = SimpleNamespace(
            prefill_cp_config=SimpleNamespace(is_enabled=lambda: True)
        )
        module.layer_idx = 7
        module._kv_owner_tokens_per_block = 64
        module._fuse_logits_head_gate = False
        module._prefill_cp_fused_quant_enabled = Mock(return_value=False)
        module._get_q_k_bf16 = Mock(return_value=(None, None))
        module._quantize_q_k = Mock(return_value=(None, None))
        module._get_logits_head_gate = Mock(return_value=None)
        module.indexer_op = SimpleNamespace(
            _get_topk_ragged_cp=Mock(return_value=self.output)
        )
        params = SimpleNamespace(
            total_global_ids=torch.tensor([0, 3, 4]),
            total_local_ids=torch.tensor([4, 0, 2]),
            cu_kv_seqlens_global=torch.tensor([0, 5, 8]),
            total_kv_len=8,
            precomputed_ks=None,
            precomputed_ke=None,
            precomputed_lengths=None,
            precomputed_topk_off=None,
            cp_size=8,
            cp_rank=3,
            kv_cache_sharded=True,
            total_local_ids_is_identity=False,
        )
        result = module.forward(
            None,
            None,
            None,
            None,
            SimpleNamespace(is_prefill=True),
            False,
            cp_params=params,
        )
        self.assertIs(result, self.output)
        call = module.indexer_op._get_topk_ragged_cp.call_args
        self.assertIs(call.kwargs["total_global_ids"], params.total_global_ids)
        self.assertIs(call.args[5], params.total_local_ids)
        self.assertEqual(call.args[12:15], (True, 8, 3))

    @staticmethod
    def decode_context(joint=False, rows=2):
        context = _Context(mqa_enabled=joint)
        context.context.identity = {
            "model_role": "target",
            "requests": [
                {"phase": "decode", "q_len": 1, "kv_len": length - 1}
                for length in (3, 70)[:rows]
            ],
        }
        return context

    def paged(
        self,
        context,
        *,
        rows=2,
        schedule="present",
        multi_tokens=1,
        mqa_effect=None,
        topk_effect=None,
        workspace_error=None,
        production_layout=False,
        metadata_error=None,
        **flags,
    ):
        count = rows * multi_tokens
        heads = self.op.index_n_heads
        q = torch.arange(count * heads * 128, dtype=torch.float32).reshape(
            count, heads, 128
        )
        q = (q % 8).to(torch.float8_e4m3fn)
        cache = (
            torch.arange(8 * 64 * 132, dtype=torch.int32)
            .to(torch.uint8)
            .reshape(8, 64, 132)
        )
        cache_before = cache.clone()
        weights = torch.arange(count * heads, dtype=torch.float32).reshape(
            count, heads, 1
        )
        table = torch.tensor([[5, 1], [3, 7]], dtype=torch.int32)[:rows]
        lengths = torch.tensor([3, 70], dtype=torch.int32)[:rows].repeat_interleave(
            multi_tokens
        )
        metadata = (
            torch.zeros((149, 2), dtype=torch.int32)
            if production_layout
            else torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        )
        if production_layout:
            # SM90 assigns the two one-split requests to the final two SMs;
            # the last entry is the one-past-batch sentinel, not another idle SM.
            metadata[-rows:, 0] = torch.arange(1, rows + 1, dtype=torch.int32)
        params = SimpleNamespace(
            kvlen_d=lengths,
            expanded_seq_lens=lengths,
            schedule_metadata=(
                metadata
                if schedule == "present"
                else (
                    None if schedule == "missing" else torch.empty(0, dtype=torch.int32)
                )
            ),
        )
        inputs = SimpleNamespace(
            tensor_capture_context=context,
            is_prefill=False,
            decode_cu_seqlens_d=torch.arange(rows + 1, dtype=torch.int32),
            kv_cache_kernel_block_id_device=table,
        )
        inputs.__dict__.update(flags)
        scores = torch.arange(count * 256, dtype=torch.float32).reshape(count, 256)
        logits = scores[:, :128] if production_layout else scores[:, ::2]
        logits_before = logits.clone()
        workspace = torch.full(
            (1 << 20 if production_layout else 137,), 193, dtype=torch.uint8
        )
        workspace_before = workspace.clone()

        def mqa(*args, **kwargs):
            if mqa_effect is not None:
                mqa_effect(*args, **kwargs)
            return logits

        def topk(scores, actual_lengths, output, actual_workspace, k, max_seq_len):
            _cpu_topk(scores, torch.zeros_like(actual_lengths), actual_lengths, output)
            if topk_effect is not None:
                topk_effect(
                    scores, actual_lengths, output, actual_workspace, k, max_seq_len
                )

        distribution = SimpleNamespace(
            metadata={"Name": "deep_gemm"},
            version="2.8.0+122e18b.cu132",
            locate_file=lambda name: Path(self.s.deep.__file__).resolve(),
        )
        with (
            patch.object(
                tensor_capture_indexer.package_metadata,
                "distributions",
                return_value=[distribution],
                side_effect=metadata_error,
            ) as version_metadata,
            patch.object(
                self.s.deep,
                "__version__",
                "2.8.0",
                create=True,
            ),
            patch.object(
                self.s.deep, "get_num_sms", return_value=148, create=True
            ) as sms,
            patch.object(
                self.s.deep,
                "get_paged_mqa_logits_metadata",
                return_value=metadata,
                create=True,
            ) as build_metadata,
            patch.object(
                torch.cuda, "get_device_capability", return_value=(9, 0)
            ) as capability,
            patch.object(
                self.s.deep, "fp8_paged_mqa_logits", side_effect=mqa
            ) as mqa_op,
            patch.object(self.op, "_paged_topk_op", side_effect=topk) as topk_op,
            patch.object(
                self.s.base,
                "_get_topk_workspace",
                return_value=workspace,
                side_effect=workspace_error,
            ) as get_workspace,
        ):
            result = self.op._get_topk_paged(
                q, weights, SimpleNamespace(kv_scale_base=cache), params, inputs
            )
        return SimpleNamespace(
            result=result,
            mqa=mqa_op,
            topk=topk_op,
            q=q,
            cache=cache,
            cache_before=cache_before,
            logits=logits,
            logits_before=logits_before,
            workspace=workspace,
            workspace_before=workspace_before,
            get_workspace=get_workspace,
            table=table,
            lengths=lengths,
            metadata=metadata,
            build_metadata=build_metadata,
            sms=sms,
            capability=capability,
            version_metadata=version_metadata,
        )

    def test_paged_decode_actual_inputs_backend_workspace_and_raw_outputs(self):
        for backend in ("dsv4_persistent", "topk_v3"):
            for joint in (False, True):
                for rows in (1, 2):
                    with self.subTest(backend=backend, joint=joint, rows=rows):
                        with patch.dict(
                            os.environ, {"GLM5_INDEXER_TOPK_BACKEND": backend}
                        ):
                            self.op = self.s.base.IndexerOp(2, 128, 3, 64, layer_idx=7)
                        context = self.decode_context(joint, rows)
                        with patch.dict(
                            os.environ,
                            {"GLM5_INDEXER_TOPK_BACKEND": "invalid-after-init"},
                        ):
                            run = self.paged(context, rows=rows)
                        expected_backend = (
                            "topk_v3"
                            if backend == "topk_v3"
                            else "dsv4_persistent_topk"
                        )
                        self.assertEqual(self.op._paged_topk_backend, expected_backend)
                        self.assertEqual(
                            run.result.tolist(), [[2, 1, 0], [69, 68, 67]][:rows]
                        )
                        run.mqa.assert_called_once()
                        run.topk.assert_called_once()
                        run.get_workspace.assert_called_once_with(torch.device("cpu"))
                        self.assertIs(run.topk.call_args.args[3], run.workspace)
                        torch.testing.assert_close(run.cache, run.cache_before)
                        torch.testing.assert_close(run.workspace, run.workspace_before)
                        params = (
                            context.tokens[0].parameters
                            if joint
                            else context.begin_paged_topk.call_args.kwargs["parameters"]
                        )
                        self.assertEqual(
                            params,
                            dict(
                                k=3,
                                max_seq_len=128,
                                index_space="sequence_relative",
                                workspace_bytes=137,
                                stride0=256,
                                stride1=2,
                            ),
                        )
                        self.assertNotIn("workspace", context.calls[0])
                        if joint:
                            begin = context.begin_paged_mqa.call_args.kwargs
                            for name, actual in zip(
                                (
                                    "q_fp8",
                                    "kv_cache_fp8",
                                    "weights",
                                    "context_lens",
                                    "block_table",
                                    "schedule_metadata",
                                ),
                                run.mqa.call_args.args[:6],
                            ):
                                self.assertIs(begin[name], actual)
                            self.assertEqual(begin["q_fp8"].shape, (rows, 1, 2, 128))
                            self.assertEqual(
                                begin["kv_cache_fp8"].shape, (8, 64, 1, 132)
                            )
                            self.assertEqual(begin["kv_cache_fp8"].dtype, torch.uint8)
                            self.assertIs(begin["lengths"], run.topk.call_args.args[1])
                            self.assertEqual(
                                begin["schedule_environment"],
                                dict(
                                    deep_gemm_version="2.8.0+122e18b.cu132",
                                    device_capability=[9, 0],
                                    num_sms=148,
                                ),
                            )
                            context.begin_paged_topk.assert_not_called()
                            run.sms.assert_called_once()
                            run.capability.assert_called_once_with(torch.device("cpu"))
                        else:
                            context.begin_paged_mqa.assert_not_called()
                            run.sms.assert_not_called()
                            run.capability.assert_not_called()
                        context.tokens[0].finish.assert_called_once()
                        context.tokens[0].cancel.assert_not_called()
                        context.begin_mqa.assert_not_called()
                        context.begin_topk.assert_not_called()

    def test_paged_decode_schedule_reuses_actual_num_sms_and_tensor(self):
        for schedule in ("present", "missing", "empty"):
            for joint in (False, True):
                with self.subTest(schedule=schedule, joint=joint):
                    context = self.decode_context(joint)
                    run = self.paged(context, schedule=schedule)
                    self.assertIs(run.mqa.call_args.args[5], run.metadata)
                    self.assertEqual(
                        run.build_metadata.call_count, int(schedule != "present")
                    )
                    self.assertEqual(
                        run.sms.call_count, int(joint or schedule != "present")
                    )
                    if schedule != "present":
                        args = run.build_metadata.call_args.args
                        self.assertIs(args[0], run.mqa.call_args.args[3])
                        self.assertEqual(args[1:], (64, 148))

    def test_paged_decode_disabled_has_no_capture_or_device_queries(self):
        with (
            patch.object(
                self.op,
                "_begin_paged_mqa_capture",
                side_effect=AssertionError("capture entered"),
            ),
            patch.object(
                self.op,
                "_capture_skip_reason",
                side_effect=AssertionError("capture guard"),
            ),
            patch.object(
                self.s.base,
                "indexer_capture_context",
                side_effect=AssertionError("adapter entered"),
            ),
        ):
            run = self.paged(None)
        run.version_metadata.assert_not_called()
        run.capability.assert_not_called()
        run.sms.assert_not_called()
        self.s.compute.cuda_graph_warmup_forward_enabled.assert_not_called()
        self.s.compute.cuda_graph_capture_forward_enabled.assert_not_called()
        run.mqa.assert_called_once()
        run.topk.assert_called_once()

    def test_paged_decode_phase_mapping_cp_and_graph_guards(self):
        for flags, reason in (
            ({"is_prefill": True}, "phase"),
            ({"context_parallel_info": object()}, "cp"),
            ({"is_cuda_graph": True}, "graph"),
            ({"prefill_cuda_graph_copy_params": object()}, "graph"),
        ):
            with self.subTest(flags=flags):
                context = self.decode_context(True)
                run = self.paged(context, **flags)
                context.note_skip.assert_called_once_with(reason, layer=7)
                context.begin_paged_mqa.assert_not_called()
                context.begin_paged_topk.assert_not_called()
                run.capability.assert_not_called()
                run.sms.assert_not_called()
                run.topk.assert_called_once()
        for identity, reason in (
            ({"model_role": "draft", "requests": []}, "phase"),
            ({"model_role": "target"}, "missing_row_mapping"),
            (
                {"model_role": "target", "requests": [{"phase": "decode", "q_len": 1}]},
                "missing_row_mapping",
            ),
            (
                {
                    "model_role": "target",
                    "requests": [{"phase": "decode", "q_len": 2}] * 2,
                },
                "missing_row_mapping",
            ),
            (
                {
                    "model_role": "target",
                    "requests": [
                        {"phase": "decode", "q_len": 1},
                        {"phase": "prefill", "q_len": 1},
                    ],
                },
                "phase",
            ),
        ):
            with self.subTest(identity=identity):
                context = self.decode_context(True)
                context.context.identity = identity
                run = self.paged(context)
                context.note_skip.assert_called_once_with(reason, layer=7)
                context.begin_paged_mqa.assert_not_called()
                context.begin_paged_topk.assert_not_called()
                run.capability.assert_not_called()
        for flag, reason in (
            ("cuda_graph_warmup_forward_enabled", "warmup"),
            ("cuda_graph_capture_forward_enabled", "graph"),
        ):
            with (
                self.subTest(flag=flag),
                patch.object(self.s.compute, flag, return_value=True),
            ):
                context = self.decode_context(True)
                run = self.paged(context)
                context.note_skip.assert_called_once_with(reason, layer=7)
                context.begin_paged_mqa.assert_not_called()
                context.begin_paged_topk.assert_not_called()
                run.capability.assert_not_called()
                run.sms.assert_not_called()

    def test_paged_verify_and_draft_keep_expanded_kernel_inputs_without_capture(self):
        for flag in ("is_target_verify", "is_draft_extend"):
            with self.subTest(flag=flag):
                context = self.decode_context(True)
                run = self.paged(
                    context, multi_tokens=3, is_prefill=True, **{flag: True}
                )
                args = run.mqa.call_args.args
                self.assertEqual(args[0].shape, (6, 1, 2, 128))
                torch.testing.assert_close(
                    args[4], run.table.repeat_interleave(3, dim=0)
                )
                torch.testing.assert_close(args[3], run.lengths.reshape(-1, 1))
                context.note_skip.assert_called_once_with("paged", layer=7)
                context.begin_paged_mqa.assert_not_called()
                context.begin_paged_topk.assert_not_called()
                run.mqa.assert_called_once()
                run.topk.assert_called_once()

    def test_paged_native_mtp_keeps_grouped_inputs_without_capture(self):
        for flag in ("is_target_verify", "is_draft_extend"):
            with self.subTest(flag=flag):
                context = self.decode_context(True)
                grouped_lens = torch.tensor(
                    [[3, 3, 3], [70, 70, 70]], dtype=torch.int32
                )
                with patch.object(
                    self.s.base, "paged_mqa_context_lens", return_value=grouped_lens
                ) as layout:
                    run = self.paged(
                        context,
                        multi_tokens=3,
                        schedule="missing",
                        is_prefill=True,
                        **{flag: True},
                    )
                layout.assert_called_once()
                self.assertIs(layout.call_args.args[1].expanded_seq_lens, run.lengths)
                args = run.mqa.call_args.args
                self.assertEqual(args[0].shape, (2, 3, 2, 128))
                self.assertEqual(args[0].stride(), run.q.view(2, 3, 2, 128).stride())
                self.assertEqual(args[0].data_ptr(), run.q.data_ptr())
                torch.testing.assert_close(
                    args[0].float(), run.q.view(2, 3, 2, 128).float()
                )
                self.assertIs(args[3], grouped_lens)
                torch.testing.assert_close(grouped_lens.flatten(), run.lengths)
                self.assertIs(args[4], run.table)
                self.assertEqual(args[4].shape, (2, 2))
                run.build_metadata.assert_called_once()
                self.assertIs(run.build_metadata.call_args.args[0], args[3])
                self.assertEqual(run.build_metadata.call_args.args[1:], (64, 148))
                self.assertIs(args[5], run.metadata)
                self.assertIs(run.topk.call_args.args[1], run.lengths)
                self.assertEqual(run.result.shape, (6, 3))
                context.note_skip.assert_called_once_with("paged", layer=7)
                context.begin_paged_mqa.assert_not_called()
                context.begin_paged_topk.assert_not_called()
                self.assertEqual(context.tokens, [])
                run.version_metadata.assert_not_called()
                run.capability.assert_not_called()
                run.sms.assert_called_once()
                run.mqa.assert_called_once()
                run.topk.assert_called_once()

    def test_paged_decode_snapshots_precede_mqa_and_topk_mutation(self):
        context = self.decode_context(True)

        def mutate_inputs(q, cache, weights, *args, **kwargs):
            q.copy_(torch.zeros(q.shape).to(q.dtype))
            cache.zero_()
            weights.zero_()

        def mutate_logits(logits, *args):
            logits.fill_(-99)

        run = self.paged(context, mqa_effect=mutate_inputs, topk_effect=mutate_logits)
        self.assertTrue(torch.any(context.calls[0]["kv_cache_fp8"] != 0))
        self.assertTrue(torch.any(context.calls[0]["q_fp8"].float() != 0))
        self.assertTrue(torch.any(context.calls[0]["weights"] != 0))
        torch.testing.assert_close(context.tokens[0].logits, run.logits_before)
        self.assertEqual(context.tokens[0].output.tolist(), [[2, 1, 0], [69, 68, 67]])

    def test_paged_decode_capture_failures_and_joint_rejection_preserve_inference(self):
        baseline = self.paged(None).result
        for stage in ("budget", "begin", "prepare_false", "prepare_error", "finish"):
            with self.subTest(stage=stage):
                context = self.decode_context(True)
                token = _Token()
                if stage in ("budget", "begin"):
                    context.begin_paged_mqa.side_effect = (
                        RuntimeError("begin") if stage == "begin" else None
                    )
                    context.begin_paged_mqa.return_value = None
                else:
                    context.begin_paged_mqa.side_effect = None
                    context.begin_paged_mqa.return_value = token
                    if stage == "prepare_false":
                        token.prepare_topk.side_effect = None
                        token.prepare_topk.return_value = False
                    elif stage == "prepare_error":
                        token.prepare_topk.side_effect = RuntimeError("prepare")
                    else:
                        token.finish.side_effect = RuntimeError("finish")
                run = self.paged(context)
                torch.testing.assert_close(run.result, baseline)
                context.begin_paged_topk.assert_not_called()
                run.mqa.assert_called_once()
                run.topk.assert_called_once()
                if stage not in ("budget", "begin"):
                    token.cancel.assert_called_once_with(reason="capture_error")
        context = self.decode_context()
        context.begin_paged_topk.side_effect = TypeError("adapter parameter failure")
        run = self.paged(context)
        torch.testing.assert_close(run.result, baseline)
        context.note_skip.assert_called_once_with("capture_error", layer=7)

    def test_paged_decode_kernel_and_workspace_failures_cancel_once(self):
        for stage in ("mqa", "workspace", "topk", "prepare"):
            for error in (RuntimeError(stage), KeyboardInterrupt()):
                with self.subTest(stage=stage, error=type(error)):
                    context = self.decode_context(True)
                    token = _Token()
                    context.begin_paged_mqa.side_effect = None
                    context.begin_paged_mqa.return_value = token
                    failure = Mock(side_effect=error)
                    if stage == "prepare":
                        token.prepare_topk.side_effect = error
                    if stage == "prepare" and isinstance(error, Exception):
                        run = self.paged(context)
                        run.topk.assert_called_once()
                    else:
                        with self.assertRaises(type(error)) as raised:
                            self.paged(
                                context,
                                mqa_effect=failure if stage == "mqa" else None,
                                topk_effect=failure if stage == "topk" else None,
                                workspace_error=error if stage == "workspace" else None,
                            )
                        self.assertIs(raised.exception, error)
                    token.cancel.assert_called_once_with(
                        reason="capture_error" if stage == "prepare" else "kernel_error"
                    )
                    token.finish.assert_not_called()
                    context.begin_paged_topk.assert_not_called()

    def test_paged_decode_schedule_environment_failure_is_capture_only(self):
        context = self.decode_context(True)
        baseline = self.paged(None).result
        run = self.paged(context, metadata_error=RuntimeError("version unavailable"))
        torch.testing.assert_close(run.result, baseline)
        context.begin_paged_mqa.assert_not_called()
        context.begin_paged_topk.assert_not_called()
        context.note_skip.assert_called_once_with("unknown_paged_schedule_abi", layer=7)
        run.capability.assert_not_called()
        run.sms.assert_not_called()
        run.mqa.assert_called_once()
        run.topk.assert_called_once()

    def test_paged_decode_excluded_layer_precedes_metadata_queries(self):
        context = self.decode_context(True)
        context.context.selected.return_value = False
        baseline = self.paged(None).result
        run = self.paged(context)
        torch.testing.assert_close(run.result, baseline)
        context.context.selected.assert_called_once_with("indexer.mqa_topk", 7)
        run.version_metadata.assert_not_called()
        run.capability.assert_not_called()
        run.sms.assert_not_called()
        context.begin_paged_mqa.assert_not_called()
        context.begin_paged_topk.assert_not_called()
        run.mqa.assert_called_once()
        run.topk.assert_called_once()

    def test_paged_decode_real_roundtrip_and_budget_rejection(self):
        for joint in (False, True):
            for budget in (False, True):
                with self.subTest(joint=joint, budget=budget):
                    self.op = self.s.base.IndexerOp(32, 128, 512, 64, layer_idx=7)
                    config = {"phases": ["decode"]}
                    if budget:
                        config["max_bundle_bytes"] = 1
                    requests = [
                        {
                            "phase": "decode",
                            "request_id": "r1",
                            "q_len": 1,
                            "kv_len": 2,
                        },
                        {
                            "phase": "decode",
                            "request_id": "r2",
                            "q_len": 1,
                            "kv_len": 69,
                        },
                    ]
                    with self.recorder(
                        mqa=joint,
                        config_overrides=config,
                        identity_overrides={
                            "requests": requests,
                            "server_role": "DECODE",
                        },
                    ) as (capture, recorder, context, root):
                        self.assertIsNotNone(context)
                        baseline = self.paged(None, production_layout=True)
                        run = self.paged(context, production_layout=True)
                        torch.testing.assert_close(run.result, baseline.result)
                        torch.testing.assert_close(run.cache, run.cache_before)
                        self.assertTrue(recorder.close())
                        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                        self.assertEqual(
                            recorder.stats()["completed"],
                            0 if budget else 1,
                            recorder.stats(),
                        )
                        paths = list(root.rglob("manifest.json"))
                        self.assertEqual(len(paths), 0 if budget else 1)
                        if budget:
                            self.assertEqual(recorder.stats()["reserved_bytes"], 0)
                            self.assertEqual(
                                recorder.stats()["skip_reasons"]["oversize"], 1
                            )
                        else:
                            manifest, tensors = capture.load_capture(paths[0])
                            self.assertNotIn("workspace", tensors)
                            self.assertEqual(
                                manifest["parameters"]["workspace_bytes"], 1 << 20
                            )
                            self.assertEqual(
                                manifest["parameters"]["index_space"],
                                "sequence_relative",
                            )
                            torch.testing.assert_close(tensors["indices"], run.result)
                            if joint:
                                self.assertEqual(
                                    tensors["packed_kv_pages"].shape, (3, 64, 1, 132)
                                )
                                self.assertEqual(
                                    set(tensors["page_ids"].tolist()), {3, 5, 7}
                                )
                                self.assertNotIn("kv_cache_fp8", tensors)

    def test_paged_decode_dp_owner_and_prior_length_contract(self):
        for joint in (False, True):
            for prior_matches in (False, True):
                with self.subTest(joint=joint, prior_matches=prior_matches):
                    self.op = self.s.base.IndexerOp(32, 128, 512, 64, layer_idx=7)
                    identity = {
                        "server_role": "DECODE",
                        "dp_rank": 3,
                        "owner_id": "owner3",
                        "replica_id": "replica3",
                        "requests": [
                            {
                                "phase": "decode",
                                "request_id": "r1",
                                "q_len": 1,
                                "kv_len": 2 if prior_matches else 1,
                            },
                            {
                                "phase": "decode",
                                "request_id": "r2",
                                "q_len": 1,
                                "kv_len": 69,
                            },
                        ],
                    }
                    with self.recorder(
                        tp_rank=0,
                        mqa=joint,
                        config_overrides={"phases": ["decode"]},
                        identity_overrides=identity,
                    ) as (_, recorder, context, root):
                        baseline = self.paged(None, production_layout=True)
                        run = self.paged(context, production_layout=True)
                        torch.testing.assert_close(run.result, baseline.result)
                        self.assertTrue(recorder.close())
                        self.assertEqual(
                            recorder.stats()["completed"],
                            int(prior_matches),
                            recorder.stats(),
                        )
                        self.assertEqual(recorder.stats()["inflight_bytes"], 0)
                        self.assertEqual(
                            len(list(root.rglob("manifest.json"))), int(prior_matches)
                        )
                        if not prior_matches:
                            self.assertEqual(recorder.stats()["reserved_bytes"], 0)
                            self.assertTrue(recorder.stats()["skip_reasons"])
                        self.s.base.all_gather.assert_not_called()
                        self.s.base.barrier.assert_not_called()

    def test_missing_shared_indices_remains_an_error(self):
        attention = object.__new__(self.s.attention.MlaAttention)
        attention.layer_idx = 7
        attention.reuse_topk_indices = True
        with self.assertRaisesRegex(RuntimeError, "needs previous top-k"):
            attention._run_sparse_indexer(
                None, None, None, None, SimpleNamespace(is_sparse=lambda: True)
            )


if __name__ == "__main__":
    unittest.main()
