"""V4.1 global/QKV stream ordering, graph lifetime, and native pool writes."""

import contextlib
import os
import sys
import threading
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv41.fp8.decode import full_overlap
from rtp_llm.models_py.modules.dsv41.fp8.decode import global_overlap as overlap
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_decode_global import (
    _case,
    _clone_case,
)
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_qkv_rope_cache import (
    fused,
    make_case,
)

_ENV = {
    "DSV41_OVERLAP_GLOBAL_QKV": "1",
    "DSV41_FUSED_DECODE_GLOBAL": "1",
    "DSV41_FUSED_DECODE_INDEXER": "1",
    "DSV41_FUSED_QKV_ROPE_CACHE": "1",
    "MOEDBG": "0",
}


class GlobalOverlapCPUTest(unittest.TestCase):
    def test_fused_paged_global_skips_end_tensor_and_keeps_projection(self):
        from rtp_llm.models_py.modules.dsv41.attn_type import INDEXER_KV
        from rtp_llm.models_py.modules.dsv41.fp8 import (
            _v41_decode_global,
            _v41_decode_indexer,
        )

        class StartsWithoutAddition:
            def __add__(self, _other):
                raise AssertionError("fused paged decode must not compute request ends")

        starts = StartsWithoutAddition()
        x = torch.empty(2, 3, 8)
        positions = torch.empty(6, dtype=torch.int64)
        request_ids = torch.empty(6, dtype=torch.int32)
        pool = torch.zeros(3, 128, 68, dtype=torch.uint8)
        table = torch.ones(2, 1, dtype=torch.int32)
        projections = (torch.empty(6, 512), torch.empty(6, 512))
        for ratio in (1, 2):
            for lengths in (None, torch.tensor([2, 4], dtype=torch.int32)):
                for projected in (None, projections):
                    with self.subTest(
                        ratio=ratio,
                        packed=lengths is not None,
                        projected=projected is not None,
                    ):
                        owner = SimpleNamespace(
                            compress_ratio=ratio,
                            _rope_max_seq_len=128,
                            _cp_ctx=None,
                            _kv_cache=SimpleNamespace(
                                get_kernel_seq_size_per_block=lambda _tag: 128
                            ),
                            _block_tables_by_type={INDEXER_KV: table},
                            _source_pool=lambda _tag: pool,
                            _shared_attention={"decode_input_lengths": lengths},
                            layer_id=2,
                            index_n_heads=32,
                            index_head_dim=128,
                        )
                        with patch.object(
                            _v41_decode_global, "try_produce_global", return_value=True
                        ) as producer, patch.object(
                            _v41_decode_indexer, "is_supported", return_value=True
                        ):
                            AttentionV41FP8._produce_global_decode(
                                owner,
                                x,
                                positions,
                                request_ids,
                                starts,
                                projected=projected,
                            )
                        producer.assert_called_once_with(
                            owner,
                            x,
                            positions,
                            request_ids,
                            starts,
                            projected=projected,
                            input_lengths=lengths,
                        )
                        keys = owner._shared_attention["global"][2]
                        self.assertIs(keys.pool, pool)
                        self.assertEqual(keys.logical_entries_per_block, 128 // ratio)

    def test_query_preparation_keeps_unsupported_scorer_on_original_path(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer

        q = torch.ones(1, 4096, dtype=torch.bfloat16)
        keys = _v41_decode_indexer.DecodeIndexerKeys(
            torch.zeros(2, 128, 68, dtype=torch.uint8),
            torch.zeros(1, 1, dtype=torch.int32),
            128,
            128,
        )
        attn = SimpleNamespace(
            is_index_source=True,
            kv_source_layer_id=2,
            index_n_heads=32,
            index_head_dim=128,
            rope_head_dim=64,
            index_weights=torch.ones(32, 5120, dtype=torch.bfloat16),
            freqs_cis=torch.ones(1, 32, dtype=torch.complex64),
            _shared_attention={"global": {2: keys}},
        )
        query, weights, packed = AttentionV41FP8._prepare_indexer_query_decode(
            attn,
            torch.ones(1, 1, 5120, dtype=torch.bfloat16),
            torch.ones(1, 1, 1280, dtype=torch.bfloat16),
            torch.zeros(1, dtype=torch.int64),
            index_q=q,
        )
        torch.testing.assert_close(query, q.view(1, 32, 128), rtol=0, atol=0)
        torch.testing.assert_close(
            weights, torch.full((1, 32), 5120, dtype=torch.bfloat16), rtol=0, atol=0
        )
        self.assertIsNone(packed)

    def test_default_cpu_and_draft_gates(self):
        attn = SimpleNamespace(is_kv_source=True, compress_ratio=1, layer_id=2)
        x = torch.empty(4, 6, 5120, dtype=torch.bfloat16)
        for env, ratio in (("0", 1), ("1", 0), ("1", 1)):
            attn.compress_ratio = ratio
            with patch.dict(
                os.environ, {"DSV41_OVERLAP_GLOBAL_QKV": env}
            ), patch.object(overlap._v41_decode_global, "is_supported") as native:
                self.assertFalse(overlap.is_supported(attn, x, None, None, None))
                native.assert_not_called()

    def _mock_streams(self):
        self.current = Mock(name="current")
        self.current.cuda_stream = 51
        self.auxiliary = Mock(name="auxiliary")
        self.auxiliary.cuda_stream = 917
        self.attn = SimpleNamespace(_produce_global_decode=Mock())
        self.x = torch.empty(4, 6, 8)
        self.metadata = torch.arange(24)
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(overlap, "is_supported", return_value=True))
        stack.enter_context(
            patch.object(
                overlap,
                "_GLOBAL_COMPUTE_STREAMS",
                SimpleNamespace(streams={(0, 17, 51): self.auxiliary}),
            )
        )
        stack.enter_context(
            patch(
                "rtp_llm.models_py.modules.dsv41.fp8.attention._cuda_device_index",
                return_value=0,
            )
        )
        stack.enter_context(
            patch("torch.cuda.current_stream", return_value=self.current)
        )
        stack.enter_context(
            patch(
                "rtp_llm.models_py.modules.dsv41.fp8._v41_wo_a_quant._context",
                return_value=17,
            )
        )
        self.device_guard = stack.enter_context(
            patch("torch.cuda.device", side_effect=lambda _: contextlib.nullcontext())
        )
        stack.enter_context(
            patch("torch.cuda.stream", side_effect=lambda _: contextlib.nullcontext())
        )
        self.addCleanup(stack.close)

    def _start(self):
        return overlap.start_global_decode(
            self.attn, self.x, self.metadata, self.metadata, self.metadata
        )

    def test_fork_join_retains_inputs_and_is_idempotent(self):
        self._mock_streams()
        with patch("torch.cuda.is_current_stream_capturing", return_value=False):
            work = self._start()
        self.assertIsNotNone(work)
        self.device_guard.assert_called_once_with(self.x.device)
        self.auxiliary.wait_stream.assert_called_once_with(self.current)
        self.assertIs(work.inputs[0], self.x)
        self.current.wait_stream.assert_not_called()
        work.finish()
        work.finish()
        self.current.wait_stream.assert_called_once_with(self.auxiliary)
        self.assertEqual(work.inputs, ())

    def test_cold_capture_falls_back_and_warm_capture_reuses_stream(self):
        self._mock_streams()
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            self.assertIsNone(self._start())
        self.attn._produce_global_decode.assert_not_called()
        with patch("torch.cuda.is_current_stream_capturing", return_value=False):
            self._start().finish()
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            self._start().finish()
            self.x = torch.empty(8, 6, 8)
            self.assertIsNone(self._start())
        self.assertEqual(self.attn._produce_global_decode.call_count, 2)

    def test_producer_failure_joins_before_propagating(self):
        self._mock_streams()
        self.attn._produce_global_decode.side_effect = RuntimeError("producer failed")
        with patch("torch.cuda.is_current_stream_capturing", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "producer failed"):
                self._start()
        self.current.wait_stream.assert_called_once_with(self.auxiliary)
        self.assertFalse(hasattr(self.attn, "_global_decode_overlap_warmed"))

    def test_attention_prepares_index_query_before_join_and_joins_on_failure(self):
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV

        x = torch.empty(1, 1, 5120)
        shared_input_quant = (object(), object())
        metadata = SimpleNamespace(
            position_ids=torch.zeros(1, dtype=torch.int32),
            req_id_per_token=torch.zeros(1, dtype=torch.int32),
            start_pos=torch.zeros(1, dtype=torch.int32),
            swa_global_slots=torch.zeros(1, 128, dtype=torch.int32),
            pool_write_slot_mappings={SWA_KV: torch.zeros(1, dtype=torch.int64)},
        )
        for fail_qkv in (False, True):
            with self.subTest(fail_qkv=fail_qkv):
                calls = []
                work = SimpleNamespace(finish=lambda: calls.append("join"))
                qkv = SimpleNamespace(
                    kv=x,
                    qr=x,
                    qr_quantized=None,
                    index_q=None,
                    swa_written=False,
                    head_weights=None,
                    index_prepared=None,
                )

                def project(*args, **kwargs):
                    self.assertIs(kwargs["shared_input_quant"], shared_input_quant)
                    calls.append("qkv")
                    if fail_qkv:
                        raise RuntimeError("qkv stop")
                    return qkv

                prepared_query = object()

                def prepare(*args):
                    calls.append("index_query")
                    return prepared_query

                def select(*args, **kwargs):
                    calls.append("indexer")
                    self.assertIs(kwargs.get("prepared_query"), prepared_query)
                    raise RuntimeError("indexer stop")

                attn = SimpleNamespace(
                    _begin_forward=lambda: None,
                    _shared_attention={},
                    freqs_cis=torch.zeros(1, 32, dtype=torch.complex64),
                    compress_ratio=1,
                    is_kv_source=True,
                    is_index_source=True,
                    index_topk=128,
                    _mega_attention=SimpleNamespace(
                        supports_input=lambda _: True,
                        supports_decode=lambda *args: True,
                        supports_pool=lambda *args: True,
                    ),
                    _source_pool=lambda _: None,
                    _global_region=lambda: SWA_KV,
                    _pool_view_3d_fp8=lambda _: None,
                    _decode_write_swa_fp8=lambda *args: calls.append("swa"),
                    _prepare_indexer_query_decode=prepare,
                    _select_indices_decode=select,
                    _produce_global_decode=Mock(),
                )
                with patch.dict(
                    os.environ, {"DSV41_FULL_DECODE_MULTISTREAM": "0"}
                ), patch.object(
                    overlap, "start_global_decode", return_value=work
                ), patch(
                    "rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv.decode_compute_qkv",
                    side_effect=project,
                ):
                    with self.assertRaisesRegex(RuntimeError, "qkv stop|indexer stop"):
                        AttentionV41FP8._forward_decode_body(
                            attn, x, metadata, shared_input_quant=shared_input_quant
                        )
                self.assertEqual(
                    calls,
                    (
                        ["qkv", "join"]
                        if fail_qkv
                        else ["qkv", "swa", "index_query", "join", "indexer"]
                    ),
                )
                attn._produce_global_decode.assert_not_called()


class FullOverlapCPUTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("DSV41_FULL_DECODE_MULTISTREAM", None)
        os.environ.pop("DSV41_FULL_DECODE_QB_PDL", None)

    def test_default_multistream_preserves_native_eligibility(self):
        attn = SimpleNamespace(
            is_kv_source=True,
            is_index_source=True,
            layer_id=2,
            kv_source_layer_id=2,
            tp_size=1,
            index_n_heads=32,
            index_head_dim=128,
            index_topk=512,
        )
        x = SimpleNamespace(is_cuda=True)
        for supported in (False, True):
            with self.subTest(supported=supported), patch.object(
                full_overlap, "supports_paged_global", return_value=supported
            ) as native:
                self.assertEqual(
                    full_overlap.is_supported(attn, x, None, None, None), supported
                )
                native.assert_called_once_with(attn, x, None, None, None)

    def test_attention_body_preserves_shared_input_in_full_path(self):
        from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV

        x = torch.empty(1, 1, 5120)
        metadata = SimpleNamespace(
            position_ids=torch.zeros(1, dtype=torch.int32),
            req_id_per_token=torch.zeros(1, dtype=torch.int32),
            start_pos=torch.zeros(1, dtype=torch.int32),
            swa_global_slots=torch.zeros(1, 128, dtype=torch.int32),
            pool_write_slot_mappings={},
        )
        mega = SimpleNamespace(
            supports_input=lambda _: True,
            supports_decode=lambda *args: True,
            supports_pool=lambda *args: True,
            merged_qb=object(),
        )
        attn = SimpleNamespace(
            _begin_forward=lambda: None,
            _shared_attention={"global": {2: object()}},
            freqs_cis=torch.ones(1, 32, dtype=torch.complex64),
            compress_ratio=1,
            is_kv_source=True,
            is_index_source=True,
            layer_id=2,
            kv_source_layer_id=2,
            tp_size=1,
            index_topk=512,
            index_weights=object(),
            _mega_attention=mega,
            _source_pool=lambda _: None,
            _global_region=lambda: SWA_KV,
            _pool_view_3d_fp8=lambda _: None,
            _produce_global_decode=Mock(),
        )
        main, glob, aux = Mock(), Mock(), Mock()
        main.cuda_stream = 51
        glob.cuda_stream, aux.cuda_stream = 123, 456
        glob.device = aux.device = torch.device("cuda", 0)
        attn._full_decode_overlap_warmed = {(0, tuple(x.shape), 123, 456)}
        with patch.dict(
            sys.modules, {"deep_gemm": SimpleNamespace(get_pdl=lambda: True)}
        ), patch.object(full_overlap, "is_supported", return_value=True), patch.object(
            full_overlap,
            "_GLOBAL_COMPUTE_STREAMS",
            SimpleNamespace(streams={(0, 17, 51): glob}),
        ), patch.object(
            full_overlap,
            "_INDEXER_STREAMS",
            SimpleNamespace(streams={(0, 17, 51): aux}),
        ), patch(
            "rtp_llm.models_py.modules.dsv41.fp8.attention._cuda_device_index",
            return_value=0,
        ), patch(
            "rtp_llm.models_py.modules.dsv41.fp8._v41_wo_a_quant._context",
            return_value=17,
        ), patch(
            "torch.cuda.current_stream", return_value=main
        ), patch(
            "torch.cuda.device", side_effect=lambda _: contextlib.nullcontext()
        ), patch(
            "torch.cuda.stream", side_effect=lambda _: contextlib.nullcontext()
        ), patch.object(
            full_overlap.F, "linear", return_value=Mock()
        ), patch.object(
            full_overlap._v41_decode_indexer, "get_decode_metadata", return_value=None
        ):
            for capturing in (False, True):
                for quantized in (None, (object(), object())):
                    with self.subTest(
                        capturing=capturing, shared=quantized is not None
                    ), patch(
                        "torch.cuda.is_current_stream_capturing", return_value=capturing
                    ), patch.object(
                        full_overlap,
                        "decode_compute_qkv",
                        side_effect=RuntimeError("qkv stop"),
                    ) as project:
                        with self.assertRaisesRegex(RuntimeError, "qkv stop"):
                            AttentionV41FP8._forward_decode_body(
                                attn, x, metadata, shared_input_quant=quantized
                            )
                        self.assertIs(
                            project.call_args.kwargs["shared_input_quant"], quantized
                        )
                        self.assertTrue(project.call_args.kwargs["qb_pdl"])

    def test_qb_pdl_gate_does_not_change_deepgemm_policy(self):
        dg = SimpleNamespace(get_pdl=Mock(return_value=True), set_pdl=Mock())
        mega = SimpleNamespace(merged_qb=object())
        with patch.dict(sys.modules, {"deep_gemm": dg}):
            with patch.dict(os.environ, {"DSV41_FULL_DECODE_QB_PDL": "0"}):
                self.assertFalse(full_overlap._qb_pdl_enabled(mega))
            dg.get_pdl.assert_not_called()
            for env in ({}, {"DSV41_FULL_DECODE_QB_PDL": "1"}):
                with self.subTest(env=env), patch.dict(os.environ, env):
                    dg.get_pdl.reset_mock()
                    dg.get_pdl.return_value = True
                    self.assertFalse(full_overlap._qb_pdl_enabled(None))
                    self.assertFalse(full_overlap._qb_pdl_enabled(SimpleNamespace()))
                    dg.get_pdl.assert_not_called()
                    self.assertTrue(full_overlap._qb_pdl_enabled(mega))
                    dg.get_pdl.return_value = False
                    self.assertFalse(full_overlap._qb_pdl_enabled(mega))
            dg.set_pdl.assert_not_called()
        with patch.dict(sys.modules, {"deep_gemm": SimpleNamespace()}):
            self.assertFalse(full_overlap._qb_pdl_enabled(mega))

    def test_gate_rejects_cpu_reuse_tp_and_disabled_without_native_access(self):
        attn = SimpleNamespace(
            is_kv_source=True,
            is_index_source=True,
            layer_id=2,
            kv_source_layer_id=2,
            tp_size=1,
            index_n_heads=32,
            index_head_dim=128,
            index_topk=512,
        )
        x = SimpleNamespace(is_cuda=True)
        for attribute, value in (
            ("is_kv_source", False),
            ("is_index_source", False),
            ("kv_source_layer_id", 1),
            ("tp_size", 2),
            ("index_topk", 128),
        ):
            with self.subTest(attribute=attribute), patch.dict(
                os.environ, {"DSV41_FULL_DECODE_MULTISTREAM": "1"}
            ), patch.object(attn, attribute, value), patch.object(
                full_overlap, "supports_paged_global"
            ) as native:
                self.assertFalse(full_overlap.is_supported(attn, x, None, None, None))
                native.assert_not_called()
        with patch.dict(os.environ, {"DSV41_FULL_DECODE_MULTISTREAM": "0"}):
            self.assertIsNone(
                full_overlap.try_full_decode(None, None, None, None, None, None, None)
            )
        with patch.dict(os.environ, {"DSV41_FULL_DECODE_MULTISTREAM": "1"}):
            self.assertFalse(
                full_overlap.is_supported(attn, torch.empty(1), None, None, None)
            )

    def test_candidate_fork_and_exit_join_preserve_lifetimes(self):
        main, glob, aux = Mock(), Mock(), Mock()
        logits, lengths, candidates, flags = (Mock() for _ in range(4))
        work = full_overlap.FullDecodeWork(main, glob, aux)
        with patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ), patch.object(
            full_overlap._v41_decode_topk,
            "select_candidates",
            return_value=(candidates, flags),
        ):
            self.assertEqual(
                work.select_candidates(logits, lengths, 8, 2048), (candidates, flags)
            )
        aux.wait_stream.assert_called_once_with(main)
        main.wait_stream.assert_not_called()
        self.assertEqual(work.inputs, [logits, lengths])
        work.finish()
        work.finish()
        self.assertEqual(main.wait_stream.call_count, 2)
        candidates.record_stream.assert_called_once_with(main)
        flags.record_stream.assert_called_once_with(main)
        self.assertEqual(work.inputs, [])

    def test_sorted_candidate_fork_without_bitmap(self):
        main, glob, aux = Mock(), Mock(), Mock()
        logits, lengths, candidates = (Mock() for _ in range(3))
        work = full_overlap.FullDecodeWork(main, glob, aux)
        with patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ), patch.object(
            full_overlap._v41_decode_topk,
            "select_candidates",
            return_value=(candidates, None),
        ) as select:
            self.assertEqual(
                work.select_candidates(
                    logits, lengths, 8, 2048, complete=True, build_flags=False
                ),
                (candidates, None),
            )
        select.assert_called_once_with(
            logits, lengths, 8, 2048, complete=True, build_flags=False
        )
        self.assertEqual(work.outputs, [candidates])
        work.finish()
        candidates.record_stream.assert_called_once_with(main)

    def test_cold_capture_and_failed_producer(self):
        x = torch.empty(1, 1, 5120)
        metadata = SimpleNamespace(
            req_id_per_token=torch.zeros(1), start_pos=torch.zeros(1)
        )
        attn = SimpleNamespace(
            is_kv_source=True,
            is_index_source=True,
            layer_id=2,
            kv_source_layer_id=2,
            tp_size=1,
            _produce_global_decode=Mock(side_effect=RuntimeError("producer failed")),
        )
        main, glob, aux = Mock(), Mock(), Mock()
        main.cuda_stream = 51
        glob.cuda_stream, aux.cuda_stream = 123, 456
        glob.device = aux.device = torch.device("cuda", 0)
        with patch.dict(
            os.environ, {"DSV41_FULL_DECODE_MULTISTREAM": "1"}
        ), patch.object(full_overlap, "is_supported", return_value=True), patch.object(
            full_overlap,
            "_GLOBAL_COMPUTE_STREAMS",
            SimpleNamespace(streams={(0, 17, 51): glob}),
        ), patch.object(
            full_overlap,
            "_INDEXER_STREAMS",
            SimpleNamespace(streams={(0, 17, 51): aux}),
        ), patch(
            "rtp_llm.models_py.modules.dsv41.fp8.attention._cuda_device_index",
            return_value=0,
        ), patch(
            "rtp_llm.models_py.modules.dsv41.fp8._v41_wo_a_quant._context",
            return_value=17,
        ), patch(
            "torch.cuda.current_stream", return_value=main
        ), patch(
            "torch.cuda.device", side_effect=lambda _: contextlib.nullcontext()
        ), patch(
            "torch.cuda.stream", side_effect=lambda _: contextlib.nullcontext()
        ):
            with patch("torch.cuda.is_current_stream_capturing", return_value=True):
                self.assertIsNone(
                    full_overlap.try_full_decode(
                        attn, x, metadata, None, None, None, None
                    )
                )
            attn._produce_global_decode.assert_not_called()
            with patch("torch.cuda.is_current_stream_capturing", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "producer failed"):
                    full_overlap.try_full_decode(
                        attn, x, metadata, None, None, None, None
                    )
            self.assertEqual(main.wait_stream.call_count, 2)
            self.assertFalse(hasattr(attn, "_full_decode_overlap_warmed"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GlobalOverlapCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("native global producer requires Blackwell")

    def setUp(self):
        env = patch.dict(os.environ, _ENV)
        env.start()
        self.addCleanup(env.stop)

    @torch.inference_mode()
    def test_merged_projection_preserves_packed_request_boundaries(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_global
        from rtp_llm.models_py.modules.dsv41.fp8.compressor import (
            _linear_bf16_bf16_fp32,
        )

        lengths = torch.tensor([1, 2, 6], device="cuda", dtype=torch.int32)
        chosen = torch.tensor([0, 6, 7, 12, 13, 14, 15, 16, 17], device="cuda")
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                reference = _case("cuda", ratio, 3, 6)
                reference.x = (
                    reference.x.flatten(0, 1)
                    .index_select(0, chosen)
                    .reshape(3, 3, -1)
                    .contiguous()
                )
                reference.positions = reference.positions.index_select(0, chosen)
                reference.req_ids = reference.req_ids.index_select(0, chosen)
                candidate = _clone_case(reference)
                flat = candidate.x.reshape(9, -1)
                values = _linear_bf16_bf16_fp32(flat, candidate.owner.global_wkv)
                scores = (
                    _linear_bf16_bf16_fp32(flat, candidate.owner.global_wgate)
                    if ratio == 2
                    else values
                )
                # FULL's merged projection returns row-strided column views.
                storage = torch.cat((values, scores), dim=1)
                projected = (storage[:, :512], storage[:, 512:])
                self.assertGreater(projected[0].stride(0), 512)
                self.assertTrue(
                    _v41_decode_global.try_produce_global(
                        reference.owner,
                        reference.x,
                        reference.positions,
                        reference.req_ids,
                        reference.starts,
                        input_lengths=lengths,
                    )
                )
                with patch(
                    "rtp_llm.models_py.modules.dsv41.fp8.compressor."
                    "_linear_bf16_bf16_fp32",
                    side_effect=AssertionError("merged projection must be reused"),
                ):
                    self.assertTrue(
                        _v41_decode_global.try_produce_global(
                            candidate.owner,
                            candidate.x,
                            candidate.positions,
                            candidate.req_ids,
                            candidate.starts,
                            projected=projected,
                            input_lengths=lengths,
                        )
                    )
                for tag, actual in candidate.owner._test_pools.items():
                    torch.testing.assert_close(
                        actual, reference.owner._test_pools[tag], rtol=0, atol=0
                    )

    def test_independent_threads_capture_with_separate_auxiliary_streams(self):
        import traceback

        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        inputs = [torch.ones(1, 1, 8, device="cuda") * (i + 1) for i in range(2)]
        outputs = [torch.empty_like(x) for x in inputs]
        metadata = torch.zeros(1, device="cuda", dtype=torch.int32)
        owners = [
            SimpleNamespace(
                _produce_global_decode=lambda x, *args, i=i: outputs[i].copy_(x)
            )
            for i in range(2)
        ]
        ready = threading.Barrier(2)
        forked = threading.Event()
        attempted = threading.Event()
        errors = []

        def work(i):
            try:
                with torch.cuda.device(0), torch.cuda.stream(streams[i]):
                    w = overlap.start_global_decode(
                        owners[i], inputs[i], metadata, metadata, metadata
                    )
                    w.finish()
                    streams[i].synchronize()
                    ready.wait(timeout=10)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(
                        graph, stream=streams[i], capture_error_mode="thread_local"
                    ):
                        if i == 1:
                            assert forked.wait(10)
                        try:
                            w = overlap.start_global_decode(
                                owners[i], inputs[i], metadata, metadata, metadata
                            )
                            if i == 0:
                                forked.set()
                                assert attempted.wait(10)
                            w.finish()
                        finally:
                            if i == 1:
                                attempted.set()
                    graph.replay()
                    streams[i].synchronize()
                    torch.testing.assert_close(outputs[i], inputs[i], rtol=0, atol=0)
            except BaseException as error:
                errors.append((i, str(error)))
                traceback.print_exc()
                attempted.set()
                forked.set()

        with patch.object(overlap, "is_supported", return_value=True):
            threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
        assert not any(t.is_alive() for t in threads), "thread timed out"
        if errors:
            raise AssertionError(errors)

    def _prepare(self, case):
        case.owner.is_kv_source = True
        case.owner._produce_global_decode = MethodType(
            AttentionV41FP8._produce_global_decode, case.owner
        )
        case.swa = make_case(4, 6, 8)
        case.swa["kv_norm"].fill_(1)

    def _run(self, case, concurrent):
        attn = case.owner
        work = (
            overlap.start_global_decode(
                attn, case.x, case.positions, case.req_ids, case.starts
            )
            if concurrent
            else None
        )
        if concurrent:
            self.assertIsNotNone(work)
        try:
            q = case.x[..., :512].unsqueeze(-2).expand(4, 6, 8, 512).contiguous()
            output = fused.try_fused_qkv_rope_cache(
                q,
                case.x[..., :512],
                case.swa["kv_norm"],
                case.positions,
                attn.freqs_cis,
                case.swa["pool_3d"],
                case.swa["slots"],
            )
        finally:
            if work is not None:
                work.finish()
        if not concurrent:
            attn._produce_global_decode(
                case.x, case.positions, case.req_ids, case.starts
            )
        # These main-stream reads must observe every auxiliary cache write.
        return output, {key: value.clone() for key, value in attn._test_pools.items()}

    @torch.no_grad()
    def test_native_graph_replays_updated_inputs_and_pool_reads(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                baseline = _case("cuda", ratio, 4, 6)
                candidate = _clone_case(baseline)
                self._prepare(baseline)
                self._prepare(candidate)
                initial = {
                    key: value.clone()
                    for key, value in baseline.owner._test_pools.items()
                }
                main = torch.cuda.Stream()
                main.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(main):
                    for _ in range(3):
                        self._run(candidate, True)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=main):
                        actual, observed = self._run(candidate, True)
                torch.cuda.current_stream().wait_stream(main)
                self.assertIsNotNone(actual)
                for delta in (0, 1, 3):
                    candidate.x.copy_(torch.randn_like(candidate.x))
                    baseline.x.copy_(candidate.x)
                    for case in (baseline, candidate):
                        case.starts.fill_(127 + delta)
                        case.positions.copy_(
                            (
                                case.starts[:, None] + torch.arange(6, device="cuda")
                            ).flatten()
                        )
                        case.swa["pool_3d"].fill_(173)
                        for key, pool in case.owner._test_pools.items():
                            pool.copy_(initial[key])
                    expected, expected_pools = self._run(baseline, False)
                    graph.replay()
                    for got, reference in (
                        (actual.q, expected.q),
                        (actual.kv, expected.kv),
                        (actual.freqs_cis, expected.freqs_cis),
                        (candidate.swa["pool_3d"], baseline.swa["pool_3d"]),
                    ):
                        torch.testing.assert_close(got, reference, rtol=0, atol=0)
                    for key, pool in observed.items():
                        torch.testing.assert_close(
                            pool, expected_pools[key], rtol=0, atol=0
                        )


def full_attention_case(ratio, batch, span, context):
    """Real TP1 FULL projection sizes and paged caches, without model loading."""
    from rtp_llm.models_py.modules.dsv41.attn_type import CSA_STATE, INDEXER_KV, SWA_KV
    from rtp_llm.models_py.modules.dsv41.fp8 import _v41_mega_attention as mega
    from rtp_llm.models_py.modules.dsv41.fp8._v41_merged_qb import MergedQB
    from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
        quantize_and_insert_swa_k_cache,
    )
    from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear
    from rtp_llm.utils.model_weight import W

    torch.manual_seed(1919)
    case = _case("cuda", ratio, batch, span)
    attn = case.owner
    pages = (context + 127) // 128
    for region in (attn._global_region(), INDEXER_KV):
        table = torch.arange(1, batch * pages + 1, device="cuda").view(batch, pages)
        if region == INDEXER_KV:
            table = table.flip(1)
        attn._block_tables_by_type[region] = table.int().contiguous()
        entries, width = (128, 68) if region == INDEXER_KV else (128 // ratio, 288)
        pool = torch.randint(
            0,
            256,
            (batch * pages + 1, entries, width),
            dtype=torch.uint8,
            device="cuda",
        )
        if region == INDEXER_KV:
            pool.view(pool.shape[0], -1)[:, entries * 64 :].fill_(123)
        else:
            pool[..., 256:].fill_(40)
        attn._test_pools[region] = pool
    swa = torch.empty(batch + 1, 256, 528, device="cuda", dtype=torch.uint8)
    quantize_and_insert_swa_k_cache(
        torch.randn((batch + 1) * 256, 512, device="cuda", dtype=torch.bfloat16),
        swa,
        torch.arange((batch + 1) * 256, device="cuda"),
    )
    attn._test_pools[SWA_KV] = swa
    attn._rope_max_seq_len = context
    angles = torch.outer(
        torch.arange(context + 128, device="cuda").float(),
        torch.linspace(0.001, 0.1, 32, device="cuda"),
    )
    attn.freqs_cis = torch.polar(torch.ones_like(angles), angles)

    def linear(n, k):
        weight = torch.randint(-8, 9, (n, k), device="cuda").to(torch.float8_e4m3fn)
        scale = torch.full((n // 32, k // 32), 2**-8, device="cuda").to(
            torch.float8_e8m0fnu
        )
        return V41MXFP8Linear(weight, scale), scale

    attn.dim, attn.n_heads, attn.n_groups, attn.o_lora_rank = 5120, 64, 8, 1024
    attn.q_lora_rank, attn.index_topk, attn.tp_size, attn.window_size = (
        1280,
        512,
        1,
        128,
    )
    attn.skip_post_q_norm = attn.is_kv_source = attn.is_index_source = True
    attn.decode_sparse_mqa = True
    attn.v41_config = dict(
        candidate_source_layer_id=20, candidate_topk_blocks=2048, candidate_block_size=8
    )
    attn.q_norm = torch.ones(1280, device="cuda", dtype=torch.bfloat16)
    attn.kv_norm = torch.ones(512, device="cuda", dtype=torch.bfloat16)
    attn.wq_a_wkv, _ = linear(1792, 5120)
    attn.wq_b, _ = linear(32768, 1280)
    attn.index_wq, index_scale = linear(4096, 1280)
    attn.wo_b, _ = linear(5120, 8192)
    attn.index_weights = (
        torch.randn(32, 5120, device="cuda", dtype=torch.bfloat16) / 5120**0.5
    )
    attn._wo_a_stk_w = torch.randint(-8, 9, (8, 1024, 4096), device="cuda").to(
        torch.float8_e4m3fn
    )
    attn._wo_a_stk_s = torch.full(
        (8, 32, 1024), 0x77777777, dtype=torch.int32, device="cuda"
    ).transpose(1, 2)
    attn.attn_sink = torch.zeros(64, device="cuda")
    attn.softmax_scale = 512**-0.5
    attn._mega_attention = mega.prepare(attn)
    if attn._mega_attention is None:
        raise unittest.SkipTest("FlashMLA mega attention backend unavailable")
    from rtp_llm.models_py.modules.dsv41.fp8._v41_output_projection import (
        warmup_quantized_output,
    )

    case.wo_a_quant_ready = warmup_quantized_output(
        attn._mega_attention.output_weight, attn._mega_attention.output_scale
    )
    attn._mega_attention.merged_qb = MergedQB.create(
        attn,
        attn._mega_attention,
        {W.v4_indexer_wq_b_w: attn.index_wq.weight, W.v4_indexer_wq_b_s: index_scale},
    )
    assert attn._mega_attention.merged_qb is not None
    attn._shared_attention["layers"] = {attn.layer_id: attn}
    for method in (
        "_begin_forward",
        "_produce_global_decode",
        "_select_indices_decode",
        "_decode_global_slots",
        "_mega_position_chunks",
        "_forward_decode_body",
    ):
        setattr(attn, method, MethodType(getattr(AttentionV41FP8, method), attn))
    attn._pool_view_3d_fp8 = attn._source_pool
    attn._prefill_output_all_reduce = lambda _: None  # TP1 has no collective.
    case.starts.fill_(context - span - 2)
    case.positions.copy_(
        (case.starts[:, None] + torch.arange(span, device="cuda")).flatten()
    )
    writes = (case.req_ids + 1) * 256 + case.positions % 256
    swa_indices = (
        (case.positions[:, None] - torch.arange(127, -1, -1, device="cuda")) % 256
        + (case.req_ids[:, None] + 1) * 256
    ).int()
    case.metadata = SimpleNamespace(
        req_id_per_token=case.req_ids.int(),
        start_pos=case.starts,
        position_ids=case.positions,
        swa_global_slots=swa_indices,
        pool_write_slot_mappings={SWA_KV: writes},
    )
    return case


def run_full_attention(case, concurrent):
    with patch.dict(
        os.environ,
        {
            "DSV41_FULL_DECODE_MULTISTREAM": str(int(concurrent)),
            "DSV41_OVERLAP_GLOBAL_QKV": "0",
        },
    ):
        out = case.owner._forward_decode_body(case.x, case.metadata)
    shared = case.owner._shared_attention
    results = {"output": out, "tokens": shared["topk"][case.owner.layer_id]}
    if shared.get("candidates") is not None:
        results.update(
            candidates=shared["candidates"], flags=shared["candidate_mask"][1]
        )
    return results


def update_full_positions(case):
    from rtp_llm.models_py.modules.dsv41.attn_type import SWA_KV

    span = case.x.shape[1]
    case.positions.copy_(
        (case.starts[:, None] + torch.arange(span, device=case.x.device)).flatten()
    )
    case.metadata.pool_write_slot_mappings[SWA_KV].copy_(
        (case.req_ids + 1) * 256 + case.positions % 256
    )
    case.metadata.swa_global_slots.copy_(
        (case.positions[:, None] - torch.arange(127, -1, -1, device=case.x.device))
        % 256
        + (case.req_ids[:, None] + 1) * 256
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FullOverlapCudaTest(unittest.TestCase):
    @torch.inference_mode()
    def test_merged_projection_multistream_graph_refreshes_inputs_and_weights(self):
        from rtp_llm.models_py.modules.dsv41.fp8 import _v41_full_projection

        for ratio in (1, 2):
            case = full_attention_case(ratio, 3, 6, 32768)
            _v41_full_projection.prepare(case.owner)
            case.metadata.start_pos = case.starts.int()
            initial = {k: v.clone() for k, v in case.owner._test_pools.items()}
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run_full_attention(case, True)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    actual = run_full_attention(case, True)
                    observed = {k: v.clone() for k, v in case.owner._test_pools.items()}
            torch.cuda.current_stream().wait_stream(stream)
            for step in range(3):
                case.x.normal_()
                case.starts.sub_(1)
                case.metadata.start_pos.copy_(case.starts)
                update_full_positions(case)
                case.owner.global_wkv.mul_(0.5)
                case.owner.index_weights.mul_(0.5)
                _v41_full_projection.prepare(case.owner)
                for table in case.owner._block_tables_by_type.values():
                    table.copy_(table.roll(1, 1))
                for key, value in initial.items():
                    case.owner._test_pools[key].copy_(value)
                expected = run_full_attention(case, True)
                expected_pools = {
                    k: v.clone() for k, v in case.owner._test_pools.items()
                }
                for key, value in initial.items():
                    case.owner._test_pools[key].copy_(value)
                graph.replay()
                for key in expected:
                    if expected[key] is None:
                        self.assertIsNone(actual[key])
                    else:
                        torch.testing.assert_close(
                            actual[key], expected[key], rtol=0, atol=0
                        )
                for key in observed:
                    torch.testing.assert_close(
                        observed[key], expected_pools[key], rtol=0, atol=0
                    )

    def test_qb_pdl_dynamic_graph(self):
        import deep_gemm

        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("native global producer requires Blackwell")
        previous_pdl = deep_gemm.get_pdl()
        self.addCleanup(deep_gemm.set_pdl, previous_pdl)
        deep_gemm.set_pdl(True)
        with patch.dict(os.environ, {"DSV41_FULL_DECODE_QB_PDL": "1"}):
            self.test_real_full_graph_replay_and_changed_inputs()

    @torch.inference_mode()
    def test_real_full_graph_replay_and_changed_inputs(self):
        if torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("native global producer requires Blackwell")
        for ratio, batch, span in (
            (1, 3, 5),
            (2, 3, 5),
            (2, 1, 1),
            (2, 7, 6),
            (2, 1, 8),
        ):
            with self.subTest(ratio=ratio, batch=batch, span=span):
                case = full_attention_case(ratio, batch, span, 32768)
                initial = {k: v.clone() for k, v in case.owner._test_pools.items()}
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run_full_attention(case, True)
                    self.assertTrue(case.owner._full_decode_overlap_warmed)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = run_full_attention(case, True)
                        observed = {
                            k: v.clone() for k, v in case.owner._test_pools.items()
                        }
                torch.cuda.current_stream().wait_stream(stream)
                for iteration in range(3):
                    case.x.copy_(torch.randn_like(case.x))
                    case.starts.sub_(1)
                    update_full_positions(case)
                    # Change physical page mappings without changing graph pointers.
                    for table in case.owner._block_tables_by_type.values():
                        table.copy_(table.roll(1, 1))
                    for key, value in initial.items():
                        case.owner._test_pools[key].copy_(value)
                    expected = run_full_attention(case, False)
                    expected_pools = {
                        k: v.clone() for k, v in case.owner._test_pools.items()
                    }
                    for key, value in initial.items():
                        case.owner._test_pools[key].copy_(value)
                    graph.replay()
                    for key in expected:
                        # Complete sorted-prefix candidates deliberately omit
                        # the bitmap; both paths must preserve that contract.
                        if actual[key] is None or expected[key] is None:
                            self.assertIsNone(actual[key], key)
                            self.assertIsNone(expected[key], key)
                            continue
                        self.assertTrue(torch.isfinite(actual[key]).all())
                        # Both selectors permit tie ordering; comparing it exactly here
                        # also catches timing-sensitive nondeterminism in this fixture.
                        torch.testing.assert_close(
                            actual[key], expected[key], rtol=0, atol=0, msg=key
                        )
                    for key in observed:
                        torch.testing.assert_close(
                            observed[key], expected_pools[key], rtol=0, atol=0
                        )

    def test_unfused_metadata_and_legacy_topk_graph(self):
        with patch.dict(
            os.environ,
            {"DSV41_FUSE_INDEXER_METADATA": "0", "DSV41_DECODE_DEEPSELECT": "0"},
        ):
            self.test_real_full_graph_replay_and_changed_inputs()


if __name__ == "__main__":
    unittest.main()
