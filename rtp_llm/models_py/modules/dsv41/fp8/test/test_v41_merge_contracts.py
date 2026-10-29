"""CPU contracts at the upstream/PR query, producer and stream merge seams."""

import contextlib
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41 import _v41_query_quant
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer as indexer
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_front_gemm as front
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_qb_kv as qb_kv
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_qkv_rope_cache_triton as cache
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv41.fp8.decode import (
    compute_qkv,
    global_overlap,
    ready_qkv,
)


class MergeContractsCPU(unittest.TestCase):
    def test_projection_apis_use_runtime_deepgemm(self):
        import deep_gemm

        self.assertIs(qb_kv._ops(), deep_gemm)

    def test_prepared_reindex_query_reuses_payload_and_head_weights(self):
        x = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
        qr = torch.zeros(1, 2, 16, dtype=torch.bfloat16)
        weights = torch.ones(2, 32, dtype=torch.bfloat16)
        prepared = (
            torch.zeros(1, 2, 32, 64, dtype=torch.int8),
            torch.zeros(1, 2, 32, dtype=torch.int32),
            torch.ones(1, 2, 32, dtype=torch.bfloat16),
        )
        keys = indexer.DecodeIndexerKeys(
            torch.zeros(2, 128, 68, dtype=torch.uint8),
            torch.ones(1, 1, dtype=torch.int32),
            128,
            128,
        )
        attn = SimpleNamespace(
            is_index_source=True,
            kv_source_layer_id=20,
            index_n_heads=32,
            index_head_dim=128,
            rope_head_dim=64,
            index_weights=torch.ones(32, 8, dtype=torch.bfloat16),
            _shared_attention={"global": {20: keys}},
            _lin=Mock(side_effect=AssertionError("query reprojected")),
        )
        with patch.object(indexer, "is_supported", return_value=True), patch.object(
            _v41_query_quant, "try_project_indexer_q", side_effect=AssertionError
        ), patch.object(
            indexer, "prepare_indexer_q_and_weights", side_effect=AssertionError
        ):
            q, actual_weights, actual = AttentionV41FP8._prepare_indexer_query_decode(
                attn,
                x,
                qr,
                torch.zeros(2, dtype=torch.int64),
                head_weights=weights,
                index_prepared=prepared,
            )
        self.assertIsNone(q)
        self.assertIs(actual_weights, weights)
        self.assertIs(actual, prepared)
        attn._lin.assert_not_called()

        # Unsupported/nonpaged consumers still get the original BF16 query.
        attn._shared_attention["global"][20] = torch.zeros(1, 128, 128)
        attn.index_wq = object()
        bf16_q = torch.zeros(2, 4096, dtype=torch.bfloat16)
        attn._lin = Mock(return_value=bf16_q)
        q, actual_weights, actual = AttentionV41FP8._prepare_indexer_query_decode(
            attn,
            x,
            qr,
            torch.zeros(2, dtype=torch.int64),
            head_weights=weights,
            index_prepared=prepared,
        )
        self.assertEqual(q.shape, (2, 32, 128))
        self.assertIs(actual_weights, weights)
        self.assertIsNone(actual)
        attn._lin.assert_called_once()

    def test_native_cache_and_raw_kv_producers_are_mutually_exclusive(self):
        x = torch.zeros(1, 2, 8, dtype=torch.bfloat16)
        qr = torch.zeros(1, 2, 128, dtype=torch.bfloat16)
        raw_kv = torch.ones(1, 2, 512, dtype=torch.bfloat16)
        cached_kv = raw_kv + 10
        quantized = (torch.zeros_like(qr), torch.ones(2, 1, dtype=torch.int32))
        main_q = torch.zeros(1, 2, 64 * 512, dtype=torch.bfloat16)
        index_q, prepared = object(), (object(), object(), object())
        weights = torch.ones(2, 32, dtype=torch.bfloat16)
        positions = torch.zeros(2, dtype=torch.int64)
        freqs = torch.ones(2, 32, dtype=torch.complex64)
        slots = torch.ones(2, 128, dtype=torch.int32)
        for route in (
            "native",
            "native_fallback",
            "reindex",
            "pdl",
            "ready",
            "ready_fused",
            "qb_kv",
        ):
            with self.subTest(route=route), contextlib.ExitStack() as stack:
                stack.enter_context(
                    patch.dict(
                        os.environ,
                        {
                            "DSV41_FUSED_QB_KV": str(int(route == "qb_kv")),
                        },
                    )
                )
                attn = SimpleNamespace(
                    is_index_source=True,
                    is_kv_source=route != "reindex",
                    rope_head_dim=64,
                    n_heads=64,
                    head_dim=512,
                    skip_post_q_norm=True,
                    freqs_cis=freqs,
                    wq_b=object(),
                    kv_norm=torch.ones(512),
                    eps=1e-6,
                    _lin=Mock(side_effect=AssertionError("unexpected projection")),
                )
                merged = SimpleNamespace(
                    try_forward_prepared=Mock(return_value=(main_q, prepared)),
                    forward_quantized=Mock(return_value=(main_q, index_q)),
                )
                mega = SimpleNamespace(merged_qb=merged, padded_heads=64)
                native = stack.enter_context(
                    patch.object(
                        _v41_query_quant,
                        "try_project_qkv_to_cache",
                        return_value=(
                            None
                            if route == "native_fallback"
                            else (qr, cached_kv, quantized)
                        ),
                    )
                )
                original = stack.enter_context(
                    patch.object(
                        _v41_query_quant,
                        "try_project_quantized_qkv",
                        return_value=(qr, raw_kv, quantized),
                    )
                )
                fused_front = stack.enter_context(
                    patch.object(
                        front,
                        "project",
                        return_value=(qr, raw_kv, quantized, weights),
                    )
                )
                ready = stack.enter_context(
                    patch.object(
                        qb_kv,
                        "try_project_kv_index",
                        return_value=(
                            (main_q, cached_kv, slots)
                            if route == "ready_fused"
                            else None
                        ),
                    )
                )
                fused_qb = stack.enter_context(
                    patch.object(qb_kv, "try_project_kv", return_value=None)
                )
                rope = stack.enter_context(
                    patch.object(
                        cache,
                        "try_fused_qkv_rope_cache",
                        return_value=cache.FusedQKVRopeCache(
                            main_q.unflatten(-1, (64, 512)),
                            cached_kv,
                            freqs,
                        ),
                    )
                )
                stack.enter_context(
                    patch.object(
                        compute_qkv, "swa_kv_norm_rope", side_effect=AssertionError
                    )
                )
                result = compute_qkv.decode_compute_qkv(
                    attn,
                    x,
                    positions,
                    freqs_cis=freqs,
                    mega_attention=mega,
                    qb_pdl=route == "pdl",
                    ready_plan=object() if route in ("ready", "ready_fused") else None,
                    swa_pool=object(),
                    swa_slots=object(),
                )
                self.assertTrue(result.swa_written)
                self.assertIs(result.kv, cached_kv)
                if route == "native":
                    native.assert_called_once()
                    original.assert_not_called()
                    fused_front.assert_not_called()
                    rope.assert_not_called()
                else:
                    if route == "native_fallback":
                        native.assert_called_once()
                    else:
                        native.assert_not_called()
                    if route == "reindex":
                        fused_front.assert_called_once()
                        original.assert_not_called()
                        self.assertIs(result.index_prepared, prepared)
                    else:
                        original.assert_called_once()
                    if route == "ready_fused":
                        rope.assert_not_called()
                        self.assertIs(result.global_slots, slots)
                        self.assertEqual(result.q.shape, (1, 2, 64, 512))
                        merged.forward_quantized.assert_not_called()
                    else:
                        self.assertIs(rope.call_args.args[1], raw_kv)
                if route in ("ready", "ready_fused"):
                    ready.assert_called_once()
                    self.assertIs(ready.call_args.args[3], raw_kv)
                    self.assertEqual(
                        ready.call_args.kwargs, {"eps": attn.eps, "wait_pdl": False}
                    )
                if route == "qb_kv":
                    fused_qb.assert_called_once()
                    self.assertIs(fused_qb.call_args.args[3], raw_kv)
                if route == "pdl":
                    self.assertTrue(original.call_args.kwargs["pdl"])
                    self.assertTrue(
                        merged.forward_quantized.call_args.kwargs["wait_pdl"]
                    )

    def test_ready_plan_requires_only_the_qkv_index_producer(self):
        rows = 128
        req = torch.zeros(rows, dtype=torch.int32)
        table = torch.ones(1, 2, dtype=torch.int32)
        local = torch.zeros(rows, 128, dtype=torch.int32)
        pool = torch.empty(3, 64, 288, dtype=torch.uint8)
        attn = SimpleNamespace(
            compress_ratio=8,
            is_kv_source=False,
            is_index_source=False,
            _cp_ctx=None,
            index_source_layer_id=20,
            _shared_attention={"topk": {20: local}},
            _global_region=lambda: "global",
            _source_pool=lambda _: pool,
            _block_tables_by_type={"global": table},
            _kv_cache=object(),
        )
        mega = SimpleNamespace(
            heads=64,
            padded_heads=64,
            groups=8,
            merged_qb=None,
            query_weight=torch.empty(0),
            supports_decode=Mock(return_value=True),
        )
        ops = SimpleNamespace(
            dsv41_qb_out=Mock(),
            dsv41_qb_kv_index_out=Mock(),
        )
        with patch.dict(
            os.environ, {"DSV41_FUSED_READY_QKV": "1"}, clear=True
        ), patch.object(qb_kv, "_device_supported", return_value=True), patch.object(
            qb_kv, "_ops", return_value=ops
        ), patch(
            "rtp_llm.models_py.modules.dsv41.fp8._kv_cache_utils.require_pool_tokens_per_block",
            return_value=512,
        ):
            plan = ready_qkv.make_ready_plan(
                attn,
                mega,
                SimpleNamespace(req_id_per_token=req),
                rows,
                object(),
                object(),
            )
        self.assertIsNotNone(plan)
        self.assertIs(plan.table, table)
        self.assertIs(plan.local, local)
        self.assertEqual((plan.logical_entries, plan.entries), (64, 64))
        ops.dsv41_qb_kv_index_out.assert_not_called()

    def test_ready_producer_binding_has_no_outgoing_pdl_argument(self):
        rows = 128
        data = torch.empty(rows, 1280, dtype=torch.float8_e4m3fn)
        raw = torch.empty(rows, 512, dtype=torch.bfloat16)
        positions = torch.zeros(rows, dtype=torch.int32)
        pool = torch.empty(3, 64, 528, dtype=torch.uint8)
        plan = ready_qkv.ReadyIndexPlan(
            torch.zeros(rows, dtype=torch.int32),
            torch.ones(1, 2, dtype=torch.int32),
            torch.zeros(rows, 128, dtype=torch.int32),
            64,
            64,
        )
        mega = SimpleNamespace(
            query_weight=torch.empty(32768, 1280, dtype=torch.float8_e4m3fn),
            query_scale=torch.empty(10, 32768, dtype=torch.int32).T,
        )
        ops = SimpleNamespace(dsv41_qb_kv_index_out=Mock())
        with patch.dict(
            os.environ, {"DSV41_FUSED_READY_QKV": "1"}, clear=True
        ), patch.object(qb_kv, "_config", return_value=(0, 64)), patch.object(
            qb_kv, "_ops", return_value=ops
        ):
            q, kv, slots = qb_kv.try_project_kv_index(
                mega,
                data,
                torch.empty(10, rows, dtype=torch.int32).T,
                raw,
                torch.ones(512, dtype=torch.bfloat16),
                positions,
                torch.ones(1, 32, dtype=torch.complex64),
                pool,
                positions,
                plan,
                eps=1e-6,
            )
        args = ops.dsv41_qb_kv_index_out.call_args.args
        self.assertEqual(len(args), 23)
        self.assertFalse(args[15])  # No incoming PDL on the ordinary REUSE path.
        self.assertIs(args[17], plan.req)
        self.assertIs(args[19], plan.local)
        self.assertIs(args[20], slots)
        self.assertEqual(args[21:], (64, 64))
        self.assertEqual(q.shape, (rows, 32768))
        self.assertEqual(kv.shape, (rows, 512))

    def test_compute_stream_cache_isolates_context_host_stream_and_thread(self):
        local = threading.local()
        current = SimpleNamespace(cuda_stream=51)
        with patch(
            "rtp_llm.models_py.modules.dsv41.fp8.attention._cuda_device_index",
            return_value=0,
        ), patch(
            "rtp_llm.models_py.modules.dsv41.fp8._v41_wo_a_quant._context",
            return_value=17,
        ) as context, patch(
            "torch.cuda.Stream", side_effect=lambda **_: object()
        ) as allocate:
            get = lambda capturing: global_overlap._get_compute_stream(
                local, torch.device("cuda", 0), current, capturing=capturing
            )
            self.assertIsNone(get(True))
            allocate.assert_not_called()
            main = get(False)
            self.assertIs(get(True), main)
            current.cuda_stream = 52
            self.assertIsNone(get(True))
            current.cuda_stream = 51
            context.return_value = 18
            self.assertIsNone(get(True))
            context.return_value = 17
            values = []

            def worker():
                values.extend((get(True), get(False)))

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
            self.assertIsNone(values[0])
            self.assertIsNot(values[1], main)
            self.assertIs(get(True), main)


if __name__ == "__main__":
    unittest.main()
