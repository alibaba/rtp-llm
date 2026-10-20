"""Production decode producer dispatch and original rounding/cache contracts."""

import os
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41._fused_rmsnorm_rope_triton import (
    fused_rmsnorm_rope,
)
from rtp_llm.models_py.modules.dsv41._v41_query_quant import query_norm_quant
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_indexer
from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
    quantize_and_insert_swa_k_cache,
)
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv41.fp8.decode import global_overlap
from rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv import (
    _apply_v41_q_rope,
    decode_compute_qkv,
)
from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_decode_global import (
    _assert_result,
    _candidate,
    _case,
    _clone_case,
    _reference,
    _snapshot,
)
from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear


def exact(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    torch.testing.assert_close(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DeepGEMMProducerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import deep_gemm

        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("DeepGEMM producers require SM10x")
        cls.api = deep_gemm
        torch.manual_seed(411021)

        def linear(n, k):
            weight = (torch.randn(n, k, device="cuda") / 32).to(torch.float8_e4m3fn)
            scales = torch.ones(n // 32, k // 32, device="cuda").to(
                torch.float8_e8m0fnu
            )
            return V41MXFP8Linear(weight, scales)

        angles = torch.randn(4096, 32, device="cuda")
        cls.attn = SimpleNamespace(
            wq_a_wkv=linear(1792, 5120),
            wq_b=linear(32768, 1280),
            q_norm=(1 + torch.randn(1280, device="cuda") / 8).bfloat16(),
            kv_norm=(1 + torch.randn(512, device="cuda") / 8).bfloat16(),
            freqs_cis=torch.polar(torch.ones_like(angles), angles),
            q_lora_rank=1280,
            rope_head_dim=64,
            n_heads=64,
            head_dim=512,
            skip_post_q_norm=True,
            eps=1e-6,
            _lin=lambda module, value: module(value),
            is_index_source=True,
            kv_source_layer_id=20,
            index_n_heads=32,
            index_head_dim=128,
            index_wq=linear(4096, 1280),
            index_weights=torch.randn(32, 5120, device="cuda").bfloat16(),
            _shared_attention={
                "global": {
                    20: _v41_decode_indexer.DecodeIndexerKeys(
                        torch.zeros(2, 128, 68, device="cuda", dtype=torch.uint8),
                        torch.ones(64, 1, device="cuda", dtype=torch.int32),
                        128,
                        128,
                    )
                }
            },
        )

    @torch.inference_mode()
    def test_decode_calls_native_qkv_and_preserves_original_outputs(self):
        attn = self.attn
        for batch, span in ((1, 1), (11, 6), (32, 6), (64, 6), (512, 1)):
            with self.subTest(batch=batch, span=span):
                rows = batch * span
                x = torch.randn(batch, span, 5120, device="cuda").bfloat16()
                positions = torch.arange(rows, device="cuda", dtype=torch.int32) * 7
                freqs = attn.freqs_cis[positions.long()].contiguous()
                slots = torch.arange(rows, device="cuda", dtype=torch.int64)
                slots[0] = -1
                cache = torch.full(
                    ((rows + 127) // 128 + 1, 128, 528),
                    173,
                    device="cuda",
                    dtype=torch.uint8,
                )
                expected_cache = cache.clone()
                projected = attn.wq_a_wkv(x)
                qr, pair = query_norm_quant(
                    projected[..., :1280], attn.q_norm, attn.eps
                )
                q = attn.wq_b.forward_quantized(*pair).unflatten(-1, (64, 512))
                _apply_v41_q_rope(q, freqs, 64)
                kv = fused_rmsnorm_rope(
                    projected[..., 1280:], attn.kv_norm, freqs, 64, eps=attn.eps
                )
                quantize_and_insert_swa_k_cache(
                    kv.reshape(rows, 512), expected_cache, slots
                )
                with patch.dict(
                    os.environ,
                    {"DSV4_FP8_QUANT_KERNEL": "auto", "DSV41_FUSED_QUERY_QUANT": "1"},
                ), patch.object(
                    self.api, "dsv41_qkv", wraps=self.api.dsv41_qkv
                ) as native:
                    actual = decode_compute_qkv(
                        attn,
                        x,
                        positions,
                        freqs,
                        swa_pool=cache,
                        swa_slots=slots,
                    )
                self.assertEqual(native.call_count, 1, "production native QKV dispatch")
                self.assertTrue(actual.swa_written)
                exact(self, actual.qr, qr)
                exact(self, actual.q, q)
                exact(self, actual.kv, kv)
                exact(self, actual.qr_quantized[0], pair[0])
                exact(self, actual.qr_quantized[1], pair[1])
                exact(self, cache, expected_cache)

    @torch.inference_mode()
    def test_indexer_query_uses_native_epilogue_without_bf16_projection(self):
        attn = self.attn
        for batch, span in ((1, 1), (64, 6)):
            with self.subTest(batch=batch, span=span):
                rows = batch * span
                x = torch.randn(batch, span, 5120, device="cuda").bfloat16()
                positions = torch.arange(rows, device="cuda", dtype=torch.int64) * 7
                qr, pair = query_norm_quant(
                    attn.wq_a_wkv(x)[..., :1280], attn.q_norm, attn.eps
                )
                q = attn.index_wq.forward_quantized(*pair).view(batch, span, 32, 128)
                weights = F.linear(x.reshape(rows, 5120), attn.index_weights)
                expected = _v41_decode_indexer.prepare_indexer_q_and_weights(
                    q,
                    weights.view(batch, span, 32),
                    attn.freqs_cis,
                    positions,
                    weights_out_dtype=torch.bfloat16,
                )
                with patch.object(
                    self.api,
                    "fp8_indexer_q_rope_fp4",
                    wraps=self.api.fp8_indexer_q_rope_fp4,
                ) as native:
                    raw, actual_weights, packed = (
                        AttentionV41FP8._prepare_indexer_query_decode(
                            attn, x, qr, positions, pair
                        )
                    )
                self.assertEqual(native.call_count, 1, "production Indexer Q dispatch")
                self.assertIsNone(
                    raw, "native epilogue must not store a BF16 projection"
                )
                exact(self, actual_weights, weights)
                for actual, wanted in zip(packed, expected):
                    exact(self, actual, wanted)

                from rtp_llm.models_py.modules.dsv41.fp8 import _v41_decode_topk as topk
                from rtp_llm.models_py.modules.dsv41.fp8 import (
                    _v41_sparse_decode as sparse,
                )
                from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
                    quantize_indexer_k_fp4,
                )

                capacity = 8192
                pool = torch.zeros(65, 128, 68, device="cuda", dtype=torch.uint8)
                keys = torch.randn(capacity, 128, device="cuda").bfloat16()
                slots = torch.arange(capacity, device="cuda", dtype=torch.int64) + 128
                quantize_indexer_k_fp4(keys, slots, pool)
                table = (
                    torch.arange(1, 65, device="cuda", dtype=torch.int32)
                    .expand(batch, -1)
                    .contiguous()
                )
                lengths = torch.full(
                    (batch, span), capacity, device="cuda", dtype=torch.int32
                )
                lengths[0, 0] = 513
                candidates = (
                    torch.arange(0, 1024, 2, device="cuda", dtype=torch.int32)
                    .expand(rows, -1)
                    .contiguous()
                )
                plan = sparse.make_plan(
                    candidates, lengths.flatten(), table, query_len=span
                )

                def select(prepared):
                    query = q if prepared is None else None
                    logits = _v41_decode_indexer.score_decode_indexer(
                        query,
                        weights.view(batch, span, 32),
                        attn.freqs_cis,
                        pool,
                        table,
                        lengths,
                        max_ctx_len=capacity,
                        positions=positions,
                        prepared_query=prepared,
                    )
                    self.assertIsNotNone(logits)
                    return (
                        logits,
                        topk.select_tokens(logits, lengths.flatten(), 512),
                        sparse.select_decode(
                            query,
                            weights.view(batch, span, 32),
                            attn.freqs_cis,
                            positions,
                            pool,
                            plan,
                            rope_head_dim=64,
                            prepared_query=prepared,
                        ),
                    )

                expected_selection = select(None)
                actual_selection = select(packed)
                for actual, wanted in zip(actual_selection, expected_selection):
                    exact(self, actual, wanted)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    select(packed)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        captured_selection = select(packed)
                torch.cuda.current_stream().wait_stream(stream)
                graph.replay()
                for actual, wanted in zip(captured_selection, expected_selection):
                    exact(self, actual, wanted)

    @torch.inference_mode()
    def test_indexer_k_native_preserves_c1_c2_caches_and_ring_state(self):
        for ratio in (1, 2):
            for batch in (1, 32, 64):
                with self.subTest(ratio=ratio, batch=batch):
                    case = _case("cuda", ratio, batch, 6)
                    case.starts.clamp_(max=500)
                    case.positions.copy_(
                        (
                            case.starts[:, None] + torch.arange(6, device="cuda")
                        ).flatten()
                    )
                    old = _clone_case(case)
                    initial = _snapshot(case)
                    _reference(old)
                    with patch.object(
                        self.api,
                        "bf16_indexer_k_rope_fp4",
                        wraps=self.api.bf16_indexer_k_rope_fp4,
                    ) as native:
                        _candidate(case)
                    self.assertEqual(
                        native.call_count, 1, "production Indexer K dispatch"
                    )
                    _assert_result(self, old, case, initial)

    @torch.inference_mode()
    def test_native_qkv_preserves_merged_query_projection(self):
        from rtp_llm.models_py.modules.dsv41.fp8._v41_merged_qb import MergedQB
        from rtp_llm.models_py.modules.dsv41.fp8.test.test_v41_merged_qb import fixture

        attn, mega, weights = fixture()
        mega.merged_qb = MergedQB.create(attn, mega, weights)
        for name, value in vars(self.attn).items():
            if not hasattr(attn, name):
                setattr(attn, name, value)
        x = torch.randn(64, 6, 5120, device="cuda").bfloat16()
        positions = torch.arange(384, device="cuda") % 256
        slots = torch.arange(384, device="cuda")
        cache = torch.full((4, 128, 528), 173, device="cuda", dtype=torch.uint8)

        def run():
            qkv = decode_compute_qkv(
                attn,
                x,
                positions,
                swa_pool=cache,
                swa_slots=slots,
                mega_attention=mega,
            )
            prepared = AttentionV41FP8._prepare_indexer_query_decode(
                attn, x, qkv.qr, positions, qkv.qr_quantized, qkv.index_q
            )
            return qkv, prepared

        with patch.object(self.api, "dsv41_qkv", None):
            expected, expected_query = run()
            expected_cache = cache.clone()
        cache.fill_(173)
        with patch.object(
            self.api, "dsv41_qkv", wraps=self.api.dsv41_qkv
        ) as native, patch.object(
            self.api,
            "fp8_indexer_q_rope_fp4",
            wraps=self.api.fp8_indexer_q_rope_fp4,
        ) as separate:
            actual, actual_query = run()
        self.assertEqual(native.call_count, 1)
        self.assertEqual(
            separate.call_count, 0, "merged Q-B already produced Indexer Q"
        )
        for name in ("q", "qr", "kv", "index_q"):
            exact(self, getattr(actual, name), getattr(expected, name))
        for got, wanted in zip(actual_query[2], expected_query[2]):
            exact(self, got, wanted)
        exact(self, cache, expected_cache)

    @torch.inference_mode()
    def test_global_production_entry_falls_back_on_other_architecture(self):
        from rtp_llm.models_py.modules.dsv41.attn_type import INDEXER_KV
        from rtp_llm.models_py.modules.dsv41.fp8._v41_decode_global import (
            try_produce_global,
        )

        case = _case("cuda", 1, 1, 1)
        cache = case.owner._source_pool(INDEXER_KV)
        original = cache.clone()
        with patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ), patch.object(
            self.api,
            "bf16_indexer_k_rope_fp4",
            wraps=self.api.bf16_indexer_k_rope_fp4,
        ) as native:
            self.assertFalse(
                try_produce_global(
                    case.owner, case.x, case.positions, case.req_ids, case.starts
                )
            )
        self.assertEqual(native.call_count, 0)
        exact(self, cache, original)

    @torch.inference_mode()
    def test_native_wk_matches_flashinfer_backend_at_fp4_boundary(self):
        import flashinfer

        from rtp_llm.models_py.modules.dsv41.fp8._v41_decode_global import (
            _try_native_indexer_k,
        )
        from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
            quantize_indexer_k_fp4,
        )
        from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import rms_norm

        bits = [
            0x4129,
            0x41B8,
            0x438E,
            0x402C,
            0x3B78,
            0x4108,
            0x4251,
            0x449A,
            0x446D,
            0x3AF2,
            0x44F8,
            0x41F7,
            0x404A,
            0x3F5A,
            0x4131,
            0x4266,
        ]
        weight = torch.zeros(128, 512, device="cuda", dtype=torch.bfloat16)
        weight[:, :128] = torch.eye(128, device="cuda", dtype=torch.bfloat16)
        gamma = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
        gamma[0], gamma[16] = 85.0, 0.1982421875
        owner = SimpleNamespace(
            index_wk=weight,
            index_k_norm=gamma,
            eps=1e-6,
            freqs_cis=torch.ones(1, 32, device="cuda", dtype=torch.complex64),
        )
        # FlashInfer initializes its backend at import time. Exercise the CUDA
        # and CuTe variants in separate processes via FLASHINFER_USE_CUDA_NORM.
        for rows in (1, 17, 384):
            with self.subTest(cuda_norm=flashinfer.norm._USE_CUDA_NORM, rows=rows):
                x = torch.zeros(rows, 512, device="cuda", dtype=torch.bfloat16)
                x[:, :128:8] = torch.tensor(
                    bits, device="cuda", dtype=torch.int16
                ).view(torch.bfloat16)
                positions = torch.zeros(rows, device="cuda", dtype=torch.int64)
                slots = torch.arange(rows, device="cuda", dtype=torch.int64)
                actual = torch.full(
                    (rows, 1, 68), 173, device="cuda", dtype=torch.uint8
                )
                expected = actual.clone()
                normalized = rms_norm(F.linear(x, weight), gamma, owner.eps)
                quantize_indexer_k_fp4(normalized, slots, expected)
                self.assertTrue(
                    _try_native_indexer_k(owner, x, positions, slots, actual)
                )
                exact(self, actual, expected)

    def test_new_host_thread_does_not_inherit_native_warmup(self):
        import threading

        from rtp_llm.models_py.modules.dsv41 import _v41_query_quant as quant

        x = torch.randn(1, 1, 5120, device="cuda").bfloat16()
        freqs = self.attn.freqs_cis[:1].contiguous()
        cache = torch.zeros(1, 128, 528, device="cuda", dtype=torch.uint8)
        slots = torch.zeros(1, device="cuda", dtype=torch.int64)
        torch.cuda.synchronize()
        errors = []

        def worker(capture):
            try:
                with torch.cuda.device(x.device):
                    stream = torch.cuda.Stream()
                    with torch.cuda.stream(stream):
                        if capture:
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph, stream=stream):
                                self.assertIsNone(
                                    quant.try_project_qkv_to_cache(
                                        self.attn, x, freqs, cache, slots
                                    )
                                )
                        else:
                            self.assertIsNotNone(
                                quant.try_project_qkv_to_cache(
                                    self.attn, x, freqs, cache, slots
                                )
                            )
                    stream.synchronize()
            except BaseException as error:
                errors.append(error)

        # Simulate recycled native thread IDs only at the adapter boundary.
        with patch.object(quant, "threading", SimpleNamespace(get_ident=lambda: 7)):
            for capture in (False, True):
                thread = threading.Thread(target=worker, args=(capture,))
                thread.start()
                thread.join(90)
                self.assertFalse(thread.is_alive(), "worker timed out")
        if errors:
            raise errors[0]

    def _prepare_chain(self, case):
        owner = case.owner
        for name in (
            "wq_a_wkv",
            "wq_b",
            "q_norm",
            "kv_norm",
            "q_lora_rank",
            "n_heads",
            "skip_post_q_norm",
            "index_wq",
            "index_weights",
            "_lin",
        ):
            setattr(owner, name, getattr(self.attn, name))
        owner.is_kv_source = owner.is_index_source = True
        owner._produce_global_decode = MethodType(
            AttentionV41FP8._produce_global_decode, owner
        )
        rows = case.x.shape[0] * case.x.shape[1]
        case.swa = torch.full(
            ((rows + 127) // 128 + 1, 128, 528),
            173,
            device="cuda",
            dtype=torch.uint8,
        )
        case.swa_slots = torch.arange(rows, device="cuda", dtype=torch.int64)

    def _chain(self, case, concurrent):
        owner = case.owner
        work = (
            global_overlap.start_global_decode(
                owner, case.x, case.positions, case.req_ids, case.starts
            )
            if concurrent
            else None
        )
        if concurrent:
            self.assertIsNotNone(work, "complete-chain overlap dispatch")
        try:
            qkv = decode_compute_qkv(
                owner,
                case.x,
                case.positions,
                owner.freqs_cis[case.positions].contiguous(),
                swa_pool=case.swa,
                swa_slots=case.swa_slots,
            )
            if not concurrent:
                owner._produce_global_decode(
                    case.x, case.positions, case.req_ids, case.starts
                )
            _, _, packed = AttentionV41FP8._prepare_indexer_query_decode(
                owner, case.x, qkv.qr, case.positions, qkv.qr_quantized, qkv.index_q
            )
        finally:
            if work is not None:
                work.finish()
        # Read auxiliary outputs on the calling stream immediately after join.
        return (
            qkv.q,
            qkv.qr,
            qkv.kv,
            *qkv.qr_quantized,
            *packed,
            case.swa.clone(),
            *(pool.clone() for pool in owner._test_pools.values()),
        )

    @torch.inference_mode()
    def test_default_complete_qkv_global_chain_graph_with_changed_inputs(self):
        env = {
            "DSV41_FUSED_DECODE_GLOBAL": "1",
            "DSV41_FUSED_DECODE_INDEXER": "1",
            "DSV41_FUSED_QUERY_QUANT": "1",
            "DSV4_FP8_QUANT_KERNEL": "auto",
            "MOEDBG": "0",
        }
        with patch.dict(os.environ, env):
            os.environ.pop("DSV41_OVERLAP_GLOBAL_QKV", None)
            for ratio in (1, 2):
                with self.subTest(ratio=ratio):
                    reference = _case("cuda", ratio, 64, 6)
                    reference.starts.clamp_(max=500)
                    reference.positions.copy_(
                        (
                            reference.starts[:, None] + torch.arange(6, device="cuda")
                        ).flatten()
                    )
                    candidate = _clone_case(reference)
                    initial = _snapshot(reference)
                    for case in (reference, candidate):
                        self._prepare_chain(case)
                    with patch.dict(os.environ, {"DSV41_OVERLAP_GLOBAL_QKV": "0"}):
                        self.assertIsNone(
                            global_overlap.start_global_decode(
                                candidate.owner,
                                candidate.x,
                                candidate.positions,
                                candidate.req_ids,
                                candidate.starts,
                            )
                        )
                    current = torch.cuda.current_stream()
                    stream = torch.cuda.Stream()
                    stream.wait_stream(current)
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            self._chain(candidate, True)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            actual = self._chain(candidate, True)
                    current.wait_stream(stream)
                    for delta in (0, 1, 3):
                        candidate.x.copy_(torch.randn_like(candidate.x))
                        reference.x.copy_(candidate.x)
                        for case in (reference, candidate):
                            case.starts.fill_(127 + delta)
                            case.positions.copy_(
                                (
                                    case.starts[:, None]
                                    + torch.arange(6, device="cuda")
                                ).flatten()
                            )
                            case.swa.fill_(173)
                            for region, pool in case.owner._test_pools.items():
                                pool.copy_(initial[region])
                        expected = self._chain(reference, False)
                        graph.replay()
                        for got, wanted in zip(actual, expected):
                            exact(self, got, wanted)


if __name__ == "__main__":
    unittest.main()
