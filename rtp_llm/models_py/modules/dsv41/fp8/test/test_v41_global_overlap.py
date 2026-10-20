"""V4.1 global/QKV stream ordering, graph lifetime, and native pool writes."""

import contextlib
import os
import threading
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import AttentionV41FP8
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
                    kv=x, qr=x, qr_quantized=None, index_q=None, swa_written=False
                )

                def project(*args, **kwargs):
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
                with patch.object(
                    overlap, "start_global_decode", return_value=work
                ), patch(
                    "rtp_llm.models_py.modules.dsv41.fp8.decode.compute_qkv.decode_compute_qkv",
                    side_effect=project,
                ):
                    with self.assertRaisesRegex(RuntimeError, "qkv stop|indexer stop"):
                        AttentionV41FP8._forward_decode_body(attn, x, metadata)
                self.assertEqual(
                    calls,
                    (
                        ["qkv", "join"]
                        if fail_qkv
                        else ["qkv", "swa", "index_query", "join", "indexer"]
                    ),
                )
                attn._produce_global_decode.assert_not_called()


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


if __name__ == "__main__":
    unittest.main()
