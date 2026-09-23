"""V4.1 global/QKV stream ordering, graph lifetime, and native pool writes."""

import contextlib
import os
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.fp8.decode import global_overlap as overlap
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_decode_global import (
    _case,
    _clone_case,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_qkv_rope_cache import (
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
        self.auxiliary = Mock(name="auxiliary")
        self.auxiliary.cuda_stream = 917
        self.attn = SimpleNamespace(_produce_global_decode=Mock())
        self.x = torch.empty(4, 6, 8)
        self.metadata = torch.arange(24)
        stack = contextlib.ExitStack()
        stack.enter_context(patch.object(overlap, "is_supported", return_value=True))
        stack.enter_context(
            patch.object(overlap, "_GLOBAL_COMPUTE_STREAMS", {0: self.auxiliary})
        )
        stack.enter_context(
            patch(
                "rtp_llm.models_py.modules.dsv4.fp8.attention._cuda_device_index",
                return_value=0,
            )
        )
        stack.enter_context(
            patch("torch.cuda.current_stream", return_value=self.current)
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

    def test_attention_body_joins_before_indexer_and_after_qkv_failure(self):
        from rtp_llm.models_py.modules.dsv4.attn_type import SWA_KV

        x = torch.empty(1, 1, 5120)
        metadata = SimpleNamespace(
            position_ids=torch.zeros(1, dtype=torch.int32),
            req_id_per_token=torch.zeros(1, dtype=torch.int32),
            start_pos=torch.zeros(1, dtype=torch.int32),
            pool_write_slot_mappings={SWA_KV: torch.zeros(1, dtype=torch.int64)},
        )
        for fail_qkv in (False, True):
            with self.subTest(fail_qkv=fail_qkv):
                calls = []
                work = SimpleNamespace(finish=lambda: calls.append("join"))
                qkv = SimpleNamespace(kv=x, qr=x, qr_quantized=None, swa_written=False)

                def project(*args, **kwargs):
                    calls.append("qkv")
                    if fail_qkv:
                        raise RuntimeError("qkv stop")
                    return qkv

                def select(*args):
                    calls.append("indexer")
                    raise RuntimeError("indexer stop")

                attn = SimpleNamespace(
                    _begin_forward=lambda: None,
                    _shared_attention={},
                    freqs_cis=torch.zeros(1, 32, dtype=torch.complex64),
                    compress_ratio=1,
                    is_kv_source=True,
                    _pool_view_3d_fp8=lambda _: None,
                    _decode_write_swa_fp8=lambda *args: calls.append("swa"),
                    _select_indices_decode=select,
                    _produce_global_decode=Mock(),
                )
                with patch.object(
                    overlap, "start_global_decode", return_value=work
                ), patch(
                    "rtp_llm.models_py.modules.dsv4.fp8.decode.compute_qkv.decode_compute_qkv",
                    side_effect=project,
                ):
                    with self.assertRaisesRegex(RuntimeError, "qkv stop|indexer stop"):
                        AttentionV41FP8._forward_decode_body(attn, x, metadata)
                self.assertEqual(
                    calls,
                    ["qkv", "join"] if fail_qkv else ["qkv", "swa", "join", "indexer"],
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
