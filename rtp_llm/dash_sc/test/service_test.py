"""Unit tests for ``rtp_llm.dash_sc.service``.

Covers:
- ``iter_real_model_stream_infer``: success, empty-list fallback, exception propagation.
- ``DashScGrpcInferenceServicer.ModelStreamInfer``: fake mode, real mode, missing input_ids,
  request_id snowflake scheme alignment with HTTP ``generate_request_id``.
- ``_iter_enqueue_sync`` cancel/timeout propagation to the backend pump.
"""

from __future__ import annotations

import asyncio
import struct
import threading
import time
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.dash_sc import service as bg_svc
from rtp_llm.dash_sc.codec import OtherParams, SamplingParams
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.dash_sc.service import (
    DashScGrpcInferenceServicer,
    _iter_enqueue_sync,
    iter_real_model_stream_infer,
    stream_log_tag,
)
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


def _add_input_tensor(
    req: predict_v2_pb2.ModelInferRequest,
    name: str,
    datatype: str,
    shape: list[int],
    raw: bytes,
) -> None:
    inp = req.inputs.add()
    inp.name = name
    inp.datatype = datatype
    inp.shape[:] = shape
    req.raw_input_contents.append(raw)


def _unpack_int32_le(raw: bytes) -> list[int]:
    return list(struct.unpack("<%di" % (len(raw) // 4), raw))


class IterRealModelStreamInferTest(TestCase):
    def _minimal_request(self) -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = "trace-real"
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [2], struct.pack("<2i", 1, 2))
        return req

    def test_yields_one_chunk_from_mock_enqueue(self) -> None:
        req = self._minimal_request()
        sampling = SamplingParams()
        other = OtherParams()

        def fake_sync(_visitor, _gi):
            out = GenerateOutput(
                output_ids=torch.tensor([3, 4], dtype=torch.int32),
                finished=True,
                aux_info=AuxInfo(input_len=2, reuse_len=0),
            )
            return [GenerateOutputs(generate_outputs=[out])]

        chunks = list(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                sampling,
                other,
                MagicMock(),
                rtp_llm_request_id=1,
                run_enqueue_sync=fake_sync,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertFalse(chunks[0].error_message)
        infer = chunks[0].infer_response
        self.assertEqual(infer.id, "trace-real")
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [3, 4])

    def test_empty_list_yields_error_response(self) -> None:
        req = self._minimal_request()

        def empty_sync(_v, _g):
            return []

        chunks = list(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                OtherParams(),
                MagicMock(),
                rtp_llm_request_id=1,
                run_enqueue_sync=empty_sync,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("empty outputs_list", chunks[0].error_message)

    def test_enqueue_exception_yields_error_message(self) -> None:
        req = self._minimal_request()

        def boom(_v, _g):
            raise RuntimeError("backend down")

        chunks = list(
            iter_real_model_stream_infer(
                req,
                [1, 2],
                SamplingParams(),
                OtherParams(),
                MagicMock(),
                rtp_llm_request_id=1,
                run_enqueue_sync=boom,
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn("backend down", chunks[0].error_message)


class DashScGrpcInferenceServicerTest(TestCase):
    def _valid_infer_request(self) -> predict_v2_pb2.ModelInferRequest:
        req = predict_v2_pb2.ModelInferRequest()
        req.id = "srv-1"
        req.model_name = "default"
        _add_input_tensor(req, "input_ids", "INT32", [1], struct.pack("<i", 42))
        return req

    def test_fake_mode_returns_incremented_ids(self) -> None:
        servicer = DashScGrpcInferenceServicer(backend_visitor=None)
        req = self._valid_infer_request()
        responses = list(servicer.ModelStreamInfer(iter([req]), MagicMock()))
        self.assertEqual(len(responses), 1)
        infer = responses[0].infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [142])

    def test_missing_input_ids_error(self) -> None:
        servicer = DashScGrpcInferenceServicer(backend_visitor=None)
        bad = predict_v2_pb2.ModelInferRequest()
        bad.id = "x"
        bad.model_name = "m"
        responses = list(servicer.ModelStreamInfer(iter([bad]), MagicMock()))
        self.assertEqual(len(responses), 1)
        self.assertIn("input_ids", responses[0].error_message)

    @patch.object(bg_svc, "_iter_enqueue_sync")
    def test_real_mode_uses_enqueue(self, mock_iter: MagicMock) -> None:
        out = GenerateOutput(
            output_ids=torch.tensor([9], dtype=torch.int32),
            finished=True,
            aux_info=AuxInfo(input_len=1, reuse_len=0),
        )
        mock_iter.side_effect = lambda *a, **k: iter(
            [GenerateOutputs(generate_outputs=[out])]
        )

        servicer = DashScGrpcInferenceServicer(backend_visitor=MagicMock())
        req = self._valid_infer_request()
        responses = list(servicer.ModelStreamInfer(iter([req]), MagicMock()))
        self.assertEqual(len(responses), 1)
        mock_iter.assert_called_once()
        infer = responses[0].infer_response
        by_name = {
            infer.outputs[i].name: infer.raw_output_contents[i]
            for i in range(len(infer.outputs))
        }
        self.assertEqual(_unpack_int32_le(by_name["generated_ids"]), [9])

    def test_real_mode_passes_context_to_enqueue_pump(self) -> None:
        """Servicer must bind grpc ``ServicerContext`` into ``_iter_enqueue_sync`` via partial."""
        captured: dict = {}

        def fake_iter(visitor, generate_input, *, context=None):
            captured["context"] = context
            out = GenerateOutput(
                output_ids=torch.tensor([1], dtype=torch.int32),
                finished=True,
                aux_info=AuxInfo(input_len=1, reuse_len=0),
            )
            yield GenerateOutputs(generate_outputs=[out])

        ctx = MagicMock()
        servicer = DashScGrpcInferenceServicer(backend_visitor=MagicMock())
        with patch.object(bg_svc, "_iter_enqueue_sync", side_effect=fake_iter):
            list(servicer.ModelStreamInfer(iter([self._valid_infer_request()]), ctx))
        self.assertIs(captured["context"], ctx)

    def test_real_mode_request_id_matches_generate_request_id(self) -> None:
        """Backend ``GenerateInput.request_id`` follows the same snowflake scheme as HTTP path."""
        from rtp_llm.frontend import request_id_generator as rig

        captured: list[int] = []

        def _capture(_visitor, gi, *, context=None):
            captured.append(gi.request_id)
            return []

        servicer = DashScGrpcInferenceServicer(
            backend_visitor=MagicMock(),
            ip="10.0.0.1",
            port=12345,
            server_id="srv-xyz",
        )
        with patch.object(rig.time, "time", return_value=1_700_000_000.0), patch.object(
            bg_svc, "_iter_enqueue_sync", side_effect=_capture
        ):
            list(servicer.ModelStreamInfer(iter([self._valid_infer_request()]), MagicMock()))
            expected = rig.generate_request_id("10.0.0.1", 12345, "srv-xyz", 1)

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0], expected)


class _FakeAsyncStream:
    """Simple async iterator of pre-built chunks, with optional per-item delay."""

    def __init__(self, chunks, delay: float = 0.0, raise_after: int | None = None):
        self._chunks = list(chunks)
        self._delay = delay
        self._raise_after = raise_after
        self._emitted = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._raise_after is not None and self._emitted >= self._raise_after:
            raise RuntimeError("backend stream error")
        if self._emitted >= len(self._chunks):
            raise StopAsyncIteration
        if self._delay:
            await asyncio.sleep(self._delay)
        item = self._chunks[self._emitted]
        self._emitted += 1
        return item


class _FakeVisitor:
    """Async ``enqueue`` that returns a ``_FakeAsyncStream``."""

    def __init__(self, stream: _FakeAsyncStream):
        self._stream = stream
        self.enqueue_called = 0

    async def enqueue(self, _generate_input):
        self.enqueue_called += 1
        return self._stream


class IterEnqueueSyncTest(TestCase):
    """Exercise the real ``_iter_enqueue_sync`` pump on a dedicated asyncio loop.

    These tests don't go through gRPC — they drive the pump directly via the
    process-level fallback loop (``_get_async_loop``) so we observe cancel and
    exception propagation without starting a real server.
    """

    def tearDown(self) -> None:
        # Reset module-level fallback loop between tests to avoid cross-test residue.
        if bg_svc._async_loop is not None and bg_svc._async_loop.is_running():
            bg_svc._async_loop.call_soon_threadsafe(bg_svc._async_loop.stop)
            if bg_svc._async_loop_thread is not None:
                bg_svc._async_loop_thread.join(timeout=5.0)
        bg_svc._async_loop = None
        bg_svc._async_loop_thread = None
        bg_svc._enqueue_loop = None

    def _gi(self) -> object:
        return MagicMock()

    def test_yields_chunks_in_order(self) -> None:
        chunks = ["c1", "c2", "c3"]
        visitor = _FakeVisitor(_FakeAsyncStream(chunks))
        ctx = MagicMock()
        ctx.is_active.return_value = True
        got = list(_iter_enqueue_sync(visitor, self._gi(), context=ctx))
        self.assertEqual(got, chunks)
        self.assertEqual(visitor.enqueue_called, 1)

    def test_backend_exception_propagates(self) -> None:
        """Backend error inside ``enqueue``/stream must bubble out of the sync iterator."""
        visitor = _FakeVisitor(_FakeAsyncStream(["c1"], raise_after=1))
        ctx = MagicMock()
        ctx.is_active.return_value = True
        with self.assertRaises(RuntimeError):
            list(_iter_enqueue_sync(visitor, self._gi(), context=ctx))

    def test_cancel_mid_stream_stops_iteration(self) -> None:
        """When ``context.is_active()`` flips to False the pump is cancelled promptly.

        We feed a slow stream and toggle ``is_active`` to False after the first chunk;
        the iterator must stop without waiting for the remaining items.
        """
        chunks = ["c1", "c2", "c3", "c4"]
        # Delay each chunk longer than the poll interval so cancel has time to kick in.
        stream = _FakeAsyncStream(chunks, delay=0.3)
        visitor = _FakeVisitor(stream)

        ctx = MagicMock()
        active_flag = [True]

        def is_active() -> bool:
            return active_flag[0]

        ctx.is_active.side_effect = is_active

        cancel_cbs: list = []

        def add_cb(cb):
            cancel_cbs.append(cb)

        ctx.add_callback.side_effect = add_cb

        got: list = []
        start = time.time()

        def consumer() -> None:
            for x in _iter_enqueue_sync(visitor, self._gi(), context=ctx):
                got.append(x)
                active_flag[0] = False  # cancel after first chunk

        t = threading.Thread(target=consumer)
        t.start()
        t.join(timeout=5.0)
        elapsed = time.time() - start

        self.assertFalse(t.is_alive(), "iterator did not stop within 5s after cancel")
        self.assertEqual(got, ["c1"])
        # Must exit well before the remaining 3 * 0.3s of chunks would be produced.
        self.assertLess(elapsed, 2.0)
        # Cancel callback must have been registered on the context.
        self.assertEqual(len(cancel_cbs), 1)

    def test_add_callback_fires_cancel(self) -> None:
        """Invoking the registered ``add_callback`` handler cancels the pump future."""
        chunks = ["c1", "c2", "c3"]
        stream = _FakeAsyncStream(chunks, delay=0.3)
        visitor = _FakeVisitor(stream)

        ctx = MagicMock()
        ctx.is_active.return_value = True
        registered: list = []

        def add_cb(cb):
            registered.append(cb)

        ctx.add_callback.side_effect = add_cb

        got: list = []

        def consumer() -> None:
            for x in _iter_enqueue_sync(visitor, self._gi(), context=ctx):
                got.append(x)

        t = threading.Thread(target=consumer)
        t.start()
        # Wait for first chunk.
        deadline = time.time() + 2.0
        while not got and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(got, ["c1"])
        # Trigger cancel via the callback (simulates gRPC peer RESET_STREAM / deadline).
        self.assertEqual(len(registered), 1)
        registered[0]()
        t.join(timeout=5.0)
        self.assertFalse(t.is_alive(), "iterator did not stop after cancel callback")

    def test_no_context_uses_default_active(self) -> None:
        """``context=None`` (unit-test fake) is treated as permanently active."""
        chunks = ["a", "b"]
        visitor = _FakeVisitor(_FakeAsyncStream(chunks))
        got = list(_iter_enqueue_sync(visitor, self._gi()))
        self.assertEqual(got, chunks)


class ResolveLoopForEnqueueTest(TestCase):
    def tearDown(self) -> None:
        if bg_svc._async_loop is not None and bg_svc._async_loop.is_running():
            bg_svc._async_loop.call_soon_threadsafe(bg_svc._async_loop.stop)
            if bg_svc._async_loop_thread is not None:
                bg_svc._async_loop_thread.join(timeout=5.0)
        bg_svc._async_loop = None
        bg_svc._async_loop_thread = None
        bg_svc._enqueue_loop = None

    def test_app_loop_preferred_when_running(self) -> None:
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, daemon=True)
        t.start()
        try:
            bg_svc.set_dash_sc_grpc_enqueue_event_loop(loop)
            self.assertIs(bg_svc.resolve_loop_for_enqueue(), loop)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            t.join(timeout=5.0)
            loop.close()
            bg_svc._enqueue_loop = None

    def test_fallback_loop_when_app_loop_missing(self) -> None:
        bg_svc._enqueue_loop = None
        loop = bg_svc.resolve_loop_for_enqueue()
        self.assertTrue(loop.is_running())
        self.assertIs(loop, bg_svc._async_loop)


if __name__ == "__main__":
    main()
