"""End-to-end smoke: ``client -> proxy gRPC server -> inference gRPC server``.

Production topology in miniature: two real ``grpc.aio.Server`` processes
co-resident in one event loop, glued via the ``SERVICE_ROUTE`` env. Verifies
the wire-level invariants the in-memory proxy unit test (which mocks the stub)
can't:

* Request frames + response chunks survive a real HTTP/2 round trip on both
  legs (client->proxy, proxy->inference).
* ``DashScProxyServicer._buffered_iter`` correctly buffers the first frame,
  flushes both once the second arrives, and streams the rest.
* gRPC stream completes cleanly when the inference servicer emits a
  ``finished=True`` chunk — no late client-cancel race.
* Proxy + inference both shut down without dangling channels.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import unittest
from unittest.mock import patch

import grpc
import torch

from rtp_llm.config.py_config_modules import DASH_SC_GRPC_SERVER_PORT_OFFSET
from rtp_llm.dash_sc.client import (
    build_model_infer_request,
    decode_finish_reason,
)
from rtp_llm.dash_sc.codec import SamplingParams
from rtp_llm.dash_sc.inference.servicer import (
    DashScInferenceServicer,
    build_think_runtime,
)
from rtp_llm.dash_sc.proto import predict_v2_pb2_grpc
from rtp_llm.dash_sc.proxy.servicer import DashScProxyServicer
from rtp_llm.dash_sc.server import DashScGrpcServer
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateOutput,
    GenerateOutputs,
)


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _free_port_with_http_offset_room() -> int:
    """Pick a free port whose ``port - DASH_SC_GRPC_SERVER_PORT_OFFSET`` is also a
    plausible TCP port. Service discovery treats the env-supplied port as the
    *http* port and computes ``grpc_port = http + offset``, so the inference
    server has to bind on a port that round-trips through that math."""
    while True:
        p = _free_port()
        if p - DASH_SC_GRPC_SERVER_PORT_OFFSET > 1024:
            return p


def _go_chunk(token_ids, finished, prompt_len=4):
    out = GenerateOutput(
        output_ids=torch.tensor(token_ids, dtype=torch.int32),
        finished=finished,
        aux_info=AuxInfo(input_len=prompt_len, output_len=len(token_ids)),
    )
    return GenerateOutputs(generate_outputs=[out])


class _FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        c = self._chunks[self._i]
        self._i += 1
        return c

    async def aclose(self):
        pass


class _FakeVisitor:
    def __init__(self, chunks):
        self._chunks = chunks
        self.captured = []

    async def enqueue(self, generate_input):
        self.captured.append(generate_input)
        return _FakeAsyncStream(self._chunks)


class DashScProxyInferenceE2ESmokeTest(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self) -> None:
        # Inference leg: fake visitor with three chunks (third = finished).
        # Three chunks ensures _buffered_iter flushes both buffered frames AND
        # the trailing one — covers the hot path, not just a short-circuit.
        self.visitor = _FakeVisitor([
            _go_chunk([100, 101], finished=False),
            _go_chunk([102],      finished=False),
            _go_chunk([103],      finished=True),
        ])

        # Bind inference server on an "http+offset" port so SERVICE_ROUTE can
        # advertise the matching http_port and the proxy resolves to us.
        self.inference_grpc_port = _free_port_with_http_offset_room()
        inference_http_port = (
            self.inference_grpc_port - DASH_SC_GRPC_SERVER_PORT_OFFSET
        )

        self.inference_server = DashScGrpcServer()
        self.inference_servicer = DashScInferenceServicer(
            backend_visitor=self.visitor,
            think_runtime=build_think_runtime(None, None, "test"),
        )
        await self.inference_server.start(
            port=self.inference_grpc_port,
            servicer=self.inference_servicer,
            server_id="0",
        )

        # Proxy leg: SERVICE_ROUTE points at the inference's *http* port; the
        # discovery code adds DASH_SC_GRPC_SERVER_PORT_OFFSET back.
        service_route = json.dumps({
            "type": "ip_port_list",
            "address": f"127.0.0.1:{inference_http_port}",
        })
        self._env_patcher = patch.dict(
            os.environ, {"SERVICE_ROUTE": service_route}, clear=False,
        )
        self._env_patcher.start()

        self.proxy_port = _free_port()
        while self.proxy_port == self.inference_grpc_port:
            self.proxy_port = _free_port()
        self.proxy_servicer = DashScProxyServicer()
        self.proxy_server = DashScGrpcServer()
        await self.proxy_server.start(
            port=self.proxy_port,
            servicer=self.proxy_servicer,
            server_id="0",
        )

    async def asyncTearDown(self) -> None:
        try:
            if self.proxy_server._server is not None:
                await self.proxy_server._server.stop(None)
            await self.proxy_servicer.close()
            if self.inference_server._server is not None:
                await self.inference_server._server.stop(None)
        finally:
            self._env_patcher.stop()

    async def test_basic_stream_through_proxy(self) -> None:
        async with grpc.aio.insecure_channel(
            f"127.0.0.1:{self.proxy_port}"
        ) as channel:
            stub = predict_v2_pb2_grpc.GRPCInferenceServiceStub(channel)
            request = build_model_infer_request(
                request_id="trace-e2e",
                model_name="fake-model",
                input_ids=[1, 2, 3, 4],
                sampling=SamplingParams(max_new_tokens=16, top_k=1),
            )

            async def _req_iter():
                yield request

            responses = []
            async for resp in stub.ModelStreamInfer(_req_iter()):
                responses.append(resp)

        # ---- response shape -------------------------------------------------
        # Inference emitted three chunks; proxy._buffered_iter holds the first
        # until the second arrives but yields both before the third — net
        # frame count must equal what inference produced.
        self.assertEqual(len(responses), 3)
        # trace_id is round-tripped end to end (not rewritten by either leg).
        self.assertEqual(responses[0].infer_response.id, "trace-e2e")
        for resp in responses:
            self.assertFalse(
                resp.error_message,
                msg=f"unexpected error_message: {resp.error_message!r}",
            )

        # ---- final-frame contract ------------------------------------------
        last = {
            o.name: (o, raw)
            for o, raw in zip(
                responses[-1].infer_response.outputs,
                responses[-1].infer_response.raw_output_contents,
            )
        }
        self.assertIn("finished", last)
        self.assertEqual(last["finished"][1], b"\x01")
        self.assertIn("finish_reason", last)
        fr_out, fr_raw = last["finish_reason"]
        # Natural finish without max_new_tokens hit -> STOP (1). LENGTH (2)
        # would only appear if cumulative_sent_ids >= max_new_tokens, which
        # 4 tokens vs max_new_tokens=16 won't trigger.
        self.assertEqual(decode_finish_reason(fr_out, fr_raw), 1)

        # ---- intermediate frames stream content ---------------------------
        first_outputs = {
            o.name: (o, raw)
            for o, raw in zip(
                responses[0].infer_response.outputs,
                responses[0].infer_response.raw_output_contents,
            )
        }
        self.assertIn("output_ids", first_outputs)
        first_ids = list(struct.unpack(
            "<%di" % (len(first_outputs["output_ids"][1]) // 4),
            first_outputs["output_ids"][1],
        ))
        self.assertEqual(first_ids, [100, 101])

        # ---- enqueue side: backend received what the codec built -----------
        self.assertEqual(len(self.visitor.captured), 1)
        gi = self.visitor.captured[0]
        self.assertEqual(gi.token_ids.tolist(), [1, 2, 3, 4])
        self.assertEqual(gi.generate_config.max_new_tokens, 16)


if __name__ == "__main__":
    unittest.main()
