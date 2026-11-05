"""Smoke-only coordination of a large PD cohort over the existing bidi RPC."""

import asyncio
from concurrent.futures import Future, TimeoutError
import json
import threading
import time

import grpc
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub

_GRPC_OPTIONS = (("grpc.max_send_message_length", -1),
                 ("grpc.max_receive_message_length", -1),
                 ("grpc.enable_http_proxy", 0))


class PdGroupRelay:
    """Bound concurrent LOADs and release GENERATE only after all LOADs succeed.

    Only control messages pass through this loopback relay. The Decode service
    still reads the original Prefill cache directly over RDMA.
    """

    def __init__(self, decode_endpoint, cohort_size, artifact, *, max_loads=3):
        self.decode_endpoint = decode_endpoint
        self.cohort_size = cohort_size
        self.artifact = artifact
        self.max_loads = max_loads
        self.startup = Future()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self.thread.start()
        try:
            self.port = self.startup.result(timeout=10)
        except TimeoutError:
            if not self.startup.cancel():
                self.__exit__(TimeoutError)
            raise
        return self

    def __exit__(self, exc_type, *_):
        self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(timeout=10)
        if self.thread.is_alive() and exc_type is None:
            raise RuntimeError("PD group relay did not stop")

    def _run(self):
        try:
            asyncio.run(self._serve())
        except Exception as exc:
            if self.startup.done():
                raise
            self.startup.set_exception(exc)

    async def _serve(self):
        self.loop = asyncio.get_running_loop()
        self.stop = asyncio.Event()
        self.ready = asyncio.Event()
        self.load_slots = asyncio.Semaphore(self.max_loads)
        self.members = set()
        self.ready_members = set()
        self.failed = False
        async with grpc.aio.insecure_channel(
            self.decode_endpoint, options=_GRPC_OPTIONS
        ) as channel:
            self.decode = RpcServiceStub(channel)
            server = grpc.aio.server(options=_GRPC_OPTIONS)
            method = grpc.stream_stream_rpc_method_handler(
                self._relay, request_deserializer=pb.GenerateRequestPB.FromString,
                response_serializer=pb.GenerateOutputsPB.SerializeToString)
            service = pb.DESCRIPTOR.services_by_name["RpcService"].full_name
            server.add_generic_rpc_handlers((
                grpc.method_handlers_generic_handler(service, {"RemoteGenerate": method}),
            ))
            port = server.add_insecure_port("127.0.0.1:0")
            with self.artifact.open("x") as stream:
                self.log = stream
                self._record("start", decode_endpoint=self.decode_endpoint,
                             relay_port=port, max_loads=self.max_loads)
                await server.start()
                try:
                    if self.startup.set_running_or_notify_cancel():
                        self.startup.set_result(port)
                        await self.stop.wait()
                finally:
                    await server.stop(0)

    def _record(self, phase, **fields):
        self.log.write(json.dumps(dict(phase=phase, time_ns=time.time_ns(), **fields)) + "\n")
        self.log.flush()

    async def _stage_response(self, call, context):
        response = await call.read()
        if response is grpc.aio.EOF:
            code = await call.code()
            if code != grpc.StatusCode.OK:
                await context.abort(code, await call.details(), await call.trailing_metadata())
            await context.abort(grpc.StatusCode.DATA_LOSS, "Decode stage response missing")
        return response

    async def _relay(self, requests, context):
        request_id = 0
        call = self.decode.RemoteGenerate(timeout=context.time_remaining(),
                                          metadata=context.invocation_metadata())
        try:
            allocate = await anext(requests)
            request_id = allocate.request_id
            if (allocate.stage != pb.ALLOCATE or request_id in self.members
                    or len(self.members) >= self.cohort_size):
                raise ValueError("unexpected grouped Decode allocation")
            self.members.add(request_id)
            await call.write(allocate)
            response = await self._stage_response(call, context)
            yield response
            if response.error_info.error_code:
                raise ValueError(f"Decode ALLOCATE error {response.error_info.error_code}")

            load = await anext(requests)
            # Prefill uses the proto default stage for this positional message.
            if (load.request_id != request_id or load.stage not in (pb.LOAD, pb.ALLOCATE)
                    or load.HasField("input")):
                raise ValueError("unexpected grouped Decode load message")
            async with self.load_slots:
                if self.failed:
                    raise ValueError("another cohort member failed")
                self._record("load_start", request_id=request_id)
                await call.write(load)
                response = await self._stage_response(call, context)
                self._record("load_end", request_id=request_id, code=response.error_info.error_code)
            yield response
            if response.error_info.error_code:
                raise ValueError(f"Decode LOAD error {response.error_info.error_code}")

            generate = await anext(requests)
            if generate.request_id != request_id or generate.stage != pb.GENERATE:
                raise ValueError("unexpected grouped Decode generation message")
            self.ready_members.add(request_id)
            if len(self.ready_members) == self.cohort_size:
                self.ready.set()
            await self.ready.wait()
            if self.failed:
                raise ValueError("another cohort member failed")
            self._record("generate", request_id=request_id)
            await call.write(generate)
            await call.done_writing()
            while (response := await call.read()) is not grpc.aio.EOF:
                yield response
            code = await call.code()
            if code != grpc.StatusCode.OK:
                await context.abort(code, await call.details(), await call.trailing_metadata())
        except BaseException as exc:
            self.failed = True
            self.ready.set()
            self._record("error", request_id=request_id, error=f"{type(exc).__name__}: {exc}")
            if isinstance(exc, grpc.aio.AioRpcError):
                await context.abort(exc.code(), exc.details(), exc.trailing_metadata())
            if isinstance(exc, (ValueError, StopAsyncIteration)):
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise
        finally:
            if not call.done():
                call.cancel()
