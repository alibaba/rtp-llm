import asyncio
import logging
import os
import socket
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

logger = logging.getLogger("vllm.navi_sched.prepare")
PREPARE_PATH = "/navi_sched/native_prepare"


@dataclass
class _PreparationProgress:
    active_batches: int = 0
    active_requests: int = 0
    ready_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    cancelled_requests: int = 0
    _last_log_time: float = float("-inf")

    def log(self, role, cache_bytes=-1):
        try:
            now = time.monotonic()
            if now - self._last_log_time < 1.0:
                return
            self._last_log_time = now
            rss_bytes = -1
            try:
                with open("/proc/self/statm") as statm:
                    rss_bytes = int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            except (OSError, ValueError, IndexError):
                pass
            logger.info(
                "NAVI_SCHED_VLLM_PREPARATION_PROGRESS role=%s pid=%d "
                "active_batches=%d active_requests=%d ready_waiting_requests=%d "
                "completed_requests=%d failed_requests=%d cancelled_requests=%d "
                "processor_cache_bytes=%d rss_bytes=%d",
                role, os.getpid(), self.active_batches, self.active_requests,
                self.ready_requests, self.completed_requests, self.failed_requests,
                self.cancelled_requests, cache_bytes, rss_bytes,
            )
        except Exception:
            pass


def _socket_prefix(port):
    return f"navi_sched.native_prepare.{port}."


def _worker_sockets(port):
    if not sys.platform.startswith("linux"):
        return ()
    prefix = "@" + _socket_prefix(port)
    with open("/proc/net/unix") as table:
        return tuple(sorted({
            "\0" + fields[-1][1:]
            for line in table
            if len(fields := line.split()) >= 8 and fields[3] == "00010000"
            and fields[-1].startswith(prefix)
            and fields[-1][len(prefix):].isdigit()
        }))


@dataclass
class _PreparationPeer:
    session: object
    in_flight: int = 0
    retired: bool = False


def configure_native_preparation(app):
    if os.environ.get("VLLM_NAVI_SCHED_DEFER_RENDER", "0") != "1":
        return

    import msgspec
    from starlette.requests import Request
    from starlette.responses import Response
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    from .VLLMDeferredRenderer import NativeRenderContinuation

    continuation = None
    port = int(os.environ.get("VLLM_NAVI_SCHED_SERVER_PORT", "0"))
    init_lock = asyncio.Lock()
    decoder = MsgpackDecoder(EngineCoreRequest)
    encoder = MsgpackEncoder()
    progress = _PreparationProgress()

    def log_progress():
        try:
            processor = getattr(continuation, "_processor", None)
            renderer = getattr(processor, "renderer", None)
            mm_processor = getattr(renderer, "mm_processor", None)
            cache = getattr(getattr(mm_processor, "cache", None), "_cache", None)
            progress.log("api", getattr(cache, "currsize", -1))
        except Exception:
            pass

    async def materialize(payload):
        nonlocal continuation
        request_id, schedule_batch_id, local_batch_id = "unknown", 0, 0
        phase = "decode"
        started_ns = time.perf_counter_ns()
        progress.active_requests += 1
        log_progress()
        try:
            body = msgspec.msgpack.decode(payload)
            request = decoder.decode(body["frames"])
            request_id = request.request_id
            schedule_batch_id = int(body["schedule_batch_id"])
            local_batch_id = int(body["local_batch_id"])
            if request.navi_sched_render_call is None:
                raise ValueError("native preparation requires a media continuation")
            phase = "initialize"
            async with init_lock:
                if continuation is None:
                    state = app.state
                    engine_core = getattr(state.engine_client, "engine_core", None)
                    provider = getattr(engine_core, "get_native_render_continuation", None)
                    if provider is not None:
                        continuation = await provider()
                    else:
                        tasks = await state.engine_client.get_supported_tasks()
                        continuation = NativeRenderContinuation(state.vllm_config, tasks)
            phase = "render"
            await continuation.resume_async(
                [request], schedule_batch_id=schedule_batch_id,
                local_batch_id=local_batch_id,
            )
            phase = "encode"
            frames = [bytes(frame) for frame in encoder.encode(request)]
            result = msgspec.msgpack.encode(frames)
            progress.completed_requests += 1
            return result
        except asyncio.CancelledError:
            progress.cancelled_requests += 1
            raise
        except Exception as exc:
            progress.failed_requests += 1
            logger.exception(
                "NAVI_SCHED_VLLM_NATIVE_PREPARE_SERVER request_id=%s "
                "schedule_batch_id=%d local_batch_id=%d pid=%d phase=%s "
                "total_ms=%.3f error_type=%s http_status=%s error=%r",
                request_id, schedule_batch_id, local_batch_id, os.getpid(), phase,
                (time.perf_counter_ns() - started_ns) / 1_000_000.0,
                type(exc).__name__, getattr(exc, "status", 0), str(exc)[:1024],
            )
            raise
        finally:
            progress.active_requests -= 1
            log_progress()

    async def prepare(raw_request: Request):
        if raw_request.client is None or raw_request.client.host not in ("127.0.0.1", "::1"):
            return Response(status_code=403)
        try:
            return Response(
                await materialize(await raw_request.body()), media_type="application/msgpack",
                headers={"X-Navi-Prepare-Pid": str(os.getpid())},
            )
        except Exception:
            return Response(status_code=422, headers={"X-Navi-Prepare-Pid": str(os.getpid())})

    app.add_api_route(PREPARE_PATH, prepare, methods=["POST"], include_in_schema=False)
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with original_lifespan(application) as state:
            runner = None
            listener = None
            try:
                if sys.platform.startswith("linux") and port > 0:
                    from aiohttp import web

                    async def prepare_unix(request):
                        try:
                            return web.Response(
                                body=await materialize(await request.read()),
                                content_type="application/msgpack",
                                headers={"X-Navi-Prepare-Pid": str(os.getpid())},
                            )
                        except Exception:
                            return web.Response(
                                status=422, headers={"X-Navi-Prepare-Pid": str(os.getpid())},
                            )

                    local_app = web.Application(client_max_size=0)
                    local_app.router.add_post(PREPARE_PATH, prepare_unix)
                    runner = web.AppRunner(local_app, access_log=None)
                    await runner.setup()
                    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    listener.bind("\0" + _socket_prefix(port) + str(os.getpid()))
                    listener.listen(128)
                    listener.setblocking(False)
                    await web.SockSite(runner, listener).start()
                yield state
            finally:
                if runner is not None:
                    await runner.cleanup()
                if listener is not None:
                    listener.close()
                if continuation is not None:
                    await continuation.aclose()

    app.router.lifespan_context = lifespan


class NativePreparationClient:
    """Prepare on existing local API processes; EngineCore retains admission."""

    def __init__(self, port):
        self._port = port
        self._url = f"http://127.0.0.1:{port}{PREPARE_PATH}"
        self._session = None
        self._peers = {}
        self._addresses = ()
        self._discovered_at = 0.0
        self._next_peer = 0
        self._encoder = None
        self._decoder = None
        self._progress = _PreparationProgress()

    def log_progress(self):
        self._progress.log("agent")

    async def _select_peer(self):
        import aiohttp

        now = time.monotonic()
        if now - self._discovered_at >= 1.0:
            self._addresses = _worker_sockets(self._port)
            self._discovered_at = now
        for address in tuple(self._peers):
            peer = self._peers.get(address)
            if peer is not None and address not in self._addresses:
                peer.retired = True
                if peer.in_flight == 0:
                    del self._peers[address]
                    await peer.session.close()
        if not self._addresses:
            return None, None
        for address in self._addresses:
            if address not in self._peers:
                self._peers[address] = _PreparationPeer(aiohttp.ClientSession(
                    connector=aiohttp.UnixConnector(path=address),
                    timeout=aiohttp.ClientTimeout(total=None),
                ))
            self._peers[address].retired = False
        start = self._next_peer % len(self._addresses)
        addresses = self._addresses[start:] + self._addresses[:start]
        address = min(addresses, key=lambda key: self._peers[key].in_flight)
        self._next_peer = (self._addresses.index(address) + 1) % len(self._addresses)
        peer = self._peers[address]
        peer.in_flight += 1
        return address, peer

    async def _prepare_one(self, request, schedule_batch_id, local_batch_id):
        if request.navi_sched_render_call is None:
            return request
        import aiohttp
        import msgspec
        from vllm.v1.engine import EngineCoreRequest
        from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

        if self._encoder is None:
            self._encoder = MsgpackEncoder()
            self._decoder = MsgpackDecoder(EngineCoreRequest)
        payload = msgspec.msgpack.encode({
            "frames": [bytes(frame) for frame in self._encoder.encode(request)],
            "schedule_batch_id": schedule_batch_id, "local_batch_id": local_batch_id,
        })
        address, peer = await self._select_peer()
        if peer is None:
            if self._session is None:
                self._session = aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(),
                    timeout=aiohttp.ClientTimeout(total=None),
                )
            session, url = self._session, self._url
        else:
            session, url = peer.session, "http://localhost" + PREPARE_PATH
        started_ns = time.perf_counter_ns()
        prepare_pid = "unknown"
        try:
            async with session.post(url, data=payload) as response:
                prepare_pid = response.headers.get("X-Navi-Prepare-Pid", "unknown")
                response.raise_for_status()
                prepared = self._decoder.decode(msgspec.msgpack.decode(await response.read()))
        except Exception as exc:
            logger.error(
                "NAVI_SCHED_VLLM_NATIVE_PREPARE_RPC request_id=%s "
                "schedule_batch_id=%d local_batch_id=%d peer=%r prepare_pid=%s "
                "total_ms=%.3f error_type=%s http_status=%s error=%r",
                request.request_id, schedule_batch_id, local_batch_id,
                address if peer is not None else url, prepare_pid,
                (time.perf_counter_ns() - started_ns) / 1_000_000.0,
                type(exc).__name__, getattr(exc, "status", 0), str(exc)[:1024],
            )
            raise
        finally:
            if peer is not None:
                peer.in_flight -= 1
                if peer.retired and peer.in_flight == 0:
                    if self._peers.get(address) is peer:
                        del self._peers[address]
                    await peer.session.close()
        if (
            prepared.request_id != request.request_id
            or prepared.navi_sched_render_call is not None
        ):
            raise ValueError("native API preparation changed request identity or did not finish")
        request.prompt_token_ids = prepared.prompt_token_ids
        request.prompt_embeds = prepared.prompt_embeds
        request.mm_features = prepared.mm_features
        request.navi_sched_render_call = None
        return request

    async def resume_async(
        self, requests, *, return_exceptions=False, schedule_batch_id=0, local_batch_id=0
    ):
        progress = self._progress
        ready_count = 0

        async def prepare(request):
            nonlocal ready_count
            progress.active_requests += 1
            try:
                result = await self._prepare_one(request, schedule_batch_id, local_batch_id)
                ready_count += 1
                progress.ready_requests += 1
                progress.completed_requests += 1
                return result
            except asyncio.CancelledError:
                progress.cancelled_requests += 1
                raise
            except Exception:
                progress.failed_requests += 1
                raise
            finally:
                progress.active_requests -= 1
                self.log_progress()

        progress.active_batches += 1
        try:
            results = await asyncio.gather(
                *(prepare(request) for request in requests), return_exceptions=True,
            )
            if not return_exceptions:
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
            return results
        finally:
            progress.active_batches -= 1
            progress.ready_requests -= ready_count
            self.log_progress()

    async def aclose(self):
        peers, self._peers = self._peers, {}
        for peer in peers.values():
            await peer.session.close()
        if self._session is not None:
            await self._session.close()
            self._session = None
