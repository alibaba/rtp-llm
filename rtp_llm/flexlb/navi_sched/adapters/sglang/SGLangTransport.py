from __future__ import annotations

import asyncio
from dataclasses import dataclass

from navi_sched.asyncio import AsyncClient

from .SGLangCache import prompt_block_hashes
from .SGLangForwardProtocol import SGLangForwardCodec


@dataclass
class _ResponseStream:
    rid: str
    queue: asyncio.Queue
    index: int
    done: asyncio.Future


class SGLangTransport:
    def __init__(
        self, manager, config, client=None, response_kinds=None, *, page_size=None
    ):
        self.manager = manager
        self.config = config
        self.codec = SGLangForwardCodec(config.max_payload_bytes)
        self.page_size = (
            page_size if page_size is not None else manager.server_args.page_size
        )
        if type(self.page_size) is not int or self.page_size <= 0:
            raise ValueError("SGLang scheduler must provide a positive page size")
        self._streams: dict[int, _ResponseStream] = {}
        self._active_rids: set[str] = set()
        self._request_tasks: dict[str, asyncio.Task] = {}
        self._closed = False
        self._async_client = None
        self._slots = asyncio.Semaphore(config.max_inflight_requests)
        self.snapshot = {
            "local_fallback_count": 0,
            "remote_request_count": 0,
            "response_frame_count": 0,
        }
        if client is None:
            from navi_sched import NaviSchedClient

            client = NaviSchedClient(
                config.client_biz,
                config.master_biz,
                runtime_config=config.runtime,
                enable_agent_forward=True,
                expect_agent_response_stream=True,
                wait_no_query=False,
            )
            client.init()
        if response_kinds is None:
            from navi_sched.util import naviSchedModule

            response_kinds = naviSchedModule().AgentResponseKind
        self.client = client
        self._kinds = response_kinds

    def abort(self, rid="", abort_all=False):
        matched = False
        for request_rid, task in tuple(self._request_tasks.items()):
            if abort_all or request_rid == rid:
                matched = True
                task.cancel()
        for request_id, stream in tuple(self._streams.items()):
            if abort_all or stream.rid == rid:
                matched = True
                self._fail_stream(request_id, stream, asyncio.CancelledError())
        return matched

    def _fail_stream(self, request_id, stream, error):
        self.client.abort_forward_request(request_id)
        if not stream.done.done():
            stream.done.set_exception(error)

    def _fail_responses(self, error):
        for request_id, stream in tuple(self._streams.items()):
            self._fail_stream(request_id, stream, error)

    def _read_response(self, response):
        stream = self._streams.get(int(response.request_id))
        if stream is None or stream.done.done():
            return
        if response.kind == self._kinds.ACK:
            return
        if response.kind == self._kinds.ERROR:
            self._fail_stream(
                response.request_id,
                stream,
                RuntimeError(response.error_message or "SGLang target failed"),
            )
            return
        try:
            if response.kind != self._kinds.OUTPUT:
                raise ValueError("unknown NaviSched response kind")
            for frame in response.frames:
                output = self.codec.decode_response(frame)
                stream.queue.put_nowait((stream.index, output, None))
                self.snapshot["response_frame_count"] += 1
            if response.terminal:
                stream.done.set_result(None)
        except (asyncio.QueueFull, ValueError, TypeError) as error:
            self._fail_stream(response.request_id, stream, error)

    async def _schedule_one(self, obj, request, index, outputs):
        from navi_sched.sched import makeRequest

        async with self._slots:
            if self._closed:
                raise RuntimeError("SGLang NaviSched transport is closed")
            loop = asyncio.get_running_loop()
            if self._async_client is None:
                self._async_client = AsyncClient(
                    self.client, self._read_response, self._fail_responses
                )
            payload = self.codec.encode_request(obj)
            # Media expansion depends on target-side processing. Reserve the context
            # limit instead of claiming that unexpanded text is the actual workload.
            if obj.contains_mm_input() or obj.input_embeds is not None:
                prompt_count = int(self.manager.max_req_input_len)
                hashes = []
            else:
                if obj.input_ids is None:
                    ids, _ = await self.manager._tokenize_texts(obj.text, False)
                    obj.text = None
                    obj.input_ids = ids
                    payload = self.codec.encode_request(obj)
                ids = obj.input_ids
                prompt_count = len(ids)
                hashes = (
                    []
                    if obj.lora_path or obj.session_params
                    else prompt_block_hashes(
                        ids,
                        self.page_size,
                        extra_key=obj.extra_key,
                        cache_salt=obj.cache_salt,
                    )
                )
            request_id = self.client.allocate_request_id()
            stream = _ResponseStream(obj.rid, outputs, index, loop.create_future())
            self._streams[request_id] = stream
            scheduled = makeRequest(
                prompt_count,
                hashes,
                request_id=request_id,
                block_size=self.page_size,
                forward_payload=payload,
            )
            try:
                decision = await self._async_client.schedule(
                    scheduled, self.config.target_biz, self.config.schedule_timeout_ms
                )
                if decision.target is None:
                    raise RuntimeError("NaviSched rejected request: " + decision.reason)
                self.snapshot["remote_request_count"] += 1
                self.snapshot["last_schedule"] = {
                    "request_id": request_id,
                    "prompt_tokens": prompt_count,
                    "payload_bytes": len(payload),
                    "schedule_id": getattr(decision, "schedule_id", 0),
                    "predicted_cache_hit_tokens": getattr(
                        decision, "predicted_cache_hit_tokens", 0
                    ),
                    "target_address": getattr(
                        decision.target, "address", str(decision.target)
                    ),
                }
                while not stream.done.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(stream.done), 0.25)
                    except asyncio.TimeoutError:
                        if request is not None and not obj.background:
                            if await request.is_disconnected():
                                raise asyncio.CancelledError()
                await stream.done
            finally:
                self._streams.pop(request_id, None)
                if not stream.done.done():
                    stream.done.cancel()
                elif not stream.done.cancelled():
                    stream.done.exception()
                self.client.abort_forward_request(request_id)

    async def generate(self, obj, request, original):
        rids = obj.rid if isinstance(obj.rid, list) else [obj.rid]
        if any(rid in self._active_rids for rid in rids):
            raise ValueError("Duplicate request ID detected")
        if not self.client.has_master_biz():
            self.client.report_local_fallback()
            self.snapshot["local_fallback_count"] += 1
            async for output in original(self.manager, obj, request):
                yield output
            return
        self.manager.auto_create_handle_loop()
        obj.normalize_batch_and_arguments()
        self.manager._set_default_priority(obj)
        items = [obj] if obj.is_single else [obj[i] for i in range(len(obj.rid))]
        if any(item.session_params is not None for item in items):
            raise ValueError(
                "SGLang NaviSched distributed sessions require target affinity and are not supported"
            )
        if len(self._active_rids) + len(items) > self.config.max_inflight_requests:
            raise ValueError("SGLang batch exceeds max_inflight_requests")
        if any(
            item.rid in self.manager.rid_to_state or item.rid in self._active_rids
            for item in items
        ):
            raise ValueError("Duplicate request ID detected")
        self.manager._init_req_state(obj, request)
        source_states = {
            item.rid: self.manager.rid_to_state[item.rid] for item in items
        }
        self._active_rids.update(source_states)
        outputs = asyncio.Queue(maxsize=self.config.max_buffered_responses)
        closing = False

        async def run_item(index, item):
            try:
                item.sampling_params = dict(item.sampling_params or {}, n=1)
                async with asyncio.timeout(self.config.request_timeout_ms / 1000):
                    await self._schedule_one(item, request, index, outputs)
            except asyncio.CancelledError as error:
                if closing:
                    raise
                await outputs.put((index, None, error))
                return
            except TimeoutError:
                await outputs.put(
                    (index, None, TimeoutError("SGLang NaviSched request timed out"))
                )
            except BaseException as error:
                await outputs.put((index, None, error))
            await outputs.put((index, None, None))

        tasks = [asyncio.create_task(run_item(i, item)) for i, item in enumerate(items)]
        self._request_tasks.update((item.rid, task) for item, task in zip(items, tasks))
        finished = 0
        results = [None] * len(items)
        try:
            while finished < len(items):
                index, output, error = await outputs.get()
                if error is not None:
                    raise error
                if output is None:
                    finished += 1
                    continue
                results[index] = output
                if obj.stream:
                    if not obj.is_single:
                        output["index"] = index
                    yield output
            if not obj.stream:
                if any(output is None for output in results):
                    raise RuntimeError("SGLang response stream ended without a result")
                yield results[0] if obj.is_single else results
        finally:
            closing = True
            for task in tasks:
                task.cancel()
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                self._active_rids.difference_update(item.rid for item in items)
                for item, task in zip(items, tasks):
                    if self._request_tasks.get(item.rid) is task:
                        self._request_tasks.pop(item.rid)
                    if (
                        self.manager.rid_to_state.get(item.rid)
                        is source_states[item.rid]
                    ):
                        self.manager.rid_to_state.pop(item.rid)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.abort(abort_all=True)
        if self._async_client is not None:
            self._async_client.close()
        self.client.stop()
