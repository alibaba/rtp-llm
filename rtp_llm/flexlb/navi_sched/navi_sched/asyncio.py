"""Event-loop integration for NaviSched's existing native queues."""

import asyncio
import os


class AsyncNotification:
    def __init__(self, fd, drain, fail):
        self.loop = asyncio.get_running_loop()
        self.fd = fd
        self.drain = drain
        self.fail = fail
        self.closed = False
        self._continuation = None
        self.loop.add_reader(fd, self._ready)

    def _ready(self):
        if self.closed:
            return
        if self._continuation is not None:
            self._continuation.cancel()
            self._continuation = None
        try:
            try:
                os.eventfd_read(self.fd)
            except BlockingIOError:
                pass
            # A bounded drain lets generation and cancellation tasks run between bursts.
            if self.drain():
                self._continuation = self.loop.call_soon(self._ready)
        except Exception as error:
            self.close()
            self.fail(error)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.loop.remove_reader(self.fd)
        if self._continuation is not None:
            self._continuation.cancel()
            self._continuation = None


class AsyncClient:
    def __init__(self, client, response, fail):
        self.client = client
        self.response = response
        self.fail = fail
        self.pending = {}
        self.notification = AsyncNotification(
            client.notification_fd(), self._drain, self._fail
        )

    async def schedule(self, request, target_biz, timeout_ms):
        if self.notification.closed:
            raise RuntimeError("NaviSched async client is closed")
        future = asyncio.get_running_loop().create_future()
        request_id = request.request_id
        if request_id in self.pending:
            raise ValueError("duplicate pending NaviSched schedule")
        self.pending[request_id] = future
        try:
            self.client.submit_schedule(request, target_biz, timeout_ms=timeout_ms)
        except BaseException:
            self.pending.pop(request_id, None)
            future.cancel()
            raise
        try:
            return await asyncio.wait_for(future, timeout_ms / 1000)
        except BaseException:
            # Keep the cancelled future until native completion consumes its decision.
            self.client.abort_forward_request(request_id)
            raise

    def _drain(self):
        for request_id, future in tuple(self.pending.items()):
            try:
                decision = self.client.take_schedule_result(request_id)
            except Exception as error:
                self.pending.pop(request_id)
                if not future.done():
                    future.set_exception(error)
            else:
                if decision is not None:
                    self.pending.pop(request_id)
                    if not future.done():
                        future.set_result(decision)
        for _ in range(32):
            response = self.client.take_next_forward_response(0)
            if response is None:
                return False
            self.response(response)
        return True

    def _fail(self, error):
        for future in self.pending.values():
            if not future.done():
                future.set_exception(error)
        self.close()
        self.fail(error)

    def close(self):
        self.notification.close()
        for request_id, future in self.pending.items():
            future.cancel()
            self.client.abort_forward_request(request_id)
        self.pending.clear()
