# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve PACE operations on a worker's local BTC Host slots."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Thread
from time import time

from btc_kvcm import PaceHostTransfer, shard_spec

from btc_vllm.torchtpu.rank_rpc import RankServer


class PaceShardServer:
    """A rank's remote-data endpoint. All SDK calls use one worker thread."""

    def __init__(self, worker, description, address):
        self._rows = worker.host_rows
        self._specs = {
            group: shard_spec(group, description.rank.global_rank, size)
            for group, size in enumerate(self._rows.group_bytes)
            if size
        }
        self._ready = Future()
        assert worker.region is not None and worker.host_fd is not None
        with memoryview(worker.region) as memory:
            self._transfer = PaceHostTransfer(memory, worker.host_fd)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="btc-pace-io"
        )
        self._thread = Thread(
            target=self._run,
            args=(address, description.rank.global_rank),
            name="btc-pace-rpc",
            daemon=True,
        )
        self._thread.start()
        try:
            self.endpoint = self._ready.result()
        except BaseException:
            self.close()
            raise

    def _run(self, address, rank):
        async def serve():
            self._loop = asyncio.get_running_loop()
            self._stop = asyncio.Event()
            server = await RankServer.start(address, self._command, rank=rank)
            self._ready.set_result((address, server.port, server.token))
            try:
                await self._stop.wait()
            finally:
                await server.close()

        try:
            asyncio.run(serve())
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
            else:
                raise

    async def _command(self, command):
        return await asyncio.get_running_loop().run_in_executor(
            self._executor, self._execute, command
        )

    def _execute(self, command):
        match command["op"]:
            case "connect":
                self._transfer.connect(
                    command["storage_configs"],
                    command["options"],
                    next(iter(self._specs.values())).name,
                )
                return {"failed": False}
            case "read":
                entries = [
                    (uri, self._specs[group], self._rows.bounds(group, slot)[0], digests)
                    for group, slot, uri, digests in command["entries"]
                ]
                return {"failed": not self._transfer.read_many(entries)}
            case "write":
                if time() >= command["admit_until"]:
                    return {"failed": True, "digests": None}
                group = command["group"]
                digests = self._transfer.write_spec(
                    command["uri"], self._specs[group], self._rows.bounds(group, command["slot"])[0]
                )
                return {"failed": digests is None, "digests": digests}
            case _:
                raise ValueError(f"Unknown BTC PACE operation {command['op']!r}")

    def close(self):
        if (
            self._ready.done()
            and self._ready.exception() is None
            and self._thread.is_alive()
        ):
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join()
        try:
            self._executor.submit(self._transfer.close).result()
        finally:
            self._executor.shutdown(wait=True)
