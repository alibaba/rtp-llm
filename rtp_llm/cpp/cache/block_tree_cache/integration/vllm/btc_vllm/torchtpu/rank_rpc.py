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

"""Serial rank commands with at-most-once admission across reconnects.

The framing and replay rules follow kvs_connector.load_rpc. BTC owns this
endpoint and does not need that module's request queue or load metrics.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from vllm_torchtpu.logger import init_logger

RankEndpoint = tuple[str, int, str]
Message = dict[str, Any]
_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_MAX_RECONNECTS = 3
logger = init_logger(__name__)


async def _read(reader: asyncio.StreamReader, first_byte: bytes = b"") -> Message:
    size = int.from_bytes(
        first_byte + await reader.readexactly(4 - len(first_byte)), "big"
    )
    if not 0 < size <= _MAX_MESSAGE_BYTES:
        raise ValueError("Invalid BTC rank RPC frame size")
    value = json.loads(await reader.readexactly(size))
    if not isinstance(value, dict):
        raise ValueError("BTC rank RPC frame must be a JSON object")
    return value


async def _write(writer: asyncio.StreamWriter, message: Message) -> None:
    body = json.dumps(message).encode()
    if not 0 < len(body) <= _MAX_MESSAGE_BYTES:
        raise ValueError("Invalid BTC rank RPC frame size")
    writer.write(len(body).to_bytes(4, "big") + body)
    await writer.drain()


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with suppress(ConnectionError):
        await writer.wait_closed()


class RankServer:
    def __init__(self, run: Callable[[Message], Awaitable[Message]], rank: int) -> None:
        self.token = secrets.token_hex(32)
        self._run = run
        self._rank = rank
        self._handlers: set[asyncio.Task[None]] = set()
        self._idle: set[asyncio.StreamWriter] = set()
        self._closing = False
        self._session: str | None = None
        self._sequence = 0
        self._command: Message | None = None
        self._operation: asyncio.Task[Message] | None = None

    @classmethod
    async def start(
        cls,
        host: str,
        run: Callable[[Message], Awaitable[Message]],
        *,
        rank: int,
    ) -> RankServer:
        endpoint = cls(run, rank)
        endpoint._server = await asyncio.start_server(endpoint._accept, host, 0)
        endpoint.port = int(endpoint._server.sockets[0].getsockname()[1])
        return endpoint

    def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._closing:
            writer.close()
            return
        task = asyncio.create_task(self._handle(reader, writer))
        self._handlers.add(task)
        task.add_done_callback(self._handlers.discard)

    def _resume(self, message: Message) -> asyncio.Task[Message]:
        token = message.get("token")
        if not isinstance(token, str) or not secrets.compare_digest(token, self.token):
            raise PermissionError("Invalid BTC rank RPC token")
        session, sequence, command = (
            message.get("session"),
            message.get("sequence"),
            message.get("command"),
        )
        if not isinstance(session, str) or len(session) != 32:
            raise ValueError("Invalid BTC rank RPC session")
        if type(sequence) is not int or sequence < 1 or not isinstance(command, dict):
            raise ValueError("Invalid BTC rank RPC sequence or command")
        if self._session is not None and session != self._session:
            raise ValueError("BTC rank RPC endpoint already has a scheduler")
        if sequence == self._sequence:
            if command != self._command or self._operation is None:
                raise ValueError("Conflicting BTC rank RPC replay")
            return self._operation
        if sequence != self._sequence + 1:
            raise ValueError("Stale or out-of-order BTC rank RPC sequence")
        if self._operation is not None and (
            not self._operation.done() or "fatal" in self._operation.result()
        ):
            raise ValueError("Previous BTC rank RPC did not complete safely")
        self._session, self._sequence, self._command = session, sequence, command
        self._operation = asyncio.create_task(self._execute(command))
        return self._operation

    async def _execute(self, command: Message) -> Message:
        try:
            return await self._run(command)
        except Exception as error:
            logger.exception("BTC rank %d command failed", self._rank)
            return {"fatal": str(error)}

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            while not self._closing:
                self._idle.add(writer)
                first = await reader.read(1)
                if not first:
                    break
                try:
                    async with asyncio.timeout(10):
                        message = await _read(reader, first)
                    if self._closing:
                        break
                    self._idle.discard(writer)
                    # A lost connection does not put an admitted Host slot back.
                    result = await asyncio.shield(self._resume(message))
                    reply = {**result, "sequence": message["sequence"]}
                except Exception as error:
                    reply = {"fatal": str(error)}
                self._idle.discard(writer)
                async with asyncio.timeout(10):
                    await _write(writer, reply)
                if "fatal" in reply:
                    break
        except (OSError, TimeoutError):
            writer.transport.abort()
        finally:
            self._idle.discard(writer)
            await _close(writer)

    async def close(self) -> None:
        self._closing = True
        self._server.close()
        for writer in tuple(self._idle):
            writer.transport.abort()
        # Drain admitted commands before the caller closes registered memory.
        await asyncio.gather(*tuple(self._handlers))
        await self._server.wait_closed()


class RankClient:
    """One connection and at most one admitted command per rank."""

    def __init__(self, endpoint: RankEndpoint, rank: int) -> None:
        self._endpoint = endpoint
        self._rank = rank
        self._session = secrets.token_hex(16)
        self._sequence = 0
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._failed = False

    async def call(self, command: Message, timeout: float) -> Message:
        if self._failed:
            raise RuntimeError("Cannot reuse a failed BTC rank RPC connection")
        host, port, token = self._endpoint
        self._sequence += 1
        message = {
            "token": token,
            "session": self._session,
            "sequence": self._sequence,
            "command": command,
        }
        try:
            async with asyncio.timeout(timeout):
                reply = await self._exchange(host, port, message)
                if "fatal" in reply:
                    raise RuntimeError(f"BTC rank {self._rank}: {reply['fatal']}")
                sequence = reply.pop("sequence", None)
                if type(sequence) is not int or sequence != self._sequence:
                    raise ValueError("BTC rank RPC reply does not match request")
                if type(reply.get("failed")) is not bool:
                    raise ValueError("BTC rank RPC reply needs a completion status")
                return reply
        except (Exception, asyncio.CancelledError):
            self._failed = True
            if self._writer is not None:
                self._writer.transport.abort()
            raise

    async def _exchange(self, host: str, port: int, message: Message) -> Message:
        for attempt in range(_MAX_RECONNECTS + 1):
            try:
                if self._writer is None:
                    async with asyncio.timeout(5):
                        self._reader, self._writer = await asyncio.open_connection(
                            host, port
                        )
                assert self._reader is not None
                await _write(self._writer, message)
                return await _read(self._reader)
            except (OSError, asyncio.IncompleteReadError):
                if attempt == _MAX_RECONNECTS:
                    raise
                if self._writer is not None:
                    self._writer.transport.abort()
                await self.close()
                await asyncio.sleep(0.1 * (2**attempt))
        raise AssertionError("Unreachable BTC rank RPC reconnect loop")

    async def close(self) -> None:
        if self._writer is not None:
            await _close(self._writer)
        self._reader = self._writer = None
