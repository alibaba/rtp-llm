"""Synchronized decode rounds across node-local frontends."""

from __future__ import annotations

import copy
import json
import selectors
import socket
import time
from typing import Any, Callable, Mapping, Sequence

from rtp_llm.test.perf_test.decode.scheduler import scheduler_local_batch_size

_CANCEL_POLL_SECONDS = 1.0


def _wait_socket(
    sock: socket.socket,
    events: int,
    deadline: float,
    check_cancelled: Callable[[], None],
) -> None:
    with selectors.DefaultSelector() as selector:
        selector.register(sock, events)
        while True:
            check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("decode control operation timed out")
            if selector.select(min(_CANCEL_POLL_SECONDS, remaining)):
                check_cancelled()
                return


class _ControlStream:
    """Line framing with cancellable socket waits, including partial messages."""

    def __init__(
        self,
        sock: socket.socket,
        timeout_seconds: float,
        check_cancelled: Callable[[], None],
    ) -> None:
        self._sock = sock
        self._timeout = timeout_seconds
        self._check_cancelled = check_cancelled
        self._buffer = bytearray()
        sock.setblocking(False)

    def readline(self) -> str:
        self._check_cancelled()
        deadline = time.monotonic() + self._timeout
        while b"\n" not in self._buffer:
            _wait_socket(
                self._sock, selectors.EVENT_READ, deadline, self._check_cancelled
            )
            try:
                chunk = self._sock.recv(65536)
            except BlockingIOError:
                continue
            if not chunk:
                raise RuntimeError("decode control peer disconnected")
            self._buffer.extend(chunk)
        line, _, rest = self._buffer.partition(b"\n")
        self._buffer = bytearray(rest)
        return line.decode("utf-8") + "\n"

    def write(self, value: str) -> None:
        pending = memoryview(value.encode("utf-8"))
        deadline = time.monotonic() + self._timeout
        while pending:
            _wait_socket(
                self._sock, selectors.EVENT_WRITE, deadline, self._check_cancelled
            )
            try:
                sent = self._sock.send(pending)
            except BlockingIOError:
                continue
            if not sent:
                raise RuntimeError("decode control peer disconnected")
            pending = pending[sent:]

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._sock.close()


def _send(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(json.dumps(value, separators=(",", ":")) + "\n")
    stream.flush()


def _receive(stream: Any) -> dict[str, Any]:
    line = stream.readline()
    if not line:
        raise RuntimeError("decode control peer disconnected")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise RuntimeError("decode control message must be an object")
    return value


def validate_decode_layout(
    local_world_size: int, tp_size: int, dp_size: int, ranks: Sequence[int]
) -> int:
    if tp_size < 1 or local_world_size < 1 or local_world_size % tp_size:
        raise ValueError("distributed decode requires whole local TP groups")
    local_dp = local_world_size // tp_size
    if dp_size != local_dp * len(ranks) or tuple(ranks) != tuple(
        index * local_world_size for index in range(len(ranks))
    ):
        raise ValueError("distributed decode DP layout does not match member ranks")
    return local_dp


class ClusterDecodeTransport:
    """Master-side transport with the same operations as DecodeHttpTransport."""

    def __init__(
        self,
        local_transport: Any,
        *,
        bind_host: str,
        control_port: int,
        local_world_size: int,
        tp_size: int,
        dp_size: int,
        member_world_ranks: Sequence[int],
        timeout_seconds: float = 3600,
        check_cancelled: Callable[[], None] = lambda: None,
    ) -> None:
        self._ranks = tuple(member_world_ranks)
        self._local_dp = validate_decode_layout(
            local_world_size, tp_size, dp_size, self._ranks
        )
        self._local = local_transport
        self._timeout = timeout_seconds
        self._check_cancelled = check_cancelled
        self._closed = False
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((bind_host, control_port))
        self._server.listen(len(self._ranks))
        self._server.setblocking(False)
        self.port = int(self._server.getsockname()[1])
        self._peers: dict[int, tuple[socket.socket, Any]] = {}
        self._allocations: tuple[dict[str, Any], ...] = ()
        self._sequence = 0

    def connect_workers(self) -> None:
        expected = set(self._ranks) - {0}
        deadline = time.monotonic() + self._timeout
        try:
            self._ensure_open()
            while set(self._peers) != expected:
                _wait_socket(
                    self._server, selectors.EVENT_READ, deadline, self._check_cancelled
                )
                sock, _address = self._server.accept()
                stream = _ControlStream(sock, self._timeout, self._check_cancelled)
                try:
                    hello = _receive(stream)
                    rank = hello.get("world_rank")
                    if rank not in expected or rank in self._peers:
                        raise RuntimeError(f"unexpected decode worker rank {rank!r}")
                    self._peers[rank] = (sock, stream)
                    _send(stream, {"status": "ready"})
                except BaseException:
                    stream.close()
                    raise
        except BaseException:
            self._abort()
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("decode control channel is closed")
        self._check_cancelled()

    def _abort(self) -> None:
        self._closed = True
        for sock, stream in self._peers.values():
            try:
                stream.close()
            finally:
                sock.close()
        self._peers.clear()
        self._server.close()

    def _command(
        self, operation: str, payloads: Mapping[int, Mapping[str, Any]]
    ) -> None:
        try:
            self._ensure_open()
            self._sequence += 1
            for rank, (_sock, stream) in self._peers.items():
                _send(
                    stream,
                    {
                        "sequence": self._sequence,
                        "operation": operation,
                        **payloads[rank],
                    },
                )
        except BaseException:
            self._abort()
            raise

    def _responses(self) -> dict[int, dict[str, Any]]:
        responses = {}
        deadline = time.monotonic() + self._timeout
        try:
            self._ensure_open()
            with selectors.DefaultSelector() as selector:
                for rank, (sock, _stream) in self._peers.items():
                    selector.register(sock, selectors.EVENT_READ, rank)
                while selector.get_map():
                    self._check_cancelled()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("decode worker response timed out")
                    ready = selector.select(min(_CANCEL_POLL_SECONDS, remaining))
                    for key, _events in ready:
                        rank = key.data
                        response = _receive(self._peers[rank][1])
                        if response.get("sequence") != self._sequence:
                            raise RuntimeError(
                                f"decode worker {rank} returned an out-of-order reply"
                            )
                        if response.get("status") != "ok":
                            raise RuntimeError(
                                f"decode worker {rank} failed: "
                                f"{response.get('error', 'unknown error')}"
                            )
                        responses[rank] = response
                        selector.unregister(key.fileobj)
        except BaseException:
            # A failed or incomplete round cannot be reused by the next case.
            self._abort()
            raise
        return responses

    def _execute_local(
        self, operation: Callable[[], Any]
    ) -> tuple[Any, dict[int, dict[str, Any]]]:
        try:
            result = operation()
        except Exception:
            # Workers already received this sequence. Drain their replies before
            # the caller can catch the local error and configure another case.
            self._responses()
            raise
        except BaseException:
            self._abort()
            raise
        return result, self._responses()

    def _node_allocations(self, rank: int) -> tuple[dict[str, Any], ...]:
        node_index = self._ranks.index(rank)
        start = node_index * self._local_dp
        return self._allocations[start : start + self._local_dp]

    def configure_scheduler(
        self, worker_allocations: Sequence[Mapping[str, Any]]
    ) -> None:
        self._allocations = tuple(
            copy.deepcopy(dict(item)) for item in worker_allocations
        )
        if len(self._allocations) != len(self._ranks) * self._local_dp:
            raise ValueError("decode allocation count does not match DP layout")
        scheduler_local_batch_size(
            tuple(item["local_batch_size"] for item in self._allocations)
        )
        self._command(
            "configure",
            {
                rank: {"allocations": self._node_allocations(rank)}
                for rank in self._peers
            },
        )
        local = self._node_allocations(0)
        self._execute_local(
            lambda: (
                self._local.configure_scheduler(local)
                if any(item["local_batch_size"] for item in local)
                else None
            )
        )

    def run_round(
        self,
        *,
        input_lengths: Sequence[int],
        output_length: int,
        generate_config: Mapping[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        lengths = tuple(input_lengths)
        if not self._allocations or len(lengths) != sum(
            len(item["request_indices"]) for item in self._allocations
        ):
            raise ValueError("decode round does not match configured global allocation")
        indices = {
            rank: tuple(
                index
                for allocation in self._node_allocations(rank)
                for index in allocation["request_indices"]
            )
            for rank in self._ranks
        }
        self._command(
            "round",
            {
                rank: {
                    "input_lengths": [lengths[index] for index in indices[rank]],
                    "output_length": output_length,
                    "generate_config": dict(generate_config),
                }
                for rank in self._peers
            },
        )
        local_results, responses = self._execute_local(
            lambda: (
                self._local.run_round(
                    input_lengths=tuple(lengths[index] for index in indices[0]),
                    output_length=output_length,
                    generate_config=generate_config,
                )
                if indices[0]
                else ()
            )
        )
        by_rank = {0: local_results}
        by_rank.update({rank: reply["requests"] for rank, reply in responses.items()})
        outcomes: list[dict[str, Any]] = []
        for rank in self._ranks:
            requests = by_rank[rank]
            if len(requests) != len(indices[rank]):
                raise RuntimeError(f"decode worker {rank} returned wrong request count")
            for local_index, request in enumerate(requests):
                if request["request_index"] != local_index:
                    raise RuntimeError(
                        f"decode worker {rank} returned wrong request index"
                    )
                outcomes.append(
                    {**request, "request_index": indices[rank][local_index]}
                )
        return tuple(sorted(outcomes, key=lambda item: item["request_index"]))

    def start_profile(
        self, *, trace_name: str, steps: int, enable_all_ranks: bool
    ) -> None:
        payload = {
            "trace_name": trace_name,
            "steps": steps,
            "enable_all_ranks": enable_all_ranks,
        }
        self._command("profile", {rank: payload for rank in self._peers})
        self._execute_local(lambda: self._local.start_profile(**payload))

    def close(self) -> None:
        try:
            if not self._closed and self._peers:
                self._command("close", {rank: {} for rank in self._peers})
        except (OSError, RuntimeError):
            pass
        finally:
            self._abort()


def serve_decode_worker(
    master_host: str,
    control_port: int,
    *,
    world_rank: int,
    local_world_size: int,
    tp_size: int,
    dp_size: int,
    local_transport: Any,
    timeout_seconds: float = 3600,
    check_cancelled: Callable[[], None] = lambda: None,
) -> None:
    """Serve master commands until the master closes the synchronized run."""
    if world_rank == 0:
        raise ValueError("decode worker must have a nonzero world rank")
    if tp_size < 1 or local_world_size < 1 or local_world_size % tp_size:
        raise ValueError("distributed decode worker has an invalid TP layout")
    local_dp = local_world_size // tp_size
    if dp_size < 1 or dp_size % local_dp:
        raise ValueError("distributed decode worker has an invalid DP layout")
    validate_decode_layout(
        local_world_size,
        tp_size,
        dp_size,
        tuple(index * local_world_size for index in range(dp_size // local_dp)),
    )
    if world_rank not in range(
        local_world_size, dp_size // local_dp * local_world_size, local_world_size
    ):
        raise ValueError("decode worker rank is outside the DP layout")
    deadline = time.monotonic() + timeout_seconds
    while True:
        check_cancelled()
        try:
            sock = socket.create_connection(
                (master_host, control_port),
                timeout=min(
                    _CANCEL_POLL_SECONDS, max(0.001, deadline - time.monotonic())
                ),
            )
            break
        except OSError:
            check_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("decode worker could not connect to master")
            time.sleep(min(_CANCEL_POLL_SECONDS, remaining))
    with sock:
        stream = _ControlStream(sock, timeout_seconds, check_cancelled)
        try:
            _send(stream, {"world_rank": world_rank})
            if _receive(stream).get("status") != "ready":
                raise RuntimeError("decode master rejected worker")
            while True:
                command = _receive(stream)
                sequence = command.get("sequence")
                operation = command.get("operation")
                if operation == "close":
                    return
                try:
                    if operation == "configure":
                        if any(
                            item["local_batch_size"] for item in command["allocations"]
                        ):
                            local_transport.configure_scheduler(command["allocations"])
                        result = {}
                    elif operation == "round":
                        result = {
                            "requests": (
                                local_transport.run_round(
                                    input_lengths=command["input_lengths"],
                                    output_length=command["output_length"],
                                    generate_config=command["generate_config"],
                                )
                                if command["input_lengths"]
                                else ()
                            )
                        }
                    elif operation == "profile":
                        local_transport.start_profile(
                            trace_name=command["trace_name"],
                            steps=command["steps"],
                            enable_all_ranks=command["enable_all_ranks"],
                        )
                        result = {}
                    else:
                        raise RuntimeError(f"unknown decode operation {operation!r}")
                    _send(stream, {"sequence": sequence, "status": "ok", **result})
                except Exception as error:
                    _send(
                        stream,
                        {"sequence": sequence, "status": "error", "error": str(error)},
                    )
                    raise
        finally:
            stream.close()
