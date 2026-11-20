"""Run the full native BTC tree with RTP-LLM's Torch and Python ABI."""

import json
import os
import socket
import socketserver
import sys

MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def _read(stream):
    line = stream.readline(MAX_MESSAGE_BYTES + 1)
    if not line:
        raise EOFError
    if len(line) > MAX_MESSAGE_BYTES or not line.endswith(b"\n"):
        raise ValueError("invalid BTC tree message length")
    return json.loads(line)


def _write(stream, message):
    encoded = json.dumps(message, separators=(",", ":")).encode()
    if len(encoded) >= MAX_MESSAGE_BYTES:
        raise ValueError("BTC tree message exceeds size limit")
    stream.write(encoded + b"\n")
    stream.flush()


class TreeCacheClient:
    """Keep one native TreeCache session per vLLM scheduler."""

    def __init__(self, groups: int, path: str) -> None:
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(30)
        try:
            self._socket.connect(path)
            self._stream = self._socket.makefile("rwb")
            self._call("create", groups)
        except Exception:
            self.close()
            raise

    def _call(self, operation: str, *args):
        _write(self._stream, (operation, args))
        success, result = _read(self._stream)
        if not success:
            raise RuntimeError(result)
        return result

    def configure_host(self, capacity: int) -> None:
        self._call("configure_host", capacity)

    def prepare_host_store(self, hashes, candidates, request_id: str = ""):
        plan = self._call(
            "prepare_host_store", [key.hex() for key in hashes], list(candidates), request_id
        )
        if plan is None:
            return None
        ticket, reservations, evicted = plan
        return (
            ticket,
            tuple(tuple(entry) for entry in reservations),
            tuple((group, bytes.fromhex(key), slot) for group, key, slot in evicted),
        )

    def settle_host_store(self, ticket: int, success: bool) -> bool:
        return self._call("settle_host_store", ticket, success)

    def pin_host_load(self, pages) -> int:
        return self._call("pin_host_load", list(pages))

    def release_host_load(self, ticket: int) -> bool:
        return self._call("release_host_load", ticket)

    def host_slot(self, group: int, hash_bytes: bytes) -> int | None:
        return self._call("host_slot", group, hash_bytes.hex())

    def reset_host(self) -> None:
        self._call("reset_host")

    def publish(self, keys, bindings) -> None:
        self._call("publish", [key.hex() for key in keys], list(bindings))

    def publish_device(self, keys, bindings) -> None:
        self._call("publish_device", [key.hex() for key in keys], list(bindings))

    def unpublish(self, group: int, tier: int, slot: int) -> None:
        self._call("unpublish", group, tier, slot)

    def unpublish_device(self, group: int, slot: int, generation: int) -> None:
        self._call("unpublish_device", group, slot, generation)

    def match(self, keys, endpoints):
        return self._call("match", [key.hex() for key in keys], list(endpoints))

    def match_with_join(self, keys, endpoints):
        return self._call("match_with_join", [key.hex() for key in keys], list(endpoints))

    def begin_device_load(self, keys, targets, operation_id) -> None:
        self._call("begin_device_load", [key.hex() for key in keys], list(targets), operation_id)

    def settle_device_load(self, operation_id: int, success: bool) -> bool:
        return self._call("settle_device_load", operation_id, success)

    def select_device_writebacks(self, eligible, limit):
        selected = self._call("select_device_writebacks", list(eligible), limit)
        return tuple(tuple(page) for page in selected)

    def plan_device_reclaim(self, eligible, limit):
        transaction, selected = self._call("plan_device_reclaim", list(eligible), limit)
        return transaction, [tuple(page) for page in selected]

    def settle_device_reclaim(self, transaction, released) -> None:
        self._call("settle_device_reclaim", transaction, list(released))

    def configure_disk(self, capacity: int, num_ranks: int, trace: bool) -> None:
        self._call("configure_disk", capacity, num_ranks, trace)

    def pin_disk_load(self, pages) -> int:
        return self._call("pin_disk_load", list(pages))

    def complete_disk_load(self, lease: int, success: bool) -> None:
        self._call("complete_disk_load", lease, success)

    def retry_disk_writes(self, host_pages, request_id: str) -> None:
        self._call("retry_disk_writes", list(host_pages), request_id)

    def take_disk_jobs(self) -> tuple[list[dict], list[dict]]:
        jobs, reclaims = self._call("take_disk_jobs")
        return jobs, reclaims

    def settle_disk_job(
        self, job_id: int, completions: int, failed: bool
    ) -> tuple[bool, list[dict]]:
        done, events = self._call("settle_disk_job", job_id, completions, failed)
        return done, events

    def disk_has_pending_work(self) -> bool:
        return self._call("disk_has_pending_work")

    def disk_reset_blocked(self) -> bool:
        return self._call("disk_reset_blocked")

    def reset_disk(self) -> None:
        self._call("reset_disk")

    def touch_disk(self, pages) -> None:
        self._call("touch_disk", list(pages))

    def reclaim_disk(self, protected, snapshot=False):
        victim, order = self._call("reclaim_disk", list(protected), snapshot)
        return (None if victim is None else tuple(victim)), [tuple(page) for page in order]

    def reset(self) -> None:
        self._call("reset")

    def close(self) -> None:
        try:
            if hasattr(self, "_stream"):
                self._stream.close()
        finally:
            self._socket.close()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        from btc_tree import TreeCache

        tree = None
        matches = 0
        while True:
            try:
                operation, args = _read(self.rfile)
            except (EOFError, ValueError):
                return
            try:
                result = None
                if operation == "stop":
                    _write(self.wfile, (True, None))
                    self.server.shutdown()
                    return
                if operation == "create" and tree is None:
                    (groups,) = args
                    tree = TreeCache(groups)
                elif tree is None:
                    raise ValueError("BTC tree session is not initialized")
                elif operation in ("configure_host", "reset_host"):
                    getattr(tree, operation)(*args)
                elif operation == "prepare_host_store":
                    hashes, candidates, request_id = args
                    plan = tree.prepare_host_store(
                        [bytes.fromhex(key) for key in hashes], candidates, request_id
                    )
                    if plan is not None:
                        ticket, reservations, evicted = plan
                        result = (
                            ticket,
                            reservations,
                            [(group, key.hex(), slot) for group, key, slot in evicted],
                        )
                elif operation in ("settle_host_store", "pin_host_load", "release_host_load"):
                    result = getattr(tree, operation)(*args)
                elif operation == "host_slot":
                    group, key = args
                    result = tree.host_slot(group, bytes.fromhex(key))
                elif operation == "publish":
                    keys, bindings = args
                    tree.publish([bytes.fromhex(key) for key in keys], bindings)
                elif operation == "publish_device":
                    keys, bindings = args
                    tree.publish_device([bytes.fromhex(key) for key in keys], bindings)
                elif operation == "unpublish":
                    tree.unpublish(*args)
                elif operation == "unpublish_device":
                    tree.unpublish_device(*args)
                elif operation in ("match", "match_with_join"):
                    keys, endpoints = args
                    match = tree.match_with_join if operation == "match_with_join" else tree.match
                    result = match([bytes.fromhex(key) for key in keys], endpoints)
                    matches += 1
                    print(
                        f"BTC_NATIVE_MATCH call={matches} device={result[0]} "
                        f"matched={result[1]} pages={len(result[3])}",
                        flush=True,
                    )
                    if operation == "match_with_join":
                        joined = [page for page in result[3] if page[2] == 1]
                        self._trace_join("match", matches, joined)
                elif operation == "begin_device_load":
                    keys, targets, operation_id = args
                    tree.begin_device_load([bytes.fromhex(key) for key in keys], targets, operation_id)
                    self._trace_join("begin", operation_id, targets)
                elif operation == "settle_device_load":
                    operation_id, success = args
                    result = tree.settle_device_load(operation_id, success)
                    self._trace_join("settle", operation_id, {"success": success, "accepted": result})
                elif operation == "select_device_writebacks":
                    result = tree.select_device_writebacks(*args)
                elif operation == "plan_device_reclaim":
                    result = tree.plan_device_reclaim(*args)
                    transaction, pages = result
                    if transaction:
                        self._trace_reclaim("plan", transaction, pages, matches)
                elif operation == "settle_device_reclaim":
                    transaction, pages = args
                    tree.settle_device_reclaim(transaction, pages)
                    self._trace_reclaim("settle", transaction, pages, matches)
                elif operation in (
                    "configure_disk",
                    "pin_disk_load",
                    "complete_disk_load",
                    "retry_disk_writes",
                    "take_disk_jobs",
                    "settle_disk_job",
                    "disk_has_pending_work",
                    "disk_reset_blocked",
                    "reset_disk",
                ):
                    result = getattr(tree, operation)(*args)
                    if operation == "take_disk_jobs":
                        for event in result[1]:
                            victim = event["victim"]
                            if victim is not None:
                                selected = [victim["group"], victim["source_id"]]
                                print(
                                    f"BTC_NATIVE_DISK_RECLAIM victim={json.dumps(selected, separators=(',', ':'))}",
                                    flush=True,
                                )
                elif operation == "touch_disk":
                    tree.touch_disk(*args)
                elif operation == "reclaim_disk":
                    result = tree.reclaim_disk(*args)
                    print(f"BTC_NATIVE_DISK_RECLAIM victim={json.dumps(result[0], separators=(',', ':'))}", flush=True)
                elif operation == "reset":
                    tree.reset()
                else:
                    raise ValueError(f"Unknown BTC tree operation {operation!r}")
            except Exception as error:
                _write(self.wfile, (False, f"{type(error).__name__}: {error}"))
            else:
                _write(self.wfile, (True, result))

    @staticmethod
    def _trace_join(event, identifier, payload) -> None:
        encoded = json.dumps(payload, separators=(",", ":"))
        print(f"BTC_NATIVE_JOIN event={event} id={identifier} data={encoded}", flush=True)

    @staticmethod
    def _trace_reclaim(operation, transaction, pages, matches) -> None:
        encoded = json.dumps(pages, separators=(",", ":"))
        print(
            f"BTC_NATIVE_GPU_RECLAIM operation={operation} txn={transaction} "
            f"pages={encoded} match_calls={matches}",
            flush=True,
        )


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def serve(path: str) -> None:
    from btc_tree import TreeCache

    probe = TreeCache(1)
    assert probe.match([], [])[:2] == (0, 0)
    del probe
    old_mask = os.umask(0o177)
    try:
        server = _Server(path, _Handler)
    finally:
        os.umask(old_mask)
    try:
        with server:
            print(f"BTC tree service ready at {path}", flush=True)
            server.serve_forever(poll_interval=0.1)
    finally:
        os.unlink(path)


def stop(path: str) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(5)
        channel.connect(path)
        with channel.makefile("rwb") as stream:
            _write(stream, ("stop", ()))
            assert _read(stream)[0]


if __name__ == "__main__":
    mode, path = sys.argv[1:]
    if mode == "serve":
        serve(path)
    elif mode == "stop":
        stop(path)
    else:
        raise SystemExit("Expected serve or stop")
