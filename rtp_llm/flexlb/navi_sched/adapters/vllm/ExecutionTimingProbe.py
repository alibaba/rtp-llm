"""Record distinct native Agent-label and backend-utilization clock brackets."""

import math
import time
from functools import wraps

_TPU_COMPILE_SNAPSHOT_JOB = "[navi-tpu-compile-snapshot-job]"
_TPU_COMPILE_SLOW_OUTPUT_NS = 500000000
_tpu_compile_baseline_pid = None


class _AsyncDiagnosticEmitter:
    """Bounded diagnostic delivery; producers never acquire a logging/I/O lock.

    A pool of deque tokens bounds CPython's reentrant SimpleQueue. Its atomic
    popleft/append operations require no producer mutex, including in GC callbacks.
    Exhaustion loses a diagnostic and records that loss; it never waits or logs.
    State writes happen only on the delivery thread, in the owned Pod directory.
    """

    def __init__(self, state_dir, capacity=4096):
        import itertools
        import os
        import queue
        import threading
        from collections import deque
        from pathlib import Path

        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Invalid diagnostic queue capacity")
        self.pid = os.getpid()
        self.capacity = capacity
        self.queue = queue.SimpleQueue()
        self.slots = deque([None] * capacity)
        self.sequence = itertools.count()
        self.loss_sequence = itertools.count()
        self.sequence_state_reads = self.loss_state_reads = 0
        self.delivered = self.failed = 0
        self.loss_observed = self.closed = self.state_write_failed = False
        self.last_delivery_ns = self.max_delivery_ns = 0
        self.writer_started_perf_ns = self.writer_finished_perf_ns = 0
        self.writer_native_tid = None
        stat = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        self.start_ticks = stat[19]
        self.state_path = Path(state_dir) / ("async-timing-audit-%s.json" % self.pid)
        self.thread = threading.Thread(
            target=self._deliver, name="navi-audit-log", daemon=True
        )
        self.thread.start()

    def submit(self, logger, message, args):
        import os

        # The already serialized strings cannot retain request/tensor objects.
        if (
            os.getpid() != self.pid
            or self.closed
            or type(message) is not str
            or type(args) is not tuple
            or not all(type(arg) is str for arg in args)
            or len(message) + sum(len(arg) for arg in args) > 65536
        ):
            self._lost()
            return False
        try:
            self.slots.popleft()
        except IndexError:
            self._lost()
            return False
        try:
            next(self.sequence)
            self.queue.put((logger, message, args))
            return True
        except Exception:
            self.slots.append(None)
            self._lost()
            return False

    def _lost(self):
        next(self.loss_sequence)
        self.loss_observed = True

    def _save_state(self):
        import json

        try:
            # A snapshot takes one ticket and subtracts earlier snapshot tickets.
            # Producers only call C-level next(): no mutex, read/modify/write
            # counter race, private iterator inspection or deprecated pickling.
            enqueued = next(self.sequence) - self.sequence_state_reads
            self.sequence_state_reads += 1
            losses = next(self.loss_sequence) - self.loss_state_reads
            self.loss_state_reads += 1
            state = dict(
                version=1,
                pid=self.pid,
                start_ticks=self.start_ticks,
                writer_native_tid=self.writer_native_tid,
                capacity=self.capacity,
                pending=self.capacity - len(self.slots),
                enqueued=enqueued,
                delivered=self.delivered,
                failed=self.failed,
                loss_count=losses,
                loss_observed=self.loss_observed,
                unprepared_loss_observed=_diagnostic_unprepared_loss,
                state_write_failed=self.state_write_failed,
                last_delivery_ns=self.last_delivery_ns,
                max_delivery_ns=self.max_delivery_ns,
                writer_started_perf_ns=self.writer_started_perf_ns,
                writer_finished_perf_ns=self.writer_finished_perf_ns,
                event_unix_ns=time.time_ns(),
            )
            temporary = self.state_path.with_suffix(".json.tmp")
            with temporary.open("w") as stream:
                import os

                os.fchmod(stream.fileno(), 0o600)
                # The one-shot C encoder avoids the recursive closure cycles
                # made by json.dump's streaming Python encoder. Repeating those
                # cycles here can trigger generation-2 GC while holding the GIL.
                stream.write(json.dumps(state, separators=(",", ":"), sort_keys=True))
            temporary.replace(self.state_path)
        except Exception:
            self.state_write_failed = True

    def _deliver(self):
        import threading

        self.writer_native_tid = threading.get_native_id()
        self._save_state()
        while True:
            item = self.queue.get()
            if item is None:
                self._save_state()
                return
            logger, message, args = item
            self.writer_started_perf_ns = time.perf_counter_ns()
            self._save_state()
            succeeded = False
            try:
                if message == _TPU_COMPILE_SNAPSHOT_JOB:
                    if len(args) != 1:
                        raise ValueError("Invalid owned TPU snapshot job")
                    _deliver_tpu_compile_snapshot(logger, args[0])
                else:
                    logger.info(message, *args)
                succeeded = True
            except Exception:
                self._lost()
            finally:
                self.writer_finished_perf_ns = time.perf_counter_ns()
                self.last_delivery_ns = max(
                    0, self.writer_finished_perf_ns - self.writer_started_perf_ns
                )
                self.max_delivery_ns = max(self.max_delivery_ns, self.last_delivery_ns)
                if succeeded:
                    self.delivered += 1
                else:
                    self.failed += 1
                self.slots.append(None)
                self._save_state()

    def close(self):
        # Test/shutdown control only; never flush/join from an inference call.
        self.closed = True
        self.queue.put(None)


_diagnostic_emitter = None
_diagnostic_unprepared_loss = False


def prepare_diagnostic_logging():
    """Prepare the process-owned writer during observer construction only."""
    global _diagnostic_emitter
    import os
    from pathlib import Path

    if (
        _diagnostic_emitter is not None
        and _diagnostic_emitter.pid == os.getpid()
        and not _diagnostic_emitter.closed
    ):
        return True
    try:
        owned = Path("/workspace/.navi_sched")
        if not owned.is_dir():
            return False
        directory = owned / "diagnostics"
        directory.mkdir(mode=0o700, exist_ok=True)
        directory.resolve().relative_to(owned.resolve())
        _diagnostic_emitter = _AsyncDiagnosticEmitter(directory)
        return True
    except Exception:
        return False


def _diagnostic_info(logger, message, *args):
    global _diagnostic_unprepared_loss
    emitter = _diagnostic_emitter
    if emitter is None:
        _diagnostic_unprepared_loss = True
        return False
    try:
        return emitter.submit(logger, message, args)
    except Exception:
        _diagnostic_unprepared_loss = True
        return False


class _DiagnosticLoggerProxy:
    def __init__(self, logger):
        self.logger = logger

    def info(self, message, *args):
        return _diagnostic_info(self.logger, message, *args)


_KV_PUBLISH_DIAGNOSTIC = (
    "kv_publish_event_record req_ids=%s unix_ns=%d duration_ms=%.3f"
)
_KV_PUBLISH_WAIT_DIAGNOSTIC = (
    "kv_publish_event_wait req_ids=%s unix_ns=%d duration_ms=%.3f"
)


class _RuntimeKvPublishDiagnosticLogger:
    """Offload the qualified runtime's timing line, preserving other logging."""

    def __init__(self, logger, message=_KV_PUBLISH_DIAGNOSTIC):
        import os

        if message not in (_KV_PUBLISH_DIAGNOSTIC, _KV_PUBLISH_WAIT_DIAGNOSTIC):
            raise ValueError("Unqualified runtime timing message")
        self.message = message
        # r64 returns _LazyVllmLogger, not logging.Logger. Resolve its required
        # interface now, before inference, while retaining the original wrapper.
        if not all(
            callable(getattr(logger, name, None)) for name in ("info", "isEnabledFor")
        ):
            raise TypeError("Qualified runtime logger interface is incomplete")
        self.logger, self.pid = logger, os.getpid()

    def __getattr__(self, name):
        return getattr(self.logger, name)

    def info(self, message, *args, **kwargs):
        if type(message) is not str or message != self.message:
            kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
            return self.logger.info(message, *args, **kwargs)
        import logging

        if kwargs:
            raise ValueError("Unqualified runtime timing log signature")
        if self.logger.isEnabledFor(logging.INFO):
            # Freeze the request list and producer timestamps before returning.
            # The queue only retains strings, never the mutable runtime list.
            _diagnostic_info(self.logger, "%s", message % args)


def _diagnostic_format_info(logger, message, *args):
    """Serialize primitive diagnostics before bounded delivery, without sink I/O."""
    global _diagnostic_unprepared_loss
    try:
        serialized = message % args if args else message
    except Exception:
        _diagnostic_unprepared_loss = True
        return False
    return _diagnostic_info(logger, serialized)


def _sampling_runtime_log_module(runner):
    method = inspect.unwrap(type(runner).sample_tokens)
    module = inspect.getmodule(method)
    if (
        module is None
        or module.__name__ != "vllm_torchtpu.runner.tpu_runner"
        or Path(module.__file__).resolve()
        != Path(inspect.getsourcefile(method)).resolve()
    ):
        raise ValueError("Unqualified runtime timing logger module")
    return module


_STAGE3_SEND_DIAGNOSTIC = (
    "TPURaidenConnectorScheduler Stage-3 send req_id=%s uuid=%d "
    "num_tokens=%d source_pages=%d source_controller=%s"
)
_V6D_COMPONENT_SOURCE_DIGESTS = (
    "3272b5e999be3e3126a4792eee9e8841b975bf70f74e85fe0da98e590554435f",
    "64bc9a02ef972d0427ab03493443d2a3646fc664fdf28307ea171a397b00c89c",
    "6bf76747eb12cc6f32565d0efab8400fa6baa19e286d46c6faaf15c7e0c90ba2",
    "73f79db216c99cf591b2214786b7fb29d325bee0ddc84bfc61ac1392fbff7d8e",
    "dc31c1071164352f8074aff21c60e33c27b41ebc56b63c2109db5bb35aea8ddb",
    "52299c37d58e7109468f932959e40d49cc3f8c51ee6a93655b14b8be8a11fb62",
)


_PIPELINE_DIAGNOSTIC_SPECS = (
    (
        "vllm_torchtpu.distributed.kv_transfer.tpu_connector",
        "logger",
        "a0c4c9c396eac3fac596e93e13f0372be8d31d1caa4ed78885abcbaae4cae2d8",
        _STAGE3_SEND_DIAGNOSTIC,
        False,
    ),
    (
        "kvs_connector.v6d_object_connector",
        "logger",
        _V6D_COMPONENT_SOURCE_DIGESTS[0],
        "[build_connector_meta] meta: ",
        True,
    ),
    (
        "v6d.client.client",
        "print",
        "6818417414adb2734d67baf497fcf8e6d80ebe2af4ba97fe7f89962e94bde01b",
        "[CLIENT_ACQUIRE_PROFILE] ",
        True,
    ),
    (
        "v6d.common.rpc.client",
        "print",
        "61281a4604473e45f133400a35b2b5689386561908591ec9a001c581049b122a",
        "[RPC_CALL_PROFILE] endpoint=acquire ",
        True,
    ),
)


class _RuntimePipelineDiagnosticLogger:
    """Only qualified diagnostic INFO lines leave the original caller thread."""

    def __init__(self, original, message, prefix):
        import os

        if not all(
            callable(getattr(original, n, None)) for n in ("info", "isEnabledFor")
        ):
            raise TypeError("Qualified runtime logger interface is incomplete")
        self.original, self.message, self.prefix = original, message, prefix
        self.pid = os.getpid()

    def __getattr__(self, name):
        return getattr(self.original, name)

    def info(self, message, *args, **kwargs):
        import logging
        import os

        matched = type(message) is str and (
            message.startswith(self.message) if self.prefix else message == self.message
        )
        if os.getpid() != self.pid or not matched or kwargs:
            kwargs["stacklevel"] = kwargs.get("stacklevel", 1) + 1
            return self.original.info(message, *args, **kwargs)
        if self.original.isEnabledFor(logging.INFO):
            _diagnostic_format_info(self.original, message, *args)
        return None


class _RuntimePipelineDiagnosticPrint:
    def __init__(self, original, message):
        import os

        self.original, self.message, self.pid = original, message, os.getpid()

    def __call__(self, *args, **kwargs):
        import os

        if (
            os.getpid() == self.pid
            and not kwargs
            and len(args) == 1
            and type(args[0]) is str
            and args[0].startswith(self.message)
        ):
            _diagnostic_info(self, args[0])
            return None
        return self.original(*args, **kwargs)

    def info(self, message, *args):
        # Emitter payloads are immutable strings; preserve original print framing.
        if args:
            raise ValueError("Qualified profile print payload is not serialized")
        return self.original(message)


class _RuntimePipelineDiagnosticBindings:
    VERSION = 1

    def __init__(self, bindings):
        import os

        self.pid = os.getpid()
        self.bindings = tuple(bindings)

    def active(self):
        import os

        if self.pid != os.getpid():
            return 0
        return sum(
            getattr(module, name, None) is replacement
            for module, name, replacement in self.bindings
        )


def install_engine_runtime_diagnostic_logging():
    """Qualify actual R64 modules before transactionally replacing diagnostic sinks.

    The original KV, RPC, normalization, seal, publication and return paths are
    untouched. No output is sent to a sink from these diagnostic producer calls.
    """
    import builtins
    import hashlib
    import os
    import sys
    from pathlib import Path

    pending = []
    for (
        module_name,
        attribute,
        expected_sha,
        message,
        prefix,
    ) in _PIPELINE_DIAGNOSTIC_SPECS:
        module = sys.modules.get(module_name)
        qualified_shas = (
            _V6D_COMPONENT_SOURCE_DIGESTS
            if expected_sha == _V6D_COMPONENT_SOURCE_DIGESTS[0]
            else (expected_sha,)
        )
        if (
            module is None
            or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            not in qualified_shas
        ):
            raise ValueError(
                "Unqualified R64 runtime diagnostic source: " + module_name
            )
        present = attribute in vars(module)
        original = vars(module).get(
            attribute, builtins.print if attribute == "print" else None
        )
        kind = (
            _RuntimePipelineDiagnosticPrint
            if attribute == "print"
            else _RuntimePipelineDiagnosticLogger
        )
        if isinstance(original, kind):
            if (
                original.pid != os.getpid()
                or original.message != message
                or attribute == "logger"
                and original.prefix != prefix
            ):
                raise ValueError("Runtime diagnostic binding belongs to another owner")
            replacement = original
        else:
            if attribute == "print" and present:
                raise ValueError(
                    "Qualified runtime print global already has another owner"
                )
            replacement = (
                kind(original, message)
                if attribute == "print"
                else kind(original, message, prefix)
            )
        pending.append((module, attribute, present, original, replacement))
    if not prepare_diagnostic_logging():
        raise ValueError("Runtime diagnostic emitter is not prepared")
    changed = []
    try:
        for module, attribute, present, original, replacement in pending:
            changed.append((module, attribute, present, original))
            setattr(module, attribute, replacement)
            if getattr(module, attribute) is not replacement:
                raise ValueError("Runtime diagnostic binding assignment did not stick")
    except BaseException:
        for module, attribute, present, original in reversed(changed):
            if present:
                vars(module)[attribute] = original
            else:
                vars(module).pop(attribute, None)
        raise
    return _RuntimePipelineDiagnosticBindings(
        [
            (module, attribute, replacement)
            for module, attribute, _, _, replacement in pending
        ]
    )


class _MainThreadStdioStream:
    """Delegate existing text I/O; only the owning thread records slow calls."""

    def __init__(self, original, kind, probe):
        self.original, self.kind, self.probe = original, kind, probe

    def __getattr__(self, name):
        return getattr(self.original, name)

    def write(self, *args, **kwargs):
        return self.probe.call(self.original.write, self.kind, "write", args, kwargs)

    def flush(self, *args, **kwargs):
        return self.probe.call(self.original.flush, self.kind, "flush", args, kwargs)

    def writelines(self, *args, **kwargs):
        return self.probe.call(
            self.original.writelines, self.kind, "writelines", args, kwargs
        )


class _MainThreadStdioProbe:
    VERSION = 1
    SLOW_NS = 10_000_000
    MAX_STACK_DEPTH = 32

    def __init__(self, submit):
        import os
        import sys
        import threading

        self._getpid, self._get_ident = os.getpid, threading.get_ident
        self._is_finalizing = sys.is_finalizing
        self._sys = sys
        self.handler_bindings = ()
        self.pid, self.owner_thread = self._getpid(), self._get_ident()
        self.native_tid = threading.get_native_id()
        self.submit = submit
        self.bound_streams = self.updated_handlers = 0
        self.calls = self.slow_calls = self.failures = self.dropped = 0
        self.audit_wall_ns = 0
        self.emitting = False

    def active_streams(self):
        return len(
            {
                id(stream)
                for stream in (self._sys.stdout, self._sys.stderr)
                if isinstance(stream, _MainThreadStdioStream) and stream.probe is self
            }
        )

    def active_handlers(self):
        return sum(
            vars(handler).get("stream") is stream
            for handler, stream in self.handler_bindings
        )

    def call(self, original, kind, action, args, kwargs):
        if (
            self._is_finalizing()
            or self._getpid() != self.pid
            or self._get_ident() != self.owner_thread
            or self.emitting
        ):
            return original(*args, **kwargs)
        try:
            started = time.perf_counter_ns()
            thread_started = time.thread_time_ns()
            process_started = time.process_time_ns()
        except BaseException:
            self.failures += 1
            return original(*args, **kwargs)
        self.calls += 1
        error_type = None
        try:
            return original(*args, **kwargs)
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            try:
                finished = time.perf_counter_ns()
                thread_finished = time.thread_time_ns()
                process_finished = time.process_time_ns()
                if finished - started >= self.SLOW_NS:
                    self.slow_calls += 1
                    chars = (
                        len(args[0])
                        if (
                            action == "write" and args and type(args[0]) in (str, bytes)
                        )
                        else None
                    )
                    self._record(
                        kind,
                        action,
                        started,
                        finished,
                        thread_finished - thread_started,
                        process_finished - process_started,
                        chars,
                        error_type,
                    )
            except BaseException:
                self.failures += 1

    def _record(
        self,
        kind,
        action,
        started,
        finished,
        thread_cpu,
        process_cpu,
        chars,
        error_type,
    ):
        import sys

        audit_started = time.perf_counter_ns()
        self.emitting = True
        frame = None
        try:
            # Source positions only: never retain frames, locals, text or arguments.
            frame = sys._getframe(2)
            stack = []
            walked = 0
            while frame is not None and walked < self.MAX_STACK_DEPTH:
                code = frame.f_code
                if code.co_filename != __file__:
                    stack.append(
                        dict(
                            file=code.co_filename,
                            line=frame.f_lineno,
                            function=code.co_name,
                        )
                    )
                frame = frame.f_back
                walked += 1
            event = dict(
                event="stdio_blocking_call",
                stdio_probe_version=self.VERSION,
                event_time_us=time.time_ns() // 1000,
                owner_pid=self.pid,
                owner_native_tid=self.native_tid,
                stream=kind,
                action=action,
                started_perf_ns=started,
                finished_perf_ns=finished,
                thread_cpu_bracket_ns=thread_cpu,
                process_cpu_bracket_ns=process_cpu,
                text_chars=chars,
                call_exception_type=error_type,
                source_stack=stack,
                source_stack_truncated=frame is not None,
                tracked_context_is_not_stage_ownership=True,
            )
            if not self.submit(event):
                self.dropped += 1
        finally:
            del frame
            self.emitting = False
            self.audit_wall_ns += max(0, time.perf_counter_ns() - audit_started)


class _StdioAuditBridgeLogger:
    def __init__(self, bridge):
        self.bridge = bridge

    def info(self, message, *args):
        self.bridge.log_request_lifecycle_audit(message)


def install_main_thread_stdio_probe(observer):
    """Bind diagnostics at scheduler construction; installation performs no I/O."""
    import json
    import logging
    import os
    import sys
    import threading

    emitter = _diagnostic_emitter
    if emitter is None or emitter.pid != os.getpid() or emitter.closed:
        raise ValueError("Diagnostic emitter must be prepared before stdio probe")
    if isinstance(sys.stdout, _MainThreadStdioStream) or isinstance(
        sys.stderr, _MainThreadStdioStream
    ):
        if not isinstance(sys.stdout, _MainThreadStdioStream):
            raise ValueError("Foreign stdio probe owner")
        probe = sys.stdout.probe
        if (
            probe.pid != os.getpid()
            or probe.owner_thread != threading.get_ident()
            or not isinstance(sys.stderr, _MainThreadStdioStream)
            or sys.stderr.probe is not probe
            or getattr(probe, "bridge", None) is not observer._bridge
            or getattr(probe, "session", None) != observer._request_audit_session
        ):
            raise ValueError("Foreign stdio probe owner")
        return probe
    bridge = observer._bridge
    if not callable(getattr(bridge, "log_request_lifecycle_audit", None)):
        raise ValueError("Missing native diagnostic bridge")
    sink = _StdioAuditBridgeLogger(bridge)
    session = observer._request_audit_session

    def submit(event):
        payload = json.dumps(
            dict(event, session=session, audit_failures=0),
            sort_keys=True,
            separators=(",", ":"),
        )
        return _diagnostic_info(sink, payload)

    probe = _MainThreadStdioProbe(submit)
    probe.bridge, probe.session = bridge, session
    originals = dict(stdout=sys.stdout, stderr=sys.stderr)
    proxies = {}
    for kind, stream in originals.items():
        if not all(
            callable(getattr(stream, method, None))
            for method in ("write", "flush", "writelines")
        ):
            raise ValueError("Unqualified existing text stream")
        if id(stream) not in proxies:
            proxies[id(stream)] = _MainThreadStdioStream(stream, kind, probe)
        else:
            proxies[id(stream)].kind = "stdout+stderr"
    handlers = []
    loggers = [logging.getLogger()] + [
        logger
        for logger in tuple(logging.Logger.manager.loggerDict.values())
        if isinstance(logger, logging.Logger)
    ]
    seen = set()
    for logger in loggers:
        for handler in logger.handlers:
            # Dynamic stderr properties follow sys.stderr automatically. Only
            # replace concrete stored stream references, without flushing them.
            if id(handler) in seen or not isinstance(handler, logging.StreamHandler):
                continue
            seen.add(id(handler))
            stream = vars(handler).get("stream")
            if id(stream) in proxies:
                handlers.append((handler, stream))
    changed = []
    try:
        for handler, stream in handlers:
            if vars(handler).get("stream") is not stream:
                raise ValueError("Logging stream changed during installation")
            changed.append((handler, stream))
            handler.stream = proxies[id(stream)]
            if vars(handler).get("stream") is not proxies[id(stream)]:
                raise ValueError("Logging stream assignment did not stick")
        sys.stdout = proxies[id(originals["stdout"])]
        sys.stderr = proxies[id(originals["stderr"])]
        probe.bound_streams, probe.updated_handlers = len(proxies), len(changed)
        probe.handler_bindings = tuple(
            (handler, proxies[id(stream)]) for handler, stream in changed
        )
    except BaseException:
        sys.stdout, sys.stderr = originals["stdout"], originals["stderr"]
        for handler, stream in reversed(changed):
            # Restore concrete saved attributes even if a custom setter mutated
            # its value before rejecting the diagnostic replacement.
            vars(handler)["stream"] = stream
        raise
    return probe


class _NativeSchedulerComponentProbe:
    """Observe synchronous scheduler calls; never change their result or errors.

    Wall time includes GIL reacquisition. Thread/process CPU are brackets, not
    device time or additive rank costs. Only the scheduling thread contributes.
    Summaries are emitted with an existing slow schedule event, never via I/O in
    a wrapped component. Missing/overflowing records are explicitly incomplete.
    """

    LIMIT = 64
    VERSION = 1
    SCOPE = "native_scheduler_components"
    TARGETS = (
        ("trace_fixed_request", None, "_trace_fixed_request"),
        ("prepare_native_prefill_batches", None, "_prepare_native_prefill_batches"),
        (
            "restore_native_prefill_queue_order",
            None,
            "_restore_native_prefill_queue_order",
        ),
        ("allocate_slots", "kv_cache_manager", "allocate_slots"),
        ("build_kv_connector_meta", None, "_build_kv_connector_meta"),
    )

    def __init__(self, scheduler, extra_targets=()):
        self.records = []
        self.active_thread = None
        self.failures = 0
        self.overflow = 0
        self.audit_wall_ns = 0
        self.bindings = []
        targets = []
        for label, child, name in self.TARGETS:
            owner = getattr(scheduler, child) if child else scheduler
            original = getattr(owner, name)
            if not callable(original) or getattr(
                original, "_navi_schedule_component_probe", False
            ):
                raise ValueError("Unqualified or already wrapped scheduler component")
            targets.append((label, owner, name, original))
        for label, owner, name in extra_targets:
            original = getattr(owner, name)
            if (
                not isinstance(label, str)
                or not label
                or not callable(original)
                or getattr(original, "_navi_schedule_component_probe", False)
            ):
                raise ValueError("Unqualified or already wrapped connector component")
            targets.append((label, owner, name, original))
        if len({label for label, _, _, _ in targets}) != len(targets) or len(
            {(id(owner), name) for _, owner, name, _ in targets}
        ) != len(targets):
            raise ValueError("Duplicate scheduler component target")
        try:
            for label, owner, name, original in targets:
                wrapped = self._wrap(label, original)
                setattr(owner, name, wrapped)
                self.bindings.append((owner, name, original, wrapped))
        except Exception:
            for owner, name, original, wrapped in reversed(self.bindings):
                if getattr(owner, name) is wrapped:
                    setattr(owner, name, original)
            self.bindings = []
            raise

    def _wrap(self, label, original):
        @wraps(original)
        def wrapped(*args, **kwargs):
            import threading

            if self.active_thread != threading.get_ident():
                return original(*args, **kwargs)
            if len(self.records) >= self.LIMIT:
                self.overflow += 1
                return original(*args, **kwargs)
            try:
                audit_start = time.perf_counter_ns()
                cpu_before = time.thread_time_ns()
                process_before = time.process_time_ns()
                # Reserve on entry so nested log calls cannot evict their parent.
                slot = len(self.records)
                self.records.append(None)
                started = time.perf_counter_ns()
            except Exception:
                self.failures += 1
                return original(*args, **kwargs)
            try:
                return original(*args, **kwargs)
            finally:
                try:
                    finished = time.perf_counter_ns()
                    cpu_after = time.thread_time_ns()
                    process_after = time.process_time_ns()
                    self.records[slot] = (
                        label,
                        started,
                        finished,
                        cpu_before,
                        cpu_after,
                        process_before,
                        process_after,
                    )
                    self.audit_wall_ns += (started - audit_start) + (
                        time.perf_counter_ns() - finished
                    )
                except Exception:
                    self.failures += 1

        wrapped._navi_schedule_component_probe = True
        return wrapped

    def begin(self):
        import threading

        if self.active_thread is not None:
            self.failures += 1
            return False
        self.records = []
        self.overflow = 0
        self.audit_wall_ns = 0
        self.cpu_before = time.thread_time_ns()
        self.process_before = time.process_time_ns()
        self.active_thread = threading.get_ident()
        return True

    def finish(self, started_ns, finished_ns):
        try:
            cpu_after = time.thread_time_ns()
            process_after = time.process_time_ns()
        finally:
            self.active_thread = None
        # Keep the existing event rate; short schedules do not serialize rows.
        if finished_ns - started_ns < 50_000_000:
            return None
        rows = []
        for record in self.records:
            if record is None:
                continue
            label, a, b, c, d, e, f = record
            if not (started_ns <= a <= b <= finished_ns and c <= d and e <= f):
                self.failures += 1
                continue
            rows.append(
                dict(
                    component=label,
                    started_perf_ns=a,
                    finished_perf_ns=b,
                    thread_cpu_bracket_ns=d - c,
                    process_cpu_bracket_ns=f - e,
                )
            )
        covered = 0
        last_end = started_ns
        for row in sorted(rows, key=lambda r: r["started_perf_ns"]):
            a, b = row["started_perf_ns"], row["finished_perf_ns"]
            covered += max(0, b - max(a, last_end))
            last_end = max(last_end, b)
        return dict(
            version=self.VERSION,
            scope=self.SCOPE,
            started_perf_ns=started_ns,
            finished_perf_ns=finished_ns,
            native_thread_cpu_bracket_ns=cpu_after - self.cpu_before,
            native_process_cpu_bracket_ns=process_after - self.process_before,
            bound_hooks=len(self.bindings),
            records=rows,
            all_selected_hook_records_captured=self.failures == 0
            and self.overflow == 0,
            audit_failures=self.failures,
            dropped_records=self.overflow,
            component_measured_audit_wall_ns=self.audit_wall_ns,
            recorded_component_wall_union_ns=covered,
            unattributed_wall_ns=finished_ns - started_ns - covered,
        )


class _NativeConnectorMetadataProbe(_NativeSchedulerComponentProbe):
    """Keep the unresolved composite metadata path without per-request trace work."""

    VERSION = 2
    SCOPE = "composite_connector_metadata_with_v6d_subcomponents"
    TARGETS = (("build_kv_connector_meta", None, "_build_kv_connector_meta"),)


class _NativeR78ConnectorMetadataProbe(_NativeConnectorMetadataProbe):
    VERSION = 3
    SCOPE = "r78_composite_connector_metadata_with_delayed_allocation"


def prepare_native_scheduler_component_probe(scheduler):
    """Qualify the actual fixed scheduler and its existing connector components.

    The connector implementation remains unchanged. These wrappers time only
    calls made by the active scheduling thread and preserve original I/O,
    exceptions, request state, object allocation, and Mamba protection order.
    """
    if not getattr(scheduler, "_navi_fixed_prefill_plan_class", False):
        return None
    import hashlib
    import sys
    from pathlib import Path

    module = sys.modules.get("kvs_connector.v6d_object_connector")
    if (
        module is None
        or hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        not in _V6D_COMPONENT_SOURCE_DIGESTS
    ):
        raise ValueError("Unqualified native V6D scheduler component source")
    # The deployed r64 TPUMultiConnector contains both Raiden and V6D.
    # Use its existing _connectors ownership tree, as the native cache-source
    # binder does; never mistake another connector's scheduler for V6D.
    pending = [getattr(scheduler, "connector", None)]
    visited, vendors = set(), {}
    while pending:
        connector = pending.pop()
        if connector is None or id(connector) in visited:
            continue
        visited.add(id(connector))
        candidate = getattr(connector, "scheduler", None)
        candidate_type = type(candidate)
        if (
            candidate_type.__name__ == "V6dObjectConnectorScheduler"
            and candidate_type.__module__ == module.__name__
        ):
            vendors[id(candidate)] = candidate
        pending.extend(getattr(connector, "_connectors", ()))
    if len(vendors) != 1:
        raise ValueError(
            "Expected exactly one qualified native V6D scheduler component"
        )
    vendor = next(iter(vendors.values()))
    extra = (
        ("connector_store_selection", vendor, "_get_reqs_to_store"),
        ("connector_cross_group_allocate", vendor, "_cross_group_batch_allocate"),
        ("connector_mamba_protection", vendor, "_protect_mamba_store_blocks"),
        ("connector_info_log", module.logger, "info"),
    )
    if (
        hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        in _V6D_COMPONENT_SOURCE_DIGESTS[1:]
    ):
        extra += (
            ("connector_store_planning", vendor, "_plan_stores"),
            ("connector_allocation_submit", vendor, "_submit_store_allocation"),
            ("connector_allocation_commit", vendor, "_take_ready_allocation"),
        )
        return _NativeR78ConnectorMetadataProbe(scheduler, extra)
    return _NativeConnectorMetadataProbe(scheduler, extra)


def engine_stage_event(observer, stage, started_ns, finished_ns, details):
    """Audit existing stage boundaries; tracked contexts are not stage ownership."""
    # Detailed per-context JSON is diagnostic work, never a serving default.
    if not getattr(observer, "_engine_stage_audit_enabled", False):
        return 0
    stage_code = {
        "schedule": 1,
        "update_from_output": 2,
        "route_output": 3,
        "post_step": 4,
        "process_input_queue": 5,
        "engine_round": 6,
        "prepare_execution_step": 7,
    }.get(stage)
    if stage_code is None:
        return 0
    audit_started = time.perf_counter_ns()
    emitted = 0
    try:
        contexts = {}
        for state in tuple(observer._pending_forward_batches.values()) + tuple(
            observer._scheduled_forward_batches.values()
        ):
            if state.released:
                continue
            contexts[(state.batch_id, state.schedule_batch_id)] = state
        if not contexts:
            if stage_code <= 3:
                return 0
            # A loop can finish after its last context was released. Keep this
            # global interval explicit instead of assigning it to a later batch.
            contexts[(0, 0)] = None
        if not (
            type(started_ns) is int
            and type(finished_ns) is int
            and 0 < started_ns <= finished_ns
        ):
            raise ValueError("Invalid original engine stage boundaries")
        fields = {
            "event": "engine_stage_audit",
            "event_time_us": time.time_ns() // 1000,
            "engine_stage_audit_version": 1,
            "stage_code": stage_code,
            "started_perf_ns": started_ns,
            "finished_perf_ns": finished_ns,
            "context_scope_currently_tracked": 1,
            "stage_partition_valid": False,
            "stage_audit_failures": getattr(
                observer, "_engine_stage_probe_failures", 0
            ),
            # Inclusive probe wall time is an overhead bound, not CPU self time.
            "cumulative_audit_wall_before_ns": getattr(
                observer, "_engine_stage_probe_audit_wall_ns", 0
            ),
        }
        stdio = getattr(observer, "_main_thread_stdio_probe", None)
        pipeline = getattr(observer, "_runtime_pipeline_diagnostic_bindings", None)
        fields.update(
            runtime_pipeline_diagnostic_version=pipeline.VERSION if pipeline else 0,
            runtime_pipeline_diagnostic_bound=len(pipeline.bindings) if pipeline else 0,
            runtime_pipeline_diagnostic_active=pipeline.active() if pipeline else 0,
            runtime_pipeline_diagnostic_install_failures=getattr(
                observer, "_runtime_pipeline_diagnostic_install_failures", 0
            ),
        )
        fields.update(
            stdio_probe_version=stdio.VERSION if stdio is not None else 0,
            stdio_probe_bound_streams=stdio.bound_streams if stdio is not None else 0,
            stdio_probe_active_streams=(
                stdio.active_streams() if stdio is not None else 0
            ),
            stdio_probe_bound_handlers=(
                stdio.updated_handlers if stdio is not None else 0
            ),
            stdio_probe_active_handlers=(
                stdio.active_handlers() if stdio is not None else 0
            ),
            stdio_probe_install_failures=getattr(
                observer, "_stdio_probe_install_failures", 0
            ),
            stdio_probe_dropped_records=stdio.dropped if stdio is not None else 0,
            stdio_probe_failures=stdio.failures if stdio is not None else 0,
            stdio_probe_calls=stdio.calls if stdio is not None else 0,
            stdio_probe_slow_calls=stdio.slow_calls if stdio is not None else 0,
            stdio_probe_audit_wall_ns=stdio.audit_wall_ns if stdio is not None else 0,
        )
        if stage_code >= 4:
            fields.update(
                engine_loop_probe_version=1,
                loop_bound_hook_count=getattr(
                    observer, "_engine_loop_probe_bound_hooks", 0
                ),
            )
            if type(details.get("model_executed")) is bool:
                fields["model_executed"] = details["model_executed"]
        for key in (
            "promote_ms",
            "native_ms",
            "report_ms",
            "scheduled_tokens",
            "running",
            "waiting",
            "output_count",
        ):
            value = details.get(key)
            if type(value) in (int, float) and math.isfinite(value):
                fields[key] = value
        if stage == "schedule" and all(
            key in fields for key in ("promote_ms", "native_ms", "report_ms")
        ):
            before_native = started_ns + round(fields["promote_ms"] * 1e6)
            after_native = before_native + round(fields["native_ms"] * 1e6)
            partition_end = after_native + round(fields["report_ms"] * 1e6)
            if (
                started_ns <= before_native <= after_native <= finished_ns
                and abs(partition_end - finished_ns) <= 1
            ):
                fields.update(
                    stage_partition_valid=True,
                    before_native_perf_ns=before_native,
                    after_native_perf_ns=after_native,
                )
        if stage == "schedule":
            profile = getattr(observer, "_native_scheduler_component_profile", None)
            fields["native_scheduler_component_bound_hooks"] = getattr(
                observer, "_native_scheduler_component_bound_hooks", 0
            )
            fields["native_scheduler_component_install_failures"] = getattr(
                observer, "_native_scheduler_component_install_failures", 0
            )
            if profile is not None:
                if (
                    fields.get("stage_partition_valid")
                    and abs(
                        profile["started_perf_ns"] - fields["before_native_perf_ns"]
                    )
                    <= 1
                    and abs(
                        profile["finished_perf_ns"] - fields["after_native_perf_ns"]
                    )
                    <= 1
                ):
                    fields["native_scheduler_components"] = profile
                else:
                    fields["native_scheduler_component_alignment_valid"] = False
        for state in contexts.values():
            ids = list(state.request_ids) if state is not None else []
            batch_id = state.batch_id if state is not None else 0
            schedule_batch_id = state.schedule_batch_id if state is not None else 0
            if (
                type(batch_id) is not int
                or type(schedule_batch_id) is not int
                or any(type(rid) is not int or rid <= 0 for rid in ids)
            ):
                raise ValueError("Invalid native diagnostic context")
            observer.on_request_lifecycle_event(
                dict(
                    fields,
                    native_batch_id=batch_id,
                    native_schedule_batch_id=schedule_batch_id,
                    navi_request_ids=ids,
                    global_engine_stage=state is None,
                    context_scope_currently_tracked=int(state is not None),
                )
            )
            emitted += 1
    except Exception:
        # A diagnostic sink/metadata error must never change a serving result.
        observer._engine_stage_probe_failures = (
            getattr(observer, "_engine_stage_probe_failures", 0) + 1
        )
    finally:
        observer._engine_stage_probe_audit_wall_ns = getattr(
            observer, "_engine_stage_probe_audit_wall_ns", 0
        ) + max(0, time.perf_counter_ns() - audit_started)
    return emitted


def bind_engine_loop_audit(observer, engine_core):
    """Time original bound methods without changing arguments, results or errors.

    These are inclusive method intervals. Input processing can block, an engine
    round includes execution/other stages, and none is CPU self time. Consumers
    must union overlaps and cannot treat a tracked context as exclusive ownership.
    """
    if (
        getattr(observer, "_engine_loop_probe_bound_hooks", 0)
        and getattr(observer, "_engine_loop_probe_bound_core", None) is engine_core
    ):
        return observer._engine_loop_probe_bound_hooks
    installed = []
    try:
        methods = [
            (engine_core, "post_step", "post_step"),
            (engine_core, "_process_input_queue", "process_input_queue"),
            (engine_core, "_process_engine_step", "engine_round"),
            (observer, "_prepare_execution_step", "prepare_execution_step"),
        ]
        originals = [
            (owner, name, stage, getattr(owner, name)) for owner, name, stage in methods
        ]
        if not all(callable(original) for _, _, _, original in originals):
            raise ValueError("Missing required engine-loop diagnostic boundary")

        def measured(original, stage):
            @wraps(original)
            def call(*args, **kwargs):
                try:
                    started = time.perf_counter_ns()
                except Exception:
                    observer._engine_stage_probe_failures = (
                        getattr(observer, "_engine_stage_probe_failures", 0) + 1
                    )
                    return original(*args, **kwargs)
                result = None
                try:
                    result = original(*args, **kwargs)
                    return result
                finally:
                    try:
                        finished = time.perf_counter_ns()
                        details = {}
                        if stage == "post_step":
                            value = args[0] if args else kwargs.get("model_executed")
                            if type(value) is bool:
                                details["model_executed"] = value
                        elif stage == "engine_round" and type(result) is bool:
                            details["model_executed"] = result
                        engine_stage_event(observer, stage, started, finished, details)
                    except Exception:
                        observer._engine_stage_probe_failures = (
                            getattr(observer, "_engine_stage_probe_failures", 0) + 1
                        )

            return call

        for owner, name, stage, original in originals:
            wrapper = measured(original, stage)
            setattr(owner, name, wrapper)
            installed.append((owner, name, original, wrapper))
        observer._engine_loop_probe_bound_core = engine_core
        observer._engine_loop_probe_bound_hooks = len(installed)
        return len(installed)
    except Exception:
        for owner, name, original, wrapper in reversed(installed):
            try:
                if getattr(owner, name, None) is wrapper:
                    setattr(owner, name, original)
            except Exception:
                pass
        observer._engine_stage_probe_failures = (
            getattr(observer, "_engine_stage_probe_failures", 0) + 1
        )
        return 0


def explicit_checkpoint_audit_fields(request, scheduler_block_size):
    # The TPU plugin can override a boundary independently of cache hits.
    # Emit only numeric directives, never sampling params or position arrays.
    params = getattr(request, "sampling_params", None)
    extra = getattr(params, "extra_args", None)
    control = extra.get("cache_control_params") if isinstance(extra, dict) else None
    positions = control.get("token_positions") if isinstance(control, dict) else None
    valid = (
        [p for p in positions if type(p) is int and 0 < p <= request.num_prompt_tokens]
        if isinstance(positions, list)
        else []
    )
    maximum = max(valid, default=0)
    return {
        "explicit_checkpoint_positions_present": isinstance(positions, list),
        "explicit_checkpoint_valid_position_count": len(valid),
        "explicit_checkpoint_max_position": maximum,
        "explicit_checkpoint_requested_boundary": maximum
        // scheduler_block_size
        * scheduler_block_size,
    }


def mark_entry(step):
    step.legacy_execute_entered_ns = step.started_ns


def begin(step, callback, *args):
    step.legacy_backend_begin_before_ns = time.perf_counter_ns()
    step.legacy_backend_native_call = True
    try:
        result = callback(*args)
        step.legacy_backend_begin_succeeded = result is True
        return result
    finally:
        step.legacy_backend_begin_after_ns = time.perf_counter_ns()


def end(step, callback, *args):
    step.legacy_backend_end_before_ns = time.perf_counter_ns()
    try:
        result = callback(*args)
        step.legacy_backend_end_succeeded = result is True
        return result
    finally:
        step.legacy_backend_end_after_ns = time.perf_counter_ns()


def begin_control(step):
    # Current continuous accounting intentionally omits zero-token native calls.
    # Keep their Python boundaries visible; these are not native-clock brackets.
    step.legacy_backend_native_call = False
    step.legacy_backend_begin_before_ns = time.perf_counter_ns()
    step.legacy_backend_begin_after_ns = step.legacy_backend_begin_before_ns


def end_control(step):
    if getattr(step, "legacy_backend_native_call", None) is False and not getattr(
        step, "legacy_backend_end_after_ns", 0
    ):
        step.legacy_backend_end_before_ns = time.perf_counter_ns()
        step.legacy_backend_end_after_ns = step.legacy_backend_end_before_ns


def event_fields(step):
    start_lo = getattr(step, "legacy_backend_begin_before_ns", 0)
    start_hi = getattr(step, "legacy_backend_begin_after_ns", 0)
    end_lo = getattr(step, "legacy_backend_end_before_ns", 0)
    end_hi = getattr(step, "legacy_backend_end_after_ns", 0)
    ordered = 0 < start_lo <= start_hi <= end_lo <= end_hi
    native = getattr(step, "legacy_backend_native_call", False)
    return {
        "legacy_timing_probe_version": 1,
        "execute_entered_perf_ns": getattr(step, "legacy_execute_entered_ns", 0),
        "model_submitted_perf_ns": step.submitted_ns,
        "future_completed_perf_ns": step.future_completed_ns,
        "legacy_backend_begin_before_perf_ns": start_lo,
        "legacy_backend_begin_after_perf_ns": start_hi,
        "legacy_backend_end_before_perf_ns": end_lo,
        "legacy_backend_end_after_perf_ns": end_hi,
        "legacy_backend_native_clock_bracket_valid": bool(
            ordered
            and native
            and getattr(step, "legacy_backend_begin_succeeded", False)
            and getattr(step, "legacy_backend_end_succeeded", False)
        ),
        "legacy_backend_control_round_boundaries_valid": bool(ordered and not native),
        "legacy_backend_elapsed_lower_ms": (
            (end_lo - start_hi) / 1e6 if ordered else None
        ),
        "legacy_backend_elapsed_upper_ms": (
            (end_hi - start_lo) / 1e6 if ordered else None
        ),
        "legacy_backend_clock_uncertainty_ms": (
            ((start_hi - start_lo) + (end_hi - end_lo)) / 1e6 if ordered else None
        ),
    }


def batch_begin(state, callback, *args):
    """Bracket the native startBatch wall clock, independently of backend usage."""
    state.legacy_batch_begin_before_wall_ns = time.time_ns()
    try:
        result = callback(*args)
        state.legacy_batch_begin_succeeded = result is not None
        return result
    finally:
        state.legacy_batch_begin_after_wall_ns = time.time_ns()


def batch_finish(state, callback, *args):
    """Bracket native finishBatch's currentTimeUs even with an explicit end label."""
    state.legacy_batch_end_before_wall_ns = time.time_ns()
    try:
        result = callback(*args)
        state.legacy_batch_end_succeeded = result is True
        return result
    finally:
        state.legacy_batch_end_after_wall_ns = time.time_ns()


def capture_batch_output_clock(state, completed_perf_ns, completed_wall_us):
    # Missing endpoints stay missing; never use the diagnostic observation time.
    state.legacy_output_completed_perf_ns = (
        completed_perf_ns
        if type(completed_perf_ns) is int and completed_perf_ns > 0
        else 0
    )
    state.legacy_output_completed_wall_us = (
        completed_wall_us
        if type(completed_wall_us) is int and completed_wall_us > 0
        else 0
    )


def batch_event_fields(state):
    lo = getattr(state, "legacy_batch_begin_before_wall_ns", 0)
    hi = getattr(state, "legacy_batch_begin_after_wall_ns", 0)
    end_lo = getattr(state, "legacy_batch_end_before_wall_ns", 0)
    end_hi = getattr(state, "legacy_batch_end_after_wall_ns", 0)
    valid = bool(
        0 < lo <= hi <= end_lo <= end_hi
        and getattr(state, "legacy_batch_begin_succeeded", False)
        and getattr(state, "legacy_batch_end_succeeded", False)
    )
    # autil currentTime() uses integral microseconds; include both endpoint ticks.
    tick = 1000
    return {
        "legacy_batch_output_clock_probe_version": 1,
        "legacy_batch_output_completed_perf_ns": getattr(
            state, "legacy_output_completed_perf_ns", 0
        ),
        "legacy_batch_output_completed_wall_time_us": getattr(
            state, "legacy_output_completed_wall_us", 0
        ),
        "legacy_batch_output_clock_present": bool(
            getattr(state, "legacy_output_completed_perf_ns", 0) > 0
            and getattr(state, "legacy_output_completed_wall_us", 0) > 0
        ),
        "legacy_batch_clock_probe_version": 2,
        "legacy_batch_begin_before_wall_ns": lo,
        "legacy_batch_begin_after_wall_ns": hi,
        "legacy_batch_end_before_wall_ns": end_lo,
        "legacy_batch_end_after_wall_ns": end_hi,
        "legacy_batch_call_bracket_valid": valid,
        "legacy_batch_clock_quantization_ns": tick,
        "legacy_batch_elapsed_lower_ms": (
            max(0, end_lo - hi - 2 * tick) / 1e6 if valid else None
        ),
        "legacy_batch_elapsed_upper_ms": (
            (end_hi - lo + 2 * tick) / 1e6 if valid else None
        ),
        "legacy_batch_clock_uncertainty_ms": (
            ((hi - lo) + (end_hi - end_lo) + 4 * tick) / 1e6 if valid else None
        ),
    }


# Opt-in passive observation. Never force a collection or change GC policy.
_GC_AUDIT_FLAG = "/workspace/.navi_sched/diagnostics/gc-timing-audit.enabled"
_GC_AUDIT_MIN_NS = 50_000_000
_gc_audit_pid = None
_gc_audit_hook = None
_gc_audit_before_hook = None


def install_gc_timing_audit():
    """Read the owned diagnostic flag once when each observer is constructed."""
    prepare_diagnostic_logging()
    try:
        import gc
        import logging
        import os
        from pathlib import Path

        with Path(_GC_AUDIT_FLAG).open("rb") as stream:
            if stream.read(3) != b"1\n":
                return False
        return _register_gc_timing_audit(
            gc,
            os.getpid(),
            _DiagnosticLoggerProxy(logging.getLogger("vllm.navi_sched.gc")),
        )
    except Exception:
        # Diagnostics must never prevent a serving process from starting.
        return False


def _register_gc_timing_audit(gc_module, pid, sink):
    global _gc_audit_pid, _gc_audit_hook, _gc_audit_before_hook
    if (
        _gc_audit_pid == pid
        and _gc_audit_hook in gc_module.callbacks
        and _gc_audit_before_hook in gc_module.callbacks
    ):
        return True
    for owned in (_gc_audit_hook, _gc_audit_before_hook):
        if owned is not None and owned in gc_module.callbacks:
            gc_module.callbacks.remove(owned)
    active = None
    boundaries = None

    def before_callback(phase, info):
        nonlocal boundaries
        try:
            import threading

            generation, tid = int(info["generation"]), threading.get_ident()
            wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
            if phase == "start":
                boundaries = (generation, tid, wall, cpu, None, None)
            elif (
                phase == "stop"
                and boundaries is not None
                and boundaries[:2] == (generation, tid)
            ):
                boundaries = boundaries[:4] + (wall, cpu)
        except Exception:
            boundaries = None

    def callback(phase, info):
        nonlocal active, boundaries
        try:
            import threading

            if phase == "start":
                active = (
                    int(info["generation"]),
                    threading.get_ident(),
                    time.perf_counter_ns(),
                    time.thread_time_ns(),
                )
                return
            if phase != "stop" or active is None:
                return
            generation, tid, wall_start, cpu_start = active
            early = boundaries
            active = boundaries = None
            wall_end, cpu_end = time.perf_counter_ns(), time.thread_time_ns()
            if (
                generation != info["generation"]
                or tid != threading.get_ident()
                or wall_end < wall_start
                or cpu_end < cpu_start
            ):
                return
            partition_valid = bool(
                early is not None
                and early[:2] == (generation, tid)
                and early[4] is not None
                and early[5] is not None
                and early[2] <= wall_start <= early[4] <= wall_end
                and early[3] <= cpu_start <= early[5] <= cpu_end
                and gc_module.callbacks[0] is before_callback
                and gc_module.callbacks[-1] is callback
            )
            cycle_wall = (
                wall_end - early[2] if partition_valid else wall_end - wall_start
            )
            if cycle_wall < _GC_AUDIT_MIN_NS:
                return
            import json
            import sys
            from pathlib import Path

            origins = []
            frame = sys._getframe(1)
            try:
                for _ in range(6):
                    if frame is None:
                        break
                    origins.append(
                        dict(
                            file=Path(frame.f_code.co_filename).name,
                            function=frame.f_code.co_name,
                            line=frame.f_lineno,
                        )
                    )
                    frame = frame.f_back
            finally:
                del frame
            frozen = (
                gc_module.get_freeze_count()
                if hasattr(gc_module, "get_freeze_count")
                else None
            )
            fields = dict(
                version=1,
                pid=pid,
                thread_id=tid,
                generation=generation,
                collected=int(info["collected"]),
                uncollectable=int(info["uncollectable"]),
                started_perf_ns=wall_start,
                finished_perf_ns=wall_end,
                thread_cpu_started_ns=cpu_start,
                thread_cpu_finished_ns=cpu_end,
                event_unix_ns=time.time_ns(),
                bracket_includes_other_gc_callbacks=True,
                frozen_object_count=frozen,
                origins=origins,
                gc_phase_audit_version=1,
                gc_phase_partition_valid=partition_valid,
                phase_boundary_probe_overhead_not_subtracted=True,
            )
            if partition_valid:
                fields.update(
                    cycle_started_perf_ns=early[2],
                    cycle_thread_cpu_started_ns=early[3],
                    start_callbacks_wall_ns=wall_start - early[2],
                    start_callbacks_cpu_ns=cpu_start - early[3],
                    body_wall_ns=early[4] - wall_start,
                    body_cpu_ns=early[5] - cpu_start,
                    stop_callbacks_wall_ns=wall_end - early[4],
                    stop_callbacks_cpu_ns=cpu_end - early[5],
                )
            sink.info(
                "[navi-python-gc-audit] %s",
                json.dumps(fields, separators=(",", ":"), sort_keys=True),
            )
        except Exception:
            active = boundaries = None

    # Preserve every foreign callback and its relative order. Roll back only
    # owned callbacks if registration cannot complete.
    gc_module.callbacks.append(callback)
    try:
        gc_module.callbacks.insert(0, before_callback)
    except Exception:
        gc_module.callbacks.remove(callback)
        raise
    _gc_audit_pid, _gc_audit_hook, _gc_audit_before_hook = (
        pid,
        callback,
        before_callback,
    )
    try:
        import json

        sink.info(
            "[navi-python-gc-audit-installed] %s",
            json.dumps(
                dict(
                    version=1,
                    pid=pid,
                    callback_count=len(gc_module.callbacks),
                    callbacks=[
                        dict(
                            module=getattr(c, "__module__", type(c).__module__),
                            qualname=getattr(c, "__qualname__", type(c).__qualname__),
                        )
                        for c in list(gc_module.callbacks)[:16]
                    ],
                    collection_policy_unchanged=True,
                ),
                sort_keys=True,
            ),
        )
    except Exception:
        pass
    return True


# Explicit pilot control for one full collection AFTER Worker warmup returns
# and BEFORE its warmup RPC returns to the Engine/readiness path. Automatic GC,
# thresholds, freeze state and all existing callbacks remain enabled/unchanged.
_WORKER_STARTUP_GC_SWEEP_FLAG = None
_worker_startup_gc_sweep_pid = None


def worker_startup_gc_sweep():
    from pathlib import Path

    try:
        control = (
            Path(_WORKER_STARTUP_GC_SWEEP_FLAG)
            if _WORKER_STARTUP_GC_SWEEP_FLAG is not None
            else Path(__file__).resolve().parents[4] / "worker-startup-gc-sweep.enabled"
        )
        with control.open("rb") as stream:
            enabled = stream.read(3)
    except FileNotFoundError:
        return False
    if enabled != b"1\n":
        raise ValueError("Invalid explicit Worker startup GC sweep control")
    import gc
    import logging
    import os

    return _run_worker_startup_gc_sweep(
        gc, os.getpid(), logging.getLogger("vllm.navi_sched.gc")
    )


def _run_worker_startup_gc_sweep(gc_module, pid, sink):
    global _worker_startup_gc_sweep_pid
    if _worker_startup_gc_sweep_pid == pid:
        return False
    started = cpu_started = None
    try:
        started, cpu_started = time.perf_counter_ns(), time.thread_time_ns()
    except Exception:
        pass
    # A failed enabled cleanup must fail warmup; never claim readiness with an
    # unsuccessful sweep. Do not catch a collection failure as diagnostic noise.
    collected = gc_module.collect(2)
    _worker_startup_gc_sweep_pid = pid
    try:
        import json

        finished, cpu_finished = time.perf_counter_ns(), time.thread_time_ns()
        sink.info(
            "[navi-worker-startup-gc-sweep] %s",
            json.dumps(
                dict(
                    version=1,
                    pid=pid,
                    collected=collected,
                    generation=2,
                    event_unix_ns=time.time_ns(),
                    started_perf_ns=started,
                    finished_perf_ns=finished,
                    thread_cpu_started_ns=cpu_started,
                    thread_cpu_finished_ns=cpu_finished,
                    after_worker_warmup=True,
                    before_warmup_rpc_return=True,
                    automatic_gc_policy_unchanged=True,
                ),
                sort_keys=True,
            ),
        )
    except Exception:
        pass
    return True


# Finite-task opt-in diagnostics; does not change an output, wait, or GC policy.
_OUTPUT_PHASE_FLAG = "/workspace/.navi_sched/diagnostics/output-phase-audit.enabled"
_OUTPUT_PHASE_BASE_SHA256 = (
    "26a9f8a35e77549dd0c1b3217921604511f20b37993ca3f8c6c3d01ecb98a0fa"
)
_output_phase_pid = None
_output_phase_enabled = False
_output_phase_classes = {}


def _output_phase_is_enabled():
    global _output_phase_pid, _output_phase_enabled
    import os

    pid = os.getpid()
    if _output_phase_pid != pid:
        _output_phase_pid = pid
        _output_phase_enabled = False
        try:
            from pathlib import Path

            with Path(_OUTPUT_PHASE_FLAG).open("rb") as stream:
                _output_phase_enabled = stream.read(3) == b"1\n"
        except Exception:
            pass
    return _output_phase_enabled


def _output_phase_tpu_snapshot():
    """Read r64's existing compile-cache counters; never materialize device work.

    The cache is process-wide and entries have no stable graph identifier. These
    counters locate compilation activity, not ownership of a particular future.
    Query intervals are recorded on the background writer after the triggering
    output finishes. Labels are never adjusted to remove diagnostic overhead.
    """
    import math
    import sys

    started = time.perf_counter_ns()
    try:
        stats = sys.modules["torch"].tpu._get_cache_stats()
        requests, hits = stats.num_cache_reqs, stats.num_cache_hits
        if (
            type(requests) is not int
            or type(hits) is not int
            or not 0 <= hits <= requests
        ):
            raise ValueError("Invalid r64 TPU compilation-cache counters")
        rows = []
        for entry in stats.per_entry_stats:
            seconds = entry.compilation_duration.total_seconds()
            epoch = entry.last_read.timestamp()
            reads = entry.read_count
            if (
                not math.isfinite(seconds)
                or seconds < 0
                or not math.isfinite(epoch)
                or type(reads) is not int
                or reads < 0
            ):
                raise ValueError("Invalid r64 TPU compilation-cache entry")
            rows.append(
                dict(
                    compilation_duration_ns=round(seconds * 1e9),
                    last_read_unix_ns=round(epoch * 1e9),
                    read_count=reads,
                )
            )
        result = dict(
            available=True,
            num_cache_reqs=requests,
            num_cache_hits=hits,
            entry_count=len(rows),
            retained_compilation_duration_ns=sum(
                row["compilation_duration_ns"] for row in rows
            ),
            recent_entry_stats=sorted(
                rows, key=lambda row: row["last_read_unix_ns"], reverse=True
            )[:16],
            recent_entry_limit=16,
            entries_without_stable_graph_identifier=True,
        )
    except Exception as error:
        # A diagnostic failure must neither become a zero-valued successful
        # metric nor change the original runtime's return/exception behavior.
        result = dict(available=False, error_type=type(error).__name__)
    finished = time.perf_counter_ns()
    result["query_clock_valid"] = finished >= started
    result["query_wall_ns"] = finished - started if finished >= started else None
    return result


def _deliver_tpu_compile_snapshot(logger, serialized_context):
    """Delivery-thread only: read counters after the triggering output finished."""
    import json
    import threading

    fields = json.loads(serialized_context)
    fields["collection_started_perf_ns"] = time.perf_counter_ns()
    fields["collection_started_unix_ns"] = time.time_ns()
    fields["collector_native_tid"] = threading.get_native_id()
    fields["snapshot"] = _output_phase_tpu_snapshot()
    fields["collection_finished_perf_ns"] = time.perf_counter_ns()
    fields["collection_finished_unix_ns"] = time.time_ns()
    logger.info(
        "[navi-vllm-tpu-compile-cache-audit] %s",
        json.dumps(fields, separators=(",", ":"), sort_keys=True),
    )


def _output_phase_queue_tpu_snapshot(frame, finished, success):
    """Bounded post-output diagnostics, never a fit/scheduling/availability gate."""
    global _tpu_compile_baseline_pid
    import json
    import logging

    baseline = _tpu_compile_baseline_pid != frame["pid"]
    duration = finished - frame["started_perf_ns"]
    if not baseline and duration < _TPU_COMPILE_SLOW_OUTPUT_NS:
        return False
    fields = {
        key: frame[key]
        for key in ("pid", "thread_id", "request_count", "request_set_sha256")
    }
    fields.update(
        version=1,
        trigger_output_started_perf_ns=frame["started_perf_ns"],
        trigger_output_finished_perf_ns=finished,
        trigger_output_success=bool(success),
        first_output_baseline=baseline,
        slow_output=duration >= _TPU_COMPILE_SLOW_OUTPUT_NS,
        queued_unix_ns=time.time_ns(),
    )
    queued = _diagnostic_info(
        logging.getLogger("vllm.navi_sched.tpu_compile"),
        _TPU_COMPILE_SNAPSHOT_JOB,
        json.dumps(fields, separators=(",", ":"), sort_keys=True),
    )
    if queued:
        _tpu_compile_baseline_pid = frame["pid"]
    return queued


def _output_phase_begin(output):
    try:
        import hashlib
        import json
        import os
        import threading

        wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
        process_cpu = time.process_time_ns()
        ids = sorted(output._model_runner_output.req_ids)
        if not all(isinstance(value, str) for value in ids):
            return None
        digest = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()
        ).hexdigest()
        return dict(
            pid=os.getpid(),
            thread_id=threading.get_ident(),
            started_perf_ns=wall,
            thread_cpu_started_ns=cpu,
            request_count=len(ids),
            request_set_sha256=digest,
            audit_emission_ns=0,
            process_cpu_started_ns=process_cpu,
        )
    except Exception:
        return None


def _output_phase_emit(frame, stage, wall, cpu, success):
    if frame is None:
        return
    try:
        import json
        import logging

        end, end_cpu = time.perf_counter_ns(), time.thread_time_ns()
        if end < wall or end_cpu < cpu:
            return
        fields = {
            key: frame[key]
            for key in ("pid", "thread_id", "request_count", "request_set_sha256")
        }
        process_finished = time.process_time_ns()
        process_delta = process_finished - frame["process_cpu_started_ns"]
        fields.update(
            version=2,
            stage=stage,
            started_perf_ns=wall,
            finished_perf_ns=end,
            thread_cpu_started_ns=cpu,
            thread_cpu_finished_ns=end_cpu,
            event_unix_ns=time.time_ns(),
            success=bool(success),
            prior_audit_emission_ns=frame["audit_emission_ns"],
            process_cpu_since_output_begin_ns=(
                process_delta if process_delta >= 0 else None
            ),
            process_cpu_clock_valid=process_delta >= 0,
        )
        emit_start = time.perf_counter_ns()
        try:
            _diagnostic_info(
                logging.getLogger("vllm.navi_sched.output"),
                "[navi-vllm-output-phase-audit] %s",
                json.dumps(fields, separators=(",", ":"), sort_keys=True),
            )
        finally:
            frame["audit_emission_ns"] += max(0, time.perf_counter_ns() - emit_start)
    except Exception:
        pass


def _output_phase_call(frame, stage, method, *args, **kwargs):
    if frame is None:
        return method(*args, **kwargs)
    try:
        wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
    except Exception:
        return method(*args, **kwargs)
    success = False
    try:
        value = method(*args, **kwargs)
        success = True
        return value
    finally:
        _output_phase_emit(frame, stage, wall, cpu, success)


def _output_phase_finish(frame, success):
    if frame is not None:
        finished = time.perf_counter_ns()
        _output_phase_emit(
            frame,
            "get_output_total",
            frame["started_perf_ns"],
            frame["thread_cpu_started_ns"],
            success,
        )
        try:
            _output_phase_queue_tpu_snapshot(frame, finished, success)
        except Exception:
            # Enqueue/metadata failure cannot replace the runtime's exception.
            global _diagnostic_unprepared_loss
            _diagnostic_unprepared_loss = True


def instrument_async_output(output):
    """Instrument the attested r64 get_output body without copying its logic.

    Four existing calls are bracketed in their original order; all other AST
    nodes, returned objects and exception paths remain the original runtime's.
    The source-file SHA is checked before compiling or replacing any method.
    """
    if not _output_phase_is_enabled():
        return False
    cls = type(output)
    original = getattr(cls, "get_output", None)
    if cls in _output_phase_classes:
        return original is _output_phase_classes[cls]
    try:
        import ast
        import hashlib
        import inspect
        import logging
        import textwrap
        from pathlib import Path

        if original is None or original.__code__.co_freevars:
            raise ValueError("Output method is not a plain runtime function")
        module = inspect.getmodule(original)
        if (
            module is None
            or module.__name__ != "vllm_torchtpu.runner.tpu_runner_async_output"
        ):
            raise ValueError("Unexpected output runtime module")
        if (
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            != _OUTPUT_PHASE_BASE_SHA256
        ):
            raise ValueError("Output runtime source SHA mismatch")
        source, first_line = inspect.getsourcelines(original)
        tree = ast.parse(textwrap.dedent("".join(source)))
        if any(
            isinstance(n, ast.Name) and n.id.startswith("_navi_phase_")
            for n in ast.walk(tree)
        ):
            raise ValueError("Reserved diagnostic local collides with runtime source")
        function = tree.body[0]
        if not isinstance(function, ast.FunctionDef) or function.decorator_list:
            raise ValueError("Unexpected output method definition")
        expected = {
            "self.kv_ready_event.synchronize": "kv_publish_event_wait",
            "self.copy_state.wait": "copy_state_wait",
            "self._hidden_nan_readback.consume": "hidden_nan_consume",
            "self._kv_cache_nan_readback.consume": "kv_nan_consume",
        }
        counts = dict.fromkeys(expected, 0)

        class BracketCalls(ast.NodeTransformer):
            def visit_Call(self, node):
                node = self.generic_visit(node)
                name = ast.unparse(node.func)
                if name not in expected:
                    return node
                counts[name] += 1
                return ast.copy_location(
                    ast.Call(
                        func=ast.Name(id="_navi_output_phase_call", ctx=ast.Load()),
                        args=[
                            ast.Name(id="_navi_phase_frame", ctx=ast.Load()),
                            ast.Constant(value=expected[name]),
                            node.func,
                            *node.args,
                        ],
                        keywords=node.keywords,
                    ),
                    node,
                )

        function = BracketCalls().visit(function)
        if any(value != 1 for value in counts.values()):
            raise ValueError("Expected output phase calls missing or duplicated")
        prefix = ast.parse(
            "_navi_phase_frame = _navi_output_phase_begin(self)\n_navi_phase_success = False"
        ).body

        # Mark success at each original return, including the early empty path.
        class MarkReturns(ast.NodeTransformer):
            def visit_Return(self, node):
                value = ast.Assign(
                    targets=[ast.Name(id="_navi_phase_result", ctx=ast.Store())],
                    value=(
                        node.value
                        if node.value is not None
                        else ast.Constant(value=None)
                    ),
                )
                success = ast.Assign(
                    targets=[ast.Name(id="_navi_phase_success", ctx=ast.Store())],
                    value=ast.Constant(value=True),
                )
                result = ast.Return(
                    value=ast.Name(id="_navi_phase_result", ctx=ast.Load())
                )
                return [
                    ast.copy_location(part, node) for part in (value, success, result)
                ]

        original_body = []
        for node in function.body:
            changed = MarkReturns().visit(node)
            original_body.extend(changed if isinstance(changed, list) else [changed])
        final = ast.parse(
            "_navi_output_phase_finish(_navi_phase_frame, _navi_phase_success)"
        ).body
        function.body = prefix + [
            ast.Try(body=original_body, handlers=[], orelse=[], finalbody=final)
        ]
        ast.fix_missing_locations(tree)
        ast.increment_lineno(tree, first_line - 1)
        namespace = dict(original.__globals__)
        # This is a separate runtime module from tpu_runner. Preserve the
        # attested get_output body, but remove its native timing logger from
        # the inference thread as well. Override the cloned globals only;
        # installation failure leaves the original method/logger untouched.
        namespace["logger"] = _RuntimeKvPublishDiagnosticLogger(
            original.__globals__["logger"], _KV_PUBLISH_WAIT_DIAGNOSTIC
        )
        namespace.update(
            _navi_output_phase_call=_output_phase_call,
            _navi_output_phase_begin=_output_phase_begin,
            _navi_output_phase_finish=_output_phase_finish,
        )
        exec(compile(tree, original.__code__.co_filename, "exec"), namespace)
        wrapped = namespace[original.__name__]
        wrapped.__defaults__, wrapped.__kwdefaults__ = (
            original.__defaults__,
            original.__kwdefaults__,
        )
        wrapped.__module__, wrapped.__qualname__ = (
            original.__module__,
            original.__qualname__,
        )
        wrapped.__doc__ = original.__doc__
        wrapped.__annotations__ = dict(original.__annotations__)
        wrapped.__dict__.update(original.__dict__)
        setattr(cls, "get_output", wrapped)
        _output_phase_classes[cls] = wrapped
        try:
            _diagnostic_info(
                logging.getLogger("vllm.navi_sched.output"),
                "[navi-vllm-output-phase-audit-installed] version=1 source_sha256=%s phases=4 runtime_kv_wait_delivery=bounded_async",
                _OUTPUT_PHASE_BASE_SHA256,
            )
        except Exception:
            pass
        return True
    except Exception:
        # The rollout preflight separately attests the source. At runtime an
        # unexpected diagnostic problem must preserve the original serving path.
        return False


# Qualified r64 host-only sampling partitions.
"""Passive host-clock partitions of the qualified r64 sampling path.

Original context managers, device waits, inputs and results are preserved.
These are host intervals; asynchronous device completion is not inferred.
"""
import hashlib
import inspect
import json
import logging
import os
import threading
import time
import weakref
from pathlib import Path

_SAMPLING_LOG = logging.getLogger("vllm.navi_sched.worker")
_SAMPLING_FLAG_PATH = None
_SAMPLING_RUNNER_SHA256 = (
    "a82e202721c1230c75a1183047fe2999b2a97f9dc620ba1cb72ce35b5d402ff6"
)
_SAMPLING_MARK = "_navi_r64_sampling_stage_audit"
_SAMPLING_LOCAL = threading.local()
_SAMPLING_METHODS = (
    "maybe_wait_for_kv_save",
    "get_finished_kv_transfers",
    "_build_padded_sampling_params",
    "_assemble_async_prefill_bootstrap",
    "_get_sampling_generator",
    "_update_placeholder",
)
_SAMPLING_MIN_NS = 2_000_000


def _sampling_compile_counters():
    # Read already loaded Python counters only. Never import torch, call a
    # backend getter, compile a graph or synchronize a device for diagnostics.
    import sys

    try:
        module = sys.modules.get("torch._dynamo.utils")
        counters = getattr(module, "counters", None)
        if not isinstance(counters, dict):
            return None
        result = {}
        for group, key, label in (
            ("stats", "unique_graphs", "unique_graphs"),
            ("stats", "calls_captured", "calls_captured"),
            ("frames", "total", "frames_total"),
            ("frames", "ok", "frames_ok"),
        ):
            values = counters.get(group)
            value = values.get(key) if isinstance(values, dict) else None
            result[label] = value if type(value) is int and value >= 0 else None
        return result
    except Exception:
        return None


def _sampling_clock():
    try:
        return time.perf_counter_ns(), time.thread_time_ns()
    except Exception:
        return None


def _sampling_read_clock(clock):
    try:
        value = clock()
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or any(type(x) is not int for x in value)
        ):
            return None
        return value
    except Exception:
        return None


def _sampling_sink(marker, fields):
    _diagnostic_info(
        _SAMPLING_LOG,
        "%s %s",
        marker,
        json.dumps(fields, sort_keys=True, separators=(",", ":")),
    )


def _sampling_event(stage, start, success, scope, clock, sink, minimum_ns):
    if scope is None or start is None:
        return
    end = _sampling_read_clock(clock)
    if end is None or end[0] - start[0] < minimum_ns:
        return
    try:
        fields = dict(
            scope["identity"],
            version=1,
            stage=stage,
            pid=os.getpid(),
            thread_id=threading.get_ident(),
            started_perf_ns=start[0],
            finished_perf_ns=end[0],
            thread_cpu_started_ns=start[1],
            thread_cpu_finished_ns=end[1],
            parent_started_perf_ns=scope["parent_started_perf_ns"],
            event_unix_ns=time.time_ns(),
            success=success,
            diagnostic_emit_wall_ns=scope["emit_wall_ns"],
            measurement="host_clock_original_call_or_context",
            new_device_events_or_waits=False,
            compile_counters_at_parent_start=scope["compile_counters_before"],
            compile_counters_at_event_end=_sampling_compile_counters(),
            compile_counter_scope="process_global_not_exclusive_to_substage",
        )
        before = _sampling_read_clock(clock)
        try:
            sink("[navi-r64-worker-sampling-stage-audit]", fields)
        finally:
            after = _sampling_read_clock(clock)
            if before is not None and after is not None:
                scope["emit_wall_ns"] += max(0, after[0] - before[0])
    except Exception:
        # Diagnostics cannot replace the original result or exception.
        pass


class _SamplingContext:
    def __init__(self, original, stage, scope, start, clock, sink, minimum_ns):
        self.original, self.stage, self.scope, self.start = (
            original,
            stage,
            scope,
            start,
        )
        self.clock, self.sink, self.minimum_ns = clock, sink, minimum_ns

    def __enter__(self):
        try:
            return self.original.__enter__()
        except BaseException:
            _sampling_event(
                self.stage,
                self.start,
                False,
                self.scope,
                self.clock,
                self.sink,
                self.minimum_ns,
            )
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        success = False
        try:
            result = self.original.__exit__(exc_type, exc_value, traceback)
            success = exc_type is None
            return result
        finally:
            _sampling_event(
                self.stage,
                self.start,
                success,
                self.scope,
                self.clock,
                self.sink,
                self.minimum_ns,
            )


class _SamplingStages:
    def __init__(self, original, clock, sink, minimum_ns):
        self._navi_sampling_original, self._navi_sampling_clock = original, clock
        self._navi_sampling_sink, self._navi_sampling_minimum_ns = sink, minimum_ns

    def __setattr__(self, name, value):
        if name.startswith("_navi_sampling_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._navi_sampling_original, name, value)

    def __delattr__(self, name):
        if name.startswith("_navi_sampling_"):
            object.__delattr__(self, name)
        else:
            delattr(self._navi_sampling_original, name)

    def __getattr__(self, name):
        return getattr(self._navi_sampling_original, name)

    def _make(self, method, name, *args, **kwargs):
        scope = getattr(_SAMPLING_LOCAL, "scope", None)
        start = (
            _sampling_read_clock(self._navi_sampling_clock)
            if scope is not None
            else None
        )
        original = getattr(self._navi_sampling_original, method)(name, *args, **kwargs)
        if scope is None:
            return original
        return _SamplingContext(
            original,
            method + ":" + str(name),
            scope,
            start,
            self._navi_sampling_clock,
            self._navi_sampling_sink,
            self._navi_sampling_minimum_ns,
        )

    def stage(self, name, *args, **kwargs):
        return self._make("stage", name, *args, **kwargs)

    def host_stage(self, name, *args, **kwargs):
        return self._make("host_stage", name, *args, **kwargs)


def _sampling_identity(observer_ref):
    try:
        observer = observer_ref()
        context = (
            observer.__dict__.get("_timing_context") if observer is not None else None
        )
        if not context:
            return None
        digest = context["request_set_sha256"]
        count, tokens = context["request_count"], context["scheduled_tokens"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or type(count) is not int
            or type(tokens) is not int
            or count <= 0
            or tokens <= 0
        ):
            return None
        return dict(
            request_set_sha256=digest, request_count=count, scheduled_tokens=tokens
        )
    except Exception:
        return None


def _sampling_call_wrapper(original, stage, clock, sink, minimum_ns):
    def wrapped(*args, **kwargs):
        scope = getattr(_SAMPLING_LOCAL, "scope", None)
        if scope is None:
            return original(*args, **kwargs)
        start, success = _sampling_read_clock(clock), False
        try:
            result = original(*args, **kwargs)
            success = True
            return result
        finally:
            _sampling_event(stage, start, success, scope, clock, sink, minimum_ns)

    return wrapped


def _sampling_install_runner(
    runner,
    observer,
    clock=_sampling_clock,
    sink=_sampling_sink,
    minimum_ns=_SAMPLING_MIN_NS,
):
    existing = vars(runner).get(_SAMPLING_MARK)
    if existing is not None:
        if existing["pid"] != os.getpid():
            raise ValueError("sampling probe belongs to a different process")
        return False
    # This optional probe is qualified for this exact r64 interface, not a
    # compatibility layer for other runtimes.
    original_sample = runner.sample_tokens
    original_stages = runner._stage_timing
    originals = {name: getattr(runner, name) for name in _SAMPLING_METHODS}
    assert all(callable(v) for v in originals.values())
    assert callable(original_sample)
    assert callable(original_stages.stage) and callable(original_stages.host_stage)
    observer_ref = weakref.ref(observer)
    saved = {
        name: (name in vars(runner), vars(runner).get(name))
        for name in (
            "sample_tokens",
            "_stage_timing",
            *_SAMPLING_METHODS,
            _SAMPLING_MARK,
        )
    }

    def sample(*args, **kwargs):
        identity = _sampling_identity(observer_ref)
        if identity is None:
            return original_sample(*args, **kwargs)
        prior = getattr(_SAMPLING_LOCAL, "scope", None)
        start = _sampling_read_clock(clock)
        scope = dict(
            identity=identity,
            parent_started_perf_ns=(start[0] if start else None),
            emit_wall_ns=0,
            compile_counters_before=_sampling_compile_counters(),
        )
        _SAMPLING_LOCAL.scope = scope
        success = False
        try:
            result = original_sample(*args, **kwargs)
            success = True
            return result
        finally:
            try:
                _sampling_event(
                    "sample_tokens_total",
                    start,
                    success,
                    scope,
                    clock,
                    sink,
                    minimum_ns,
                )
            finally:
                _SAMPLING_LOCAL.scope = prior

    replacements = dict(
        sample_tokens=sample,
        _stage_timing=_SamplingStages(original_stages, clock, sink, minimum_ns),
    )
    replacements.update(
        {
            name: _sampling_call_wrapper(method, name, clock, sink, minimum_ns)
            for name, method in originals.items()
        }
    )
    try:
        for name, value in replacements.items():
            setattr(runner, name, value)
            if getattr(runner, name) is not value:
                raise ValueError("sampling probe assignment did not stick")
        setattr(runner, _SAMPLING_MARK, dict(pid=os.getpid()))
    except BaseException:
        # Restore the original instance dictionary, including inherited methods.
        for name, (present, value) in saved.items():
            if present:
                vars(runner)[name] = value
            else:
                vars(runner).pop(name, None)
        raise
    return True


def _sampling_runner_source(runner):
    method = inspect.unwrap(type(runner).sample_tokens)
    path = Path(inspect.getsourcefile(method))
    return dict(
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_file=str(path),
    )


def install_worker_sampling_stage_audit(worker, observer):
    flag = (
        _SAMPLING_FLAG_PATH
        if _SAMPLING_FLAG_PATH is not None
        else Path(__file__).resolve().parents[4] / "worker-sampling-stage-audit.enabled"
    )
    try:
        if not flag.exists():
            return False
        if flag.read_bytes() != b"1\n":
            raise ValueError("invalid sampling diagnostic control")
        runner = worker.model_runner
        source = _sampling_runner_source(runner)
        if source["source_sha256"] != _SAMPLING_RUNNER_SHA256:
            raise ValueError("unqualified r64 sampling source")
        module = _sampling_runtime_log_module(runner)
        original_logger = module.logger
        already_async = isinstance(original_logger, _RuntimeKvPublishDiagnosticLogger)
        if already_async and original_logger.pid != os.getpid():
            raise ValueError("Runtime diagnostic logger belongs to a different process")
        replacement = (
            original_logger
            if already_async
            else _RuntimeKvPublishDiagnosticLogger(original_logger)
        )
        try:
            module.logger = replacement
            if module.logger is not replacement:
                raise ValueError("Runtime diagnostic logger assignment did not stick")
            installed = _sampling_install_runner(runner, observer)
            if not installed and not already_async:
                raise ValueError("Sampling probe and runtime logger disagree")
        except BaseException:
            if not already_async:
                module.logger = original_logger
            raise
        if installed:
            try:
                _sampling_sink(
                    "[navi-r64-worker-sampling-audit-installed]",
                    dict(
                        version=1,
                        pid=os.getpid(),
                        **source,
                        minimum_stage_duration_ns=_SAMPLING_MIN_NS,
                        new_device_events_or_waits=False,
                        original_timing_configuration_preserved=True,
                        runtime_kv_publish_diagnostic_delivery="bounded_async",
                        runtime_kv_publish_logger_type=type(
                            replacement.logger
                        ).__name__,
                        observed_methods=list(_SAMPLING_METHODS),
                    ),
                )
            except Exception:
                pass
        return installed
    except Exception as error:
        try:
            _sampling_sink(
                "[navi-r64-worker-sampling-audit-install-failed]",
                dict(version=1, pid=os.getpid(), error_type=type(error).__name__),
            )
        except Exception:
            pass
        return False


# Exact r64 Eagle source. This observes existing calls and tensor metadata only.
_DRAFT_COMPILE_SHA256 = (
    "6b8aed8b617e399d8afd670e5f5b5af13199563374936a541f8db44d975b9dcd"
)
_DRAFT_COMPILE_METHODS = (
    "_draft_seed_input_ids",
    "_draft_gather_carries",
    "_draft_propose_token",
    "_draft_combine_hidden_states",
)
_DRAFT_COMPILE_MARK = "_navi_draft_compile_audit"


def _draft_compile_source(drafter):
    import hashlib
    import inspect

    source = Path(
        inspect.getsourcefile(inspect.unwrap(type(drafter)._draft_seed_input_ids))
    )
    return dict(
        source_file=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )


def _draft_compile_tensor_metadata(args, kwargs):
    import sys

    try:
        tensor_type = getattr(sys.modules.get("torch"), "Tensor", None)
        if not isinstance(tensor_type, type):
            return None
        result = []
        for name, value in [(str(i), v) for i, v in enumerate(args)] + list(
            kwargs.items()
        ):
            if not isinstance(value, tensor_type):
                continue
            shape, strides = tuple(value.shape), tuple(value.stride())
            if len(shape) > 8 or len(strides) != len(shape):
                return None
            if any(type(v) is not int or v < 0 for v in (*shape, *strides)):
                return None
            result.append(
                dict(
                    argument=name,
                    shape=shape,
                    strides=strides,
                    dtype=str(value.dtype),
                    device_type=str(value.device.type),
                    has_dynamic_dim_marks=bool(
                        getattr(value, "_dynamo_dynamic_indices", None)
                    ),
                )
            )
        return result
    except Exception:
        return None


def _draft_compile_install(
    drafter, clock=_sampling_clock, sink=_sampling_sink, minimum_ns=_SAMPLING_MIN_NS
):
    existing = vars(drafter).get(_DRAFT_COMPILE_MARK)
    if existing is not None:
        if existing["pid"] != os.getpid():
            raise ValueError("draft compile probe belongs to another process")
        return False
    originals = {name: getattr(drafter, name) for name in _DRAFT_COMPILE_METHODS}
    if not all(callable(v) for v in originals.values()):
        raise ValueError("incomplete r64 draft interface")
    saved = {
        name: (name in vars(drafter), vars(drafter).get(name))
        for name in (*_DRAFT_COMPILE_METHODS, _DRAFT_COMPILE_MARK)
    }

    def wrap(original, name):
        def call(*args, **kwargs):
            parent = getattr(_SAMPLING_LOCAL, "scope", None)
            if parent is None:
                return original(*args, **kwargs)
            start, success = _sampling_read_clock(clock), False
            scope = dict(parent)
            scope["identity"] = dict(
                parent["identity"],
                compile_counters_at_call_start=_sampling_compile_counters(),
                draft_argument_metadata=_draft_compile_tensor_metadata(args, kwargs),
            )
            try:
                result = original(*args, **kwargs)
                success = True
                return result
            finally:
                try:
                    # Always retain the seed's cached calls for cold/warm comparison.
                    threshold = 0 if name == "_draft_seed_input_ids" else minimum_ns
                    _sampling_event(
                        "draft_helper:" + name,
                        start,
                        success,
                        scope,
                        clock,
                        sink,
                        threshold,
                    )
                finally:
                    parent["emit_wall_ns"] += max(
                        0, scope["emit_wall_ns"] - parent["emit_wall_ns"]
                    )

        return call

    try:
        for name, original in originals.items():
            replacement = wrap(original, name)
            setattr(drafter, name, replacement)
            if getattr(drafter, name) is not replacement:
                raise ValueError("draft compile probe assignment did not stick")
        setattr(drafter, _DRAFT_COMPILE_MARK, dict(pid=os.getpid()))
    except BaseException:
        for name, (present, value) in saved.items():
            if present:
                vars(drafter)[name] = value
            else:
                vars(drafter).pop(name, None)
        raise
    return True


def install_worker_draft_compile_audit(worker):
    flag = (
        _SAMPLING_FLAG_PATH
        if _SAMPLING_FLAG_PATH is not None
        else Path(__file__).resolve().parents[4] / "worker-sampling-stage-audit.enabled"
    )
    try:
        if not flag.exists():
            return False
        if flag.read_bytes() != b"1\n":
            raise ValueError("invalid sampling diagnostic control")
        runner = worker.model_runner
        if vars(runner).get(_SAMPLING_MARK, {}).get("pid") != os.getpid():
            raise ValueError("parent sampling probe must be active first")
        drafter = runner._async_drafter
        source = _draft_compile_source(drafter)
        if source["source_sha256"] != _DRAFT_COMPILE_SHA256:
            raise ValueError("unqualified r64 draft source")
        installed = _draft_compile_install(drafter)
        if installed:
            try:
                _sampling_sink(
                    "[navi-r64-draft-compile-audit-installed]",
                    dict(
                        version=1,
                        pid=os.getpid(),
                        **source,
                        observed_methods=list(_DRAFT_COMPILE_METHODS),
                        seed_cached_calls_recorded=True,
                        tensor_metadata_only=True,
                        original_arguments_unchanged=True,
                        new_device_events_or_waits=False,
                    ),
                )
            except Exception:
                pass
        return installed
    except Exception as error:
        try:
            _sampling_sink(
                "[navi-r64-draft-compile-audit-install-failed]",
                dict(version=1, pid=os.getpid(), error_type=type(error).__name__),
            )
        except Exception:
            pass
        return False
