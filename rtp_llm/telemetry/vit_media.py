"""Media-level tracing. CPU workers return bounded timing sidecars, not SDK objects.

Raw input URLs are intentionally exported, as requested by the operator. They
are attributes only: never span names or metric labels. No CUDA synchronization
or network export is performed by these helpers or by pool callbacks.
"""

import contextvars
import functools
import inspect
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from rtp_llm.telemetry import attributes as trace_attrs
from rtp_llm.telemetry import tracing

PREFIX = "rtp_llm.vit."
MEDIA = PREFIX + "media."
_current = contextvars.ContextVar("vit_media", default=())
_worker = contextvars.ContextVar("vit_worker_timings", default=None)
_outcome = contextvars.ContextVar("vit_stage_outcome", default=None)
_download_depth = contextvars.ContextVar("vit_download_depth", default=0)
_input_indices = contextvars.ContextVar("vit_input_indices", default=None)
_MAX_RECORDS = 256


def _end(span, error=None, end_ns=None):
    if span is None:
        return
    try:
        if error is not None:
            name = error if isinstance(error, str) else type(error).__name__
            span.set_attribute("error.type", name)
            span.set_status(tracing.trace.StatusCode.ERROR)
        else:
            span.set_status(tracing.trace.StatusCode.OK)
        span.end(end_time=end_ns)
    except Exception:
        pass


class MediaTrace:
    def __init__(self, span, index, input_id):
        self.span = span
        self.index = index
        self.input_id = input_id
        self.detached = False
        self.context = tracing.trace.set_span_in_context(span)
        self.request_trace_id = format(span.get_span_context().trace_id, "032x")
        self._lock = threading.RLock()
        self._stages = {}
        self._ended = False
        self.attribute(trace_attrs.RTP_LLM_REQUEST_TRACE_ID, self.request_trace_id)

    def attribute(self, key, value):
        try:
            with self._lock:
                if not self._ended:
                    self.span.set_attribute(key, value)
        except Exception:
            pass

    def start(self, name, attributes=None, start_ns=None, links=()):
        try:
            with self._lock:
                if self._ended:
                    return None
                return tracing.get_tracer().start_span(
                    PREFIX + name,
                    context=self.context,
                    attributes={
                        **(attributes or {}),
                        trace_attrs.RTP_LLM_REQUEST_TRACE_ID: self.request_trace_id,
                    },
                    start_time=start_ns,
                    links=links,
                )
        except Exception:
            return None

    def begin(self, name, attributes=None):
        with self._lock:
            if name not in self._stages and not self._ended:
                self._stages[name] = self.start(name, attributes)

    def end(self, name, error=None):
        with self._lock:
            span = self._stages.pop(name, None)
        _end(span, error)

    def link(self, context):
        try:
            if (
                context is not None
                and context.is_valid
                and context != self.span.get_span_context()
            ):
                self.span.add_link(context)
        except Exception:
            pass

    def finish(self, error=None):
        with self._lock:
            if self._ended:
                return
            self._ended = True
            stages, self._stages = self._stages, {}
            media_span = self.span
            # A cached work item's publication callback can retain this handle.
            # Keep only IDs/flags for links, not the SDK span and its attributes.
            self.span = tracing.trace.NonRecordingSpan(media_span.get_span_context())
            self.context = tracing.trace.set_span_in_context(self.span)
        for span in stages.values():
            _end(span, error)
        _end(media_span, error)


def current():
    return _current.get()


@contextmanager
def bind(media):
    token = _current.set(tuple(media))
    try:
        yield
    finally:
        _current.reset(token)


def _new_media(inputs, request_id):
    if not tracing.is_telemetry_active():
        return ()
    state = tracing.CURRENT_TRACE_STATE.get()
    if state is None:
        return ()
    parent = tracing.CURRENT_INTERNAL_CONTEXT.get() or state.server_context
    parent_span = tracing.trace.get_current_span(parent).get_span_context()
    if parent_span.is_valid and not parent_span.trace_flags.sampled:
        return ()
    result = []
    try:
        for index, item in enumerate(inputs):
            index = (_input_indices.get() or {}).get(id(item), index)
            span = tracing.get_tracer().start_span(
                PREFIX + "media",
                context=parent,
                attributes={
                    "request_id": str(request_id),
                    MEDIA + "index": index,
                    MEDIA + "url": item.url,
                    MEDIA + "type": int(item.mm_type),
                },
            )
            if not span.is_recording():
                span.end()
                continue
            result.append(MediaTrace(span, index, id(item)))
    except Exception:
        for media in result:
            media.finish()
        return ()
    return tuple(result)


@contextmanager
def original_input_indices(inputs):
    """Keep RPC input positions when hash lookup filters/deduplicates its work."""
    token = _input_indices.set({id(item): index for index, item in enumerate(inputs)})
    try:
        yield
    finally:
        _input_indices.reset(token)


def request(method):
    """Own synchronous observers; a cache-miss producer transfers its lifetime."""
    signature = inspect.signature(method)

    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        if current() or not tracing.is_telemetry_active():
            return method(*args, **kwargs)
        try:
            arguments = signature.bind(*args, **kwargs).arguments
            media = _new_media(
                arguments.get("mm_inputs", ()), arguments.get("request_id", 0)
            )
        except Exception:
            media = ()
        error = None
        try:
            with bind(media):
                return method(*args, **kwargs)
        except BaseException as caught:
            error = caught
            raise
        finally:
            for item in media:
                if not item.detached:
                    item.finish(error)

    return wrapped


def select_inputs(inputs):
    """Preserve occurrence identity through filtered lists and duplicate URLs."""
    available = list(current())
    selected = []
    for value in inputs:
        found = next((m for m in available if m.input_id == id(value)), None)
        selected.append(found)
        if found is not None:
            available.remove(found)
    return selected


def claim(media, entry, state):
    if media is None:
        return
    try:
        media.attribute(MEDIA + "cache_state", state)
        if state == "miss":
            media.detached = True
            entry._vit_media_producer = media
            entry._vit_media_context = media.span.get_span_context()
        else:
            media.link(getattr(entry, "_vit_media_context", None))
    except Exception:
        pass


def finish_producer(entry, error=None):
    media = getattr(entry, "_vit_media_producer", None)
    if media is not None:
        media.finish(error or getattr(entry, "error", None))
        entry._vit_media_producer = None


def compute_task(method):
    @functools.wraps(method)
    def wrapped(self, inputs, key, entry, *args, **kwargs):
        media = getattr(entry, "_vit_media_producer", None)
        if media is not None:
            media.end("compute.queue")
        with bind((media,) if media is not None else ()):
            return method(self, inputs, key, entry, *args, **kwargs)

    return wrapped


@contextmanager
def stage(name, media=None, attributes=None):
    recorder = _worker.get()
    if recorder is not None:
        with recorder.stage(name, attributes):
            yield
        return
    members = tuple(current() if media is None else media)
    if not members:
        yield
        return
    attributes = {PREFIX + "shared_execution": len(members) > 1, **(attributes or {})}
    spans = [m.start(name, attributes) for m in members if m is not None]
    outcome = {"attributes": {}, "error": None}
    token = _outcome.set(outcome)
    error = None
    try:
        yield
    except BaseException as caught:
        error = caught
        raise
    finally:
        _outcome.reset(token)
        for span in spans:
            try:
                if span is not None:
                    if "name" in outcome:
                        span.update_name(PREFIX + outcome["name"])
                    span.set_attributes(outcome["attributes"])
            except Exception:
                pass
            _end(span, error or outcome["error"])


def record_verdict(verdict):
    outcome = _outcome.get()
    if outcome is not None and verdict is not None:
        outcome["attributes"].update(
            {
                PREFIX + "greennet.code": verdict.code,
                PREFIX + "greennet.passed": verdict.passed,
            }
        )
        if not verdict.passed:
            outcome["error"] = (
                "GREENNET_REJECTED" if verdict.code == 2 else "GREENNET_PROCESS_ERROR"
            )


def download_cache_hit():
    outcome = _outcome.get()
    if outcome is not None:
        outcome["name"] = "download.cache"
        outcome["attributes"][PREFIX + "download.cache_hit"] = True


def item_media(items):
    return tuple(
        m for item in items for m in getattr(item, "media_traces", ()) if m is not None
    )


def enqueue(items):
    for media in item_media(items):
        media.begin("embedding.queue", {PREFIX + "includes_prefetch": True})


def embedding_batch(method):
    @functools.wraps(method)
    def wrapped(mm_part, items, *args, **kwargs):
        media = item_media(items)
        if not media:
            return method(mm_part, items, *args, **kwargs)
        for member in media:
            member.end("embedding.queue")
        batch = None
        spans = []
        try:
            links = [tracing.trace.Link(m.span.get_span_context()) for m in media]
            attributes = {
                PREFIX
                + "batch.media_count": sum(len(item.mm_inputs) for item in items),
                PREFIX + "batch.work_item_count": len(items),
                PREFIX + "timing_scope": "host_call",
            }
            batch = media[0].start("embedding.batch", attributes, links=links)
            if batch is not None:
                batch_id = format(batch.get_span_context().span_id, "016x")
                attributes.update(
                    {"rtp_llm.batch_id": batch_id, PREFIX + "shared_execution": True}
                )
                spans = [
                    m.start(
                        "embedding",
                        attributes,
                        links=[tracing.trace.Link(batch.get_span_context())],
                    )
                    for m in media
                ]
        except Exception:
            pass
        error = None
        try:
            return method(mm_part, items, *args, **kwargs)
        except BaseException as caught:
            error = caught
            raise
        finally:
            for span in spans:
                _end(span, error)
            _end(batch, error)

    return wrapped


@dataclass
class WorkerTimings:
    """Picklable timestamps only. No tensors, URLs, spans or SDK dependency."""

    count: int
    records: list = field(default_factory=list)
    dropped: int = 0
    index: object = None
    cpu_start: object = None
    download_depth: int = 0

    def record(self, name, start, end, error=None, attributes=None):
        if len(self.records) >= _MAX_RECORDS:
            self.dropped += 1
            return
        attrs = dict(attributes or {})
        attrs[PREFIX + "worker.pid"] = os.getpid()
        attrs[PREFIX + "shared_execution"] = self.index is None and self.count > 1
        indices = tuple(range(self.count)) if self.index is None else (self.index,)
        self.records.append((indices, name, start, max(start, end), error, attrs))

    def pause_cpu(self, error=None):
        if self.cpu_start is not None:
            self.record("preprocess", self.cpu_start, time.time_ns(), error)
            self.cpu_start = None

    @contextmanager
    def stage(self, name, attributes=None):
        start = time.time_ns()
        error = None
        outcome = {"attributes": dict(attributes or {}), "error": None, "name": name}
        token = _outcome.set(outcome)
        try:
            yield
        except BaseException as caught:
            error = type(caught).__name__
            raise
        finally:
            _outcome.reset(token)
            self.record(
                outcome["name"],
                start,
                time.time_ns(),
                error or outcome["error"],
                outcome["attributes"],
            )


def worker_spec(media, submitted_ns=None):
    if not any(m is not None for m in media):
        return None
    return {"count": len(media), "submitted_ns": submitted_ns or time.time_ns()}


@contextmanager
def collect_worker(spec, preprocess=False):
    recorder = WorkerTimings(spec["count"]) if spec else None
    token = _worker.set(recorder)
    error = None
    try:
        if recorder is not None and preprocess:
            now = time.time_ns()
            recorder.record("preprocess.queue", spec["submitted_ns"], now)
            recorder.cpu_start = now
        yield recorder
    except BaseException as caught:
        error = type(getattr(caught, "error", caught)).__name__
        raise
    finally:
        if recorder is not None:
            recorder.pause_cpu(error)
        _worker.reset(token)


@contextmanager
def input_scope(index):
    """For a model's existing multi-input loop; never split its actual work."""
    recorder = _worker.get()
    if recorder is None:
        yield
        return
    previous = recorder.index
    recorder.pause_cpu()
    recorder.index = index
    recorder.cpu_start = time.time_ns()
    error = None
    try:
        yield
    except BaseException as caught:
        error = type(caught).__name__
        raise
    finally:
        recorder.pause_cpu(error)
        recorder.index = previous
        recorder.cpu_start = time.time_ns()


def download(method):
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        recorder = _worker.get()
        if recorder is None:
            if _download_depth.get():
                return method(*args, **kwargs)
            token = _download_depth.set(1)
            try:
                with stage("download"):
                    return method(*args, **kwargs)
            finally:
                _download_depth.reset(token)
        if recorder.download_depth:
            return method(*args, **kwargs)
        recorder.pause_cpu()
        recorder.download_depth += 1
        try:
            with recorder.stage("download"):
                return method(*args, **kwargs)
        finally:
            recorder.download_depth -= 1
            recorder.cpu_start = time.time_ns()

    return wrapped


def consume(media, recorder):
    if recorder is None:
        return
    try:
        for indices, name, start, end, error, attributes in recorder.records:
            for index in indices:
                if index < len(media) and media[index] is not None:
                    span = media[index].start(name, attributes, start_ns=start)
                    _end(span, error, end)
        if recorder.dropped:
            for member in media:
                if member is not None:
                    member.attribute(
                        PREFIX + "timing_records_dropped", recorder.dropped
                    )
    except Exception:
        pass


def inspection(method):
    """Track the asynchronous branch through its verdict, not submit's return."""

    @functools.wraps(method)
    def wrapped(self, mm_inputs, *args, **kwargs):
        media = tuple(m for m in select_inputs(mm_inputs) if m is not None)
        enabled = self._greennet_enabled()
        for member in media:
            member.attribute(MEDIA + "greennet_enabled", enabled)
        if not enabled or not media:
            return method(self, mm_inputs, *args, **kwargs)
        spans = [m.start("greennet.total") for m in media]

        def finish(future):
            error = None
            try:
                verdict = future.result()
                if verdict is not None:
                    for span in spans:
                        if span is not None:
                            span.set_attribute(PREFIX + "greennet.code", verdict.code)
                            span.set_attribute(
                                PREFIX + "greennet.passed", verdict.passed
                            )
                    if not verdict.passed:
                        error = (
                            "GREENNET_REJECTED"
                            if verdict.code == 2
                            else "GREENNET_PROCESS_ERROR"
                        )
            except BaseException as caught:
                error = caught
            for span in spans:
                _end(span, error)

        try:
            with bind(media):
                result = method(self, mm_inputs, *args, **kwargs)
        except BaseException as error:
            for span in spans:
                _end(span, error)
            raise
        future = result[1]
        if future is not None:
            try:
                future.add_done_callback(finish)
            except Exception:
                for span in spans:
                    _end(span, "TELEMETRY_CALLBACK_FAILED")
        else:
            for span in spans:
                if span is not None:
                    span.set_attribute(PREFIX + "greennet.reused", True)
                _end(span)
        return result

    return wrapped
