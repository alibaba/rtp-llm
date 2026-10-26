"""Decode lookahead whose producer/consumer edges are captured in the graph."""

import os
import weakref

import torch


def enabled():
    return os.getenv("DSV41_DECODE_ENGRAM_PREFETCH", "0") == "1"


class _TableSnapshot:
    """Table generation only: no auxiliary stream is needed for sync Graphs."""

    def __init__(self, embedding):
        self.embedding = embedding
        self.uva, self.pinned = embedding._uva, embedding._pinned
        self.device_scale_lease = embedding._device_scale_resources()

    def matches(self, embedding):
        return (
            embedding is self.embedding
            and embedding._uva is self.uva
            and embedding._pinned is self.pinned
            and embedding._device_scale_lease is self.device_scale_lease
        )


class _LookupStream:
    def __init__(self, embedding, device):
        self.embedding = embedding
        self.uva = embedding._uva
        self.pinned = embedding._pinned
        self.stream = torch.cuda.Stream(device=device, priority=0)
        # An external record node also updates the event on graph replay.
        # Host-table unregister must wait for actual reads, not capture time.
        self.done = torch.cuda.Event(external=True)
        self.done.record(torch.cuda.current_stream(device))
        self.pinned._lookup_events.append(self.done)

    def matches(self, embedding):
        return (
            embedding is self.embedding
            and embedding._uva is self.uva
            and embedding._pinned is self.pinned
        )


def prepare(module):
    """Initialize streams during warmup; cold capture keeps synchronous lookup."""
    if not enabled() or not module.q_weight.is_cuda:
        return None
    embedding = module.embed_tokens
    if embedding._uva is None or embedding._pinned is None:
        return None
    state = getattr(module, "_decode_lookup_stream", None)
    if state is None or not state.matches(embedding):
        if torch.cuda.is_current_stream_capturing():
            return None
        state = _LookupStream(embedding, module.q_weight.device)
        module._decode_lookup_stream = state
    return state


class _GraphTables:
    """Metadata owns registered host tables for exactly this captured graph."""

    def __init__(self, model, entries):
        self.model = weakref.ref(model)
        self.entries = entries

    def validate(self):
        model = self.model()
        if model is None or any(
            index >= len(model.layers)
            or model.layers[index].engram is not module
            or not state.matches(module.embed_tokens)
            for index, module, state in self.entries
        ):
            raise RuntimeError(
                "Engram tables changed; recreate decode CUDA graphs and metadata"
            )


class _LookupWork:
    def __init__(self, module, ids, state):
        from rtp_llm.models_py.modules.dsv41.engram import _lookup_for_projection

        self.module, self.ids, self.state = module, ids, state
        self.current = torch.cuda.current_stream(ids.device)
        self.rows = None
        self.joined = set()
        state.stream.wait_stream(self.current)
        try:
            with torch.cuda.stream(state.stream):
                try:
                    self.rows = _lookup_for_projection(
                        module, ids, ids.device, prefetch=True
                    )
                finally:
                    state.done.record(state.stream)
        except BaseException:
            self.wait()
            raise

    def wait(self, stream=None):
        stream = self.current if stream is None else stream
        if stream not in self.joined:
            stream.wait_stream(self.state.stream)
            if self.rows is not None:
                outputs = self.rows if isinstance(self.rows, tuple) else (self.rows,)
                for tensor in outputs:
                    tensor.record_stream(stream)
            self.joined.add(stream)

    def consume(self, ids, hidden):
        outputs = self.rows if isinstance(self.rows, tuple) else (self.rows,)
        if (
            ids is not self.ids
            or any(
                t is None or hidden.shape[0] != t.shape[0] or hidden.device != t.device
                for t in outputs
            )
            or not self.state.matches(self.module.embed_tokens)
        ):
            raise RuntimeError("Engram decode lookup inputs changed within forward")
        self.wait(torch.cuda.current_stream(hidden.device))
        return self.rows


class DecodeEngramPrefetch:
    """Two-layer lookahead; Python bookkeeping is discarded after capture.

    Replay reruns the hash/lookup kernels and fork/join edges. Each capture's
    allocator owns its output buffers; different batch graphs share no outputs.
    """

    def __init__(self, model, metadata):
        self.entries, self.work = [], []
        self._capture_permit = self._capture_metadata = None
        asynchronous = enabled()
        device_scales = os.getenv("DSV41_ENGRAM_DEVICE_SCALES", "1") == "1"
        if not asynchronous and not device_scales:
            return
        capture_permit_type = None
        if device_scales and torch.cuda.is_current_stream_capturing():
            from rtp_llm.models_py.modules.dsv41.engram import _DeviceScaleCapturePermit

            capture_permit_type = _DeviceScaleCapturePermit
            # Reject before replacing an outer plan's metadata owner.
            capture_permit_type.check_available()
        resources = getattr(metadata, "_engram_decode_tables", None)
        if resources is not None:
            resources.validate()
        rows = metadata.batch_size * metadata.q_len_per_req
        table_entries = []
        for index, layer in enumerate(model.layers):
            module = layer.engram
            if module is None:
                continue
            embedding = module.embed_tokens
            if (
                device_scales
                and getattr(embedding, "_device_scale_lease", None) is not None
            ):
                table_entries.append((index, module, _TableSnapshot(embedding)))
            ids = layer.engram_hashes
            if not asynchronous or index >= 20 or ids is None:
                continue
            if (
                not ids.is_cuda
                or ids.dtype != torch.int64
                or ids.ndim != 2
                or not ids.is_contiguous()
                or ids.shape != (rows, module.layout.n_hash_cols)
                or ids.device != module.q_weight.device
                or not 0 < rows <= 32768
            ):
                continue
            state = prepare(module)
            if state is not None:
                self.entries.append((index, module, ids, state))
        # Keep stream states too: old Graphs retain their original auxiliary
        # stream and external done event even after a later generation reload.
        table_entries.extend((i, m, s) for i, m, _, s in self.entries)
        if table_entries and torch.cuda.is_current_stream_capturing():
            # Publish the future-replay owner before authorizing any lookup.
            resources = _GraphTables(model, tuple(table_entries))
            if capture_permit_type is not None:
                capture_permit_type.check_available()
            metadata._engram_decode_tables = resources
            if capture_permit_type is not None:
                self._capture_metadata = metadata
                self._capture_permit = capture_permit_type(self, resources)
                try:
                    # Last constructor action: no half-built plan gets a permit.
                    self._capture_permit.install()
                except BaseException:
                    self._capture_permit.close()
                    self._capture_permit = self._capture_metadata = None
                    raise

    def before_layer(self, index):
        for target, module, ids, state in self.entries:
            if max(0, target - 2) == index:
                if module._lookup_work is not None:
                    raise RuntimeError("Engram lookup is already pending")
                work = _LookupWork(module, ids, state)
                self.work.append(work)
                module._lookup_work = work

    def close(self):
        # Join abandoned work too, so a failed forward cannot leave an unjoined
        # capture branch or let the allocator reuse an in-flight output.
        try:
            for work in self.work:
                work.wait()
                if work.module._lookup_work is work:
                    work.module._lookup_work = None
            self.work.clear()
        finally:
            # Keep legacy __new__ CPU fixtures and make failed joins fail closed.
            permit = getattr(self, "_capture_permit", None)
            if permit is not None:
                permit.close()
                self._capture_permit = self._capture_metadata = None
