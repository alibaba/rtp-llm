"""Graph table ownership and attention-bounded decode lookup prefetch."""

import os
import weakref

import torch


def enabled():
    return os.getenv("DSV41_DECODE_ENGRAM_PREFETCH", "1") == "1"


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
    """Prepare the selected attention schedule without allocating during capture."""
    if not enabled():
        return None
    from ._engram_decode_schedule import prepare_module

    layout = getattr(module, "layout", None)
    index = getattr(module, "layer_hash_index", None)
    if (
        layout is None
        or type(index) is not int
        or not 0 <= index < len(layout.layer_ids)
    ):
        return None
    return prepare_module(module, layout.layer_ids[index])


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


class DecodeEngramPrefetch:
    """Own per-forward work and captured table generations for the phase schedule."""

    def __init__(self, model, metadata):
        self.work = []
        self._schedule = None
        self._capture_permit = self._capture_metadata = None
        asynchronous = enabled()
        device_scales = os.getenv("DSV41_ENGRAM_DEVICE_SCALES", "1") == "1"
        if not asynchronous and not device_scales:
            return
        capture_permit_type = None
        if device_scales and torch.cuda.is_current_stream_capturing():
            from rtp_llm.models_py.modules.dsv41.engram import _DeviceScaleCapturePermit

            capture_permit_type = _DeviceScaleCapturePermit
            capture_permit_type.check_available()
        resources = getattr(metadata, "_engram_decode_tables", None)
        if resources is not None:
            resources.validate()
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
        try:
            if table_entries and torch.cuda.is_current_stream_capturing():
                resources = _GraphTables(model, tuple(table_entries))
                if capture_permit_type is not None:
                    capture_permit_type.check_available()
                metadata._engram_decode_tables = resources
                if capture_permit_type is not None:
                    self._capture_metadata = metadata
                    self._capture_permit = capture_permit_type(self, resources)
                    self._capture_permit.install()
            if asynchronous:
                from ._engram_decode_schedule import DecodeEngramSchedule

                self._schedule = DecodeEngramSchedule(self, model, metadata)
        except BaseException:
            self.close()
            raise

    def before_layer(self, index):
        if self._schedule is not None:
            self._schedule.before_layer(index)

    def close(self):
        # Join abandoned work before releasing bindings or a capture permit.
        try:
            for work in self.work:
                work.wait()
                if work.module._lookup_work is work:
                    work.module._lookup_work = None
            self.work.clear()
        finally:
            try:
                schedule = getattr(self, "_schedule", None)
                if schedule is not None:
                    schedule.close()
                    self._schedule = None
            finally:
                permit = getattr(self, "_capture_permit", None)
                if permit is not None:
                    permit.close()
                    self._capture_permit = self._capture_metadata = None
