# SPDX-License-Identifier: Apache-2.0
"""Attention-bounded native lookup scheduling for positive decode token counts.

L1 launches at verify graph entry or L0 entry and joins before FFN. L14
partitions the flattened batch/query rows over eight attention pre/post windows
in L10..L13. Empty pieces launch no kernels. Unsupported producers keep the
serial lookup route.
"""

import logging

import torch

from . import _engram_decode_prefetch as prefetch
from . import _engram_quantized_slice as sliced
from . import _engram_quantized_triton as direct

L1_GRID = 120
L14_GRID = 80
WINDOWS = 8


def partition_tokens(tokens):
    """Balanced contiguous pieces, with nonempty pieces first for small M."""
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("Scheduled lookup requires a positive token count")
    size, remainder = divmod(tokens, WINDOWS)
    return tuple(
        (index * size + min(index, remainder), size + (index < remainder))
        for index in range(WINDOWS)
    )


def _eligible(module):
    embedding = module.embed_tokens
    return (
        getattr(module, "_direct_fp8_lookup", False) is True
        and module.q_weight.is_cuda
        and getattr(embedding, "supports_decode_prefetch", False)
        and module.layout.n_hash_cols == 24
        and module.layout.head_dim == 256
    )


class _ScheduleStream(prefetch._LookupStream):
    def __init__(self, module, target, stream):
        super().__init__(module.embed_tokens, module.q_weight.device, stream)
        self.target = target
        self.grid = min(
            L1_GRID if target == 1 else L14_GRID, module.embed_tokens._num_sms
        )
        self.ready = False


def prepare_module(module, target, *, model=None):
    """Warm shape-independent kernels on private buffers outside capture."""
    if not prefetch.enabled() or target not in (1, 14) or not _eligible(module):
        return None
    state = getattr(module, "_decode_scheduled_stream", None)
    if (
        state is not None
        and state.target == target
        and state.matches(module.embed_tokens)
    ):
        return state if state.ready else None
    if torch.cuda.is_current_stream_capturing():
        return None
    # L1 is consumed before L14 starts. Share their queue within one model,
    # while each table generation retains its own external completion event.
    owner = module if model is None else model
    device = module.q_weight.device
    streams = vars(owner).setdefault("_engram_decode_lookup_streams", {})
    if device not in streams:
        streams[device] = torch.cuda.Stream(device=device)
    state = _ScheduleStream(module, target, streams[device])
    state.stream.wait_stream(torch.cuda.current_stream(module.q_weight.device))
    try:
        with module.embed_tokens.lookup_guard(), torch.cuda.stream(state.stream):
            ids = torch.zeros((9, 24), dtype=torch.int64, device=module.q_weight.device)
            outputs = direct.allocate_outputs(ids)
            weight, scales = module.embed_tokens._uva
            try:
                if target == 1:
                    direct.prepare_lookup(
                        weight, scales, ids, outputs, num_sms=state.grid
                    )
                else:
                    for start, count in ((0, 2), (8, 1)):
                        sliced.prepare(
                            weight, scales, ids, outputs, start, count, grid=state.grid
                        )
            finally:
                state.done.record(state.stream)
    finally:
        state.stream.synchronize()
    state.ready = True
    module._decode_scheduled_stream = state
    logging.info(
        "Engram decode prefetch prepared: layer=%d grid=%d scale_shards=%d external=%s",
        target,
        state.grid,
        getattr(module.embed_tokens, "_scale_shard_count", 1),
        getattr(module.embed_tokens, "_external_storage", None) is not None,
    )
    return state


class _ScheduledWork:
    def __init__(self, module, ids, state):
        self.module, self.ids, self.state = module, ids, state
        self.tokens = ids.shape[0]
        self.pieces = partition_tokens(self.tokens)
        self.current = torch.cuda.current_stream(ids.device)
        self.capturing = torch.cuda.is_current_stream_capturing()
        self.outputs = direct.allocate_outputs(ids)
        self.rows = self.outputs.pair()
        self.next_token = self.epoch = 0
        self.joined = {}

    def launch(self, start, count, *, ready=None):
        if start != self.next_token or count < 0 or start + count > self.tokens:
            raise RuntimeError(
                "Scheduled lookup slices must cover the request in order"
            )
        if self.state.target == 1 and (start, count) != (0, self.tokens):
            raise RuntimeError("L1 scheduled lookup must launch the whole request")
        if count == 0:
            return
        embedding = self.module.embed_tokens
        if not self.state.matches(embedding):
            raise RuntimeError("Scheduled lookup table generation changed")
        if self.capturing:
            from .engram import _DeviceScaleCapturePermit

            _DeviceScaleCapturePermit.require(
                embedding, embedding._device_scale_resources()
            )
        if ready is None:
            self.state.stream.wait_stream(torch.cuda.current_stream(self.ids.device))
        else:
            self.state.stream.wait_event(ready)
        for tensor in (self.ids, *self.rows):
            tensor.record_stream(self.state.stream)
        # Even a failed enqueue must be drained by the enclosing plan's close.
        self.epoch += 1
        with embedding.lookup_guard(), torch.cuda.stream(self.state.stream):
            try:
                weight, scales = embedding._uva
                if self.state.target == 1:
                    direct.launch_epilogue(
                        weight, scales, self.ids, self.outputs, self.state.grid
                    )
                else:
                    sliced.launch(
                        weight,
                        scales,
                        self.ids,
                        self.outputs,
                        start,
                        count,
                        grid=self.state.grid,
                    )
            finally:
                self.state.done.record(self.state.stream)
        self.next_token += count

    def wait(self, stream=None):
        stream = self.current if stream is None else stream
        if self.epoch and self.joined.get(stream, 0) < self.epoch:
            stream.wait_stream(self.state.stream)
            for tensor in self.rows:
                tensor.record_stream(stream)
            self.joined[stream] = self.epoch

    def consume(self, ids, hidden):
        if (
            ids is not self.ids
            or self.next_token != self.tokens
            or hidden.shape[0] != self.tokens
            or hidden.device != self.ids.device
            or not self.state.matches(self.module.embed_tokens)
        ):
            raise RuntimeError(
                "Scheduled Engram lookup is incomplete or its inputs changed"
            )
        self.wait(torch.cuda.current_stream(hidden.device))
        return self.rows


class _AttentionHook:
    def __init__(self, plan, layer, target):
        self.plan, self.layer, self.target = plan, layer, target
        self.entered = self.pre_seen = self.core_seen = self.core_finished = False
        self.pending_slice = None

    def _work(self):
        return self.plan.scheduled.get(self.target)

    def _start_slice(self, work, piece):
        start, count = work.pieces[(self.layer - 10) * 2 + piece]
        if count and work.capturing:
            # Keep the fork at the window entrance, but capture the main
            # branch first so replay can reuse a queue across all windows.
            ready = torch.cuda.current_stream(work.ids.device).record_event()
            self.pending_slice = (start, count, ready)
        else:
            work.launch(start, count)

    def _submit_pending_slice(self, work):
        if self.pending_slice is not None:
            start, count, ready = self.pending_slice
            self.pending_slice = None
            work.launch(start, count, ready=ready)

    def before_attention(self):
        if self.entered:
            raise RuntimeError("Scheduled attention boundary was entered twice")
        self.entered = True

    def before_projection(self, native_core):
        work = self._work()
        if self.target != 14 or work is None:
            return
        if not self.entered or self.pre_seen:
            raise RuntimeError("Unexpected scheduled attention projection boundary")
        if not native_core:
            self.plan.cancel(self.target)
            return
        self.pre_seen = True
        self._start_slice(work, 0)

    def before_core(self):
        work = self._work()
        if work is None:
            return
        if not self.pre_seen or self.core_seen:
            raise RuntimeError("Unexpected scheduled attention core boundary")
        self.core_seen = True
        self._submit_pending_slice(work)
        work.wait(torch.cuda.current_stream())

    def after_core(self):
        work = self._work()
        if work is None:
            return
        if not self.core_seen or self.core_finished:
            raise RuntimeError("Scheduled attention core completion was repeated")
        self.core_finished = True
        self._start_slice(work, 1)

    def after_attention(self):
        try:
            work = self._work()
            if work is not None:
                if self.target == 14 and not self.core_finished:
                    raise RuntimeError(
                        "Scheduled lookup missed the native attention core"
                    )
                self._submit_pending_slice(work)
        finally:
            self.abort()

    def abort(self):
        self.pending_slice = None
        work = self._work()
        if work is not None:
            work.wait(torch.cuda.current_stream())


class DecodeEngramSchedule:
    """Phase hooks owned by one forward; graph replay retains only CUDA nodes."""

    def __init__(self, owner, model, metadata):
        self.owner = owner
        self.scheduled, self.targets, self.bindings = {}, {}, []
        self.metadata = self.l1_hook = None
        tokens = metadata.batch_size * metadata.q_len_per_req
        # Engram.forward slices larger inputs into new ID views. Keep that
        # bounded serial consumer instead of publishing a whole-request pair.
        if type(tokens) is not int or not 0 < tokens <= 32768:
            return
        try:
            for target, producers in ((1, (0,)), (14, (10, 11, 12, 13))):
                if target >= len(model.layers):
                    continue
                layer = model.layers[target]
                module, ids = layer.engram, layer.engram_hashes
                if module is None or ids is None or not _eligible(module):
                    continue
                if (
                    ids.shape != (tokens, 24)
                    or ids.dtype != torch.int64
                    or not ids.is_cuda
                    or not ids.is_contiguous()
                    or ids.device != module.q_weight.device
                ):
                    continue
                attentions = [getattr(model.layers[i], "attn", None) for i in producers]
                if any(
                    getattr(a, "tp_size", 1) != 1
                    or not getattr(a, "_supports_engram_lookup_schedule", False)
                    for a in attentions
                ):
                    continue
                if target == 14 and any(
                    not getattr(
                        getattr(a, "_mega_attention", None),
                        "_supports_engram_lookup_schedule",
                        False,
                    )
                    for a in attentions
                ):
                    continue
                state = prepare_module(module, target, model=model)
                if state is None:
                    continue
                if self.metadata is None:
                    self.metadata = metadata
                    self.previous_stream = getattr(
                        metadata, "_engram_lookup_stream", None
                    )
                    metadata._engram_lookup_stream = state.stream
                self.targets[target] = (module, ids, state)
                if torch.cuda.is_current_stream_capturing():
                    metadata._engram_decode_tables.entries += ((target, module, state),)
                for index, attn in zip(producers, attentions):
                    hook = _AttentionHook(self, index, target)
                    if target == 1:
                        self.l1_hook = hook
                    self._bind(attn, hook)
                    if target == 14:
                        self._bind(attn._mega_attention, hook)
        except BaseException:
            self.close()
            raise

    def _bind(self, module, hook):
        if getattr(module, "_engram_lookup_schedule", None) is not None:
            raise RuntimeError("Engram scheduling is not reentrant")
        self.bindings.append(
            (
                module,
                "_engram_lookup_schedule" in module.__dict__,
                module.__dict__.get("_engram_lookup_schedule"),
            )
        )
        module._engram_lookup_schedule = hook

    def before_layer(self, index):
        target = {0: 1, 10: 14}.get(index)
        if target not in self.targets:
            return
        module, ids, state = self.targets[target]
        if module._lookup_work is not None:
            raise RuntimeError("Engram lookup is already pending")
        work = _ScheduledWork(module, ids, state)
        self.owner.work.append(work)
        self.scheduled[target] = work
        module._lookup_work = work
        if target == 1:
            if work.capturing:
                # Record the hash-only fork before the C++ verify callback
                # adds the metadata dependency to the main capture stream.
                self.l1_hook.pending_slice = (
                    0,
                    ids.shape[0],
                    work.current.record_event(),
                )
            else:
                work.launch(0, ids.shape[0])

    def cancel(self, target):
        work = self.scheduled.get(target)
        if work is not None:
            work.wait(torch.cuda.current_stream(work.ids.device))
            if work.module._lookup_work is work:
                work.module._lookup_work = None
            del self.scheduled[target]

    def close(self):
        if self.metadata is not None:
            self.metadata._engram_lookup_stream = self.previous_stream
            self.metadata = None
        for module, existed, previous in reversed(self.bindings):
            if existed:
                module._engram_lookup_schedule = previous
            else:
                delattr(module, "_engram_lookup_schedule")
        self.l1_hook = None
        self.bindings.clear()
        self.scheduled.clear()
