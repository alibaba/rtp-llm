"""Forward-local source publication through the maintained L21 attention owner.

The caller owns the native allocation lease and forward serialization. This
adapter admits short eager CP1 maintained-stage controls; it does not admit
production PP execution.
"""

from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from rtp_llm.models_py.modules.dsv4.native_l20_transfer import (
    NativeL20ReplicaReceiver,
    _require_eager,
)
from rtp_llm.models_py.modules.dsv4.v41_cache import logical_to_physical_slots
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import V41L20Span
from rtp_llm.models_py.modules.dsv4.v41_pp import V41PPFrame, _validate_frame

IMPORT_KEY = "v41_imported_l20_prefill"


@dataclass(frozen=True)
class NativeL20PrefillSource:
    receiver: NativeL20ReplicaReceiver
    spans: tuple[V41L20Span, ...]
    rows: torch.Tensor
    topk: torch.Tensor
    globals_by_req: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    def _ready(self) -> None:
        for span in self.spans:
            self.receiver.require_ready(
                span.identity, span.start + span.count, span.chunk_id
            )

    def require_owner(self, attention: Any) -> None:
        """Check L21-39 source provenance before any forward cache writes."""
        self._ready()
        shared = attention._shared_attention
        layers = shared.get("layers")
        if not isinstance(layers, Mapping) or not layers or min(layers) != 21:
            raise NotImplementedError(
                "Native L20 publication requires the maintained L21 stage entry"
            )
        layer_id = attention.layer_id
        expected_index_source = 20 if layer_id < 24 else layer_id // 4 * 4
        if (
            not 21 <= layer_id < 40
            or any(not 21 <= i < 40 for i in layers)
            or layers.get(layer_id) is not attention
            or attention.kv_source_layer_id != 20
            or attention.index_source_layer_id != expected_index_source
            or attention.is_kv_source
            or attention.is_index_source != (layer_id == expected_index_source)
            or attention.tp_size != 1
        ):
            raise NotImplementedError(
                "Native L20 publication requires maintained CP1 L21-39 owners"
            )
        if expected_index_source != 20:
            producer = layers.get(expected_index_source)
            if (
                producer is None
                or producer.layer_id != expected_index_source
                or producer.index_source_layer_id != expected_index_source
                or not producer.is_index_source
                or producer.is_kv_source
                or producer.kv_source_layer_id != 20
                or producer.tp_size != 1
                or producer._shared_attention is not shared
            ):
                raise NotImplementedError(
                    "Native L20 consumer requires its maintained local index owner"
                )
        if attention._kv_cache is not self.receiver._native_cache:
            raise ValueError(
                "Native L20 publication must use the admitted replica cache"
            )
        if shared.get(IMPORT_KEY) is not self:
            raise ValueError(
                "Native L20 publication requires its current forward scope"
            )
        current_tables = getattr(attention, "_block_tables_by_type", None)
        if not isinstance(current_tables, Mapping):
            raise ValueError("Native L20 publication requires bound tagged page tables")
        for tag in ("hca_kv", "indexer_kv"):
            tables = [self.receiver.block_tables(s.identity)[tag] for s in self.spans]
            width = max(map(len, tables))
            value = current_tables.get(tag)
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.shape[0] != len(tables)
                or value.shape[1] < width
                or value.dtype != torch.int32
                or value.device != self.topk.device
                or not value.is_contiguous()
                or value.tolist()
                != [list(t) + [0] * (value.shape[1] - len(t)) for t in tables]
            ):
                raise ValueError(f"Native L20 dispatch disagrees with {tag} pages")

    def publish(self, attention: Any, common: Any) -> None:
        """Publish only at L21, after the entry owner's forward reset."""
        if attention.layer_id != 21:
            raise NotImplementedError("Native L20 sources publish only at L21")
        self.require_owner(attention)
        shared = attention._shared_attention
        if common.cp_on:
            raise NotImplementedError(
                "Native L20 publication requires eager CP1 L21 entry"
            )
        if (
            shared.get("global") != {}
            or shared.get("topk") != {}
            or shared.get("candidates") is not None
        ):
            raise ValueError("Native L20 publication must follow the forward reset")
        for name, expected in (
            ("input_lengths", [s.count for s in self.spans]),
            ("prefix_lengths", [s.start for s in self.spans]),
            ("position_ids", self.rows[:, 1].tolist()),
            ("req_id_per_token", self.rows[:, 0].tolist()),
        ):
            value = getattr(common, name, None)
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or value.dtype not in (torch.int32, torch.int64)
                or value.device != self.topk.device
                or value.tolist() != expected
            ):
                raise ValueError(f"Native L20 current dispatch disagrees with {name}")
        shared["global"] = {20: list(self.globals_by_req)}
        shared["topk"] = {20: self.topk}
        # Every admitted context has fewer than 2048 candidate blocks.
        shared["candidates"] = None


def prepare_native_l20_prefill_source(
    frame: V41PPFrame,
    receiver: NativeL20ReplicaReceiver,
    spans: Sequence[V41L20Span],
    expected_rows: torch.Tensor,
) -> NativeL20PrefillSource:
    """Snapshot current native source bytes and cloned short-forward selections."""
    if not isinstance(receiver, NativeL20ReplicaReceiver):
        raise ValueError("Native L20 publication requires an admitted native receiver")
    _require_eager(receiver.caches["hca_kv"].raw)
    spans = tuple(spans)
    _validate_frame(frame, spans, expected_rows)
    if (
        not spans
        or not expected_rows.numel()
        or bool((expected_rows[:, 2] != 1).any())
        or any(s.start + s.count > 256 for s in spans)
    ):
        raise NotImplementedError(
            "Native L20 publication requires short unpadded CP1 rows"
        )
    rows = torch.tensor(
        [
            [request, p, 1]
            for request, s in enumerate(spans)
            for p in range(s.start, s.start + s.count)
        ],
        dtype=torch.int64,
    )
    if not torch.equal(rows, expected_rows):
        raise ValueError("Native L20 frame rows disagree with request order")
    device = receiver.caches["hca_kv"].raw.device
    if frame.hidden_states.device != device:
        raise ValueError("Native L20 frame and replica must share a device")
    for tensor in frame.__dict__.values():
        if tensor.numel() and any(
            tensor.device == cache.raw.device
            and tensor.untyped_storage().data_ptr()
            == cache.raw.untyped_storage().data_ptr()
            for cache in receiver.caches.values()
        ):
            raise ValueError("Native L20 frame aliases replica storage")
    for span in spans:
        receiver.require_ready(span.identity, span.start + span.count, span.chunk_id)
    globals_by_req = []
    for span in spans:
        tables = receiver.block_tables(span.identity)
        logical = torch.arange(span.start + span.count, device=device)
        values = []
        for tag, dtype in (("hca_kv", torch.bfloat16), ("indexer_kv", torch.float32)):
            slots = logical_to_physical_slots(
                torch.tensor(tables[tag], device=device, dtype=torch.int64),
                logical,
                tag,
                1,
            )
            value = receiver.caches[tag].read(
                slots, out_dtype=dtype, backend=receiver.backend
            )
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Native L20 source contains nonfinite decoded values")
            values.append(value)
        globals_by_req.append(tuple(values))
    return NativeL20PrefillSource(
        receiver, spans, rows, frame.topk.clone(), tuple(globals_by_req)
    )


@contextmanager
def imported_native_l20_prefill(attention: Any, source: NativeL20PrefillSource):
    """Hold the source snapshot for exactly one serialized eager owner forward."""
    if not isinstance(source, NativeL20PrefillSource):
        raise ValueError("Native L20 scope requires a prepared source")
    _require_eager(source.receiver.caches["hca_kv"].raw)
    source._ready()
    shared = attention._shared_attention
    if IMPORT_KEY in shared:
        raise ValueError("Native L20 forward already has an imported source")
    shared[IMPORT_KEY] = source
    try:
        yield
    finally:
        shared.pop(IMPORT_KEY, None)
        globals_map = shared.get("global", {})
        published = globals_map.get(20)
        if (
            isinstance(published, list)
            and len(published) == len(source.globals_by_req)
            and all(a is b for a, b in zip(published, source.globals_by_req))
        ):
            globals_map.pop(20)
        topk_map = shared.get("topk", {})
        if topk_map.get(20) is source.topk:
            topk_map.pop(20)
