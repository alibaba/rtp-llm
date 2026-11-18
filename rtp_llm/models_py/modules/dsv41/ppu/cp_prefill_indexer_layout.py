"""Bounded early-CP layout adapter for the existing BF16 flat MQA provider.

The native page256 storage is planar inside each page. Two existing ATen
page gathers produce contiguous payload/SF planes in true request order.
No bytes are requantized. A single cache slot bounds retained layout storage.
"""

from dataclasses import dataclass
import weakref

import torch

from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import _source_tensor_key
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import env_value

ENV = "RTP_V41_PPU_CP8_INDEXER_FLAT_LAYOUT"
PAGE, WIDTH = 256, 68


def _stream_key(device):
    stream = torch.cuda.current_stream(device)
    return stream.device, int(stream.cuda_stream)


def _keys(bank):
    return (id(bank), bank.source, bank.epoch, bank.prefix_rows, bank.request_counts,
            tuple(_source_tensor_key(value) for value in (bank.pages, bank.request_table)))


def _stats(cache):
    value = getattr(cache, "_cp_indexer_layout_stats", None)
    if value is None:
        value = cache._cp_indexer_layout_stats = dict(
            packs=0, copies=0, readBytes=0, writeBytes=0, prefixDuplicateBytes=0,
            paddingBytes=0, retainedBytes=0, peakRetainedBytes=0, denseCalls=0,
        )
    return value


@dataclass(frozen=True)
class FlatIndexerLayout:
    bank_ref: object
    keys: tuple
    stream_key: tuple
    payload: torch.Tensor
    scales: torch.Tensor
    output_keys: tuple
    request_stride: int

    @property
    def bank(self):
        return self.bank_ref()

    @property
    def nbytes(self):
        return self.payload.numel() + self.scales.numel() * 4

    def valid(self):
        stream = _stream_key(self.payload.device)
        bank = self.bank
        return (bank is not None and bank.valid() and _keys(bank) == self.keys
                and stream == self.stream_key
                and tuple(_source_tensor_key(value) for value in (self.payload, self.scales)) == self.output_keys)


def eligible(rows):
    """Admit the qualified full CP8 ratio2 workload."""
    if (env_value(ENV) != "1" or rows.ratio != 2
            or rows.bank.source not in (2, 8, 14) or rows.state.compacted
            or rows.bank.request_counts != (65536,) * 8 or rows.bank.prefix_rows != 63488
            or rows.state.lengths != (4096,) * 8 or rows.state.query_prefixes != (126976,) * 8
            or rows.count <= 0 or rows.count % 4):
        return False
    # The existing flat kernel shares K among four Q rows. Never infer that
    # an arbitrary ragged request boundary may cross such a group.
    ids = rows.request_ids_host
    return all(ids[begin:begin + 4] == (ids[begin],) * 4 for begin in range(0, len(ids), 4))


def prepare(rows):
    """Preflight before allocations; failure after packing never retries paged."""
    from rtp_llm.models_py.modules.dsv41.ppu import cp_prefill_paged_indexer as paged

    if (not eligible(rows) or not rows.valid() or not paged._device_supported(rows.positions)
            or torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing()):
        return None
    bank, cache = rows.bank, rows.bank.cache
    stream = _stream_key(bank.pages.device)
    if not rows.valid():
        raise RuntimeError("CP Indexer rows changed during flat layout admission")
    if not paged._bank_supported(bank, rows.state, rows.ratio):
        return None
    birth_stream = getattr(rows.state, "_cp_metadata_stream", None)
    if birth_stream is None or birth_stream[0] != stream:
        return None
    keys = _keys(bank)
    previous = getattr(cache, "_cp_indexer_flat_layout", None)
    if previous is not None and type(previous) is not FlatIndexerLayout:
        raise RuntimeError("unknown CP flat Indexer layout owner")
    if previous is not None and previous.stream_key != stream:
        return None
    if previous is not None and previous.bank is bank:
        if not previous.valid():
            raise RuntimeError("stale CP flat Indexer layout")
        return previous
    # Drop the previous slot only after this source's complete preflight.
    # Its native readers precede this same-stream enqueue. Do not attach a
    # strong layout to every source bank and retain all three 34MiB copies.
    cache._cp_indexer_flat_layout = None
    del previous
    stats = _stats(cache)
    stats["retainedBytes"] = 0
    pages = bank.pages
    count, stride = pages.shape[0], pages.stride(0)
    indices = bank.request_table.view(-1)
    payload_pages = pages.as_strided((count, PAGE, 64), (stride, 64, 1))
    scale_pages = pages.as_strided((count, PAGE, 4), (stride, 4, 1),
                                  pages.storage_offset() + PAGE * 64)
    # These two gathers are the complete packing cost. Their results must
    # already be contiguous; do not hide an additional clone/contiguous copy.
    with torch.inference_mode(False), torch.no_grad():
        packed = payload_pages.index_select(0, indices)
        sf = scale_pages.index_select(0, indices)
    if not packed.is_contiguous() or not sf.is_contiguous():
        raise RuntimeError("CP flat page gather returned a noncontiguous plane")
    request_stride = bank.request_table.shape[1] * PAGE
    with torch.inference_mode(False), torch.no_grad():
        payload = packed.view(-1, 64).view(torch.int8)
        scales = sf.view(torch.int32).view(-1)
    current_stream = _stream_key(pages.device)
    if current_stream != stream or not bank.valid() or _keys(bank) != keys or not rows.valid():
        raise RuntimeError("CP Indexer source changed during flat packing")
    layout = FlatIndexerLayout(
        weakref.ref(bank), keys, stream, payload, scales,
        tuple(_source_tensor_key(value) for value in (payload, scales)), request_stride,
    )
    cache._cp_indexer_flat_layout = layout
    duplicated = (len(bank.request_counts) - 1) * bank.prefix_rows * WIDTH
    padding = (len(bank.request_counts) * request_stride - sum(bank.request_counts)) * WIDTH
    stats["packs"] += 1
    stats["copies"] += 2
    stats["readBytes"] += layout.nbytes
    stats["writeBytes"] += layout.nbytes
    stats["prefixDuplicateBytes"] += duplicated
    stats["paddingBytes"] += padding
    stats["retainedBytes"] = layout.nbytes
    stats["peakRetainedBytes"] = max(stats["peakRetainedBytes"], layout.nbytes)
    rows.state.shared.setdefault("cp_indexer_layout_calls", []).append(dict(
        kind="pack", source=bank.source, epoch=bank.epoch, rows=payload.shape[0],
        requestStride=request_stride, copies=2, readBytes=layout.nbytes,
        writeBytes=layout.nbytes, prefixDuplicateBytes=duplicated, paddingBytes=padding,
    ))
    return layout


def score(q_payload, q_sf, weights, rows, operation):
    """Keep native relative BF16 output and the original per-row visible end."""
    if not callable(operation):
        return None
    inputs = q_payload, q_sf, weights
    input_keys = tuple(_source_tensor_key(value) for value in inputs)
    layout = prepare(rows)
    if layout is None:
        return None
    starts = rows.request_ids * layout.request_stride
    ends = starts + rows.visible
    head_weights = weights.to(torch.bfloat16).contiguous()
    if (not layout.valid() or not rows.valid()
            or input_keys != tuple(_source_tensor_key(value) for value in inputs)):
        raise RuntimeError("CP flat Indexer operands changed before native scoring")
    logits = operation(
        (q_payload, q_sf), (layout.payload, layout.scales),
        head_weights, starts, ends,
        max_seqlen_k=rows.width,
    )
    if logits is None:
        # None denotes prelaunch refusal to the caller. Once the native
        # operation ran it must never trigger a second paged execution.
        raise RuntimeError("CP flat Indexer scorer returned no logits after launch")
    if not layout.valid() or not rows.valid():
        raise RuntimeError("CP flat Indexer operands changed during native scoring")
    _stats(rows.bank.cache)["denseCalls"] += 1
    rows.state.shared.setdefault("cp_indexer_layout_calls", []).append(dict(
        kind="dense", source=rows.bank.source, epoch=rows.bank.epoch,
        rows=rows.count, width=rows.width, requestStride=layout.request_stride,
    ))
    return logits
