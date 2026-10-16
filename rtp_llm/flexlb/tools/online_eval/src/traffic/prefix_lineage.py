"""Version 2 empirical prefix structure: columnar varints in an XZ envelope.

No raw text is stored. Token lengths are block aligned; arrival timestamps are
never paced here. The allocation bitmap preserves v1 complete-block labels.
"""
import json
import lzma

MAGIC = b'PFL2\0'
from traffic.lineage_model import DEFAULT_PREFIX_BLOCK_SIZE, prefix_block_size, MAX_LINEAGE_EVENTS as MAX_EVENTS
from traffic.lineage_model import LineageModel


def _varint(n):
    if type(n) is not int or not 0 <= n <= (1 << 64) - 1:
        raise ValueError('invalid unsigned varint')
    out = bytearray()
    while n >= 128:
        out.append((n & 127) | 128)
        n >>= 7
    out.append(n)
    return out


def _read(data, offset):
    n = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError('truncated lineage model')
        byte = data[offset]
        offset += 1
        n |= (byte & 127) << shift
        if not byte & 128:
            if n > (1 << 64) - 1:
                break
            return n, offset
    raise ValueError('invalid lineage varint')


def encode(events, provenance=None, *, block_size=DEFAULT_PREFIX_BLOCK_SIZE):
    """Convert [relative ms, original input length, parent, shared blocks]."""
    block_size = prefix_block_size(block_size)
    columns = [bytearray() for _ in range(4)]
    allocation = bytearray((len(events) + 7) // 8)
    total = aligned = short_count = removed = added = 0
    last = 0
    lengths = []
    if not 1 <= len(events) <= MAX_EVENTS:
        raise ValueError('invalid lineage event count')
    for i, event in enumerate(events):
        if len(event) != 4 or any(type(v) is not int for v in event):
            raise ValueError('invalid lineage event')
        ts, il, parent, shared = event
        if not last <= ts <= 9223372036854775807 or not 1 <= il <= 2147483647 or not -1 <= parent < i:
            raise ValueError('invalid lineage ordering/shape')
        if not 0 <= shared <= il // block_size or (parent == -1 and shared) or (parent >= 0 and shared > lengths[parent] // block_size):
            raise ValueError('invalid lineage parent reference')
        blocks = max(1, il // block_size)
        if il >= block_size and il % block_size:
            allocation[i // 8] |= 1 << (i % 8)
        for col, value in zip(columns, ((ts - last) * 2, blocks, 0 if parent < 0 else i - parent, shared)):
            col.extend(_varint(value))
        total += il
        aligned += blocks * block_size
        removed += max(0, il - blocks * block_size)
        added += max(0, blocks * block_size - il)
        short_count += il < block_size
        last = ts
        lengths.append(il)
    metadata = dict(version=2, block_size=block_size, count=len(events),
                    realism='EMPIRICAL_PREFIX_STRUCTURE', tail='BLOCK_ALIGNED_LOSSY_TOKENS',
                    arrival='TRUE_TIMESTAMPS_PACED_BY_CLIENT',
                    provenance=provenance or {},
                    token_adjustment=dict(original_tokens=total, aligned_tokens=aligned,
                        removed_tokens=removed, added_tokens=added, short_requests=short_count,
                        relative_change=(aligned-total)/total),
                    allocation_bitmap='reserve discarded v1 partial-block labels; no tail lengths retained')
    header = json.dumps(metadata, separators=(',', ':'), allow_nan=False).encode()
    payload = bytearray(MAGIC) + _varint(len(header)) + header
    for col in columns:
        payload += _varint(len(col)) + col
    payload += allocation
    return lzma.compress(bytes(payload), preset=9)


def _decode_payload(raw):
    decoder = lzma.LZMADecompressor(memlimit=128 * 1024 * 1024)
    data = decoder.decompress(raw, max_length=128 * 1024 * 1024)
    if not decoder.eof or decoder.unused_data or not data.startswith(MAGIC):
        raise ValueError('invalid or oversized lineage v2 envelope')
    n, offset = _read(data, len(MAGIC))
    metadata = json.loads(data[offset:offset+n])
    offset += n
    block_size = prefix_block_size(metadata.get('block_size'))
    count = metadata.get('count')
    if metadata.get('version') != 2 or type(count) is not int or not 1 <= count <= MAX_EVENTS:
        raise ValueError('invalid lineage v2 header')
    columns = []
    for _ in range(4):
        size, offset = _read(data, offset)
        end = offset + size
        values = []
        while offset < end:
            value, offset = _read(data, offset)
            values.append(value)
        if offset != end or len(values) != count:
            raise ValueError('invalid lineage column length')
        columns.append(values)
    allocation = data[offset:]
    if len(allocation) != (count+7)//8:
        raise ValueError('invalid lineage allocation bitmap')
    events = []
    ts = 0
    for i, (delta, blocks, distance, shared) in enumerate(zip(*columns)):
        ts += (delta >> 1) ^ -(delta & 1)
        parent = i-distance if distance else -1
        if ts < 0 or ts > 9223372036854775807 or (events and ts < events[-1][0]) or not 1 <= blocks <= 2147483647//block_size:
            raise ValueError('invalid lineage shape/timestamp')
        if not -1 <= parent < i or shared > blocks or (parent == -1 and shared) or (parent >= 0 and shared > events[parent][1]):
            raise ValueError('invalid lineage reference')
        events.append((ts, blocks, parent, shared, (allocation[i//8] >> (i%8)) & 1))
    return metadata, events


class BlockAlignedLineageCodec:
    """Decode the persisted format into the common, unit-safe domain model."""

    def decode(self, raw):
        return LineageModel.from_wire(*_decode_payload(raw))

    def encode(self, events, provenance=None, *, block_size=DEFAULT_PREFIX_BLOCK_SIZE):
        return encode(events, provenance, block_size=block_size)


def decode(raw):
    return BlockAlignedLineageCodec().decode(raw).to_wire()
