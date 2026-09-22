"""Version 2 empirical prefix structure: columnar varints in an XZ envelope.

No raw text is stored. Token lengths are block aligned; arrival timestamps are
never paced here. The allocation bitmap preserves v1 complete-block labels.
"""
import argparse
import gzip
import hashlib
import json
import lzma
from pathlib import Path

MAGIC = b'PFL2\0'
BLOCK = 512
MAX_EVENTS = 1_000_000


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


def encode(events, provenance=None):
    """Convert [relative ms, original input length, parent, shared blocks]."""
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
        if not 0 <= shared <= il // BLOCK or (parent == -1 and shared) or (parent >= 0 and shared > lengths[parent] // BLOCK):
            raise ValueError('invalid lineage parent reference')
        blocks = max(1, il // BLOCK)
        if il >= BLOCK and il % BLOCK:
            allocation[i // 8] |= 1 << (i % 8)
        for col, value in zip(columns, ((ts - last) * 2, blocks, 0 if parent < 0 else i - parent, shared)):
            col.extend(_varint(value))
        total += il
        aligned += blocks * BLOCK
        removed += max(0, il - blocks * BLOCK)
        added += max(0, blocks * BLOCK - il)
        short_count += il < BLOCK
        last = ts
        lengths.append(il)
    metadata = dict(version=2, block_size=BLOCK, count=len(events),
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


def decode(raw):
    decoder = lzma.LZMADecompressor(memlimit=128 * 1024 * 1024)
    data = decoder.decompress(raw, max_length=128 * 1024 * 1024)
    if not decoder.eof or decoder.unused_data or not data.startswith(MAGIC):
        raise ValueError('invalid or oversized lineage v2 envelope')
    n, offset = _read(data, len(MAGIC))
    metadata = json.loads(data[offset:offset+n])
    offset += n
    count = metadata.get('count')
    if metadata.get('version') != 2 or metadata.get('block_size') != BLOCK or type(count) is not int or not 1 <= count <= MAX_EVENTS:
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
        if ts < 0 or ts > 9223372036854775807 or (events and ts < events[-1][0]) or not 1 <= blocks <= 2147483647//BLOCK:
            raise ValueError('invalid lineage shape/timestamp')
        if not -1 <= parent < i or shared > blocks or (parent == -1 and shared) or (parent >= 0 and shared > events[parent][1]):
            raise ValueError('invalid lineage reference')
        events.append((ts, blocks, parent, shared, (allocation[i//8] >> (i%8)) & 1))
    return metadata, events


def expand(events):
    paths = []
    next_label = 1
    for ts, blocks, parent, shared, reserve in events:
        labels = paths[parent][:shared] if parent >= 0 else []
        fresh = blocks-shared
        if next_label + fresh + reserve > 2147483647:
            raise ValueError('lineage label budget exceeded')
        labels.extend(range(next_label, next_label+fresh))
        next_label += fresh + reserve
        paths.append(labels)
        yield ts, labels


def write_trace(path, parameters, namespace, base_dir, *, max_requests=None):
    p = parameters
    if set(p) != {'path', 'sha256', 'count', 'output_tokens', 'priority'}:
        raise ValueError('lineage v2 requires pinned model, count, output_tokens and priority; pacing belongs to client')
    if type(p['output_tokens']) is not int or not 1 <= p['output_tokens'] <= 2147483647 or type(p['priority']) is not int or not 1 <= p['priority'] <= 100:
        raise ValueError('invalid output length/priority')
    raw = (Path(base_dir) / p['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != p['sha256']:
        raise ValueError('lineage model checksum mismatch')
    metadata, events = decode(raw)
    if type(p['count']) is not int or len(events) != p['count']:
        raise ValueError('lineage model count mismatch')
    with Path(path).open('w') as out:
        selected = events[:max_requests] if max_requests is not None else events
        for i, (ts, labels) in enumerate(expand(selected)):
            out.write(json.dumps(dict(rid=f'{namespace}:{i}', ts=ts, il=len(labels)*BLOCK,
                ol=p['output_tokens'], priority=p['priority'], cache_key_block_size=BLOCK,
                input_token_blocks=labels), separators=(',', ':'))+'\n')
    return dict(realism=metadata['realism'], tail=metadata['tail'], arrival=metadata['arrival'],
                model_sha256=p['sha256'], provenance=metadata['provenance'],
                token_adjustment=metadata['token_adjustment'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('v1_model', type=Path)
    parser.add_argument('--fit-report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(gzip.decompress(args.v1_model.read_bytes()))
    if model.get('version') != 1 or model.get('block_size') != BLOCK:
        raise ValueError('expected lineage v1 model')
    report = json.loads(args.fit_report.read_text())
    provenance = {k:report.get(k) for k in ('missing_pod_indices','source_start','source_end','expected_pods','source_pods')}
    provenance['v1_sha256'] = hashlib.sha256(args.v1_model.read_bytes()).hexdigest()
    provenance['fit_report_sha256'] = hashlib.sha256(args.fit_report.read_bytes()).hexdigest()
    raw = encode(model['events'], provenance)
    args.out.write_bytes(raw)
    print(json.dumps(dict(path=str(args.out), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), **decode(raw)[0]), indent=2))


if __name__ == '__main__':
    main()
