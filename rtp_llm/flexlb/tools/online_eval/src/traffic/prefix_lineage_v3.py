"""Exact-length empirical lineage; filter only when projecting a replay plan."""
import json
import lzma

from traffic.lineage_model import DEFAULT_PREFIX_BLOCK_SIZE, prefix_block_size, MAX_LINEAGE_EVENTS as MAX_EVENTS
from traffic.lineage_model import LineageModel


def validate(events, block_size=DEFAULT_PREFIX_BLOCK_SIZE):
    block_size = prefix_block_size(block_size)
    if not isinstance(events, list) or not 1 <= len(events) <= MAX_EVENTS:
        raise ValueError('invalid lineage event count')
    previous = 0
    for i, event in enumerate(events):
        if len(event) != 4 or any(type(v) is not int for v in event):
            raise ValueError('invalid lineage event')
        ts, length, parent, shared = event
        if not previous <= ts <= 9223372036854775807 or not 1 <= length <= 2147483647:
            raise ValueError('invalid lineage timestamp/length')
        if not -1 <= parent < i or not 0 <= shared <= length // block_size:
            raise ValueError('invalid lineage reference')
        if (parent == -1 and shared) or (parent >= 0 and shared > events[parent][1] // block_size):
            raise ValueError('invalid lineage shared prefix')
        previous = ts


def encode(events, provenance=None, *, block_size=DEFAULT_PREFIX_BLOCK_SIZE):
    validate(events, block_size)
    metadata = dict(version=3, block_size=block_size, count=len(events),
                    realism='EMPIRICAL_PREFIX_STRUCTURE', tail='EXACT_LENGTH_PRIVATE_PARTIAL_BLOCK',
                    arrival='TRUE_TIMESTAMPS_PACED_BY_CLIENT', provenance=provenance or {})
    return lzma.compress(json.dumps(dict(metadata=metadata, events=events),
                                   separators=(',', ':'), allow_nan=False).encode(), preset=6)


def _decode_payload(raw):
    decoder = lzma.LZMADecompressor(memlimit=128*1024*1024)
    data = decoder.decompress(raw, max_length=128*1024*1024)
    if not decoder.eof or decoder.unused_data:
        raise ValueError('invalid or oversized lineage v3 envelope')
    model = json.loads(data)
    metadata, events = model['metadata'], model['events']
    block_size = prefix_block_size(metadata.get('block_size'))
    if metadata.get('version') != 3 or metadata.get('count') != len(events):
        raise ValueError('invalid lineage v3 header')
    validate(events, block_size)
    return metadata, events


class ExactLengthLineageCodec:
    """Decode the persisted format into the common, unit-safe domain model."""

    def decode(self, raw):
        return LineageModel.from_wire(*_decode_payload(raw))

    def encode(self, events, provenance=None, *, block_size=DEFAULT_PREFIX_BLOCK_SIZE):
        return encode(events, provenance, block_size=block_size)


def decode(raw):
    return ExactLengthLineageCodec().decode(raw).to_wire()
