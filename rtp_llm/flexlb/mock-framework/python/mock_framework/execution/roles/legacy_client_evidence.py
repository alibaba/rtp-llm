"""Decode historical compressed Client evidence into the JSON reader's record format."""
import struct

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import DecodeError
import zstandard

from . import client_evidence_pb2 as pb

MAX_FRAME = 2 * 1024 * 1024


def _batch(encoded):
    content_size = zstandard.frame_content_size(encoded)
    if content_size not in (zstandard.CONTENTSIZE_UNKNOWN, zstandard.CONTENTSIZE_ERROR) and content_size > MAX_FRAME:
        raise ValueError('Oversized decompressed evidence frame')
    raw = zstandard.ZstdDecompressor().decompress(encoded, max_output_size=MAX_FRAME, allow_extra_data=False)
    size = 0
    for i, value in enumerate(raw[:5]):
        size |= (value & 127) << (7 * i)
        if value < 128:
            if size != len(raw) - i - 1:
                raise ValueError('Invalid length-delimited EventBatch')
            return pb.EventBatch.FromString(raw[i + 1:])
    raise ValueError('Invalid EventBatch length')


def _legacy_batches(stream):
    while header := stream.read(4):
        if len(header) != 4:
            raise ValueError('Truncated evidence frame')
        size, = struct.unpack('>I', header)
        if not 0 < size <= MAX_FRAME:
            raise ValueError('Invalid compressed frame size')
        encoded = stream.read(size)
        if len(encoded) != size:
            raise ValueError('Truncated evidence frame')
        yield _batch(encoded)


def read_batches(stream):
    try:
        for frame, batch in enumerate(_legacy_batches(stream), 1):
            if batch.frame != frame or not (batch.events or batch.HasField('seal')):
                raise ValueError('Evidence frame identity/sequence mismatch')
            row = MessageToDict(batch, preserving_proto_field_name=True)
            del row['frame']
            events = row.pop('events', [])
            seal = row.pop('seal', None)
            yield row, events, seal
    except (DecodeError, zstandard.ZstdError) as error:
        raise ValueError(str(error)) from error
