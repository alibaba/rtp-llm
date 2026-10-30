"""Standard playback input: bounded Protobuf batches in checksummed Zstd frames."""
from pathlib import Path
import struct

from google.protobuf.message import DecodeError
import zstandard

from . import client_input_pb2 as pb

MAGIC = b'MCKINP01'
MAX_FRAME = 4 * 1024 * 1024
MAX_REQUEST = 1024 * 1024
MAX_BATCH_REQUESTS = 1024


def _known(message):
    clean = type(message)()
    clean.CopyFrom(message)
    clean.DiscardUnknownFields()
    if clean != message:
        raise ValueError('Unknown playback input fields')


def _validate(request):
    if not isinstance(request, pb.Request):
        raise TypeError('Expected client_input_pb2.Request')
    _known(request)
    if (not request.token_ids or any(token < 0 for token in request.token_ids)
            or request.max_new_tokens < 1 or request.HasField('arrival_offset_ms') and request.arrival_offset_ms < 0
            or request.HasField('priority') and request.priority < 0
            or request.ByteSize() > MAX_REQUEST):
        raise ValueError('Invalid or unsupported playback request')
    end = 0
    if len(request.images) > 256:
        raise ValueError('Too many images')
    for image in request.images:
        if (not image.url.strip() or len(image.url) > 4096 or not 1 <= image.width <= 32768
                or not 1 <= image.height <= 32768 or image.offset < end or image.length < 1
                or image.offset + image.length > len(request.token_ids)):
            raise ValueError('Invalid expanded image span or geometry')
        end = image.offset + image.length


def write_inputs(path, requests):
    """Write protobuf Request objects directly; refuse overwrite and remove failed output."""
    path = Path(path)
    stream = path.open('xb')
    try:
        with stream:
            stream.write(MAGIC)
            compressor = zstandard.ZstdCompressor(level=3, write_checksum=True, write_content_size=True)
            sequence = count = size = 0
            batch = pb.Batch()

            def flush():
                nonlocal sequence
                sequence += 1
                batch.sequence = sequence
                encoded = compressor.compress(batch.SerializeToString())
                if len(encoded) > MAX_FRAME:
                    raise ValueError('Oversized compressed input batch')
                stream.write(struct.pack('>I', len(encoded)))
                stream.write(encoded)
                batch.Clear()

            for request in requests:
                _validate(request)
                length = request.ByteSize() + 5
                if batch.requests and (size + length > MAX_REQUEST or len(batch.requests) == MAX_BATCH_REQUESTS):
                    flush()
                    size = 0
                batch.requests.add().CopyFrom(request)
                size += length
                count += 1
            if batch.requests:
                flush()
            batch.total_requests = count
            flush()
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def read_inputs(path):
    """Stream validated Request objects; reaching EOF requires a valid final seal."""
    path = Path(path)
    try:
        with path.open('rb') as stream:
            if stream.read(len(MAGIC)) != MAGIC:
                raise ValueError('Expected Protobuf+Zstd playback input v1')
            sequence = count = 0
            while True:
                header = stream.read(4)
                if len(header) != 4:
                    raise ValueError('Truncated input or missing seal')
                size, = struct.unpack('>I', header)
                if not 6 <= size <= MAX_FRAME:
                    raise ValueError('Invalid compressed input batch size')
                encoded = stream.read(size)
                if len(encoded) != size:
                    raise ValueError('Truncated input batch')
                parameters = zstandard.get_frame_parameters(encoded)
                if not 1 <= parameters.content_size <= MAX_FRAME or not parameters.has_checksum or not encoded[4] & 0x20:
                    raise ValueError('Input batch requires bounded content size and checksum')
                raw = zstandard.ZstdDecompressor().decompress(encoded, max_output_size=MAX_FRAME, allow_extra_data=False)
                batch = pb.Batch.FromString(raw)
                _known(batch)
                sequence += 1
                if batch.sequence != sequence:
                    raise ValueError('Noncontiguous input batch sequence')
                if batch.HasField('total_requests'):
                    if batch.requests or batch.total_requests != count or stream.read(1):
                        raise ValueError('Input seal count mismatch or trailing data')
                    return
                if not 1 <= len(batch.requests) <= MAX_BATCH_REQUESTS:
                    raise ValueError('Invalid input batch request count')
                for request in batch.requests:
                    _validate(request)
                count += len(batch.requests)
                yield from batch.requests
    except (ValueError, DecodeError, zstandard.ZstdError) as error:
        raise ValueError(f'Invalid playback input at {path}: {error}') from error
