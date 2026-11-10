"""Encode pre-indexed anonymous replay rows into the Client's MCKINDEX format."""

import hashlib
import os
from pathlib import Path
import struct
import tempfile
import zlib

import zstandard


HEADER = struct.Struct(">8s9q32s")
REQUEST_PAGE = struct.Struct(">qii")
REQUESTS_PER_PAGE = 1024
CHAIN = struct.Struct(">qqi")
CHAIN_PAGE = struct.Struct(">qqii")
CHAINS_PER_PAGE = 256
MEDIA = struct.Struct(">ii")
REQUEST = struct.Struct(">qqiiH")
IMAGE = struct.Struct(">qii")
MAX_TOKENS = 1024 * 1024


def number(value, minimum, maximum=(1 << 31) - 1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"expected integer in [{minimum}, {maximum}]")
    return value


def record(table, data):
    table.write(data)
    table.write(struct.pack(">I", zlib.crc32(data)))


class RequestPages:
    """Write independent request pages and their payload offsets."""

    def __init__(self, index, payload):
        self.index = index
        self.payload = payload
        self.rows = []
        self.compressor = zstandard.ZstdCompressor(level=3, write_checksum=True)

    def append(self, row):
        self.rows.append(row)
        if len(self.rows) == REQUESTS_PER_PAGE:
            self.finish()

    def finish(self):
        if not self.rows:
            return
        offsets = bytearray()
        position = 0
        for row in self.rows:
            offsets.extend(struct.pack(">I", position))
            position += len(row)
        raw = offsets + b"".join(self.rows)
        encoded = self.compressor.compress(raw)
        record(self.index, REQUEST_PAGE.pack(self.payload.tell(), len(encoded), len(raw)))
        self.payload.write(encoded)
        self.rows.clear()


def write(target, requests, chains, dimensions):
    """Write one file. Requests carry arrival, leaf, observed length, priority, cap and media."""
    with (tempfile.TemporaryFile(dir=target.parent) as index,
          tempfile.TemporaryFile(dir=target.parent) as payload):
        caps = {}
        pages = RequestPages(index, payload)
        count = missing_output = 0
        last_arrival = -1
        for arrival, leaf, observed, priority, maximum, images in requests:
            if arrival < last_arrival or arrival < 0:
                raise ValueError("arrival offsets must be nondecreasing")
            observed = -1 if observed is None else number(observed, 0)
            cap = 0 if maximum is None else caps.setdefault(number(maximum, 1), len(caps) + 1)
            if cap > 65535 or maximum is not None and observed > maximum:
                raise ValueError("invalid output cap")
            row = bytearray(REQUEST.pack(arrival, leaf, observed, priority, cap))
            for image in images:
                row.extend(IMAGE.pack(*image))
            pages.append(row)
            count += 1
            missing_output += observed == -1 and cap == 0
            last_arrival = arrival
        if not count:
            raise ValueError("empty replay dataset")
        pages.finish()

        compressor = zstandard.ZstdCompressor(level=3, write_checksum=True)
        chain_pages = bytearray()
        for start in range(0, len(chains), CHAINS_PER_PAGE):
            page = chains[start:start + CHAINS_PER_PAGE]
            encoded = compressor.compress(b"".join(CHAIN.pack(*chain) for chain in page))
            entry = CHAIN_PAGE.pack(page[0][0], payload.tell(), len(encoded), len(page))
            chain_pages.extend(entry + zlib.crc32(entry).to_bytes(4, "big"))
            payload.write(encoded)
        chain_offset = HEADER.size + index.tell()
        media_offset = chain_offset + len(chain_pages)
        payload_offset = media_offset + len(dimensions) * (MEDIA.size + 4) + len(caps) * 8
        page_count = len(chain_pages) // (CHAIN_PAGE.size + 4)

        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".mckidx-", delete=False) as output:
            pending = Path(output.name)
            try:
                output.write(bytes(HEADER.size))
                digest = hashlib.sha256()

                def emit(data):
                    output.write(data)
                    digest.update(data)

                def emit_record(data):
                    emit(data)
                    emit(zlib.crc32(data).to_bytes(4, "big"))

                index.seek(0)
                while chunk := index.read(1024 * 1024):
                    emit(chunk)
                emit(chain_pages)
                for width, height in dimensions:
                    emit_record(MEDIA.pack(number(width, 1, 32768), number(height, 1, 32768)))
                for maximum in caps:
                    emit_record(struct.pack(">i", maximum))
                payload.seek(0)
                while chunk := payload.read(1024 * 1024):
                    emit(chunk)
                output.seek(0)
                output.write(HEADER.pack(b"MCKINDEX", count, chain_offset, page_count, media_offset,
                                         len(dimensions), len(caps), payload_offset, last_arrival, missing_output,
                                         digest.digest()))
                output.flush()
                os.fsync(output.fileno())
                os.link(pending, target)
            finally:
                pending.unlink()
