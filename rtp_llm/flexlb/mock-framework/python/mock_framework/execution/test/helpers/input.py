"""Write small MCKIDX fixtures from explicit prefix chains and request leaves."""

import hashlib
import struct
import zlib

import zstandard


def _checked_record(data):
    return data + struct.pack(">I", zlib.crc32(data))


def write_inputs(path, requests, *, chains=((1, 0, 1),), media=()):
    requests = list(requests)
    caps = {}
    missing = 0
    payload = bytearray()
    request_index = bytearray()
    chain_index = bytearray()
    compressor = zstandard.ZstdCompressor(level=3, write_checksum=True)

    for start in range(0, len(requests), 1024):
        offsets = bytearray()
        rows = bytearray()
        for leaf, arrival, output, priority, images in requests[start : start + 1024]:
            observed = output.get("observed_output_tokens", -1)
            maximum = output.get("max_output_tokens")
            cap_id = 0 if maximum is None else caps.setdefault(maximum, len(caps) + 1)
            missing += observed == -1 and cap_id == 0
            offsets += struct.pack(">I", len(rows))
            rows += struct.pack(">qqiiH", arrival, leaf, observed, priority, cap_id)
            for image in images:
                rows += struct.pack(">qii", *image)
        raw = offsets + rows
        encoded = compressor.compress(raw)
        request_index += _checked_record(
            struct.pack(">qii", len(payload), len(encoded), len(raw))
        )
        payload += encoded

    for start in range(0, len(chains), 256):
        page = chains[start : start + 256]
        raw = b"".join(struct.pack(">qqi", *chain) for chain in page)
        encoded = compressor.compress(raw)
        chain_index += _checked_record(
            struct.pack(">qqii", page[0][0], len(payload), len(encoded), len(page))
        )
        payload += encoded

    media_table = b"".join(_checked_record(struct.pack(">ii", *size)) for size in media)
    cap_table = b"".join(_checked_record(struct.pack(">i", cap)) for cap in caps)
    header = struct.Struct(">8s9q32s")
    chain_offset = header.size + len(request_index)
    media_offset = chain_offset + len(chain_index)
    payload_offset = media_offset + len(media_table) + len(cap_table)
    body = request_index + chain_index + media_table + cap_table + payload
    with path.open("xb") as stream:
        stream.write(
            header.pack(
                b"MCKINDEX",
                len(requests),
                chain_offset,
                len(chain_index) // 28,
                media_offset,
                len(media),
                len(caps),
                payload_offset,
                requests[-1][1],
                missing,
                hashlib.sha256(body).digest(),
            )
        )
        stream.write(body)
