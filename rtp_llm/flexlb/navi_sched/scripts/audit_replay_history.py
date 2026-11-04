#!/usr/bin/env python3
"""Read-only audit of vLLM replay history; never treats unknown records as text.

Requires msgpack. Reads only frame zero, not the external multimodal tensors.
Checksums are not verified: this is a content audit, not a replacement for the
ReplayHistoryStore checksum validation performed when loading replay samples.
"""

import argparse
import json
import os
import struct

try:
    import msgpack
except ModuleNotFoundError:
    from pip._vendor import msgpack


HEADER = struct.Struct("<IIqQI")
MAGIC = 0x4E535248
LEGACY_RECORD_VERSION = 1
RECORD_VERSION = 2
HISTORY_PAYLOAD_VERSION = 1
COMPATIBLE_LEGACY_DATA_VERSIONS = (9, 10)
MAX_METADATA_BYTES = 8 * 1024 * 1024


def exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated record metadata")
    return data


def varint(stream, bits):
    value = 0
    for shift in range(0, bits, 7):
        byte = exact(stream, 1)[0]
        value |= (byte & 127) << shift
        if not byte & 128:
            if value >= 1 << bits:
                break
            return value
    raise ValueError("invalid varint")


class BoundedReader:
    def __init__(self, stream, remaining):
        self.stream = stream
        self.remaining = remaining

    def read(self, size=-1):
        size = self.remaining if size < 0 else min(size, self.remaining)
        data = self.stream.read(size)
        self.remaining -= len(data)
        return data


def classify(stream, payload_size, record_version=LEGACY_RECORD_VERSION):
    reader = BoundedReader(stream, payload_size)
    version, = struct.unpack("<i", exact(reader, 4))
    if record_version == LEGACY_RECORD_VERSION:
        if version not in COMPATIBLE_LEGACY_DATA_VERSIONS:
            raise ValueError("unsupported legacy request version: %d" % version)
    elif record_version == RECORD_VERSION:
        if version != HISTORY_PAYLOAD_VERSION:
            raise ValueError("unsupported history payload version: %d" % version)
    else:
        raise ValueError("unsupported record version: %d" % record_version)
    exact(reader, 8)  # Native request id is a raw uint64.
    biz_size = varint(reader, 32)
    if biz_size > 65536:
        raise ValueError("invalid biz length")
    exact(reader, biz_size)
    varint(reader, 64)  # Zigzag promptTokenCount.
    varint(reader, 64)  # Zigzag blockSize.
    hashes = varint(reader, 32)
    if hashes > 1024 * 1024:
        raise ValueError("invalid block hash count")
    exact(reader, hashes * 8)
    varint(reader, 64)  # Zigzag timeoutMs.
    forward_size = varint(reader, 32)
    if forward_size != reader.remaining:
        raise ValueError("forward payload length mismatch")
    unpacker = msgpack.Unpacker(reader, raw=False, read_size=4096,
                               max_buffer_size=MAX_METADATA_BYTES)
    frame_count = unpacker.read_array_header()
    if frame_count < 1:
        raise ValueError("empty forward frames")
    frame = unpacker.unpack()
    if not isinstance(frame, bytes):
        raise ValueError("frame zero is not bytes")
    request = msgpack.unpackb(frame, raw=False)
    if not isinstance(request, list) or len(request) < 6:
        raise ValueError("unknown EngineCoreRequest schema")
    if not isinstance(request[0], str) or not isinstance(request[1], list):
        raise ValueError("invalid request id or prompt tokens")
    features = request[2]
    if features is not None and not isinstance(features, list):
        raise ValueError("unknown mm_features schema")
    if features:
        return "vl"
    # Text requests with additional tensor frames need separate inspection.
    if frame_count != 1:
        raise ValueError("extra frames in purported text request")
    if unpacker.tell() != forward_size:
        raise ValueError("unexpected trailing forward data")
    return "text"


def audit(stream, start_time_us=0, end_time_us=2**63 - 1):
    size = os.fstat(stream.fileno()).st_size
    result = {"counts": {"text": 0, "vl": 0, "unknown": 0},
              "ranges": {}, "errors": [], "truncated_tail": False,
              "checksum_verified": False, "file_bytes": size}
    offset = 0
    while offset < size:
        stream.seek(offset)
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            result["truncated_tail"] = True
            break
        magic, version, event, length, _ = HEADER.unpack(header)
        if (magic != MAGIC or version not in (LEGACY_RECORD_VERSION, RECORD_VERSION)
                or event <= 0 or length > 256 * 1024 * 1024):
            result["errors"].append({"offset": offset, "error": "invalid record header"})
            break
        end = offset + HEADER.size + length
        if end > size:
            result["truncated_tail"] = True
            break
        if start_time_us <= event <= end_time_us:
            try:
                kind = classify(stream, length, version)
            except (ValueError, TypeError, msgpack.exceptions.UnpackException) as error:
                kind = "unknown"
                if len(result["errors"]) < 10:
                    result["errors"].append({"offset": offset, "error": str(error)})
            result["counts"][kind] += 1
            span = result["ranges"].setdefault(kind, {"earliest_us": event, "latest_us": event})
            span["earliest_us"] = min(span["earliest_us"], event)
            span["latest_us"] = max(span["latest_us"], event)
        offset = end
    result["scanned_bytes"] = offset
    result["all_selected_text"] = (result["counts"]["text"] > 0
                                   and result["counts"]["vl"] == 0
                                   and result["counts"]["unknown"] == 0
                                   and not result["errors"]
                                   and not result["truncated_tail"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--start-time-us", type=int, default=0)
    parser.add_argument("--end-time-us", type=int, default=2**63 - 1)
    args = parser.parse_args()
    if args.end_time_us <= args.start_time_us:
        parser.error("end-time-us must exceed start-time-us")
    with open(args.path, "rb") as stream:
        result = audit(stream, args.start_time_us, args.end_time_us)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["all_selected_text"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
