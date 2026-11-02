#!/usr/bin/env python3
"""Write a text-only NaviSched replay history without changing the source.

Run against a stopped client or a stable snapshot of queries.bin. Unknown or
corrupt records fail closed; the output is never published partially.
"""

import argparse
import fcntl
import json
import os

from audit_replay_history import (HEADER, LEGACY_RECORD_VERSION, MAGIC,
                                  RECORD_VERSION, classify, msgpack)


MAX_RECORD_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024 * 1024
MAX_OUTPUT_RECORDS = 40000
_CRC_TABLE = None


def crc32c(data, value=0):
    global _CRC_TABLE
    if _CRC_TABLE is None:
        table = []
        for index in range(256):
            entry = index
            for _ in range(8):
                entry = (entry >> 1) ^ (0x82F63B78 if entry & 1 else 0)
            table.append(entry)
        _CRC_TABLE = table
    crc = value ^ 0xFFFFFFFF
    for byte in data:
        crc = _CRC_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def checked_record(source, offset, file_size):
    source.seek(offset)
    header = source.read(HEADER.size)
    if len(header) != HEADER.size:
        raise ValueError("truncated record header at byte %d" % offset)
    magic, version, event, length, checksum = HEADER.unpack(header)
    if (magic != MAGIC or version not in (LEGACY_RECORD_VERSION, RECORD_VERSION)
            or event <= 0 or length > MAX_RECORD_BYTES):
        raise ValueError("invalid record header at byte %d" % offset)
    end = offset + HEADER.size + length
    if end > file_size:
        raise ValueError("truncated record payload at byte %d" % offset)
    value = 0
    remaining = length
    while remaining:
        chunk = source.read(min(65536, remaining))
        if not chunk:
            raise ValueError("truncated record payload at byte %d" % offset)
        value = crc32c(chunk, value)
        remaining -= len(chunk)
    if value != checksum:
        raise ValueError("record checksum mismatch at byte %d" % offset)
    source.seek(offset + HEADER.size)
    try:
        kind = classify(source, length, version)
    except (ValueError, TypeError, msgpack.exceptions.UnpackException) as error:
        raise ValueError("unknown request at byte %d: %s" % (offset, error)) from error
    return end, event, kind


def filter_history(source_path, output_path, max_records=MAX_OUTPUT_RECORDS,
                   max_bytes=MAX_OUTPUT_BYTES):
    if max_records <= 0 or max_bytes <= 0:
        raise ValueError("output limits must be positive")
    source_path = os.path.realpath(source_path)
    output_path = os.path.abspath(output_path)
    if source_path == os.path.realpath(output_path):
        raise ValueError("source and output must differ")
    temporary = output_path + ".tmp"
    counts = {"text": 0, "vl": 0}
    written = 0
    earliest = None
    latest = None
    created = False
    directory = os.open(os.path.dirname(output_path), os.O_RDONLY)
    try:
        fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with open(source_path, "rb") as source:
            if os.path.lexists(temporary):
                if not os.path.isfile(temporary) or os.path.islink(temporary):
                    raise ValueError("unsafe temporary output: " + temporary)
                os.unlink(temporary)
            if os.path.lexists(output_path):
                raise FileExistsError(output_path)
            file_size = os.fstat(source.fileno()).st_size
            with open(temporary, "xb") as output:
                created = True
                offset = 0
                while offset < file_size:
                    end, event, kind = checked_record(source, offset, file_size)
                    counts[kind] += 1
                    if kind == "text":
                        earliest = event if earliest is None else min(earliest, event)
                        latest = event if latest is None else max(latest, event)
                        record_size = end - offset
                        if counts["text"] > max_records or written + record_size > max_bytes:
                            raise ValueError("text-only history exceeds output limits")
                        source.seek(offset)
                        remaining = record_size
                        while remaining:
                            chunk = source.read(min(65536, remaining))
                            if not chunk:
                                raise ValueError("source changed during copy")
                            output.write(chunk)
                            remaining -= len(chunk)
                        written += record_size
                    offset = end
                if not counts["text"]:
                    raise ValueError("no text requests to replay")
                if os.fstat(source.fileno()).st_size != file_size:
                    raise ValueError("source changed during filtering")
                output.flush()
                os.fsync(output.fileno())
        os.link(temporary, output_path)
        os.fsync(directory)
        return {"counts": counts, "earliest_event_time_us": earliest,
                "latest_event_time_us": latest, "output_bytes": written,
                "output": output_path}
    finally:
        if created:
            os.unlink(temporary)
        os.close(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="stable source queries.bin")
    parser.add_argument("output", help="new text-only queries.bin; never overwritten")
    parser.add_argument("--max-records", type=int, default=MAX_OUTPUT_RECORDS)
    parser.add_argument("--max-bytes", type=int, default=MAX_OUTPUT_BYTES)
    args = parser.parse_args()
    print(json.dumps(filter_history(args.source, args.output,
                                    args.max_records, args.max_bytes), sort_keys=True))


if __name__ == "__main__":
    main()
