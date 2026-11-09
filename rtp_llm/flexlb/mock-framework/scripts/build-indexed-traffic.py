#!/usr/bin/env python3
"""Seal one shared indexed replay file from anonymous text and media JSONL."""
import argparse
from array import array
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import zlib

import zstandard


HEADER = struct.Struct(">8s9q32s")
INDEX = struct.Struct(">qi")
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


def text_template(db, tokens, next_node):
    """Find a prefix in a disk radix tree; splitting a lookup edge preserves its node IDs."""
    if not isinstance(tokens, list) or len(tokens) > MAX_TOKENS:
        raise ValueError("invalid anonymous text template")
    packed = array("I", (number(token, 0) for token in tokens))
    if sys.byteorder == "little":
        packed.byteswap()
    values = packed.tobytes()
    parent = offset = 0
    while offset < len(values):
        symbol = values[offset:offset + 4]
        edge = db.execute("SELECT first, symbols FROM edges WHERE parent=? AND symbol=?",
                          (parent, symbol)).fetchone()
        if edge is None:
            remaining = values[offset:]
            db.execute("INSERT INTO edges VALUES (?, ?, ?, ?)", (parent, symbol, next_node, remaining))
            size = len(remaining) // 4
            return next_node + size - 1, (next_node, parent, size)
        first, previous = edge
        available = min(len(values) - offset, len(previous)) // 4
        if values[offset:offset + 4 * available] == previous[:4 * available]:
            shared = 4 * available
        else:
            low, high = 1, available - 1
            while low < high:
                mid = (low + high + 1) // 2
                if values[offset:offset + 4 * mid] == previous[:4 * mid]:
                    low = mid
                else:
                    high = mid - 1
            shared = 4 * low
        if shared < len(previous) and offset + shared < len(values):
            db.execute("UPDATE edges SET symbols=? WHERE parent=? AND symbol=?",
                       (previous[:shared], parent, symbol))
            db.execute("INSERT INTO edges VALUES (?, ?, ?, ?)",
                       (first + shared // 4 - 1, previous[shared:shared + 4], first + shared // 4, previous[shared:]))
        parent = first + shared // 4 - 1
        offset += shared
    return parent, None


def build(source, target):
    max_id = (1 << 63) - 1
    with (tempfile.TemporaryDirectory(dir=target.parent, prefix=".mckidx-tables-") as scratch,
          closing(sqlite3.connect(Path(scratch) / "prefix.db")) as db,
          tempfile.TemporaryFile(dir=target.parent) as index,
          tempfile.TemporaryFile(dir=target.parent) as chains,
          tempfile.TemporaryFile(dir=target.parent) as pages,
          tempfile.TemporaryFile(dir=target.parent) as media,
          tempfile.TemporaryFile(dir=target.parent) as requests):
        db.execute("PRAGMA cache_size=-16384")
        db.execute("CREATE TABLE edges (parent INTEGER, symbol BLOB, first INTEGER, symbols BLOB,"
                   " PRIMARY KEY(parent, symbol)) WITHOUT ROWID")
        count = chain_count = media_count = missing_output = 0
        caps = {}
        next_node = 1
        last_arrival = -1
        sealed = False
        for line_no, line in enumerate(source, 1):
            try:
                kind, *fields = json.loads(line)
                if sealed:
                    raise ValueError("data after seal")
                if kind == "media":
                    media_id, width, height = fields
                    if number(media_id, 1, max_id) != media_count + 1:
                        raise ValueError("noncontiguous media ID")
                    record(media, MEDIA.pack(number(width, 1, 32768), number(height, 1, 32768)))
                    media_count += 1
                elif kind == "tokens":
                    tokens, arrival, output, priority, images = fields
                    arrival = number(arrival, max(0, last_arrival), max_id)
                    if not isinstance(images, list) or len(images) > 256:
                        raise ValueError("invalid image list")
                    leaf, added = text_template(db, tokens, next_node)
                    length = len(tokens)
                    if added:
                        if leaf > max_id or chain_count >= (1 << 31) - 3:
                            raise ValueError("text chain limit")
                        chains.write(CHAIN.pack(*added))
                        chain_count += 1
                        next_node += added[2]
                    if count % 256 == 255:
                        db.commit()
                    if not 1 <= length + len(images) <= MAX_TOKENS:
                        raise ValueError("invalid input length")
                    if not isinstance(output, dict) or output.keys() - {"observed_output_tokens", "max_output_tokens"}:
                        raise ValueError("invalid output fields")
                    observed = output.get("observed_output_tokens")
                    maximum = output.get("max_output_tokens")
                    observed = -1 if observed is None else number(observed, 0)
                    cap_id = 0
                    if maximum is not None:
                        maximum = number(maximum, 1)
                        if observed > maximum:
                            raise ValueError("observed output exceeds request maximum")
                        cap_id = caps.setdefault(maximum, len(caps) + 1)
                        if cap_id > 65535:
                            raise ValueError("too many distinct output caps")
                    if observed == -1 and cap_id == 0:
                        missing_output += 1
                    row = bytearray(REQUEST.pack(arrival, leaf, observed, number(priority, 0), cap_id))
                    previous = 0
                    for media_id, offset, expected in images:
                        offset = number(offset, previous, length)
                        row += IMAGE.pack(number(media_id, 1, media_count), offset, number(expected, 0))
                        previous = offset
                    record(index, INDEX.pack(requests.tell(), len(row) + 4))
                    record(requests, row)
                    count += 1
                    last_arrival = arrival
                elif kind == "end":
                    if len(fields) != 1 or number(fields[0], 1, max_id) != count:
                        raise ValueError("request count mismatch")
                    sealed = True
                else:
                    raise ValueError("unknown traffic record")
            except (ValueError, TypeError, struct.error) as error:
                raise ValueError(f"line {line_no}: {error}") from error
        if not sealed:
            raise ValueError("missing final seal")
        chain_offset = HEADER.size + count * (INDEX.size + 4)
        chains.seek(0)
        compressor = zstandard.ZstdCompressor(level=3, write_checksum=True)
        page_count = 0
        for start in range(0, chain_count, CHAINS_PER_PAGE):
            size = min(CHAINS_PER_PAGE, chain_count - start)
            raw = chains.read(size * CHAIN.size)
            encoded = compressor.compress(raw)
            record(pages, CHAIN_PAGE.pack(struct.unpack_from(">q", raw)[0], requests.tell(), len(encoded), size))
            requests.write(encoded)
            page_count += 1
        media_offset = chain_offset + page_count * (CHAIN_PAGE.size + 4)
        payload = media_offset + media_count * (MEDIA.size + 4) + len(caps) * 8
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".mckidx-", delete=False) as out:
            pending = Path(out.name)
            try:
                out.write(bytes(HEADER.size))
                media.seek(0, 2)
                for cap in caps:
                    record(media, struct.pack(">i", cap))
                for table in (index, pages, media, requests):
                    table.seek(0)
                    while chunk := table.read(1024 * 1024):
                        digest.update(chunk)
                        out.write(chunk)
                out.seek(0)
                out.write(HEADER.pack(b"MCKINDEX", count, chain_offset, page_count, media_offset,
                                      media_count, len(caps), payload, last_arrival, missing_output, digest.digest()))
                out.flush()
                os.fsync(out.fileno())
                out.close()
                os.link(pending, target)
            finally:
                pending.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.suffix != ".mckidx":
        parser.error("output must end with .mckidx")
    try:
        with args.source.open(encoding="utf-8") as source:
            build(source, args.output)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
