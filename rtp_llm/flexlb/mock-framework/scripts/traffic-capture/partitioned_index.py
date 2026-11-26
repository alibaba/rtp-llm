"""Stage checked segments and build an exact replay index from sorted runs."""

import base64
import heapq
import json
import os
import shutil
import struct
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from itertools import groupby, islice

import mckidx_writer
from layout import media_spans, text_at
from prefix_index import PrefixIndex, SegmentLoss, common_labels
from verify import KEY_BYTES, frontend, geometry, inventory, require

PARTITIONS = 16
FAN_IN = 8
RUN_HEADER = struct.Struct(">III")
MIN_FREE_BYTES = 1024**3


def partition(labels):
    return labels[0] >> 4 if labels else 0


def durable_json(path, payload):
    pending = path.with_suffix(".part")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    sync_directory(path.parent)


def sync_directory(path):
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_run(path, rows):
    previous = b""
    with path.open("xb", buffering=1024 * 1024) as stream:
        for labels, order, metadata in rows:
            shared = common_labels(labels, 0, previous) * KEY_BYTES
            suffix = labels[shared:]
            record = json.dumps([order, *metadata], separators=(",", ":")).encode()
            stream.write(RUN_HEADER.pack(shared, len(suffix), len(record)))
            stream.write(suffix)
            stream.write(record)
            previous = labels
        stream.flush()
        os.fsync(stream.fileno())


def read_run(path):
    previous = b""
    with path.open("rb", buffering=256 * 1024) as stream:
        while header := stream.read(RUN_HEADER.size):
            shared, suffix_size, record_size = RUN_HEADER.unpack(header)
            require(
                shared <= len(previous)
                and shared % KEY_BYTES == 0
                and suffix_size % KEY_BYTES == 0
                and shared + suffix_size <= mckidx_writer.MAX_TOKENS * KEY_BYTES
                and 0 < record_size <= 65536,
                "invalid capture run record",
            )
            suffix = stream.read(suffix_size)
            metadata = stream.read(record_size)
            require(
                len(suffix) == suffix_size and len(metadata) == record_size,
                "truncated capture run",
            )
            labels = previous[:shared] + suffix
            require(labels >= previous, "unordered capture run")
            order, *fields = json.loads(metadata)
            yield labels, order, fields
            previous = labels


def merge_runs(paths):
    return heapq.merge(*(read_run(path) for path in paths), key=lambda row: row[:2])


class PartitionedIndex:
    """Keep immutable runs on disk and serialize compaction for each partition."""

    def __init__(self, path, sort_bytes=64 * 1024**2):
        require(sort_bytes > 0, "sort buffer must be positive")
        self.path = path
        self.sort_bytes = sort_bytes
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.manifest = self.path / "manifest.json"
        self.state = (
            json.loads(self.manifest.read_text())
            if self.manifest.exists()
            else {"segments": [], "runs": []}
        )
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.compaction = None
        self.compaction_seconds = 0
        self.peak_bytes = self.bytes_used()
        # A manifest update is the commit point for staging and compaction.
        # Abandoned outputs are safe to remove on an exclusively owned workspace.
        referenced = {row[0] for row in self.state["runs"]}
        for run in self.path.glob("*.run"):
            if run.name not in referenced:
                run.unlink()

    def bytes_used(self):
        total = 0
        for path in self.path.iterdir():
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def sample(self):
        self.peak_bytes = max(self.peak_bytes, self.bytes_used())

    def progress(self):
        with self.lock:
            runs = len(self.state["runs"])
        return {
            "index_bytes": self.bytes_used(),
            "index_runs": runs,
            "compaction_seconds": round(self.compaction_seconds, 3),
        }

    def check_disk(self):
        require(
            shutil.disk_usage(self.path).free >= MIN_FREE_BYTES,
            "capture working disk below 1 GiB reserve",
        )

    def new_run(self, rows):
        self.check_disk()
        target = self.path / (uuid.uuid4().hex + ".run")
        try:
            write_run(target, rows)
            self.sample()
            return target
        except BaseException:
            target.unlink(missing_ok=True)
            raise

    def ingest(self, front, vit, start, end, workers, vit_pods, max_missing_rate):
        if self.compaction is not None and self.compaction.done():
            self.compaction.result()
        with self.lock:
            segments = self.state["segments"]
        previous = segments[-1] if segments else None
        order = sum(row["report"]["requests"] for row in segments)
        front_paths, vit_paths = inventory(front, vit, workers, vit_pods)
        counts = Counter(
            frontend_shards=len(front_paths), image_references=0, missing_geometry=0
        )
        shards = set()
        window, images = (
            geometry(vit_paths, None, shards, counts) if vit_paths else (None, {})
        )
        if window is None:
            window = frontend_window(front_paths)
        require(
            window[:2] == (start, end), "archive window differs from scheduled segment"
        )
        require(
            previous is None
            or (previous["report"]["end_ms"], previous["key"]) == (start, window[2]),
            "noncontiguous windows or changed capture key",
        )
        runs, pending, referenced = [], [], set()
        byte_count = 0

        def flush():
            nonlocal byte_count
            for bucket, rows in groupby(
                sorted(pending, key=lambda row: row[:2]),
                key=lambda row: partition(row[0]),
            ):
                path = self.new_run(rows)
                runs.append([path.name, bucket, 0])
            pending.clear()
            byte_count = 0

        def add(arrival, length, priority, cap, media, labels, block, route, observed):
            nonlocal order, byte_count
            mckidx_writer.number(length, 1, mckidx_writer.MAX_TOKENS)
            require(len(media) <= 256, "request exceeds replay media limit")
            if previous is not None:
                report = previous["report"]
                require(
                    (block, route)
                    == (report["base_block_tokens"], report["routing_block_tokens"]),
                    "cache key block sizes changed",
                )
            pending.append(
                (labels, order, [arrival, length, priority, cap, media, observed])
            )
            order += 1
            referenced.update(key for key, _, _ in media)
            byte_count += len(labels) + 128 + 100 * len(media)
            if byte_count >= self.sort_bytes:
                flush()

        try:
            block, route = frontend(
                None, front_paths, window, shards, images, counts, add
            )
            rate = (
                counts["missing_requests"] / counts["arrivals"]
                if counts["arrivals"]
                else 0
            )
            result = {
                "start_ms": start,
                "end_ms": end,
                **counts,
                "missing_rate": rate,
                "base_block_tokens": block,
                "routing_block_tokens": route,
            }
            if (
                any(counts[k] for k in ("dropped", "errors", "missing_geometry"))
                or rate > max_missing_rate
            ):
                raise SegmentLoss(result)
            require(
                not any(shards.intersection(row["shards"]) for row in segments),
                "repeated archive shard",
            )
            flush()
            media_path = self.path / f"media-{start}.json"
            durable_json(
                media_path, [[key, *images[key]] for key in sorted(referenced)]
            )
            entry = {
                "report": result,
                "key": window[2],
                "media": media_path.name,
                "shards": sorted(shards),
            }
            with self.lock:
                committed = {
                    "segments": [*segments, entry],
                    "runs": [*self.state["runs"], *runs],
                }
                durable_json(self.manifest, committed)
                self.state = committed
        except BaseException:
            # If directory fsync failed after replace, leave files for recovery.
            if not self.manifest.exists() or not any(
                row["report"]["start_ms"] == start
                for row in json.loads(self.manifest.read_text())["segments"]
            ):
                for name, _, _ in runs:
                    (self.path / name).unlink(missing_ok=True)
            raise
        self.sample()
        if self.compaction is None or self.compaction.done():
            self.compaction = self.pool.submit(self.compact)
        return result

    def compact(self):
        started = time.monotonic()
        while True:
            with self.lock:
                groups = {}
                for entry in self.state["runs"]:
                    groups.setdefault(tuple(entry[1:]), []).append(entry)
                selected = next(
                    (rows[:FAN_IN] for rows in groups.values() if len(rows) >= FAN_IN),
                    None,
                )
            if selected is None:
                self.compaction_seconds += time.monotonic() - started
                return
            bucket, level = selected[0][1:]
            paths = [self.path / row[0] for row in selected]
            merged = self.new_run(merge_runs(paths))
            with self.lock:
                runs = [row for row in self.state["runs"] if row not in selected]
                state = {
                    **self.state,
                    "runs": [*runs, [merged.name, bucket, level + 1]],
                }
                durable_json(self.manifest, state)
                self.state = state
            for path in paths:
                path.unlink()

    def close(self):
        self.pool.shutdown(wait=True)
        if self.compaction is not None:
            self.compaction.result()

    def seal(self, target, workers, vit_pods):
        self.close()
        self.compact()
        entries = self.state["segments"]
        require(entries, "no accepted capture segments")
        # A retry rebuilds only the derived output tables. The manifest remains authoritative.
        database = self.path / "replay.sqlite"
        database.unlink(missing_ok=True)
        output = PrefixIndex(database)
        counts = Counter()
        node = 1
        block = entries[0]["report"]["base_block_tokens"]
        route = entries[0]["report"]["routing_block_tokens"]
        with closing(output.connect()) as db:
            for entry in entries:
                with db:
                    output.register_media(
                        db, json.loads((self.path / entry["media"]).read_text())
                    )
                    segment = entry["report"]
                    db.execute(
                        "INSERT INTO segments VALUES (?,?,?,?)",
                        (
                            segment["start_ms"],
                            segment["end_ms"],
                            entry["key"],
                            json.dumps(segment),
                        ),
                    )
            for bucket in range(PARTITIONS):
                paths = [
                    self.path / row[0] for row in self.state["runs"] if row[1] == bucket
                ]
                require(len(paths) < 64, "capture partition exceeds final merge fan-in")
                previous = b""
                chain = []
                reader = iter(merge_runs(paths))
                while batch := list(islice(reader, 512)):
                    self.check_disk()
                    with db:
                        for labels, order, fields in batch:
                            node, chain = append_replay(
                                db,
                                output,
                                counts,
                                node,
                                block,
                                route,
                                previous,
                                chain,
                                labels,
                                order,
                                fields,
                            )
                            previous = labels
                    self.sample()
            with db:
                summary = {
                    **entries[0]["report"],
                    **counts,
                    "requests": entries[0]["report"]["requests"],
                }
                db.execute(
                    "UPDATE segments SET report=? WHERE start=?",
                    (json.dumps(summary), entries[0]["report"]["start_ms"]),
                )
        require(
            counts["requests"] == sum(row["report"]["requests"] for row in entries),
            "capture run count changed",
        )
        validation, conversion = output.seal(target, workers, vit_pods)
        for entry in entries:
            validation.update(
                {
                    key: validation.get(key, 0) + value
                    for key, value in entry["report"].items()
                    if key.startswith("requests_") or key == "observed_outputs"
                }
            )
        self.sample()
        conversion.update(
            compaction_seconds=round(self.compaction_seconds, 3),
            working_peak_bytes=self.peak_bytes,
        )
        return validation, conversion


def frontend_window(paths):
    from verify import header, records

    with closing(records(paths[0])) as rows:
        return header(rows, "header", None, set())


def append_replay(
    db, index, counts, node, block, route, previous, chain, labels, order, fields
):
    arrival, length, priority, cap, media, observed = fields
    depth = common_labels(labels, 0, previous)
    blocks = len(labels) // KEY_BYTES
    stride = route // block
    counts.update(
        requests=1,
        distinct_base_blocks=blocks - depth,
        distinct_routing_blocks=blocks // stride - depth // stride,
    )
    split = depth * block
    prefix_text = text_at(media, split)
    if split:
        left = [(a, b, key) for a, b, key in media_spans(chain[0]) if a < split]
        right = [(a, b, key) for a, b, key in media_spans(media) if a < split]
        require(
            [(a, b) for a, b, _ in left] == [(a, b) for a, b, _ in right],
            "incompatible shared image layout",
        )
        counts["shared_prefix_edges_with_distinct_media_affinity_ids"] += left != right
        for (_, _, a), (_, _, b) in zip(left, right):
            a, b = index.canonical(db, a), index.canonical(db, b)
            if a != b:
                db.execute(
                    "UPDATE images SET alias=? WHERE key=?", (min(a, b), max(a, b))
                )
    retained = []
    for end, first in chain[1:] if chain else ():
        start = retained[-1][0] if retained else 0
        if start >= prefix_text:
            break
        retained.append((min(end, prefix_text), first))
    require(
        (retained[-1][0] if retained else 0) == prefix_text,
        "invalid shared text prefix",
    )
    parent = (
        retained[-1][1]
        + prefix_text
        - (retained[-2][0] if len(retained) > 1 else 0)
        - 1
        if retained
        else 0
    )
    text_length = length - sum(image[2] for image in media)
    remaining = text_length - prefix_text
    leaf = node + remaining - 1 if remaining else parent
    if remaining:
        db.execute(
            "INSERT INTO chains(first,parent,length) VALUES (?,?,?)",
            (node, parent, remaining),
        )
        retained.append((text_length, node))
    db.execute(
        "INSERT INTO requests(arrival,leaf,priority,cap,media,observed,capture_order) VALUES (?,?,?,?,?,?,?)",
        (
            arrival,
            leaf,
            priority,
            cap,
            json.dumps(media, separators=(",", ":")),
            observed,
            order,
        ),
    )
    index.record_media_blocks(db, media, labels, block)
    return node + remaining, [media, *retained]
