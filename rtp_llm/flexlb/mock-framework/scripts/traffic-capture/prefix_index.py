"""Build the replay prefix tree incrementally in a coordinator-local SQLite file."""

import json
import sqlite3
from collections import Counter
from contextlib import closing
from functools import lru_cache

import mckidx_writer
from bundle import checksum
from layout import media_spans, text_at
from verify import (
    KEY_BYTES,
    frontend,
    geometry,
    header,
    inventory,
    records,
    require,
    source_summary,
)

SCHEMA = """
CREATE TABLE segments (start INTEGER PRIMARY KEY, end INTEGER, key TEXT, report TEXT);
CREATE TABLE shard_ids (id TEXT PRIMARY KEY) WITHOUT ROWID;
CREATE TABLE requests (
    id INTEGER PRIMARY KEY, arrival INTEGER, leaf INTEGER, priority INTEGER, cap INTEGER, media TEXT,
    prefix_text INTEGER, chain_first INTEGER, parent INTEGER, shared_blocks INTEGER, labels BLOB,
    observed INTEGER, capture_order INTEGER
);
CREATE INDEX arrival_order ON requests(arrival, id);
CREATE TABLE edges (key BLOB PRIMARY KEY, parent BLOB, depth INTEGER, stop INTEGER, owner INTEGER) WITHOUT ROWID;
CREATE TABLE chains (id INTEGER PRIMARY KEY, first INTEGER, parent INTEGER, length INTEGER);
CREATE TABLE images (
    key TEXT PRIMARY KEY, width INTEGER NOT NULL, height INTEGER NOT NULL, tokens INTEGER NOT NULL,
    alias TEXT, referenced INTEGER DEFAULT 0, replay INTEGER
) WITHOUT ROWID;
CREATE TABLE image_blocks (
    parent BLOB, depth INTEGER, start INTEGER, end INTEGER, image TEXT, child BLOB NOT NULL,
    PRIMARY KEY (parent, depth, start, end, image)
) WITHOUT ROWID;
CREATE TABLE replay_images (id INTEGER PRIMARY KEY, key TEXT UNIQUE, width INTEGER, height INTEGER);
"""


class Chains:
    """Expose the indexed slices the existing writer consumes a page at a time."""

    def __init__(self, db):
        self.db = db

    def __len__(self):
        return self.db.execute("SELECT coalesce(max(id),0) FROM chains").fetchone()[0]

    def __getitem__(self, page):
        return self.db.execute(
            "SELECT first,parent,length FROM chains WHERE id > ? AND id <= ? ORDER BY id",
            (page.start, page.stop),
        ).fetchall()


class Dimensions:
    def __init__(self, db):
        self.db = db

    def __len__(self):
        return self.db.execute(
            "SELECT coalesce(max(id),0) FROM replay_images"
        ).fetchone()[0]

    def __iter__(self):
        return iter(
            self.db.execute("SELECT width,height FROM replay_images ORDER BY id")
        )


def key_at(labels, depth):
    if depth < 0:
        return b""
    start = depth * KEY_BYTES
    return labels[start : start + KEY_BYTES]


def common_labels(labels, depth, edge):
    lo, hi = 0, min(len(labels) // KEY_BYTES - depth, len(edge) // KEY_BYTES)
    at = depth * KEY_BYTES
    if labels.startswith(edge, at):
        return len(edge) // KEY_BYTES
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if labels.startswith(edge[: mid * KEY_BYTES], at):
            lo = mid
        else:
            hi = mid - 1
    return lo


class PrefixIndex:
    def __init__(self, path):
        self.path = path
        with closing(self.connect()) as db:
            db.executescript(SCHEMA)

    def connect(self):
        db = sqlite3.connect(self.path)
        db.execute("PRAGMA cache_size=-65536")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def ingest(self, front, vit, start, end, workers, vit_pods, max_missing_rate):
        with closing(self.connect()) as db, db:
            previous = db.execute(
                "SELECT end,key,report FROM segments ORDER BY start DESC LIMIT 1"
            ).fetchone()
            front_paths, geometry_paths = inventory(front, vit, workers, vit_pods)
            counts = Counter(
                frontend_shards=len(front_paths), image_references=0, missing_geometry=0
            )
            shards = set()
            if geometry_paths:
                window, images = geometry(geometry_paths, None, shards, counts)
            else:
                with closing(records(front_paths[0])) as rows:
                    window, images = header(rows, "header", None, set()), {}
            require(
                window[:2] == (start, end),
                "archive window differs from scheduled segment",
            )
            require(
                previous is None or (previous[0], previous[1]) == (start, window[2]),
                "noncontiguous capture windows or changed capture key",
            )
            db.executemany(
                """
                INSERT INTO images(key,width,height,tokens) VALUES(?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET width=NULL
                WHERE (width,height,tokens)<>(excluded.width,excluded.height,excluded.tokens)
                """,
                ((key, *size) for key, size in images.items()),
            )
            latest = db.execute(
                "SELECT first,length FROM chains ORDER BY id DESC LIMIT 1"
            ).fetchone()
            next_node = sum(latest) if latest else 1

            def index_request(
                arrival,
                length,
                priority,
                cap,
                media,
                labels,
                block,
                route_block,
                observed,
            ):
                nonlocal next_node
                if previous is not None:
                    require(
                        (block, route_block) == previous_sizes,
                        "cache key block sizes changed",
                    )
                self.register_media(db, ((key, *images[key]) for key, _, _ in media))
                next_node = self.insert(
                    db,
                    counts,
                    next_node,
                    arrival,
                    length,
                    priority,
                    cap,
                    media,
                    labels,
                    block,
                    route_block,
                    observed=observed,
                )

            if previous is not None:
                report = json.loads(previous[2])
                previous_sizes = (
                    report["base_block_tokens"],
                    report["routing_block_tokens"],
                )
            block, route_block = frontend(
                None, front_paths, window, shards, images, counts, index_request
            )
            db.executemany(
                "INSERT INTO shard_ids VALUES (?)", ((shard,) for shard in shards)
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
                "routing_block_tokens": route_block,
            }
            if (
                any(counts[k] for k in ("dropped", "errors", "missing_geometry"))
                or rate > max_missing_rate
            ):
                raise SegmentLoss(result)
            db.execute(
                "INSERT INTO segments VALUES (?,?,?,?)",
                (start, end, window[2], json.dumps(result)),
            )
        return result

    @staticmethod
    def register_media(db, images):
        for key, width, height, tokens in images:
            prior = db.execute(
                "SELECT width,height,tokens FROM images WHERE key=?", (key,)
            ).fetchone()
            require(
                prior is None or prior == (width, height, tokens),
                "image geometry changed between segments",
            )
            db.execute(
                "INSERT OR IGNORE INTO images(key,width,height,tokens) VALUES (?,?,?,?)",
                (key, width, height, tokens),
            )

    @staticmethod
    def canonical(db, image):
        while True:
            alias = db.execute(
                "SELECT alias FROM images WHERE key=?", (image,)
            ).fetchone()[0]
            if alias is None:
                return image
            image = alias

    def insert(
        self,
        db,
        counts,
        next_node,
        arrival,
        length,
        priority,
        cap,
        media,
        labels,
        block,
        route_block,
        observed=None,
    ):
        mckidx_writer.number(length, 1, mckidx_writer.MAX_TOKENS)
        mckidx_writer.number(priority, 0)
        if cap is not None:
            mckidx_writer.number(cap, 1)
        require(len(media) <= 256, "request exceeds replay media limit")
        depth, blocks = 0, len(labels) // KEY_BYTES
        witness = None
        # Each edge points into the introducing request's unique suffix. A split
        # only rewrites edge offsets and leaves the stored labels intact.
        while depth < blocks:
            key = key_at(labels, depth)
            row = db.execute(
                """
                SELECT e.parent,e.depth,e.stop,e.owner,
                       substr(r.labels,(e.depth-r.shared_blocks)*?+1,(e.stop-e.depth)*?),
                       r.media,r.prefix_text,r.chain_first,r.parent
                FROM edges e JOIN requests r ON r.id=e.owner WHERE e.key=?
                """,
                (KEY_BYTES, KEY_BYTES, key),
            ).fetchone()
            if row is None:
                break
            parent_key, edge_start, stop, owner, edge, *witness = row
            require(
                edge_start == depth and parent_key == key_at(labels, depth - 1),
                "capture key has inconsistent ancestry",
            )
            matched = common_labels(labels, depth, edge)
            depth += matched
            if depth < stop:
                if depth < blocks:
                    db.execute("UPDATE edges SET stop=? WHERE key=?", (depth, key))
                    db.execute(
                        "INSERT INTO edges VALUES (?,?,?,?,?)",
                        (
                            key_at(edge, matched),
                            key_at(edge, matched - 1),
                            depth,
                            stop,
                            owner,
                        ),
                    )
                break
        counts["distinct_base_blocks"] += blocks - depth
        stride = route_block // block
        counts["distinct_routing_blocks"] += blocks // stride - depth // stride
        split = depth * block
        prefix_text = text_at(media, split)
        parent = 0
        if witness is not None:
            old_media, old_prefix, old_first, old_parent = witness
            old_media = json.loads(old_media)
            left = [(a, b, k) for a, b, k in media_spans(old_media) if a < split]
            right = [(a, b, k) for a, b, k in media_spans(media) if a < split]
            require(
                [(a, b) for a, b, _ in left] == [(a, b) for a, b, _ in right],
                "incompatible shared image layout",
            )
            require(prefix_text >= old_prefix, "matched prefix precedes its text chain")
            parent = (
                old_parent
                if prefix_text == old_prefix
                else old_first + prefix_text - old_prefix - 1
            )
            counts["shared_prefix_edges_with_distinct_media_affinity_ids"] += (
                left != right
            )
            for (_, _, a), (_, _, b) in zip(left, right):
                if a != b:
                    a, b = self.canonical(db, a), self.canonical(db, b)
                    if a != b:
                        db.execute(
                            "UPDATE images SET alias=? WHERE key=?",
                            (min(a, b), max(a, b)),
                        )
        remaining = length - sum(image[2] for image in media) - prefix_text
        leaf = next_node + remaining - 1 if remaining else parent
        if remaining:
            db.execute(
                "INSERT INTO chains(first,parent,length) VALUES (?,?,?)",
                (next_node, parent, remaining),
            )
        owner = db.execute(
            """
            INSERT INTO requests(arrival,leaf,priority,cap,media,prefix_text,chain_first,parent,shared_blocks,labels,observed)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                arrival,
                leaf,
                priority,
                cap,
                json.dumps(media, separators=(",", ":")),
                prefix_text,
                next_node,
                parent,
                depth,
                labels[depth * KEY_BYTES :],
                observed,
            ),
        ).lastrowid
        if depth < blocks:
            db.execute(
                "INSERT INTO edges VALUES (?,?,?,?,?)",
                (
                    key_at(labels, depth),
                    key_at(labels, depth - 1),
                    depth,
                    blocks,
                    owner,
                ),
            )
        self.record_media_blocks(db, media, labels, block)
        return next_node + remaining

    @staticmethod
    def record_media_blocks(db, media, labels, block):
        for start, end, image in media_spans(media):
            db.execute(
                "UPDATE images SET referenced=1 WHERE key=? AND referenced=0", (image,)
            )
            db.executemany(
                """
                INSERT INTO image_blocks VALUES (?,?,?,?,?,?)
                ON CONFLICT(parent,depth,start,end,image) DO UPDATE SET child=NULL WHERE child<>excluded.child
                """,
                (
                    (key_at(labels, i - 1), i, start, end, image, key_at(labels, i))
                    for i in range((start + block - 1) // block, end // block)
                ),
            )

    def seal(self, target, workers, vit_pods):
        with closing(self.connect()) as db, db:
            segments = [
                json.loads(row[0])
                for row in db.execute("SELECT report FROM segments ORDER BY start")
            ]
            first, last = segments[0], segments[-1]
            stat_fields = (
                "requests",
                "arrivals",
                "dropped",
                "errors",
                "missing_requests",
                "missing_geometry",
                "frontend_shards",
                "vit_shards",
                "image_references",
                "multimodal_requests",
                "oversize_media_requests",
                "conflicting_geometry_ids",
                "expanded_tokens",
                "base_blocks",
                "routing_blocks",
                "distinct_base_blocks",
                "distinct_routing_blocks",
                "shared_prefix_edges_with_distinct_media_affinity_ids",
            )
            totals = {k: sum(row.get(k, 0) for row in segments) for k in stat_fields}
            validation = {
                "start_ms": first["start_ms"],
                "end_ms": last["end_ms"],
                "frontend_pods": len(workers),
                "vit_pods": vit_pods,
                "base_block_tokens": first["base_block_tokens"],
                "routing_block_tokens": first["routing_block_tokens"],
                "unique_vit_images": db.execute(
                    "SELECT count(*) FROM images"
                ).fetchone()[0],
                **totals,
                "production_source": source_summary(
                    row["production_source"]
                    for row in segments
                    if "production_source" in row
                ),
                "segments": segments,
                "passed": not totals["missing_requests"],
            }
            for image, width, height, tokens in db.execute(
                "SELECT key,width,height,tokens FROM images WHERE referenced=1 ORDER BY key"
            ):
                root = self.canonical(db, image)
                require(
                    db.execute(
                        "SELECT width,height,tokens FROM images WHERE key=?", (root,)
                    ).fetchone()
                    == (width, height, tokens),
                    "shared image aliases disagree in geometry",
                )
                db.execute(
                    "INSERT OR IGNORE INTO replay_images(key,width,height) VALUES (?,?,?)",
                    (root, width, height),
                )
                db.execute(
                    "UPDATE images SET replay=(SELECT id FROM replay_images WHERE key=?) WHERE key=?",
                    (root, image),
                )
            conflict = db.execute(
                """
                SELECT 1 FROM image_blocks b JOIN images i ON b.image=i.key
                GROUP BY b.parent,b.depth,b.start,b.end,i.replay
                HAVING min(b.child) <> max(b.child) LIMIT 1
                """
            ).fetchone()
            require(conflict is None, "aliased image-only branches diverge")

            @lru_cache(maxsize=8192)
            def replay_id(image):
                return db.execute(
                    "SELECT replay FROM images WHERE key=?", (image,)
                ).fetchone()[0]

            request_rows = (
                (
                    arrival - first["start_ms"],
                    leaf,
                    observed,
                    priority,
                    cap,
                    [
                        (replay_id(key), offset, tokens)
                        for key, offset, tokens in json.loads(media)
                    ],
                )
                for arrival, leaf, observed, priority, cap, media in db.execute(
                    "SELECT arrival,leaf,observed,priority,cap,media FROM requests ORDER BY arrival,coalesce(capture_order,id),id"
                )
            )
            chains, dimensions = Chains(db), Dimensions(db)
            mckidx_writer.write(target, request_rows, chains, dimensions)
            base = (totals["base_blocks"] - totals["distinct_base_blocks"]) * first[
                "base_block_tokens"
            ]
            routing = (
                totals["routing_blocks"] - totals["distinct_routing_blocks"]
            ) * first["routing_block_tokens"]
            count, positions = db.execute(
                "SELECT count(*),coalesce(sum(length),0) FROM chains"
            ).fetchone()
            conversion = {
                "dataset_requests": totals["requests"],
                "dataset_bytes": target.stat().st_size,
                "theoretical_base": base / totals["expanded_tokens"],
                "repeat_tokens_base": base,
                "theoretical_routing": routing / totals["expanded_tokens"],
                "repeat_tokens_routing": routing,
                "image_references": totals["image_references"],
                "multimodal_requests": totals["multimodal_requests"],
                "unique_referenced_images": db.execute(
                    "SELECT count(*) FROM images WHERE referenced=1"
                ).fetchone()[0],
                "replay_media_ids": len(dimensions),
                "collapsed_proven_media_aliases": db.execute(
                    "SELECT count(*) FROM images WHERE alias IS NOT NULL"
                ).fetchone()[0],
                "shared_prefix_edges_with_distinct_media_affinity_ids": totals[
                    "shared_prefix_edges_with_distinct_media_affinity_ids"
                ],
                "text_chains": count,
                "synthetic_text_positions": positions,
                "sha256": checksum(target),
            }
        return validation, conversion


class SegmentLoss(ValueError):
    def __init__(self, report):
        super().__init__(f"segment {report['start_ms']} failed acceptance")
        self.report = report
