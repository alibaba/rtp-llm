#!/usr/bin/env python3
"""Validate a complete anonymous Frontend capture and its ViT geometry."""

import argparse
import base64
import gzip
import json
import re
import tempfile
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

KEY_BYTES = 16
PREFIX_BUCKETS = 64
EDGE_BYTES = 2 * KEY_BYTES + 1
SERVICE_ID = re.compile(r"v-[a-f0-9]{8}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(path):
    with gzip.open(path, "rt") as stream:
        yield from map(json.loads, stream)


def header(rows, kind, window, shards):
    tag, version, start, end, identity, key_check = next(rows)
    require(tag == kind and version == 4 and start < end, "invalid shard header")
    require(
        re.fullmatch(r"[a-f0-9]{32}", identity) and identity not in shards,
        "duplicate or invalid shard ID",
    )
    require(re.fullmatch(r"[a-f0-9]{32}", key_check), "invalid capture key check")
    current = (start, end, key_check)
    require(window is None or window == current, "mixed capture windows or keys")
    shards.add(identity)
    return current


def geometry(paths, window, shards, stats):
    images = {}
    conflicts = set()
    for path in paths:
        rows = records(path)
        window = header(rows, "geometry", window, shards)
        local = set()
        for tag, *fields in rows:
            if tag == "end":
                written, dropped = fields
                require(
                    written == len(local) and dropped == 0 and next(rows, None) is None,
                    "invalid or lossy ViT seal",
                )
                break
            key, width, height, rows_count = fields
            size = (width, height, rows_count)
            require(
                tag == "image"
                and re.fullmatch(r"[a-f0-9]{32}", key)
                and key not in local
                and all(type(n) is int and 1 <= n <= 2**31 - 1 for n in (width, height))
                and type(rows_count) is int
                and rows_count > 0,
                "invalid ViT geometry",
            )
            if images.setdefault(key, size) != size:
                conflicts.add(key)
            local.add(key)
        else:
            raise ValueError("missing ViT seal")
    # ViT archives can include geometry from requests outside the accepted
    # Frontend cohort. A referenced ambiguous key still fails as missing.
    for key in conflicts:
        del images[key]
    stats["conflicting_geometry_ids"] = len(conflicts)
    stats["vit_shards"] = len(paths)
    stats["unique_vit_images"] = len(images)
    return window, images


def source_summary(reports):
    """Combine counters from worker seals or already validated segment reports."""
    counts = Counter()
    missing = shards = 0
    for report in reports:
        counts.update(report["by_service_id"])
        missing += report["missing_service_id"]
        shards += report["shards_with_source_stats"]
    return {
        "first_frames": sum(counts.values()) + missing,
        "missing_service_id": missing,
        "by_service_id": dict(sorted(counts.items())),
        "shards_with_source_stats": shards,
    }


def source_seal(extra):
    require(len(extra) <= 1, "invalid Frontend seal fields")
    if not extra:
        return None
    data = extra[0]
    require(isinstance(data, list) and len(data) == 2, "invalid source seal")
    missing, counts = data
    require(
        type(missing) is int
        and missing >= 0
        and isinstance(counts, dict)
        and all(
            SERVICE_ID.fullmatch(key) and type(value) is int and value > 0
            for key, value in counts.items()
        ),
        "invalid source counts",
    )
    return {
        "by_service_id": counts,
        "missing_service_id": missing,
        "shards_with_source_stats": 1,
    }


def frontend(buckets, paths, window, shards, images, stats, index_request=None):
    block_sizes = None
    sources = []
    for path in paths:
        rows = records(path)
        start, end, _ = header(rows, "header", window, shards)
        written = 0
        for tag, *fields in rows:
            if tag == "end":
                arrivals, dropped, errors, *extra = fields
                require(
                    arrivals >= written and next(rows, None) is None,
                    "invalid Frontend seal",
                )
                stats.update(
                    arrivals=arrivals,
                    dropped=dropped,
                    errors=errors,
                    missing_requests=arrivals - written,
                )
                if (source := source_seal(extra)) is not None:
                    sources.append(source)
                break
            require(tag == "request", "unexpected Frontend row")
            (
                arrival,
                length,
                priority,
                maximum,
                media,
                encoded,
                route_count,
                block,
                route_block,
                *extra,
            ) = fields
            require(len(extra) in (0, 2, 3), "invalid request metadata")
            observed, interface = extra[:2] if extra else (None, "unknown")
            inline = extra[2] if len(extra) == 3 else []
            require(
                interface
                in ("openai_http", "native_http", "dashscope_grpc", "unknown"),
                "invalid capture interface",
            )
            require(
                maximum is None or type(maximum) is int and maximum > 0,
                "invalid output cap",
            )
            require(
                observed is None
                or type(observed) is int
                and observed > 0
                and (maximum is None or observed <= maximum),
                "invalid observed output length",
            )
            media_ids = {item[0] for item in media}
            for key, width, height, size in inline:
                require(
                    key in media_ids
                    and re.fullmatch(r"[a-f0-9]{32}", key)
                    and all(type(n) is int and n > 0 for n in (width, height, size)),
                    "invalid inline image geometry",
                )
                require(
                    images.setdefault(key, (width, height, size))
                    == (width, height, size),
                    "conflicting inline image geometry",
                )
            labels = base64.b64decode(encoded, validate=True)
            require(
                start <= arrival < end and length > 0 and priority >= 0,
                "invalid Frontend request",
            )
            require(
                type(block) is int
                and block > 0
                and type(route_block) is int
                and route_block >= block
                and route_block % block == 0,
                "invalid cache key block sizes",
            )
            if block_sizes is None:
                block_sizes = (block, route_block)
            require(block_sizes == (block, route_block), "mixed cache key block sizes")
            require(
                len(labels) == KEY_BYTES * (length // block)
                and route_count == length // route_block,
                "Frontend block count differs from token length",
            )
            stride = route_block // block
            media_length = sum(image[2] for image in media)
            previous = 0
            missing = 0
            oversize = False
            for key, offset, count in media:
                require(
                    re.fullmatch(r"[a-f0-9]{32}", key)
                    and previous <= offset <= length - media_length
                    and count > 0,
                    "invalid media offsets",
                )
                missing += key not in images
                if key in images:
                    oversize |= max(images[key][:2]) > 32768
                require(
                    key not in images or images[key][2] == count,
                    "ViT expansion differs from captured request",
                )
                previous = offset

            stats["missing_geometry"] += missing
            written += 1
            if oversize:
                require(
                    index_request is not None, "image exceeds MCKIDX dimension limit"
                )
                stats.update(oversize_media_requests=1, missing_requests=1)
                continue
            stats["image_references"] += len(media)
            if index_request is not None:
                if not missing:
                    index_request(
                        arrival,
                        length,
                        priority,
                        maximum,
                        media,
                        labels,
                        block,
                        route_block,
                        observed,
                    )
            else:
                parent = bytes(KEY_BYTES)
                flags = 2
                for index, at in enumerate(range(0, len(labels), KEY_BYTES), 1):
                    key = labels[at : at + KEY_BYTES]
                    route = index % stride == 0
                    buckets[key[0] % PREFIX_BUCKETS].write(
                        key + parent + bytes([flags | route])
                    )
                    parent, flags = key, 0
            stats["routing_blocks"] += route_count
            stats["base_blocks"] += len(labels) // KEY_BYTES
            stats.update(
                requests=1, expanded_tokens=length, multimodal_requests=bool(media)
            )
            stats["requests_" + interface] += 1
            stats["observed_outputs"] += observed is not None
        else:
            raise ValueError("missing Frontend seal")
    require(block_sizes is not None, "no Frontend requests captured")
    stats["production_source"] = source_summary(sources)
    return block_sizes


def inventory(front, vit, frontend_workers, vit_pods):
    groups = sorted(front.iterdir())
    require(
        {p.name for p in groups} == set(frontend_workers)
        and all(p.is_dir() for p in groups),
        "incomplete Frontend Pod inventory",
    )
    frontend_paths = []
    for group in groups:
        files = sorted(group.glob("worker-*.jsonl.gz"))
        require(
            sorted(int(f.name.split("-")[1]) for f in files)
            == list(range(frontend_workers[group.name])),
            "missing or duplicate Frontend worker",
        )
        frontend_paths.extend(files)
    groups = sorted(vit.iterdir()) if vit.exists() else []
    require(
        len(groups) == vit_pods and all(p.is_dir() for p in groups),
        "incomplete ViT Pod inventory",
    )
    geometry_paths = []
    for group in groups:
        files = list(group.glob("vit-*.jsonl.gz"))
        require(len(files) == 1, "missing or duplicate ViT shard")
        geometry_paths.extend(files)

    return frontend_paths, geometry_paths


def verify_series(segments, frontend_workers, vit_pods, scratch):
    stats = Counter()
    shards = set()
    image_ids = set()
    segment_reports = []
    previous_end = key_check = block_sizes = None
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        paths = [Path(tmp) / str(i) for i in range(PREFIX_BUCKETS)]
        with ExitStack() as stack:
            buckets = [
                stack.enter_context(path.open("wb", buffering=65536)) for path in paths
            ]
            for front, vit, start, end in segments:
                try:
                    front_paths, geometry_paths = inventory(
                        front, vit, frontend_workers, vit_pods
                    )
                    counts = Counter(
                        frontend_shards=len(front_paths),
                        image_references=0,
                        missing_geometry=0,
                    )
                    window, images = geometry(geometry_paths, None, shards, counts)
                    require(
                        (start, end) == window[:2]
                        and (previous_end is None or start == previous_end),
                        "noncontiguous capture windows",
                    )
                    require(
                        key_check is None or window[2] == key_check,
                        "capture key changed",
                    )
                    sizes = frontend(
                        buckets, front_paths, window, shards, images, counts
                    )
                    require(
                        block_sizes is None or sizes == block_sizes,
                        "cache key block sizes changed",
                    )
                except (ValueError, OSError, StopIteration) as error:
                    raise ValueError(f"segment {start}: {error}") from error
                block_sizes = sizes
                previous_end, key_check = end, window[2]
                image_ids.update(images)
                stats.update(
                    {k: v for k, v in counts.items() if k != "production_source"}
                )
                segment_reports.append(
                    {
                        "start_ms": start,
                        "end_ms": end,
                        **counts,
                        "missing_rate": counts["missing_requests"] / counts["arrivals"],
                    }
                )
        for path in paths:
            parents = {}
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(EDGE_BYTES * 65536), b""):
                    require(len(chunk) % EDGE_BYTES == 0, "truncated prefix bucket")
                    for at in range(0, len(chunk), EDGE_BYTES):
                        key = chunk[at : at + KEY_BYTES]
                        record = chunk[at + KEY_BYTES : at + EDGE_BYTES]
                        require(
                            parents.setdefault(key, record) == record,
                            "capture key has inconsistent ancestry",
                        )
            stats["distinct_base_blocks"] += len(parents)
            stats["distinct_routing_blocks"] += sum(
                record[-1] & 1 for record in parents.values()
            )
            path.unlink()
    checks = ("dropped", "errors", "missing_requests", "missing_geometry")
    start = segment_reports[0]["start_ms"]
    stats["unique_vit_images"] = len(image_ids)
    return {
        "start_ms": start,
        "end_ms": previous_end,
        "frontend_pods": len(frontend_workers),
        "vit_pods": vit_pods,
        "base_block_tokens": block_sizes[0],
        "routing_block_tokens": block_sizes[1],
        **stats,
        "production_source": source_summary(
            report["production_source"] for report in segment_reports
        ),
        "segments": segment_reports,
        "passed": bool(stats["requests"] and not any(stats[k] for k in checks)),
    }


def verify(front, vit, frontend_workers, vit_pods):
    _, geometry_paths = inventory(front, vit, frontend_workers, vit_pods)
    first = next(records(geometry_paths[0]))
    return verify_series(
        [(front, vit, first[2], first[3])], frontend_workers, vit_pods, None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frontend", type=Path)
    parser.add_argument("vit", type=Path)
    parser.add_argument("--frontend-workers", required=True, type=int)
    parser.add_argument("--vit-pods", required=True, type=int)
    args = parser.parse_args()
    result = verify(
        args.frontend,
        args.vit,
        {p.name: args.frontend_workers for p in args.frontend.iterdir()},
        args.vit_pods,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
