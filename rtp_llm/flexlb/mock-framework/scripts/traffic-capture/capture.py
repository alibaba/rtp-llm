"""Operate one sealed capture batch from a single Frontend ASI session."""

import argparse
import base64
import fcntl
import hashlib
import hmac
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from ipaddress import IPv4Address
from pathlib import Path

import bundle
from partitioned_index import PartitionedIndex
from prefix_index import SegmentLoss

JOBS = Path(os.environ.get("MOCK_CAPTURE_JOBS", "/tmp/mock-capture"))
MAX_SEGMENT_MS = 300_000
ARM_LEAD_MS = 30_000
GIB = 1024**3
DOMAINS = {"frontend": "MOCK_CAPTURE_EXPORT_VIP", "vit": "MOCK_CAPTURE_VIT_EXPORT_VIP"}
NAMES = re.compile(r"vit-[0-9a-f]{32}\.jsonl\.gz")


def millis():
    return time.time_ns() // 1_000_000


def status(root, phase, **fields):
    current = root / "status.json"
    previous = json.loads(current.read_text()) if current.exists() else {}
    bundle.write_json(
        current, {**previous, "batch": root.name, "phase": phase, **fields}
    )


def post(ip, port, path, payload, headers=None):
    conn = http.client.HTTPConnection(ip, port, timeout=4)
    try:
        conn.request(
            "POST",
            path,
            payload,
            {"Content-Type": "application/json", **(headers or {})},
        )
        with conn.getresponse() as response:
            raw = response.read(5 * 1024 * 1024 + 1)
            if response.status != 200 or len(raw) > 5 * 1024 * 1024:
                raise OSError(f"capture_http_{response.status}")
        return json.loads(raw)
    finally:
        conn.close()


def frontend(ip, operation, key, fields):
    path = "/capture/" + operation
    payload = json.dumps({**fields, "expires": int(time.time()) + 60}).encode()
    return post(
        ip,
        bundle.PORT,
        path,
        payload,
        {"X-Capture-Signature": bundle.signature(path.encode() + b"\n" + payload, key)},
    )


def vit(ip, port, operation, key, fields):
    args = {**fields, "expires": int(time.time()) + 60}
    order = (
        ("run", "start", "end", "expires")
        if operation == "arm"
        else ("start", "owner", "expires", "offset", "size")
    )
    version = 2 if operation == "arm" else 1
    payload = "\n".join(
        [f"vit-capture-{operation}-v{version}", *(str(args[k]) for k in order)]
    )
    args["signature"] = hmac.digest(key, payload.encode("ascii"), "sha256").hex()
    return post(
        ip,
        port,
        "/invoke",
        json.dumps(
            {
                "interface": "vit/capture/" + operation,
                "timeout_ms": 3000,
                "args": args,
            }
        ),
    )["result"]


def membership(expected):
    members = {}
    for role, env in DOMAINS.items():
        if not expected[role]:
            members[role] = {}
            continue
        endpoints = bundle.vip_members(os.environ[env])
        addresses = {str(IPv4Address(ip)): port for ip, port in endpoints}
        if role == "vit" and (
            len(endpoints) != len(addresses)
            or any(not 1 <= p <= 65535 for p in addresses.values())
        ):
            raise ValueError("ViT VIP must advertise one control port per Pod")
        if len(addresses) != expected[role]:
            raise ValueError(f"{role} VIP count {len(addresses)} != {expected[role]}")
        members[role] = (
            dict.fromkeys(addresses, bundle.PORT) if role == "frontend" else addresses
        )
    return members


def parallel(pool, targets, callback, ready, seconds):
    deadline = time.monotonic() + seconds
    pending = set(targets)
    results = {}
    errors = {}
    while pending and time.monotonic() < deadline:

        def attempt(target):
            try:
                return callback(*target)
            except (KeyError, ValueError, OSError, http.client.HTTPException) as error:
                errors[target] = str(error)
                return None

        batch = sorted(pending)
        for target, row in zip(batch, pool.map(attempt, batch)):
            if row is not None and ready(row):
                results[target] = row
                pending.remove(target)
        if pending:
            time.sleep(0.25)
    if pending:
        details = ", ".join(
            f"{role}/{ip}: {errors.get((role, ip), 'not ready')}"
            for role, ip in sorted(pending)[:5]
        )
        raise RuntimeError(f"{len(pending)} capture targets incomplete: {details}")
    return results


def fetch(role, ip, port, key, window, armed, shards):
    address = hashlib.sha256(ip.encode()).hexdigest()
    start = window["start"]
    manifest = (
        frontend(ip, "archive", key, {"start": start})
        if role == "frontend"
        else vit(
            ip,
            port,
            "read",
            key,
            {"start": start, "owner": address, "offset": 0, "size": 1},
        )
    )
    if (
        manifest["start"] != start
        or type(manifest["bytes"]) is not int
        or manifest["bytes"] <= 0
        or not re.fullmatch(r"[a-f0-9]{64}", manifest["sha256"])
    ):
        raise ValueError("invalid archive manifest")
    actual = manifest if role == "frontend" else vit(ip, port, "arm", key, window)
    if actual["instance"] != armed["instance"]:
        raise ValueError("capture instance changed")
    name = "pod.tar" if role == "frontend" else manifest["name"]
    if role == "vit" and not NAMES.fullmatch(name):
        raise ValueError("invalid ViT archive name")
    group = shards / role / ip
    group.mkdir(exist_ok=True)
    partial = group / "archive.part"
    archive = group / name
    offset = 0
    with partial.open("wb") as stream:
        while offset < manifest["bytes"]:
            size = min(bundle.CHUNK_BYTES, manifest["bytes"] - offset)
            request = {"start": start, "offset": offset, "size": size}
            row = (
                frontend(ip, "archive", key, {**request, "sha256": manifest["sha256"]})
                if role == "frontend"
                else vit(ip, port, "read", key, {**request, "owner": address})
            )
            if (row["start"], row["bytes"], row["sha256"], row["offset"]) != (
                start,
                manifest["bytes"],
                manifest["sha256"],
                offset,
            ):
                raise ValueError("archive manifest changed")
            if role == "frontend" and row["instance"] != armed["instance"]:
                raise ValueError("Frontend archive instance changed")
            if role == "vit" and (row["owner"] != address or row["name"] != name):
                raise ValueError("ViT archive identity changed")
            chunk = base64.b64decode(row["data_base64"], validate=True)
            if (
                len(chunk) != size
                or hashlib.sha256(chunk).hexdigest() != row["chunk_sha256"]
            ):
                raise ValueError("archive chunk checksum mismatch")
            stream.write(chunk)
            offset += size
    if bundle.checksum(partial) != manifest["sha256"]:
        raise ValueError("archive SHA-256 mismatch")
    partial.replace(archive)
    if role == "frontend":
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            parsed = [bundle.SHARD.fullmatch(m.name) for m in members]
            if (
                not all(parsed)
                or sorted(int(m[1]) for m in parsed) != list(range(armed["workers"]))
                or not all(m.isfile() for m in members)
            ):
                raise ValueError("unexpected Frontend shard membership")
            for member in members:
                with tar.extractfile(member) as source, (group / member.name).open(
                    "wb"
                ) as dest:
                    while chunk := source.read(1024 * 1024):
                        dest.write(chunk)
        archive.unlink()
    return {
        "bytes": manifest["bytes"],
        "sha256": manifest["sha256"],
        "instance": armed["instance"],
    }


def windows(start, end, segment_ms):
    return [
        {"run": start, "start": offset, "end": min(offset + segment_ms, end)}
        for offset in range(start, end, segment_ms)
    ]


def arm_segment(window, members, targets, key):
    def arm(role, ip):
        if role == "frontend":
            return frontend(ip, "arm", key, window)
        return vit(ip, members[role][ip], "arm", key, window)

    start = window["start"]
    with ThreadPoolExecutor(max_workers=16) as pool:
        armed = parallel(
            pool,
            targets,
            arm,
            lambda row: row["state"] == "armed" and row["start"] == start,
            max(0, (start - millis()) / 1000 - 3),
        )
    if millis() >= start or len({r["instance"] for r in armed.values()}) != len(armed):
        raise ValueError("could not arm a complete inventory before the segment")
    return armed


def fetch_segment(root, window, members, targets, armed, key):
    shards = root / "shards"
    for role in DOMAINS:
        (shards / role).mkdir(parents=True)

    def fetch_role(role):
        began = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = parallel(
                pool,
                [x for x in targets if x[0] == role],
                lambda role, ip: fetch(
                    role, ip, members[role][ip], key, window, armed[role, ip], shards
                ),
                lambda row: True,
                90,
            )
        return round(time.monotonic() - began, 3), rows

    began = time.monotonic()
    timing = {}
    checks = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {role: pool.submit(fetch_role, role) for role in DOMAINS}
        for role, future in jobs.items():
            timing[role + "_fetch_seconds"], rows = future.result()
            checks.extend(
                {"role": role, "ip": ip, **row}
                for (role, ip), row in sorted(rows.items())
            )
    timing["parallel_fetch_seconds"] = round(time.monotonic() - began, 3)
    report = {
        "start_ms": window["start"],
        "end_ms": window["end"],
        "timing": timing,
        "archive_checks": checks,
    }
    bundle.write_json(root / "archive-checks.json", report)
    return report


def collect_segment(
    root, window, members, targets, armed, key, index, max_missing_rate, keep_shards
):
    if shutil.disk_usage(root.parent).free < GIB:
        raise OSError("workspace has less than 1 GiB free")
    report = fetch_segment(root, window, members, targets, armed, key)
    workers = {
        ip: row["workers"] for (role, ip), row in armed.items() if role == "frontend"
    }
    began = time.monotonic()
    try:
        report["validation"] = index.ingest(
            root / "shards/frontend",
            root / "shards/vit",
            window["start"],
            window["end"],
            workers,
            len(members["vit"]),
            max_missing_rate,
        )
    except SegmentLoss as error:
        bundle.write_json(root / "validation.json", error.report)
        raise
    report["timing"]["index_seconds"] = round(time.monotonic() - began, 3)
    report.update(index.progress())
    bundle.write_json(root / "archive-checks.json", report)
    if not keep_shards:
        shutil.rmtree(root / "shards")
    return report


def execute(root, plan):
    start, end, expected = plan["start"], plan["end"], plan["expected"]
    key = bundle.configured_key()
    if key is None:
        raise ValueError("MOCK_CAPTURE_KEY is missing")
    work = Path(plan["work"])
    index = PartitionedIndex(work / "index", plan["sort_bytes"])
    max_missing_rate = plan["max_missing_rate"]
    members = membership(expected)
    targets = [
        (role, ip) for role, addresses in members.items() for ip in sorted(addresses)
    ]
    segments = windows(start, end, plan["segment_ms"])
    paths = {window["start"]: work / str(window["start"]) for window in segments}
    events = sorted(
        [(window["start"] - ARM_LEAD_MS, "arm", window["start"]) for window in segments]
        + [
            (window["end"] + bundle.DRAIN_MS, "fetch", window["start"])
            for window in segments
        ]
    )
    by_start = {window["start"]: window for window in segments}
    armed = {}
    first = None
    identities = None
    timing = {"arm_seconds": 0}
    pending, completed = {}, {}
    status(
        root,
        "collecting",
        start_ms=start,
        end_ms=end,
        segment_count=len(segments),
        work_dir=str(work),
    )

    def collect_completed(wait=False):
        previous_count = len(completed)
        for offset, future in list(pending.items()):
            if wait or future.done():
                try:
                    completed[offset] = future.result()
                except Exception as error:
                    raise RuntimeError(
                        f"segment {offset} archive fetch or indexing failed: {error}"
                    ) from error
                del pending[offset]
        if len(completed) != previous_count:
            status(
                root,
                "collecting",
                sealed_segments=len(completed),
                verified_segments=sorted(completed),
                pending_segments=len(pending),
                **index.progress(),
            )

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        for at, action, offset in events:
            while millis() < at:
                collect_completed()
                time.sleep(min(2, max(0, (at - millis()) / 1000)))
            collect_completed()
            window = by_start[offset]
            status(root, "collecting", segment=offset, action=action)
            if shutil.disk_usage(work).free < GIB:
                raise OSError(f"segment {offset}: workspace has less than 1 GiB free")
            if action == "fetch":
                if len(pending) >= 2:
                    raise RuntimeError(
                        "two sealed segments are still awaiting acceptance"
                    )
                pending[offset] = pool.submit(
                    collect_segment,
                    paths[offset],
                    window,
                    members,
                    targets,
                    armed[offset],
                    key,
                    index,
                    max_missing_rate,
                    plan["keep_shards"],
                )
                continue
            began = time.monotonic()
            try:
                if membership(expected) != members:
                    raise ValueError("deployment membership changed")
                rows = arm_segment(window, members, targets, key)
                current = {
                    target: (row["instance"], row.get("workers"), row.get("geometry"))
                    for target, row in rows.items()
                }
                if first is None:
                    first, identities = rows, current
                elif current != identities:
                    raise ValueError(
                        "capture instance, worker count or geometry changed"
                    )
            except Exception as error:
                raise RuntimeError(
                    f"segment {offset} could not arm: {error}"
                ) from error
            armed[offset] = rows
            status(root, "collecting", armed_segments=len(armed))
            timing["arm_seconds"] += time.monotonic() - began
        collect_completed(wait=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        index.close()

    if membership(expected) != members:
        raise ValueError("deployment membership changed during capture")
    workers = {
        ip: row["workers"] for (role, ip), row in first.items() if role == "frontend"
    }
    if any(type(count) is not int or count <= 0 for count in workers.values()):
        raise ValueError("invalid reported Frontend worker count")
    geometries = [row["geometry"] for (role, _), row in first.items() if role == "vit"]
    if any(geometry != geometries[0] for geometry in geometries):
        raise ValueError("ViT Pods have different geometry configurations")
    timing["arm_seconds"] = round(timing["arm_seconds"], 3)
    for name in (
        "frontend_fetch_seconds",
        "vit_fetch_seconds",
        "parallel_fetch_seconds",
        "index_seconds",
    ):
        timing[name] = round(
            sum(item["timing"][name] for item in completed.values()), 3
        )

    status(root, "converting", timing=timing)
    mark = time.monotonic()
    output = work / "replay.mckidx"
    validation, conversion = index.seal(output, workers, expected["vit"])
    bundle.write_json(work / "validation.json", validation)
    timing["conversion_seconds"] = round(time.monotonic() - mark, 3)
    timing["capture_seconds"] = (end - start) / 1000
    timing["drain_seconds"] = bundle.DRAIN_MS / 1000
    report = {
        "batch": root.name,
        "timing": timing,
        "validation": validation,
        "conversion": conversion,
        "vit_geometry": geometries[0] if geometries else None,
        "frontend_workers": workers,
        "max_missing_rate": max_missing_rate,
        "segments": [completed[w["start"]] for w in segments],
    }
    bundle.write_json(work / "report.json", report)
    partial, packed = work / "result.tar.part", work / "result.tar"
    with partial.open("wb") as stream:
        with tarfile.open(fileobj=stream, mode="w") as tar:
            for path in (output, work / "report.json"):
                tar.add(path, arcname=path.name, recursive=False)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(packed)
    directory = os.open(work, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    digest = bundle.checksum(packed)
    shutil.rmtree(index.path)
    status(
        root,
        "ready_with_gaps" if validation["missing_requests"] else "ready",
        timing=timing,
        verified=expected,
        missing_requests=validation["missing_requests"],
        missing_geometry=0,
        result=str(packed),
        sha256=digest,
        replay=str(output),
        report=str(work / "report.json"),
    )


def seconds(value):
    match = re.fullmatch(r"(\d+)([smh]?)", value)
    if not match:
        raise argparse.ArgumentTypeError("use seconds or a duration like 20s, 5m, 1h")
    return int(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[match[2]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)
    run = sub.add_parser("run")
    run.add_argument("--duration", type=seconds, required=True)
    run.add_argument("--segment", type=seconds, default=120)
    run.add_argument("--sort-mib", type=int, default=64)
    run.add_argument(
        "--keep-shards",
        action="store_true",
        help="retain accepted archives for offline comparison",
    )
    run.add_argument("--work-dir", type=Path, default=JOBS)
    run.add_argument(
        "--min-free-gib", type=int, help="override the estimated workspace requirement"
    )
    run.add_argument("--max-missing-rate", type=float, default=0.01)
    run.add_argument("--start-ms", type=int)
    run.add_argument("--frontend-pods", type=int, required=True)
    run.add_argument("--vit-pods", type=int, default=0)
    for op in ("status", "_worker"):
        sub.add_parser(op).add_argument("batch", type=int)
    args = parser.parse_args()
    if args.op == "run":
        now = millis()
        start = args.start_ms or now + 30_000
        if args.duration <= 0 or not now + 15_000 <= start <= now + 240_000:
            parser.error("duration must be positive and start 15..240s in the future")
        if not 120 <= args.segment <= MAX_SEGMENT_MS // 1000:
            parser.error("segment must be 120..300s; a shorter job uses one segment")
        if not 0 <= args.max_missing_rate <= 1:
            parser.error("max missing rate must be a fraction in [0, 1]")
        expected = {"frontend": args.frontend_pods, "vit": args.vit_pods}
        if expected["frontend"] <= 0 or expected["vit"] < 0 or args.sort_mib <= 0:
            parser.error("invalid Pod counts or sort buffer")
        JOBS.mkdir(mode=0o700, parents=True, exist_ok=True)
        work_base = args.work_dir.resolve()
        # Reserve space for persistent runs, a staged segment and final output.
        estimated_mib = 9 * args.duration + 16 * min(args.duration, args.segment) + 2048
        if args.keep_shards:
            estimated_mib += 16 * args.duration
        minimum = (
            args.min_free_gib
            if args.min_free_gib is not None
            else (estimated_mib + 1023) // 1024
        )
        if (
            minimum < 2
            or not work_base.is_dir()
            or shutil.disk_usage(work_base).free < minimum * GIB
        ):
            parser.error(f"workspace {work_base} requires at least {minimum} GiB free")
        with (JOBS / ".capture.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("a capture batch is already running")
            root = JOBS.resolve() / str(start)
            root.mkdir(mode=0o700)
            work = work_base / str(start)
            if work != root:
                work.mkdir(mode=0o700)
            bundle.write_json(
                root / "job.json",
                {
                    "start": start,
                    "end": start + args.duration * 1000,
                    "expected": expected,
                    "work": str(work),
                    "segment_ms": args.segment * 1000,
                    "sort_bytes": args.sort_mib * 1024**2,
                    "keep_shards": args.keep_shards,
                    "max_missing_rate": args.max_missing_rate,
                },
            )
            status(root, "starting")
            with (root / "job.log").open("wb") as log:
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "_worker",
                        str(start),
                    ],
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(lock.fileno(),),
                    env={**os.environ, "SQLITE_TMPDIR": str(work)},
                )
        print((root / "status.json").read_text())
        return
    root = JOBS / str(args.batch)
    if args.op == "status":
        print((root / "status.json").read_text())
        return
    try:
        execute(root, json.loads((root / "job.json").read_text()))
    except Exception as error:
        status(root, "failed", error=str(error)[:1000], log=str(root / "job.log"))
        raise


if __name__ == "__main__":
    main()
