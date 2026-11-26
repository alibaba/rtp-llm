"""Build or compare replay indexes from sealed capture archives and import verified results."""

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import bundle
from mckidx_writer import HEADER
from partitioned_index import PartitionedIndex, durable_json
from prefix_index import PrefixIndex
from verify import require


class DiskPeak:
    """Sample allocated working files, including unlinked SQLite and encoder files on Linux."""

    def __init__(self, directory):
        self.directory = directory.resolve()
        self.peak = 0
        self.stop = threading.Event()

    def sample(self):
        files = {}
        paths = list(self.directory.rglob("*"))
        descriptors = Path("/proc/self/fd")
        if descriptors.is_dir():
            for descriptor in descriptors.iterdir():
                try:
                    if os.readlink(descriptor).startswith(str(self.directory) + "/"):
                        paths.append(descriptor)
                except FileNotFoundError:
                    continue
        for path in paths:
            try:
                stat = path.stat()
                files[stat.st_dev, stat.st_ino] = stat.st_blocks * 512
            except FileNotFoundError:
                continue
        self.peak = max(self.peak, sum(files.values()))

    def _poll(self):
        while not self.stop.wait(0.05):
            self.sample()

    def __enter__(self):
        self.thread = threading.Thread(target=self._poll)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.sample()


def offline_build(plan, output, sqlite=False):
    output.mkdir(mode=0o700)
    index = (
        PrefixIndex(output / "prefix.sqlite")
        if sqlite
        else PartitionedIndex(output / "index", plan["sort_mib"] * 1024**2)
    )
    timings = []
    target = output / "replay.mckidx"
    workers, vit_pods = plan["frontend_workers"], plan["vit_pods"]
    # The SQLite sorter may create anonymous disk files for arrival order.
    os.environ["SQLITE_TMPDIR"] = str(output.resolve())
    with DiskPeak(output) as disk:
        try:
            for segment in plan["segments"]:
                shards = Path(segment["shards"])
                mark = time.monotonic()
                row = index.ingest(
                    shards / "frontend",
                    shards / "vit",
                    segment["start_ms"],
                    segment["end_ms"],
                    workers,
                    vit_pods,
                    plan["max_missing_rate"],
                )
                timings.append(
                    {
                        "start_ms": row["start_ms"],
                        "requests": row["requests"],
                        "ingest_seconds": time.monotonic() - mark,
                    }
                )
            mark = time.monotonic()
            validation, conversion = index.seal(target, workers, vit_pods)
            finishing_seconds = time.monotonic() - mark
        finally:
            if not sqlite:
                index.close()
    result = {
        "segments": timings,
        "finishing_seconds": finishing_seconds,
        "work_peak_allocated_bytes": disk.peak,
        "validation": validation,
        "conversion": conversion,
    }
    durable_json(output / "report.json", result)
    return result


def read_plan(path):
    plan = json.loads(path.read_text())
    for segment in plan["segments"]:
        segment["shards"] = str((path.parent / segment["shards"]).resolve())
    require(
        plan["segments"]
        and plan["frontend_workers"]
        and plan["vit_pods"] >= 0
        and plan["sort_mib"] > 0
        and 0 <= plan["max_missing_rate"] <= 1,
        "invalid capture build plan",
    )
    return plan


def recover(index_path, output, expected_frontend, expected_vit):
    output.mkdir(mode=0o700)
    index = PartitionedIndex(index_path)
    try:
        validation, conversion = index.seal(
            output / "replay.mckidx", range(expected_frontend), expected_vit
        )
        report = {"partial": True, "validation": validation, "conversion": conversion}
        durable_json(output / "report.json", report)
        return report
    finally:
        index.close()


def import_result(archive, checksum, shared_root, name):
    require(bundle.checksum(archive) == checksum, "result archive SHA-256 mismatch")
    require(
        Path(name).name == name
        and name not in ("", ".", "..")
        and name.endswith(".mckidx"),
        "dataset name must be a .mckidx file name",
    )
    destination = shared_root / "data" / "traffic"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    report_target = target.with_suffix(".report.json")
    with tarfile.open(archive, "r:") as tar:
        members = tar.getmembers()
        require(
            sorted(row.name for row in members) == ["replay.mckidx", "report.json"]
            and all(row.isfile() for row in members),
            "invalid result members",
        )
        report = json.load(tar.extractfile("report.json"))
        with tempfile.TemporaryDirectory(dir=destination) as work:
            staged = Path(work) / "replay.mckidx"
            staged_report = Path(work) / "report.json"
            with tar.extractfile("replay.mckidx") as source, staged.open(
                "wb"
            ) as stream:
                shutil.copyfileobj(source, stream)
                stream.flush()
                os.fsync(stream.fileno())
            with staged.open("rb") as stream:
                magic, count, *_ = HEADER.unpack(stream.read(HEADER.size))
            conversion = report["conversion"]
            require(
                magic == b"MCKINDEX"
                and count == conversion["dataset_requests"]
                and bundle.checksum(staged) == conversion["sha256"],
                "replay checksum or request count mismatch",
            )
            durable_json(staged_report, report)
            os.link(staged, target)
            try:
                os.link(staged_report, report_target)
            except OSError:
                target.unlink()
                raise
    return {
        "dataset": str(target),
        "report": str(report_target),
        "sha256": conversion["sha256"],
        "requests": count,
        "bytes": target.stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("build", "compare"):
        command = commands.add_parser(name)
        command.add_argument("plan", type=Path)
        command.add_argument("output", type=Path)
    command = commands.add_parser("recover")
    command.add_argument("index", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("--frontend-pods", type=int, required=True)
    command.add_argument("--vit-pods", type=int, default=0)
    command = commands.add_parser("import")
    command.add_argument("archive", type=Path)
    command.add_argument("--sha256", required=True)
    command.add_argument("--shared-root", type=Path, required=True)
    command.add_argument("--name", required=True)
    args = parser.parse_args()
    if args.command == "import":
        result = import_result(args.archive, args.sha256, args.shared_root, args.name)
    elif args.command == "recover":
        result = recover(args.index, args.output, args.frontend_pods, args.vit_pods)
    else:
        plan = read_plan(args.plan)
        if args.command == "build":
            result = offline_build(plan, args.output)
        else:
            args.output.mkdir(mode=0o700)
            baseline = offline_build(plan, args.output / "sqlite", sqlite=True)
            partitioned = offline_build(plan, args.output / "partitioned")
            reference, actual = baseline["conversion"], partitioned["conversion"]
            fields = (
                "dataset_requests",
                "repeat_tokens_base",
                "repeat_tokens_routing",
                "image_references",
                "replay_media_ids",
            )
            equal = all(reference[key] == actual[key] for key in fields)
            result = {
                "equivalent_counts": equal,
                "routing_reuse_loss_pp": 100
                * (reference["theoretical_routing"] - actual["theoretical_routing"]),
                "sqlite": baseline,
                "partitioned": partitioned,
            }
            durable_json(args.output / "comparison.json", result)
            require(equal, "capture conversion changed replay counts or reuse")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
