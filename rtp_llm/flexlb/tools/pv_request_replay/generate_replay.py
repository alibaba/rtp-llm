#!/usr/bin/env python3
"""Collect FlexLB PV logs and build a request replay report."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from build_html import build_html
from build_workbook import build_workbook
from collect_pv_log import collect_logs


LOCAL_TZ = ZoneInfo("Asia/Shanghai")
ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = os.environ.get("FLEXLB_REPLAY_WORKSPACE", "ai-lab-test")
DEFAULT_DEPLOYMENT = os.environ.get(
    "FLEXLB_REPLAY_DEPLOYMENT",
    "flexlb-hongyi-test-v1-flexlb",
)
DEFAULT_TEMPLATE = ROOT / "replay_template.html"


def parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid time {value!r}; use YYYY-MM-DD HH:MM:SS[.sss][+08:00]"
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(LOCAL_TZ)


def parse_duration(value: str) -> timedelta:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", value.lower())
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid duration {value!r}; examples: 30s, 10m, 1h"
        )
    amount = float(match.group(1))
    unit = match.group(2) or "s"
    seconds = amount * {"s": 1, "m": 60, "h": 3600}[unit]
    return timedelta(seconds=seconds)


def time_text(value: datetime) -> str:
    return value.astimezone(LOCAL_TZ).isoformat(timespec="milliseconds")


def default_output_dir(deployment: str, start: datetime, end: datetime) -> Path:
    configured = os.environ.get("FLEXLB_REPLAY_OUTPUT_ROOT")
    if configured:
        base = Path(configured).expanduser()
    else:
        workspace_tmp = Path.home() / "workspace" / "tmp"
        base = workspace_tmp if workspace_tmp.is_dir() else Path(tempfile.gettempdir())
        base = base / "flexlb-pv-replay"
    safe_deployment = re.sub(r"[^A-Za-z0-9_.-]+", "-", deployment).strip("-")
    window = (
        start.astimezone(LOCAL_TZ).strftime("%Y%m%d-%H%M%S")
        + "_"
        + end.astimezone(LOCAL_TZ).strftime("%H%M%S")
    )
    return base / f"{safe_deployment}-{window}"


def validate_window(start: datetime, end: datetime) -> None:
    if end <= start:
        raise ValueError(f"end must be later than start: {time_text(start)} >= {time_text(end)}")


def json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return time_text(value)
    if isinstance(value, timedelta):
        return str(value.total_seconds())
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return loaded


def collect_manifest_path(input_path: Path) -> Path | None:
    if input_path.is_file() and input_path.name == "collect_manifest.json":
        return input_path
    candidate = input_path / "collect_manifest.json"
    return candidate if candidate.is_file() else None


def snapshot_sources(input_path: Path) -> tuple[list[Path], dict[str, Any] | None]:
    manifest_path = collect_manifest_path(input_path)
    if manifest_path is None:
        if not input_path.is_file():
            raise FileNotFoundError(
                f"{input_path} is neither a PV log file nor a collection directory"
            )
        return [input_path], None

    manifest = load_json(manifest_path)
    sources: list[Path] = []
    for entry in manifest.get("snapshots", []):
        if not isinstance(entry, dict) or not entry.get("path"):
            continue
        path = Path(entry["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        sources.append(path)
    if not sources:
        raise ValueError(f"no snapshots recorded in {manifest_path}")
    missing = [str(source) for source in sources if not source.is_file()]
    if missing:
        raise FileNotFoundError("snapshot files are missing: " + ", ".join(missing))
    return sources, manifest


def manifest_window(manifest: dict[str, Any] | None) -> tuple[datetime | None, datetime | None]:
    if not manifest:
        return None, None
    requested = manifest.get("requested_window") or manifest.get("requestedWindow") or {}
    if not isinstance(requested, dict):
        return None, None
    start = parse_time(str(requested["start"])) if requested.get("start") else None
    end = parse_time(str(requested["end"])) if requested.get("end") else None
    return start, end


def run_collect(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = collect_logs(
        workspace=args.workspace,
        deployment=args.deployment,
        instances=args.instance,
        start=args.start,
        end=args.end,
        output_dir=output_dir,
        log_dir=args.log_dir,
        log_name=args.log_name,
        lead_grace=args.lead_grace,
        tail_grace=args.tail_grace,
        strict=args.strict,
        page_lines=args.page_lines,
        workers=args.workers,
    )
    manifest_path = output_dir / "collect_manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    return manifest


def run_build(
    input_path: Path,
    output_dir: Path,
    start: datetime | None,
    end: datetime | None,
    template: Path,
    strict: bool = False,
) -> dict[str, Any]:
    sources, collect_manifest = snapshot_sources(input_path)
    manifest_start, manifest_end = manifest_window(collect_manifest)
    start = start or manifest_start
    end = end or manifest_end
    if start is None or end is None:
        raise ValueError("--start and --end are required when the input has no collection manifest")
    validate_window(start, end)

    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / "analysis.xlsx"
    html_path = output_dir / "replay.html"
    workbook_summary = build_workbook(
        sources=input_path,
        destination=workbook_path,
        start=start,
        end=end,
    )
    request_count = int(workbook_summary.get("request_count") or 0)
    complete_count = int(workbook_summary.get("complete_request_count") or 0)
    joins_complete = request_count == complete_count
    collection_status = (
        str(collect_manifest.get("status")) if collect_manifest else "not_applicable"
    )
    collection_complete = collect_manifest is None or collection_status == "complete"
    report_status = "complete" if collection_complete and joins_complete else "partial"

    collection_summary = None
    if collect_manifest:
        collection_summary = {
            "status": collection_status,
            "snapshot_truncated": bool(collect_manifest.get("snapshot_truncated")),
            "errors": collect_manifest.get("errors") or [],
            "resolved_instances": (
                (collect_manifest.get("source") or {}).get("resolved_instances") or []
            ),
            "instance_coverage": [
                {
                    "instance": item.get("instance"),
                    "status": item.get("status"),
                    "coverage": item.get("coverage"),
                }
                for item in collect_manifest.get("instances", [])
                if isinstance(item, dict)
            ],
        }

    manifest = {
        "schema_version": 1,
        "status": report_status,
        "requested_window": {"start": time_text(start), "end": time_text(end)},
        "input": {
            "collection_manifest": str(collect_manifest_path(input_path) or ""),
            "snapshots": [str(source) for source in sources],
        },
        "collection": collection_summary,
        "join": {
            "complete": joins_complete,
            "request_count": request_count,
            "complete_request_count": complete_count,
            "incomplete_request_count": request_count - complete_count,
        },
        "workbook": workbook_summary,
        "html": None,
        "outputs": {
            "analysis_xlsx": str(workbook_path),
            "replay_html": None,
        },
    }
    if strict and report_status != "complete":
        manifest["status"] = "failed"
        write_json(output_dir / "manifest.json", manifest)
        raise RuntimeError(
            "strict report completeness check failed: "
            f"collection={collection_status}, complete joins={complete_count}/{request_count}"
        )

    html_summary = build_html(
        input_path=workbook_path,
        template_path=template,
        output_path=html_path,
    )
    manifest["html"] = html_summary
    manifest["outputs"]["replay_html"] = str(html_path)
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def add_window_arguments(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument("--start", type=parse_time, required=required)
    parser.add_argument("--end", type=parse_time, required=required)


def add_collect_arguments(parser: argparse.ArgumentParser) -> None:
    add_window_arguments(parser)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--deployment", default=DEFAULT_DEPLOYMENT)
    parser.add_argument(
        "--instance",
        action="append",
        default=[],
        help="explicit FlexLB instance; repeat for multiple instances",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--log-dir", default="/home/admin/logs")
    parser.add_argument("--log-name", default="pv.log")
    parser.add_argument("--lead-grace", type=parse_duration, default=timedelta(minutes=5))
    parser.add_argument("--tail-grace", type=parse_duration, default=timedelta(minutes=10))
    parser.add_argument("--page-lines", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on incomplete log coverage or incomplete request joins",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="collect PV logs only")
    add_collect_arguments(collect_parser)

    all_parser = subparsers.add_parser("all", help="collect logs and build XLSX + HTML")
    add_collect_arguments(all_parser)
    all_parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)

    build_parser_ = subparsers.add_parser("build", help="build XLSX + HTML from collected logs")
    build_parser_.add_argument("--input", type=Path, required=True)
    add_window_arguments(build_parser_, required=False)
    build_parser_.add_argument("--output-dir", type=Path)
    build_parser_.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    build_parser_.add_argument(
        "--strict",
        action="store_true",
        help="fail when any route lacks complete cache/WorkerStatus first-token telemetry",
    )

    html_parser = subparsers.add_parser("html", help="build only HTML from an XLSX workbook")
    html_parser.add_argument("--input-xlsx", type=Path, required=True)
    html_parser.add_argument("--output-html", type=Path, required=True)
    html_parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "html":
            summary = build_html(
                input_path=args.input_xlsx,
                template_path=args.template,
                output_path=args.output_html,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
            return 0

        if args.command == "build":
            sources, collect_manifest = snapshot_sources(args.input)
            del sources
            manifest_start, manifest_end = manifest_window(collect_manifest)
            start = args.start or manifest_start
            end = args.end or manifest_end
            if start is None or end is None:
                raise ValueError("--start and --end are required for a raw PV log input")
            output_dir = args.output_dir or (
                args.input if args.input.is_dir() else args.input.parent / "replay-output"
            )
            manifest = run_build(
                args.input,
                output_dir,
                start,
                end,
                args.template,
                strict=args.strict,
            )
            print(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default))
            if manifest["status"] != "complete":
                print(
                    "WARNING: report was built from incomplete collection or joins; "
                    "see manifest.json",
                    file=sys.stderr,
                )
                return 1
            return 0

        validate_window(args.start, args.end)
        output_dir = args.output_dir or default_output_dir(
            args.deployment,
            args.start,
            args.end,
        )
        collect_manifest = run_collect(args, output_dir)
        if args.command == "collect":
            print(json.dumps(collect_manifest, ensure_ascii=False, indent=2, default=json_default))
            print(f"Collected PV logs under {output_dir}")
            return 0 if collect_manifest.get("status") == "complete" else 1

        manifest = run_build(
            output_dir,
            output_dir,
            args.start,
            args.end,
            args.template,
            strict=args.strict,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2, default=json_default))
        print(f"Built {output_dir / 'analysis.xlsx'}")
        print(f"Built {output_dir / 'replay.html'}")
        if manifest["status"] != "complete":
            print(
                "WARNING: report was built from incomplete collection or joins; "
                "see manifest.json",
                file=sys.stderr,
            )
            return 1
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    sys.exit(main())
