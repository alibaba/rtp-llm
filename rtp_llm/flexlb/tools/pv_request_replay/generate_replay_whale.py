#!/usr/bin/env python3
"""Collect FlexLB PV logs from a Whale role and build a request replay report.

This is the whale-on-spectrum counterpart of ``generate_replay.py``, whose remote
collector drives ``dashctl``. Whale zones are carbon roles behind a Spectrum dep shell
that ``dashctl`` cannot enumerate, so pod resolution goes through ``whale`` and log
reads go through ``asicli``. Collection then joins the shared local-input path, which
means ``build`` and ``html`` stay exactly as documented for ``generate_replay.py``:

    python3 generate_replay_whale.py all --service dash_pd \\
        --deployment beijing_RTX_PRO_5000_72GB_p4tp_d2tp \\
        --start '2026-09-19 19:55:00' --end '2026-09-19 20:21:00'

    python3 generate_replay.py build --input outputs/<run> --start ... --end ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from collect_pv_log import collect_logs
from generate_replay import (
    json_default,
    parse_duration,
    parse_time,
    run_build,
    time_text,
    validate_window,
    write_json,
)
from whale_pv_source import (
    DEFAULT_CONTAINER,
    DEFAULT_LOG_DIR,
    DEFAULT_PAGE_LINES,
    DEFAULT_ROLE,
    WhalePod,
    WhaleSourceError,
    collect_whale_snapshots,
    default_runner,
    resolve_deployment_id,
    resolve_role_pods,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "outputs"
DEFAULT_SERVICE = os.environ.get("FLEXLB_REPLAY_WHALE_SERVICE", "")
DEFAULT_DEPLOYMENT = os.environ.get("FLEXLB_REPLAY_WHALE_DEPLOYMENT", "")
DEFAULT_TEMPLATE = ROOT / "replay_template.html"


def default_output_dir(label: str, start: datetime, end: datetime) -> Path:
    configured = os.environ.get("FLEXLB_REPLAY_OUTPUT_ROOT")
    base = Path(configured).expanduser() if configured else DEFAULT_OUTPUT_ROOT
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-")
    window = start.strftime("%Y%m%d-%H%M%S") + "_" + end.strftime("%H%M%S")
    return base / f"{safe_label}-{window}"


def output_label(args: argparse.Namespace) -> str:
    target = args.service or args.deployment_id or "whale"
    return f"{target}-{args.role}"


def resolve_pods(args: argparse.Namespace) -> list[WhalePod]:
    """Resolve the target role's pods, preferring an explicit deployment id.

    ``whale deployment describe`` is rate-limited locally, so this is called once per
    run and its result is passed along rather than re-resolved per stage.
    """
    if args.deployment_id:
        deployment_id = args.deployment_id
    else:
        if not args.service or not args.deployment:
            raise WhaleSourceError(
                "either --deployment-id, or both --service and --deployment, is required"
            )
        deployment_id = resolve_deployment_id(
            default_runner, args.service, args.deployment, args.context
        )
    pods = resolve_role_pods(
        default_runner, deployment_id, args.role, args.context
    )
    if not args.instance:
        return pods
    wanted = set(args.instance)
    selected = [pod for pod in pods if pod.pod_name in wanted]
    missing = wanted - {pod.pod_name for pod in selected}
    if missing:
        raise WhaleSourceError(
            f"role {args.role!r} has no such pods: {', '.join(sorted(missing))}"
        )
    return selected


def run_collect(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    pods = resolve_pods(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    collection_start = args.start - args.lead_grace
    collection_end = args.end + args.tail_grace
    local_inputs, whale_manifest = collect_whale_snapshots(
        pods=pods,
        output_dir=output_dir,
        start=collection_start,
        end=collection_end,
        container=args.container,
        log_dir=args.log_dir,
        log_name=args.log_name,
        page_lines=args.page_lines,
    )
    whale_manifest.update(
        {
            "requested_window": {"start": time_text(args.start), "end": time_text(args.end)},
            "collection_window": {
                "start": time_text(collection_start),
                "end": time_text(collection_end),
            },
            "role": args.role,
            "service": args.service or None,
            "deployment": args.deployment or None,
            "deployment_id": args.deployment_id or None,
        }
    )
    write_json(output_dir / "whale_manifest.json", whale_manifest)

    manifest = collect_logs(
        workspace="",
        deployment=None,
        instances=[pod.instance for pod in pods],
        start=args.start,
        end=args.end,
        output_dir=output_dir,
        log_name=args.log_name,
        lead_grace=args.lead_grace,
        tail_grace=args.tail_grace,
        strict=args.strict,
        workers=args.workers,
        local_inputs=local_inputs,
    )
    manifest_path = output_dir / "collect_manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    return manifest


def add_whale_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", type=parse_time, required=True)
    parser.add_argument("--end", type=parse_time, required=True)
    parser.add_argument(
        "--service",
        default=DEFAULT_SERVICE,
        help="Whale biz name or service id holding the deployment",
    )
    parser.add_argument(
        "--deployment",
        default=DEFAULT_DEPLOYMENT,
        help="deployment_name or logic_deployment_name; must match exactly one",
    )
    parser.add_argument(
        "--deployment-id",
        default="",
        help="skip service/deployment resolution and use this deployment id",
    )
    parser.add_argument("--role", default=DEFAULT_ROLE, help="carbon role, e.g. master")
    parser.add_argument("--context", default=None, help="whale config context override")
    parser.add_argument(
        "--instance",
        action="append",
        default=[],
        help="restrict to these pod names; repeat for multiple pods",
    )
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--log-name", default="pv.log")
    parser.add_argument(
        "--page-lines",
        type=int,
        default=DEFAULT_PAGE_LINES,
        help="lines per asicli exec read; lower it if a chunk hits the stdout cap",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lead-grace", type=parse_duration, default=timedelta(minutes=5))
    parser.add_argument("--tail-grace", type=parse_duration, default=timedelta(minutes=10))
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
    add_whale_arguments(collect_parser)

    all_parser = subparsers.add_parser("all", help="collect logs and build XLSX + HTML")
    add_whale_arguments(all_parser)
    all_parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_window(args.start, args.end)
        output_dir = args.output_dir or default_output_dir(
            output_label(args), args.start, args.end
        )
        collect_manifest = run_collect(args, output_dir)
        if args.command == "collect":
            print(
                json.dumps(
                    collect_manifest, ensure_ascii=False, indent=2, default=json_default
                )
            )
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
