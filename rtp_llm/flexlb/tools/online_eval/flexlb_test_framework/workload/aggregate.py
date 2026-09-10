"""Prepare the existing stress aggregator's inputs from workload evidence."""

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

from online_eval.requests import request_success
from stress.consolidate_run_outputs import parse_grouped_prometheus_timeseries

ROOT = Path(__file__).resolve().parents[2]


def client_row(record, anchor, engine_events=()):
    if "send_start_epoch_ms" in record:
        return dict(record)
    if "issued_s" not in record:
        raise ValueError("request has no issue timestamp")
    row = dict(
        rid=record["wire_request_id"],
        request_id=record["wire_request_id"],
        send_start_epoch_ms=(
            record["issued_s"] - anchor["monotonic_s"] + anchor["epoch_s"]
        )
        * 1000,
        status="ok" if request_success(record) else "request_error",
        error=(
            ""
            if request_success(record)
            else str(
                record.get("business_error_message")
                or "request did not complete successfully"
            )
        ),
        prefill=record.get("prefill_addr"),
    )
    terminal = record.get("transport_terminal_s")
    if terminal is not None:
        row["total_ms"] = (terminal - record["issued_s"]) * 1000
        row["wall_clock_ts"] = terminal - anchor["monotonic_s"] + anchor["epoch_s"]
    schedule = record["schedule"]
    if schedule.get("ended_s") is not None and schedule.get("started_s") is not None:
        row["schedule_ms"] = (schedule["ended_s"] - schedule["started_s"]) * 1000
    for name in ("input_len", "output_len"):
        if name in record:
            row[name] = record[name]
        elif name in record.get("request_shape", {}):
            row[name] = record["request_shape"][name]
    if record.get("send_due_s") is not None:
        row["pacing_lag_ms"] = max(0, record["issued_s"] - record["send_due_s"]) * 1000
    for role in ("prefill", "decode"):
        if not row.get(role):
            addresses = {
                event["engine_address"]
                for event in engine_events
                if event.get("event") == role + "_done"
                and event.get("engine_address")
                and not event.get("cancelled")
                and not event.get("error_code")
            }
            if len(addresses) == 1:
                row[role] = next(iter(addresses))
    return row


def run_canvas(out, timeout_s):
    with (out / "canvas.json").open("w") as output, (out / "aggregate.stderr").open(
        "w"
    ) as errors:
        run = subprocess.run(
            [sys.executable, str(ROOT / "stress/aggregate_canvas_run.py")],
            cwd=out,
            stdout=output,
            stderr=errors,
            timeout=timeout_s,
        )
    if run.returncode:
        raise RuntimeError(
            f"stress aggregate exited {run.returncode}: "
            + (out / "aggregate.stderr").read_text()[-2000:]
        )
    render_metadata = []
    if (out / "run_meta.json").is_file():
        params = json.loads((out / "run_meta.json").read_text())["params"]
        for field, option in (
            ("n_prefill", "--p-engines"),
            ("n_decode", "--d-engines"),
            ("load_client_workers", "--shards"),
        ):
            if field in params:
                render_metadata.extend([option, str(params[field])])
    with (out / "render.log").open("w") as log:
        render = subprocess.run(
            [
                sys.executable,
                str(ROOT / "stress/canvas_report_gen.py"),
                "--aggregate",
                str(out / "canvas.json"),
                "--out",
                str(out / "report.html"),
                *render_metadata,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    if render.returncode:
        raise RuntimeError(
            "stress report rendering failed: "
            + (out / "render.log").read_text()[-2000:]
        )
    return json.loads((out / "canvas.json").read_text())


def aggregate_workload(
    directory,
    joined,
    environments,
    anchor,
    timeout_s,
    master_log_directories=None,
    environment_metadata=None,
):
    directory = Path(directory)
    results = []
    for epoch, env_directory in environments.items():
        out = directory / "aggregate" / epoch
        out.mkdir(parents=True, exist_ok=True)
        try:
            if epoch in (environment_metadata or {}):
                (out / "run_meta.json").write_text(
                    json.dumps(dict(params=environment_metadata[epoch]))
                )
            environment_requests = [
                r for r in joined["requests"] if r["env_epoch"] == epoch
            ]
            preconditioning = [
                r
                for r in environment_requests
                if r["original"].get("purpose") == "preconditioning"
            ]
            requests = [
                r
                for r in environment_requests
                if r["original"].get("purpose") != "preconditioning"
            ]
            (out / "preconditioning-requests.json").write_text(
                json.dumps(preconditioning)
            )
            (out / "request-scope.json").write_text(
                json.dumps(
                    dict(
                        measured_requests=len(requests),
                        preconditioning_requests=len(preconditioning),
                        scope="Measured requests only; direct cache seeds retain full lineage in preconditioning-requests.json",
                    )
                )
            )
            if not requests:
                raise ValueError("no request evidence for environment")
            rows = [
                client_row(r["original"], anchor, r.get("engine_events", []))
                for r in requests
            ]
            (out / "client_events.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            shutil.copyfile(
                Path(env_directory) / "engine_events.jsonl", out / "engine_events.jsonl"
            )
            env_path = Path(env_directory)
            source_logs = {}
            for name, log_directory in (
                (master_log_directories or {}).get(epoch, {}).items()
            ):
                source_logs[name] = sorted(Path(log_directory).glob("application.log*"))
                if not source_logs[name]:
                    raise ValueError(f"no owned application log for master {name}")
            logs = [path for paths in source_logs.values() for path in paths]
            if not source_logs:
                logs = sorted(env_path.glob("**/application.log*"))
                if not logs:
                    logs = sorted(env_path.glob("flexlb_master*.log"))
            if not logs:
                raise ValueError("no owned master log for aggregation")

            def copy_logs(paths, destination):
                with destination.open("w") as sink:
                    for path in paths:
                        sink.write(path.read_text(errors="replace"))
                        sink.write("\n")

            copy_logs(logs, out / "master.log")
            (out / "master-log-sources.json").write_text(
                json.dumps(
                    {
                        name: [str(path) for path in paths]
                        for name, paths in source_logs.items()
                    }
                )
            )
            telemetry = directory / "telemetry" / epoch
            masters = {}
            for path in telemetry.glob("master-*.prom"):
                masters[path.stem] = parse_grouped_prometheus_timeseries(path)
            # Keep masters separate: summing independently timestamped samples
            # would create a false arrival timeline during failover.
            (out / "master-sources.json").write_text(json.dumps(masters))
            if len(masters) == 1:
                (out / "master.json").write_text(
                    json.dumps({"prometheus_timeseries": next(iter(masters.values()))})
                )
            mock = parse_grouped_prometheus_timeseries(telemetry / "mock.prom")
            with gzip.open(out / "mock_per_engine_timeseries.json.gz", "wt") as stream:
                json.dump(mock, stream)
            final_snapshots = {}
            for snapshot in (directory / "telemetry" / epoch).glob(
                "server-latency-*.json"
            ):
                final_snapshots[snapshot.stem.removeprefix("server-latency-")] = (
                    json.loads(snapshot.read_text())
                )
            if len(final_snapshots) == 1 and len(masters) == 1:
                (out / "load_client").mkdir(exist_ok=True)
                (out / "load_client/server_latency.json").write_text(
                    json.dumps(next(iter(final_snapshots.values())))
                )
            aggregate = run_canvas(out, timeout_s)
            planes = {}
            if len(masters) > 1:
                for name, groups in masters.items():
                    plane = out / "masters" / name
                    plane.mkdir(parents=True, exist_ok=True)
                    for file in (
                        "client_events.jsonl",
                        "engine_events.jsonl",
                        "master.log",
                        "mock_per_engine_timeseries.json.gz",
                    ):
                        shutil.copyfile(out / file, plane / file)
                    if (out / "run_meta.json").is_file():
                        shutil.copyfile(out / "run_meta.json", plane / "run_meta.json")
                    master_name = name.removeprefix("master-")
                    if master_name in source_logs:
                        copy_logs(source_logs[master_name], plane / "master.log")
                    (plane / "master.json").write_text(
                        json.dumps({"prometheus_timeseries": groups})
                    )
                    if master_name in final_snapshots:
                        (plane / "load_client").mkdir(exist_ok=True)
                        (plane / "load_client/server_latency.json").write_text(
                            json.dumps(final_snapshots[master_name])
                        )
                    run_canvas(plane, timeout_s)
                    planes[name] = dict(
                        path=str(plane / "canvas.json"),
                        report=str(plane / "report.html"),
                        client_scope="whole environment",
                        master_metrics_scope=name,
                    )
            results.append(
                dict(
                    env_epoch=epoch,
                    status="GENERATED",
                    path=str(out / "canvas.json"),
                    report=str(out / "report.html"),
                    test_valid=aggregate.get("summary", {}).get("test_valid"),
                    master_sources=list(masters),
                    master_aggregates=planes,
                    master_aggregation="separate sources; no implicit cross-source sum",
                )
            )
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            results.append(dict(env_epoch=epoch, status="ERROR", error=str(exc)))
    return results
