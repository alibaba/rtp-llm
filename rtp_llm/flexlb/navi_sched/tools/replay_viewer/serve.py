#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import urlopen


VIEWER_ID = "navi_sched_replay_viewer"


def replay_metadata(replay):
    scheduler_info = replay.get("scheduler_info") or {}
    optimization_runs = scheduler_info.get("optimization_runs") or []
    optimization_run = optimization_runs[0] if optimization_runs else {}
    requests = replay.get("requests", [])
    nodes = replay.get("nodes", [])
    decisions = replay.get("decisions", [])
    node_matches = replay.get("node_matches", [])
    batch_sizes = {}
    expected_batch_sizes = {}
    for decision in decisions:
        batch_id = str(decision.get("batch_id", "-"))
        batch_sizes[batch_id] = batch_sizes.get(batch_id, 0) + 1
        expected_batch_sizes[batch_id] = int(
            decision.get("expected_batch_size", 0)
        )
    prompt_tokens = [
        int(item.get("request", {}).get("prompt_token_count", 0))
        for item in requests
    ]
    node_indexes = {}
    for index, node in enumerate(nodes):
        for address in (
            node.get("agent_id"),
            node.get("collector_address"),
            node.get("server_address"),
        ):
            if address:
                node_indexes[address] = index
    selected_cache_hit_blocks = 0
    selected_cache_hit_tokens = 0
    for request_index, decision in enumerate(decisions):
        target = decision.get("target") or {}
        node_index = node_indexes.get(
            target.get("agent_address") or target.get("address")
        )
        if node_index is None:
            continue
        try:
            matched_blocks = int(
                node_matches[request_index][node_index].get(
                    "matched_block_count", 0
                )
            )
        except (IndexError, TypeError, AttributeError):
            matched_blocks = 0
        selected_cache_hit_blocks += matched_blocks
        block_size = int(
            requests[request_index].get("request", {}).get("block_size", 0)
        )
        selected_cache_hit_tokens += min(
            matched_blocks * block_size, prompt_tokens[request_index]
        )
    queue_values = [
        float(node.get("queue_total_predicted_latency_ms", 0))
        for node in nodes
    ]
    revisions = [
        int(node["latency_model_revision"])
        for node in nodes
        if node.get("latency_model_revision") is not None
    ]
    actual_batch_sizes = list(batch_sizes.values())
    expected_sizes = list(expected_batch_sizes.values())
    return {
        "schedule_time_us": replay.get("schedule_time_us"),
        "biz_name": replay.get("biz_name"),
        "scheduler": scheduler_info.get("scheduler", "-"),
        "lambda": scheduler_info.get("lambda"),
        "alpha": scheduler_info.get("alpha"),
        "alpha_decay": scheduler_info.get("alpha_decay"),
        "min_alpha": scheduler_info.get("min_alpha"),
        "max_loop_count": scheduler_info.get("max_loop_count"),
        "actual_loop_count": optimization_run.get("actual_loop_count"),
        "optimization_stop_reason": optimization_run.get("stop_reason"),
        "optimization_time_budget_us": scheduler_info.get(
            "optimization_time_budget_us"
        ),
        "parallelism": scheduler_info.get("parallelism"),
        "request_count": len(requests),
        "agent_count": len(nodes),
        "batch_count": len(actual_batch_sizes),
        "mean_batch_size": (
            sum(actual_batch_sizes) / len(actual_batch_sizes)
            if actual_batch_sizes else 0
        ),
        "min_batch_size": min(actual_batch_sizes, default=0),
        "max_batch_size": max(actual_batch_sizes, default=0),
        "min_expected_batch_size": min(expected_sizes, default=0),
        "max_expected_batch_size": max(expected_sizes, default=0),
        "prompt_token_count": sum(prompt_tokens),
        "min_prompt_tokens": min(prompt_tokens, default=0),
        "max_prompt_tokens": max(prompt_tokens, default=0),
        "selected_cache_hit_blocks": selected_cache_hit_blocks,
        "selected_cache_hit_token_rate": (
            selected_cache_hit_tokens / sum(prompt_tokens)
            if sum(prompt_tokens) else 0
        ),
        "queue_mean_ms": (
            sum(queue_values) / len(queue_values) if queue_values else 0
        ),
        "queue_max_ms": max(queue_values, default=0),
        "inflight_request_count": sum(
            int(node.get("inflight_requests", 0)) for node in nodes
        ),
        "running_request_count": sum(
            int(node.get("running_requests", 0)) for node in nodes
        ),
        "min_model_revision": min(revisions, default=0),
        "max_model_revision": max(revisions, default=0),
    }


class ReplayViewerHandler(SimpleHTTPRequestHandler):
    def __init__(
        self, *args, static_dir, replay_dir, report_file, **kwargs
    ):
        self.replay_dir = replay_dir
        self.report_file = report_file
        super().__init__(*args, directory=str(static_dir), **kwargs)

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        if path == "/api/health":
            self.send_json({"viewer": VIEWER_ID})
            return
        if path == "/api/replays":
            self.send_replay_list()
            return
        if path == "/api/comparison":
            self.send_comparison()
            return
        if path.startswith("/replays/"):
            self.send_replay(
                path[len("/replays/"):], parse_qs(parsed_url.query)
            )
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, value, status=200):
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_replay_list(self):
        entries = []
        if self.replay_dir.is_dir():
            for path in self.replay_dir.iterdir():
                if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
                    continue
                stat = path.stat()
                entry = {
                    "name": path.name,
                    "size": stat.st_size,
                    "modified_ns": stat.st_mtime_ns,
                    "url": "/replays/" + quote(path.name),
                }
                if path.suffix == ".json":
                    try:
                        with path.open(encoding="utf-8") as replay_file:
                            replay = json.load(replay_file)
                        entries.append({**entry, **replay_metadata(replay)})
                    except (OSError, ValueError):
                        entries.append(entry)
                    continue
                with path.open("rb") as replay_file:
                    sample_index = 0
                    for line_number, line in enumerate(replay_file, 1):
                        if not line.strip():
                            continue
                        sample_index += 1
                        metadata = {}
                        try:
                            metadata = replay_metadata(json.loads(line))
                        except ValueError:
                            pass
                        entries.append({
                            **entry,
                            **metadata,
                            "name": f"{path.name} #{sample_index}",
                            "size": len(line),
                            "url": entry["url"] + f"?line={line_number}",
                        })
        entries.sort(key=lambda entry: entry["name"])
        self.send_json(entries)

    def send_replay(self, encoded_name, query):
        name = unquote(encoded_name)
        if name != Path(name).name:
            self.send_error(404)
            return
        path = self.replay_dir / name
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            self.send_error(404)
            return
        if path.suffix == ".json":
            payload = path.read_bytes()
        else:
            try:
                line_number = int(query.get("line", ["1"])[0])
            except ValueError:
                self.send_error(400, "invalid replay line")
                return
            if line_number <= 0:
                self.send_error(400, "invalid replay line")
                return
            with path.open("rb") as replay_file:
                payload = next(
                    (line for index, line in enumerate(replay_file, 1)
                     if index == line_number),
                    None,
                )
            if payload is None or not payload.strip():
                self.send_error(404)
                return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_comparison(self):
        if not self.report_file.is_file():
            self.send_json({"runs": []})
            return
        payload = self.report_file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, message, *args):
        sys.stderr.write("viewer: " + message % args + "\n")


def is_viewer_running(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.25) as response:
            value = json.loads(response.read().decode("utf-8"))
            return value.get("viewer") == VIEWER_ID
    except (OSError, URLError, ValueError):
        return False


def port_available(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def serve(static_dir, replay_dir, report_file, host, port):
    def handler(*args, **kwargs):
        return ReplayViewerHandler(
            *args,
            static_dir=static_dir,
            replay_dir=replay_dir,
            report_file=report_file,
            **kwargs,
        )

    server = ThreadingHTTPServer((host, port), handler)
    server.serve_forever()


def launch(args, static_dir, replay_dir, report_file):
    replay_dir.mkdir(parents=True, exist_ok=True)
    for port in range(args.port, args.port + 20):
        if is_viewer_running(port):
            print(f"http://127.0.0.1:{port}")
            return 0
        if not port_available(port):
            continue
        log_path = replay_dir.parent / "replay_viewer.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--serve",
            "--host",
            args.host,
            "--port",
            str(port),
            "--replay-dir",
            str(replay_dir),
            "--report-file",
            str(report_file),
        ]
        with log_path.open("ab") as log:
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
                close_fds=True,
            )
        for unused in range(40):
            if is_viewer_running(port):
                print(f"http://127.0.0.1:{port}")
                return 0
            time.sleep(0.05)
        print(f"replay viewer failed to start; see {log_path}", file=sys.stderr)
        return 1
    print("no available replay viewer port", file=sys.stderr)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18780)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--report-file", default="")
    args = parser.parse_args()

    static_dir = Path(__file__).resolve().parent
    replay_dir = Path(os.path.abspath(args.replay_dir))
    report_file = Path(
        os.path.abspath(
            args.report_file
            or replay_dir.parent / "comparison_report.json"
        )
    )
    if args.serve:
        serve(static_dir, replay_dir, report_file, args.host, args.port)
        return 0
    return launch(args, static_dir, replay_dir, report_file)


if __name__ == "__main__":
    raise SystemExit(main())
