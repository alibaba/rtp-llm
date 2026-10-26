"""Bounded original-video DP4 load sweep; retain the service until STOP exists."""

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import statistics
import threading
import time
from pathlib import Path

import psutil
import requests

SMOKE_ARGS = (
    "--use_local 1 --role_type PDFUSION --vit_separation 0 "
    "--world_size 4 --tp_size 1 --dp_size 4 --ep_size 4 --act_type BF16 "
    "--load_method auto --max_seq_len 131072 --enable_cuda_graph 0 "
    "--reuse_cache 1 --think_mode 0 --seq_size_per_block 2048 "
    "--kernel_seq_size_per_block 64 --kv_cache_mem_mb 16384 "
    "--reserver_runtime_mem_mb 24576 --concurrency_limit 32"
)
VIDEO_SHA = (
    "d8b10b3fd5669c6d4db8cab1d0b2cd970d4e9db09a2ce09059b3a3e3a76987b0",
    "1208f65839ed292872a7abbeb951177714a77a5491c70964a2de4da650db966b",
    "4c2dbe94c50b704e906b538375ce6504bc74e84f845f45179ea254fd20d1e67f",
    "1ab8106f662755ad8ee8753652a86bc3b1f27be20a61a9ccc7b4f2b6ae3caa56",
)
PREPROCESS = dict(
    fps=6,
    min_frames=-1,
    max_frames=180,
    min_pixels=2500000,
    max_pixels=73728000,
    resized_width=1280,
    resized_height=704,
)
PROMPT_TOKENS = (24601, 8979, 30104, 40644)
VIDEO_TOKENS = (20424, 7992, 27530, 36419)


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def distribution(values):
    if not values:
        return None
    return dict(
        mean=statistics.mean(values),
        p50=percentile(values, 0.5),
        p90=percentile(values, 0.9),
        p99=percentile(values, 0.99),
        min=min(values),
        max=max(values),
    )


class AttachedServer:
    """A non-owning handle: this class never starts or stops a model process."""

    def __init__(self, live_path):
        live_path = Path(live_path)
        self.live = json.loads(live_path.read_text())
        self.server_pid = self.live["server_pid"]
        self.port = self.live["port"]
        previous = live_path.parent
        assert not list(
            (previous / "responses").iterdir()
        ), "prior requests already sent"
        assert not (
            previous / "requests.json"
        ).exists(), "prior requests already recorded"
        progress = json.loads((previous / "progress.json").read_text())
        assert progress["status"] == "failed_service_retained"
        assert "host monitor detected external GPU load" in progress["error"]
        self.verify_identity()

    def verify_identity(self):
        process = psutil.Process(self.server_pid)
        assert process.uids().real == os.getuid()
        assert (
            process.create_time() == self.live["server_create_time"]
        ), "server PID reused"
        assert "rtp_llm.start_server" in process.cmdline()
        ranks = {
            p.pid
            for p in process.children(recursive=True)
            if p.name().startswith("rtp_llm_rank")
        }
        assert ranks == set(self.live["rank_pids"]) and len(ranks) == 4
        return process

    def start_server(self, **unused):
        self.verify_identity()
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(f"http://127.0.0.1:{self.port}/health", timeout=5)
            response.raise_for_status()
        return True


class Runner:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "responses").mkdir()
        self.thread_local = threading.local()
        self.rows = []
        self.rounds = []
        self.warm_reference = {}
        self.queries = json.loads((Path(args.reference) / "queries.json").read_text())
        assert len(self.queries) == 9
        self.queries = self.queries[:5]
        for case, name in enumerate(
            ("video.mp4", "video_02.mp4", "video_03.mp4", "video_04.mp4"), 1
        ):
            path = Path(args.video_dir) / name
            digest = hashlib.sha256()
            with path.open("rb") as src:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    digest.update(chunk)
            assert digest.hexdigest() == VIDEO_SHA[case - 1], (name, digest.hexdigest())
            query = self.queries[case]
            assert query["max_tokens"] == 8192 and query["stream"] is False
            assert query["temperature"] == 0
            media = [
                c
                for m in query["messages"]
                if isinstance(m["content"], list)
                for c in m["content"]
                if c["type"] == "video_url"
            ]
            assert len(media) == 1 and media[0]["preprocess_config"] == PREPROCESS
            assert media[0]["video_url"]["url"] == str(path)
        save(self.output / "queries.json", self.queries)
        save(
            self.output / "contract.json",
            dict(
                args=vars(args),
                smoke_args=SMOKE_ARGS,
                video_sha256=VIDEO_SHA,
                topology=dict(tp=1, dp=4, ep=4, vit="local", pd="PDFUSION", late=True),
                gpu_ids=os.environ.get("CUDA_VISIBLE_DEVICES"),
                metrics=dict(
                    rt="client POST until complete nonstream JSON response",
                    ttft="server aux_info milliseconds; not client streaming TTFT",
                ),
                limitation="Finite closed-loop batches, not sustained/open-loop capacity; small sample tail percentiles",
            ),
        )
        os.environ["TEST_UNDECLARED_OUTPUTS_DIR"] = str(self.output)
        if args.attach_live_info:
            self.server = AttachedServer(args.attach_live_info)
        else:
            from rtp_llm.test.utils.maga_server_manager import MagaServerManager

            self.server = MagaServerManager(
                env_args={
                    "LOAD_PYTHON_MODEL": "1",
                    "RTP_LLM_LOCAL_MM_LATE_PREFILL": "1",
                    "FT_SERVER_TEST": "1",
                    "LOG_LEVEL": "INFO",
                },
                role_name="late_dp4",
                process_file_name="late_dp4.process.log",
                smoke_args_str=SMOKE_ARGS,
            )

    def progress(self, status, **kw):
        value = dict(
            status=status,
            time=time.time(),
            driver_pid=os.getpid(),
            server_pid=self.server.server_pid,
            port=self.server.port,
            **kw,
        )
        save(self.output / "progress.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def session(self):
        if not hasattr(self.thread_local, "session"):
            self.thread_local.session = requests.Session()
            self.thread_local.session.trust_env = False
        return self.thread_local.session

    def request(self, phase, index, case):
        started = time.monotonic()
        row = dict(
            phase=phase, index=index, case=case, start_epoch=time.time(), ok=False
        )
        body = None
        try:
            response = self.session().post(
                f"http://127.0.0.1:{self.server.port}/v1/chat/completions",
                json=copy.deepcopy(self.queries[case]),
                timeout=self.args.request_timeout,
            )
            row["seconds"] = time.monotonic() - started
            row["http_status"] = response.status_code
            response_path = self.output / "responses" / f"{phase}.{index}.json"
            response_path.write_text(response.text)
            row["response"] = str(response_path)
            response.raise_for_status()
            body = response.json()
            assert "error" not in body and body.get("choices"), body
            choice = body["choices"][0]
            assert choice["finish_reason"] == "stop", choice["finish_reason"]
            assert choice["message"]["content"].strip()
            if case == 0:
                assert choice["message"]["content"].strip() == "测试正常"
            else:
                assert body["usage"]["prompt_tokens"] == PROMPT_TOKENS[case - 1]
                assert (
                    body["usage"]["prompt_tokens_details"]["video_tokens"]
                    == VIDEO_TOKENS[case - 1]
                )
            aux = body.get("aux_info", {})
            row.update(
                usage=body["usage"],
                server_ttft_ms=aux.get("first_token_cost_time"),
                server_total_ms=aux.get("cost_time"),
                wait_ms=aux.get("wait_time"),
                role_addrs=aux.get("role_addrs", []),
                choices_sha256=hashlib.sha256(
                    json.dumps(
                        body["choices"], sort_keys=True, ensure_ascii=False
                    ).encode()
                ).hexdigest(),
            )
            if case and phase == "cold":
                before = json.loads(
                    (Path(self.args.reference) / f"late.{case}.json").read_text()
                )
                row["dp2_reference_exact_choices"] = (
                    body["choices"] == before["choices"]
                )
                # Different EP topology is not a bitwise regression comparison.
                # Preserve the diagnostic, then establish the actual DP4 reference.
            if phase == "warm":
                self.warm_reference[case] = body["choices"]
            if phase.startswith("c") and phase != "cold":
                row["exact_dp4_warm_choices"] = (
                    body["choices"] == self.warm_reference[case]
                )
                assert row[
                    "exact_dp4_warm_choices"
                ], "output differs from same-DP4 warm serial reference"
            row["ok"] = True
        except Exception as exc:
            row["error"] = repr(exc)
        row.setdefault("seconds", time.monotonic() - started)
        row["end_epoch"] = time.time()
        return row

    def check_environment(self):
        if (self.output / "STOP").exists():
            raise RuntimeError("user stop file detected")
        if (self.output / "external_load.json").exists():
            raise RuntimeError(
                "host monitor detected external GPU load; performance samples invalid"
            )
        if (self.output / "host_monitor_error.json").exists():
            raise RuntimeError(
                "host GPU monitoring failed; see host_monitor_error.json"
            )
        monitor = json.loads((self.output / "host_monitor_status.json").read_text())
        if time.time() - monitor["epoch"] > 60:
            raise RuntimeError("host GPU monitoring stale; performance samples invalid")
        if self.args.attach_live_info:
            self.server.verify_identity()
        response = self.session().get(
            f"http://127.0.0.1:{self.server.port}/health", timeout=5
        )
        response.raise_for_status()

    def record(self, rows):
        self.rows.extend(rows)
        save(self.output / "requests.json", self.rows)
        failed = [row for row in rows if not row["ok"]]
        if failed:
            raise RuntimeError(f"request correctness/transport failure: {failed}")

    def run(self):
        self.progress("starting")
        ready = self.server.start_server(
            model_path=self.args.checkpoint, model_type="qwen35_moe", timeout=1600
        )
        if not ready:
            raise RuntimeError("DP4 server startup failed")
        process = psutil.Process(self.server.server_pid)
        children = process.children(recursive=True)
        ranks = [p.pid for p in children if p.name().startswith("rtp_llm_rank")]
        assert len(ranks) == 4, ranks
        save(
            self.output / "live_info.json",
            dict(
                server_pid=process.pid,
                driver_pid=os.getpid(),
                server_create_time=process.create_time(),
                rank_pids=ranks,
                descendant_pids=[p.pid for p in children],
                port=self.server.port,
                gpu_ids=os.environ.get("CUDA_VISIBLE_DEVICES"),
                keep_server_alive=True,
            ),
        )
        self.progress("ready", rank_pids=ranks)
        # The external host monitor must be present before any measured request.
        for _ in range(120):
            if (self.output / "host_monitor_ready.json").exists():
                break
            time.sleep(1)
        else:
            raise RuntimeError("host GPU monitor was not attached")
        self.check_environment()
        self.record([self.request("text", 0, 0)])
        for phase in ("cold", "warm"):
            for case in range(1, 5):
                self.check_environment()
                self.progress("validating", phase=phase, case=case)
                self.record([self.request(phase, case, case)])
        for concurrency in self.args.concurrencies:
            count = max(4, 2 * concurrency)
            count = int(math.ceil(count / 4) * 4)
            for repetition in range(self.args.repeats):
                self.check_environment()
                phase = f"c{concurrency}_r{repetition}"
                self.progress(
                    "measuring", phase=phase, concurrency=concurrency, count=count
                )
                started = time.monotonic()
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=concurrency
                ) as pool:
                    jobs = [
                        pool.submit(self.request, phase, i, (i + repetition) % 4 + 1)
                        for i in range(count)
                    ]
                    rows = [future.result() for future in jobs]
                seconds = time.monotonic() - started
                self.record(rows)
                self.check_environment()
                summary = dict(
                    concurrency=concurrency,
                    repetition=repetition,
                    requests=count,
                    seconds=seconds,
                    success_qps=count / seconds,
                    completion_tokens_per_second=sum(
                        r["usage"]["completion_tokens"] for r in rows
                    )
                    / seconds,
                    rt_seconds=distribution([r["seconds"] for r in rows]),
                    server_ttft_ms=distribution([r["server_ttft_ms"] for r in rows]),
                    wait_ms=distribution([r["wait_ms"] for r in rows]),
                )
                self.rounds.append(summary)
                save(self.output / "rounds.json", self.rounds)
                print("ROUND " + json.dumps(summary), flush=True)
        all_ports = sorted(
            {a["grpc_port"] for row in self.rows for a in row.get("role_addrs", [])}
        )
        assert len(all_ports) == 4, ("not all DP ranks served videos", all_ports)
        save(
            self.output / "results.json",
            dict(
                status="completed_finite_load_sweep",
                rounds=self.rounds,
                requests=len(self.rows),
                exact_dp4_warm_choices=True,
                dp_ports=all_ports,
                service_retained=True,
                formal_nonregression_accepted=False,
                scope="late DP4 only; not an eager/late performance comparison",
            ),
        )
        self.progress("completed_service_retained")

    def keep_alive(self):
        while not (self.output / "STOP").exists():
            if self.server.server_pid is None or not psutil.pid_exists(
                self.server.server_pid
            ):
                self.progress("service_exited")
                return
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--concurrencies", default="1,4,8,16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument(
        "--attach-live-info",
        help="Reuse a verified, request-free retained service; never stop it",
    )
    args = parser.parse_args()
    args.concurrencies = [int(value) for value in args.concurrencies.split(",")]
    assert args.repeats >= 3 and all(1 <= c <= 16 for c in args.concurrencies)
    runner = Runner(args)
    failed = False
    try:
        try:
            runner.run()
        except Exception as exc:
            failed = True
            save(
                runner.output / "failure.json", dict(error=repr(exc), time=time.time())
            )
            runner.progress("failed_service_retained", error=repr(exc))
        if not args.attach_live_info:
            runner.keep_alive()
    finally:
        if not args.attach_live_info:
            runner.server.stop_server()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
