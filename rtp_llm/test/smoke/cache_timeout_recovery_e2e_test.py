"""Real model/service/cache E2E using short production timeout settings.

No delay injection: concurrency and cache transfers create the workload. A
deadline sweep must actually reach the requested path; absence is a failure.
"""

import argparse
import concurrent.futures
import json
import logging
import os
import shlex
import tempfile
import time
import unittest
from pathlib import Path

import requests
from transformers import AutoTokenizer

from rtp_llm.test.smoke.remote_kvcm_server import RemoteKVCMServer
from rtp_llm.test.smoke.timeout_recovery_checks import (
    analyze_timeout_recovery,
    d2h_timeout_records,
    frozen_host_metric_samples,
)
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class CacheTimeoutRecoveryE2ETest(unittest.TestCase):
    def setUp(self):
        self.output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        self.output.mkdir(parents=True, exist_ok=True)
        self.work = tempfile.TemporaryDirectory(
            prefix="cache-timeout-e2e-", dir=os.environ.get("TEST_TMPDIR")
        )
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.backend = "kvcm" if OPTIONS.scenario.startswith("kvcm") else "local"
        self.model = os.environ.get(
            "CHECKPOINT_PATH", "/mnt/nas1/hf/Qwen2.5-0.5B-Instruct"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, local_files_only=True, trust_remote_code=True
        )
        self.prompts = self.make_prompts()
        self.remote_env = {}
        if self.backend == "kvcm":
            self.start_kvcm()
        self.observations = []
        self.addCleanup(self.save_report)

    def save_report(self):
        (self.output / "timeout_recovery_e2e.json").write_text(
            json.dumps(
                {"scenario": OPTIONS.scenario, "attempts": self.observations},
                indent=2,
                ensure_ascii=False,
            )
        )

    def make_prompts(self):
        body = "".join(
            f"Archive row {i}: books, maps and letters are stored in box {i % 17}.\n"
            for i in range(600)
        )
        body = self.tokenizer.decode(self.tokenizer.encode(body)[:4096])
        prompts = [
            f"Document {i:04d}.\n{body}\nReturn only the word READY.\nAnswer:"
            for i in range(OPTIONS.concurrency)
        ]
        for prompt in prompts:
            count = len(self.tokenizer.encode(prompt))
            self.assertGreater(count, 3000)
            self.assertLess(count + 8, 8192)
        return prompts

    def start_kvcm(self):
        binary = (
            Path(os.environ["TEST_SRCDIR"])
            / os.environ["TEST_WORKSPACE"]
            / "3rdparty/remote_kv_cache_manager/bin/kv_cache_manager_bin"
        )
        self.assertTrue(binary.is_file(), f"missing real KVCM server: {binary}")
        # Keep payload files private to this test, away from shared runfiles.
        installation = self.root / "kvcm"
        (installation / "bin").mkdir(parents=True)
        (installation / "bin/kv_cache_manager_bin").symlink_to(binary.resolve())
        self.remote = RemoteKVCMServer(
            str(installation),
            {"KVCM_LOG_LEVEL": "DEBUG"},
            str(Path.cwd() / "logs"),
            str(self.output / "kvcm_logs"),
        )
        self.addCleanup(self.remote.copy_logs)
        self.addCleanup(self.remote.stop_server)
        self.assertTrue(self.remote.start_server(), "real KVCM server startup failed")
        self.remote.post_json(
            "updateStorage",
            {
                "trace_id": self.root.name,
                "storage": self.remote.get_storage_config(),
                "force_update": True,
            },
            admin=True,
        )
        # The real client also registers its node for remote-cache control,
        # even when HBM event publishing is disabled. Give that control path
        # an event backend while keeping all payloads in the private NFS root.
        event_storage = self.root.name + "_events"
        self.remote.post_json(
            "addStorage",
            {
                "trace_id": self.root.name,
                "storage": {
                    "global_unique_name": event_storage,
                    "storage_type": 7,
                    "event_report": {
                        "heartbeat_timeout_ms": 30000,
                        "cleanup_grace_ms": 300000,
                        "liveness_check_interval_ms": 5000,
                        "snapshot_min_interval_ms": 1000,
                    },
                },
            },
            admin=True,
        )
        group = self.remote.post_json(
            "getInstanceGroup", {"name": "default"}, admin=True
        )["instance_group"]
        # Bootstrap permits an empty reclaim backend; update requires one.
        group["cache_config"]["reclaim_strategy"]["storage_unique_name"] = "nfs_01"
        version = int(group.get("version", 0))
        group["version"] = version + 1
        group["event_report_storage_candidates"] = [event_storage]
        self.remote.post_json(
            "updateInstanceGroup",
            {
                "trace_id": self.root.name,
                "instance_group": group,
                "current_version": version,
            },
            admin=True,
        )
        configured = self.remote.post_json(
            "getInstanceGroup", {"name": "default"}, admin=True
        )["instance_group"]
        self.assertEqual(configured["event_report_storage_candidates"], [event_storage])
        self.assertEqual(configured["storage_candidates"], ["nfs_01"])
        self.remote_env = self.remote.client_env()

    def server_args(self, budget, label):
        args = shlex.split(os.environ["SMOKE_ARGS"])
        d2h = OPTIONS.scenario == "local_d2h_rpc"
        args += [
            "--warm_up",
            "0",
            "--enable_cuda_graph",
            "0",
            "--act_type",
            "FP16",
            "--max_seq_len",
            "8192",
            "--reuse_cache",
            "1",
            "--seq_size_per_block",
            "64",
            "--enable_device_cache",
            "1",
            "--enable_memory_cache",
            "1" if self.backend == "local" else "0",
            "--test_block_num",
            "2048",
            "--concurrency_limit",
            "16",
            "--max_context_batch_size",
            "8",
            "--block_tree_device_evict_low_watermark_ratio",
            "0.04" if d2h else "0.0001",
            "--block_tree_device_evict_high_watermark_ratio",
            "0.065" if d2h else "0.0002",
            "--block_tree_frozen_recovery_interval_ms",
            "100",
            "--block_tree_frozen_recovery_query_timeout_ms",
            "1000",
        ]
        if d2h:
            # 仅保留 DEVICE→HOST 淘汰；足够的 HOST 容量避免冻结比例保护抢先触发。
            args += [
                "--enable_disk_cache",
                "0",
                "--enable_remote_cache",
                "0",
                "--memory_cache_size_mb",
                "2048",
                "--memory_cache_sync_timeout_ms",
                str(budget),
                "--btc_transfer_engine_async",
                "1",
            ]
        elif self.backend == "local":
            world_size = int(args[args.index("--world_size") + 1])
            disks = [self.root / label / f"disk{rank}" for rank in range(world_size)]
            for disk in disks:
                disk.mkdir(parents=True)
            args += [
                # Local DISK caching requires the full DEVICE -> HOST -> DISK
                # hierarchy. Low watermarks demote the seed through both tiers.
                "--memory_cache_size_mb",
                "2048",
                "--block_tree_memory_evict_low_watermark_ratio",
                "0.0001",
                "--block_tree_memory_evict_high_watermark_ratio",
                "0.0002",
                "--enable_disk_cache",
                "1",
                "--enable_remote_cache",
                "0",
                "--disk_cache_paths",
                ",".join(map(str, disks)),
                "--disk_cache_size_mb",
                "2048",
                "--disk_cache_buffered_io",
                "0",
                "--disk_cache_read_thread_count",
                "1",
                "--disk_cache_write_thread_count",
                "1",
                "--disk_cache_sync_timeout_ms",
                str(budget),
                "--disk_cache_evict_timeout_ms",
                "30000",
                "--block_tree_transfer_worker_count",
                "1",
                "--btc_transfer_engine_async",
                "0" if OPTIONS.legacy else "1",
            ]
        else:
            sdk_budget = budget if OPTIONS.scenario == "kvcm_sdk" else 30000
            task_budget = 30000 if OPTIONS.scenario == "kvcm_sdk" else budget
            args += [
                "--enable_disk_cache",
                "0",
                "--enable_remote_cache",
                "1",
                "--kvcm_model_sdk_config",
                '[{"type":"file"}]',
                "--kvcm_client_config",
                "",
                "--kvcm_instance_group",
                "default",
                "--kvcm_enable_vipserver",
                "false",
                "--kvcm_server_address",
                self.remote.address(),
                "--kvcm_storage_thread_num",
                "1",
                "--kvcm_asyncwrapper_thread_num",
                "1",
                "--kvcm_task_timeout_ms",
                str(task_budget),
                "--kvcm_get_timeout_ms",
                str(sdk_budget),
                "--kvcm_put_timeout_ms",
                "30000",
                "--kvcm_instance_id_salt",
                self.root.name,
            ]
        return shlex.join(args)

    def start_model(self, budget, label):
        args = self.server_args(budget, label)
        cpp_logs = self.output / (label + "_cpp")
        cpp_logs.mkdir(parents=True, exist_ok=True)
        config = self.root / (label + "_alog.conf")
        config.write_text(
            "alog.rootLogger=INFO, engineAppender\n"
            "alog.max_msg_len=2048000\n"
            "alog.appender.engineAppender=FileAppender\n"
            f"alog.appender.engineAppender.fileName={cpp_logs}/engine-%p.log\n"
            "alog.appender.engineAppender.layout=PatternLayout\n"
            "alog.appender.engineAppender.layout.LogPattern=[%%d] [%%l] [%%p:%%t] %%m\n"
            "alog.appender.engineAppender.async_flush=false\n"
            "alog.appender.engineAppender.flush=true\n"
            "alog.appender.engineAppender.compress=false\n"
        )
        args += " --ft_alog_conf_path " + shlex.quote(str(config))
        server = MagaServerManager(
            env_args={
                "FRONTEND_SERVER_COUNT": "1",
                # Per-process files avoid TP ranks interleaving long freeze
                # records in their shared stdout stream.
                "FT_SERVER_TEST": "0",
                "LOG_LEVEL": "INFO",
                **self.remote_env,
                **(
                    {"kmonitorEnableLogFileSink": "true"}
                    if OPTIONS.scenario == "local_d2h_rpc"
                    else {}
                ),
            },
            role_name=label,
            smoke_args_str=args,
        )
        server.timeout_cpp_logs = cpp_logs
        server.timeout_label = label
        self.addCleanup(server.stop_server)
        self.assertTrue(
            server.start_server(
                model_path=self.model, model_type="qwen_tool", timeout=180
            ),
            f"model startup failed: {label}",
        )
        response = requests.post(
            f"http://127.0.0.1:{server.port}/set_log_level",
            json={"log_level": "INFO"},
            timeout=10,
        )
        response.raise_for_status()
        logging.info(
            "E2E scenario=%s budget=%d args=%s", OPTIONS.scenario, budget, args
        )
        return server

    def request(self, server, index, label, reuse=True, allow_timeout_error=False):
        query = {
            "prompt": self.prompts[index],
            "generate_config": {
                "max_new_tokens": 4,
                "top_k": 1,
                "random_seed": 42,
                "reuse_cache": reuse,
                "return_output_ids": True,
                "aux_info": True,
            },
        }
        # Sessions are per call: concurrent requests do not share mutable state.
        with requests.Session() as session:
            session.trust_env = False
            response = session.post(
                f"http://127.0.0.1:{server.port}/",
                json=query,
                timeout=180,
            )
        (self.output / f"{label}_{index}.http.json").write_text(response.text)
        result = response.json()
        # A timed-out cache load can terminate generation. Accept only this
        # exact response during the short-budget workload; logs must separately
        # prove the timeout and recovery. Baseline and fresh probes stay strict.
        if (
            allow_timeout_error
            and response.status_code == 500
            and (
                (
                    result.get("error_code") == 602
                    and result.get("message")
                    == "allocator load failed: load transfer failed"
                )
                or (
                    result.get("error_code") == 8300
                    and result.get("error_code_str") == "8300_LOAD_CACHE_TIMEOUT"
                )
            )
        ):
            return result
        response.raise_for_status()
        self.assertNotIn("error_code", result, result)
        self.assertIs(result.get("finished"), True, result)
        output_ids = result.get("output_ids")
        self.assertIsInstance(output_ids, list, result)
        self.assertEqual(len(output_ids), 1, result)
        self.assertTrue(output_ids[0], result)
        self.assertLessEqual(len(output_ids[0]), 4, result)
        self.assertTrue(
            all(type(token) is int and token >= 0 for token in output_ids[0]), result
        )
        return result

    def batch(self, server, label, allow_timeout_error=False):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=OPTIONS.concurrency
        ) as executor:
            return list(
                executor.map(
                    lambda i: self.request(
                        server, i, label, allow_timeout_error=allow_timeout_error
                    ),
                    range(len(self.prompts)),
                )
            )

    def assert_outputs(self, expected, actual, allow_timeout_error=False):
        self.assertEqual(len(expected), len(actual))
        for i, (before, after) in enumerate(zip(expected, actual)):
            if allow_timeout_error and "error_code" in after:
                continue
            self.assertEqual(before["output_ids"], after["output_ids"], f"prompt {i}")

    def stress_batch(self, server, label, reference):
        before = analyze_timeout_recovery(self.log_text(server), backend=self.backend)[
            "evidence"
        ]
        results = self.batch(server, label, allow_timeout_error=True)
        self.assert_outputs(reference, results, allow_timeout_error=True)
        failures = sum("error_code" in item for item in results)
        if failures:
            evidence = analyze_timeout_recovery(
                self.log_text(server), backend=self.backend
            )["evidence"]
            self.assertTrue(
                any(
                    name.startswith(self.backend + "_")
                    and len(lines) > len(before.get(name, []))
                    for name, lines in evidence.items()
                ),
                f"cache load failed without timeout evidence in batch {label}",
            )
        self.observations.append(
            {
                "stage": "http_batch",
                "label": label,
                "successful_requests": len(results) - failures,
                "cache_load_timeout_responses": failures,
            }
        )

    def wait_for_remote_seed(self):
        # Keep the writer alive until KVCM acknowledges full publication. Model
        # probes cannot establish this while its DEVICE cache still serves hits.
        deadline = time.monotonic() + 180
        access_log = Path.cwd() / "logs" / "access.log"
        while True:
            sessions = {}
            completed = set()
            text = access_log.read_text(errors="replace") if access_log.exists() else ""
            for line in text.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A live log may end in an incomplete record.
                request = event.get("request", {})
                if not request.get("instance_id", "").startswith(self.root.name + "_"):
                    continue
                response = event.get("response", {})
                if response.get("header", {}).get("status", {}).get("code") != "OK":
                    continue
                if event.get("api_name") == "StartWriteCache":
                    keys = tuple(request.get("block_keys", []))
                    if keys:
                        sessions[response["write_session_id"]] = keys
                elif event.get("api_name") == "FinishWriteCache":
                    keys = sessions.get(request.get("write_session_id"))
                    # A prefix mask equal to the batch size confirms every block.
                    if keys and request.get("success_blocks", {}).get("offset") == len(keys):
                        completed.add(keys)
            if len(completed) >= len(self.prompts):
                self.observations.append(
                    {"stage": "remote_seed_ready", "published_batches": len(completed)}
                )
                self.save_report()
                return
            self.assertLess(
                time.monotonic(),
                deadline,
                f"remote seed publication incomplete: {len(completed)}/{len(self.prompts)} batches",
            )
            time.sleep(2)

    def log_text(self, server):
        parts = []
        for path in sorted(server.timeout_cpp_logs.glob("engine-*.log")):
            # A partial last line from a live process is not yet evidence.
            text = path.read_text(errors="replace")
            complete, separator, _ = text.rpartition("\n")
            parts.append(f"# {path.name}\n" + complete + separator)
        return "".join(parts)

    def persist_evidence(self, server, text, report):
        path = (
            self.output
            / f"{server.timeout_label}_evidence_{len(self.observations)}.log"
        )
        path.write_text(text)
        report["evidence_log"] = str(path)

    def wait_for_recovery(self, server):
        deadline = time.monotonic() + OPTIONS.recovery_seconds
        stable_since = time.monotonic()
        previous = None
        while True:
            text = self.log_text(server)
            report = analyze_timeout_recovery(text, backend=self.backend)
            now = time.monotonic()
            signature = json.dumps(report, sort_keys=True)
            if signature != previous:
                previous = signature
                stable_since = now
            # An empty snapshot immediately after generation can precede an
            # asynchronous store failure or its log flush. Require a quiet
            # evidence window as well as the release of every observed freeze.
            if not report["pending"] and now - stable_since >= 1.0:
                self.persist_evidence(server, text, report)
                return report
            if now >= deadline:
                self.persist_evidence(server, text, report)
                report["server_log"] = server.log_file_path
                report["stage"] = "recovery_wait_expired"
                self.observations.append(report)
                self.save_report()
                self.fail(f"cache recovery did not settle: {json.dumps(report)}")
            time.sleep(0.1)

    def coverage(self, report):
        evidence = report["evidence"]
        if OPTIONS.scenario.endswith("rpc"):
            return bool(report["frozen"]) and bool(
                evidence.get(self.backend + "_rpc_timeout")
            )
        if OPTIONS.scenario == "kvcm_sdk":
            return bool(evidence.get("kvcm_sdk_timeout"))
        categories = (
            ("kvcm_pre_io", "kvcm_payload_pre_io")
            if self.backend == "kvcm"
            else (
                "local_business_queue",
                "local_workloop_queue",
                "local_worker_queue",
                "local_resource_wait",
            )
        )
        return any(evidence.get(category) for category in categories)

    def wait_for_host_metric_zero(self, server, pools):
        # 从恢复已完成后的下一个完整采样窗口开始；启动时的零样本不能通过。
        after_ms = int(time.time() * 1000) + 2000
        deadline = time.monotonic() + max(30, OPTIONS.recovery_seconds)
        samples = {}
        while time.monotonic() < deadline:
            text = self.log_text(server)
            samples = frozen_host_metric_samples(text, after_ms=after_ms)
            if all(pool in samples and samples[pool]["maximum"] == 0 for pool in pools):
                report = {
                    "stage": "d2h_host_metrics_zero",
                    "after_ms": after_ms,
                    "pools": sorted(pools),
                    "samples": samples,
                }
                self.persist_evidence(server, text, report)
                self.observations.append(report)
                self.save_report()
                return
            time.sleep(0.2)
        self.fail(
            f"HOST frozen metric did not reach zero: pools={pools}, samples={samples}"
        )

    def exercise_d2h_eviction(self):
        baseline = self.start_model(30000, "d2h_baseline")
        reference = [
            self.request(baseline, i, "d2h_reference", reuse=False)
            for i in range(len(self.prompts))
        ]
        baseline.stop_server()
        for budget in OPTIONS.budgets:
            label = f"d2h_{budget}ms"
            server = self.start_model(budget, label)
            try:
                # 新进程中使用独立前缀，累积可淘汰 DEVICE 块；不靠 H2D 读取施压。
                self.assert_outputs(reference, self.batch(server, label + "_evict"))
                report = self.wait_for_recovery(server)
                report.update(stage="d2h_recovery", budget_ms=budget)
                self.observations.append(report)
                self.save_report()
                self.assertFalse(
                    [
                        error
                        for error in report["errors"]
                        if error != "no local timeout was observed"
                    ],
                    report,
                )
                records = d2h_timeout_records(report)
                if not records:
                    continue
                self.assertTrue(report["passed"], report)
                # 在同一进程恢复后做真实推理，避免仅重启后可用的假阳性。
                self.assertEqual(
                    reference[0]["output_ids"],
                    self.request(server, 0, label + "_after_recovery", reuse=False)[
                        "output_ids"
                    ],
                )
                final = self.wait_for_recovery(server)
                self.assertTrue(final["passed"], final)
                pools = {
                    pool
                    for record in d2h_timeout_records(final)
                    for pool in record["target_pools"]
                }
                self.assertTrue(pools, final)
                self.wait_for_host_metric_zero(server, pools)
                logging.info(
                    "D2H recovery passed: budget=%d traces=%s HOST pools=%s",
                    budget,
                    [record["trace"] for record in records],
                    sorted(pools),
                )
                return
            finally:
                text = self.log_text(server)
                report = analyze_timeout_recovery(text, backend="local")
                report.update(stage="d2h_final", budget_ms=budget)
                self.persist_evidence(server, text, report)
                self.observations.append(report)
                self.save_report()
                server.stop_server()
        self.fail(f"NOT COVERED: no TP2 D2H eviction RPC timeout: {self.observations}")

    def test_short_timeouts_recover_without_breaking_generation(self):
        if OPTIONS.scenario == "local_d2h_rpc":
            self.exercise_d2h_eviction()
            return
        baseline = self.start_model(30000, "baseline")
        reference = [
            self.request(baseline, i, "uncached", reuse=False)
            for i in range(len(self.prompts))
        ]
        self.assert_outputs(reference, self.batch(baseline, "seed"))
        if self.backend == "kvcm":
            self.wait_for_remote_seed()
            # Cold DEVICE cache makes remote reuse observable without changing
            # production eviction behavior. The KVCM service and identity stay.
            baseline.stop_server()
            baseline = self.start_model(30000, "baseline_remote_read")
        else:
            time.sleep(10)  # Wait for ordinary asynchronous tiered eviction.
        field = "remote_reuse_len" if self.backend == "kvcm" else "disk_reuse_len"
        reused = self.batch(baseline, "baseline_reuse")
        self.assert_outputs(reference, reused)
        self.assertTrue(
            all(item.get("aux_info", {}).get(field, 0) > 0 for item in reused),
            f"baseline did not reuse {field}: {[item.get('aux_info') for item in reused]}",
        )
        baseline.stop_server()
        # KVCM remains alive and keeps the fully published long-timeout seed.
        # Local caches are populated anew; their store/eviction budget stays long.
        covered = False
        for budget in OPTIONS.budgets:
            label = f"{OPTIONS.scenario}_{budget}ms"
            server = self.start_model(budget, label)
            try:
                if self.backend == "local":
                    self.assert_outputs(reference, self.batch(server, label + "_seed"))
                    # Let tiered eviction publish DISK seeds before the read workload.
                    time.sleep(10)
                for wave in range(3):
                    self.stress_batch(server, f"{label}_wave{wave}", reference)
                    report = self.wait_for_recovery(server)
                    report["budget_ms"] = budget
                    report["server_log"] = server.log_file_path
                    report["stage"] = f"wave{wave}_recovery"
                    self.observations.append(report)
                    self.save_report()
                    # Absence of a timeout may continue the sweep. Missing
                    # release evidence must never be discarded.
                    recovery_errors = [
                        error
                        for error in report["errors"]
                        if error != f"no {self.backend} timeout was observed"
                    ]
                    self.assertFalse(recovery_errors, report)
                if self.coverage(report):
                    self.stress_batch(server, label + "_after_recovery", reference)
                    self.assertEqual(
                        reference[0]["output_ids"],
                        self.request(server, 0, label + "_fresh", reuse=False)[
                            "output_ids"
                        ],
                    )
                    final = self.wait_for_recovery(server)
                    final["budget_ms"] = budget
                    final["server_log"] = server.log_file_path
                    final["stage"] = "after_recovery_requests"
                    self.observations.append(final)
                    self.save_report()
                    self.assertTrue(final["passed"], final)
                    covered = True
                    break
            except Exception:
                text = self.log_text(server)
                failure = analyze_timeout_recovery(text, backend=self.backend)
                self.persist_evidence(server, text, failure)
                failure.update(
                    budget_ms=budget,
                    server_log=server.log_file_path,
                    stage="request_or_recovery_failed",
                )
                self.observations.append(failure)
                self.save_report()
                raise
            finally:
                server.stop_server()
        self.assertTrue(
            covered,
            f"requested timeout path was not observed (NOT COVERED): {self.observations}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=[
            "local_queue",
            "local_rpc",
            "local_d2h_rpc",
            "kvcm_queue",
            "kvcm_rpc",
            "kvcm_sdk",
        ],
        required=True,
    )
    parser.add_argument("--budgets", type=int, nargs="+")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--recovery-seconds", type=int, default=30)
    parser.add_argument("--legacy", action="store_true")
    OPTIONS, remaining = parser.parse_known_args()
    if OPTIONS.budgets is None:
        # The real service spends this same KVCM budget on match-to-ready.
        # Sweep far enough to get past that earlier stage into queue/RPC I/O.
        OPTIONS.budgets = (
            [1, 2, 5, 10, 20, 50, 100, 200, 500]
            if OPTIONS.scenario in ("kvcm_queue", "kvcm_rpc")
            else [1, 2, 5, 10]
        )
    if not OPTIONS.budgets or any(value <= 0 for value in OPTIONS.budgets):
        parser.error("budgets must be positive milliseconds")
    if not 2 <= OPTIONS.concurrency <= 8:
        parser.error("concurrency must be between 2 and 8")
    logging.basicConfig(level=logging.INFO)
    unittest.main(argv=[__file__, *remaining])
