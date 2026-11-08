"""One257-token shared40-layer compact140 native RDMA P/D smoke role."""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

from rtp_llm.models_py.modules.dsv41.ppu.test.service_fixture import (
    healthy,
    wait_for,
    write_json,
)

AFFINITY = {str(rank): f"mlx5_bond_{(2, 2, 3, 3, 1, 1, 0, 0)[rank]}" for rank in range(8)}


def classify_rdma_failures(
    text, lifecycle, cleanup, *, launcher_pid, snapshot_id, stop_path, role, output_path
):
    """Allow only an owned weak-pointer release callback during normal stop."""
    qualified = False
    owned_pids = set()
    try:
        marker = lifecycle["stop_marker"]
        started = datetime.fromisoformat(lifecycle["started_at"])
        finished = datetime.fromisoformat(lifecycle["finished_at"])
        observed = datetime.fromisoformat(marker["observed_at"])
        tag = marker["payload"]["tag"]
        expected_marker = {
            "owner": "L0",
            "role": {"prefill": "P", "decode": "D"}[role],
            "tag": tag,
            "snapshot_id": snapshot_id,
        }
        owned_pids = {launcher_pid, *cleanup["recorded_descendants"]}
        qualified = (
            launcher_pid > 0
            and lifecycle["launcher_pid"] == launcher_pid
            and lifecycle["snapshot_id"] == snapshot_id
            and bool(snapshot_id)
            and marker["path"] == stop_path
            and bool(stop_path)
            and marker["payload"] == expected_marker
            and bool(tag)
            and all(c.isalnum() or c == "-" for c in tag)
            and Path(output_path).name == "native-" + tag
            and Path(stop_path).parent.name == "control-" + tag
            and Path(stop_path).name == "stop"
            and Path(output_path).parent == Path(stop_path).parent.parent
            and marker["inode"] > 0
            and lifecycle["launcher_running_at_start"] is True
            and lifecycle["primary_error"] is None
            and lifecycle["cleanup_error"] is None
            and lifecycle["exit_code"] == 0
            and cleanup["exit_code"] == 0
            and cleanup["forced"] is False
            and observed <= started <= finished
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    release = re.compile(
        r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6})\] "
        r"\[INFO\] \[(?P<pid>\d+):\d+\] \[RANK \d+\]\[[^\]\r\n]+\]"
        r"\[rtp_llm/cpp/disaggregate/cache_store/rdma/"
        r"RdmaServerConnectionCallback\.cpp:38\]"
        r"\[virtual void rtp_llm::RdmaServerConnectionCallback::OnConnectFailed"
        r"\(const std::shared_ptr<accl::barex::XConnection>&\)\] "
        r"rdma server connect failed from [^\s]*:\d+ channel=0x[0-9a-fA-F]+, "
        r"but server released$"
    )
    failures, callbacks = [], []
    for line in text.splitlines():
        if not re.search(
            r"rdma.*(?:read failed|connect failed)|messager client get rdma connection failed",
            line,
            re.I,
        ):
            continue
        match = release.fullmatch(line)
        accepted = False
        if qualified and match is not None:
            try:
                timestamp = datetime.fromisoformat(match["timestamp"])
                accepted = (
                    int(match["pid"]) in owned_pids and started <= timestamp <= finished
                )
            except (TypeError, ValueError):
                pass
        if accepted:
            callbacks.append(line)
        else:
            failures.append(line)
    return {
        "teardown_qualified": qualified,
        "accepted_teardown_callbacks": callbacks,
        "failures": failures,
    }


def stop_shared_owned(process, output):
    """Drain this launcher; tolerate exited children while recording ownership."""
    children = []
    records = [{"pid": process.pid, "argv": process.args, "launcher": True}]
    if process.poll() is None:
        try:
            parent = psutil.Process(process.pid)
            records[0]["created"] = parent.create_time()
            children = parent.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.ZombieProcess) as error:
            records[0]["metadata_error"] = repr(error)
    for child in children:
        record = {"pid": child.pid}
        for key, getter in (("created", child.create_time), ("argv", child.cmdline)):
            try:
                record[key] = getter()
            except (
                psutil.NoSuchProcess,
                psutil.ZombieProcess,
                psutil.AccessDenied,
            ) as error:
                record[key + "_error"] = repr(error)
        records.append(record)
    write_json(output / "owned-processes.json", records)

    def running_children():
        alive = []
        for child in children:
            try:
                # psutil checks creation time before signaling, guarding PID reuse.
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    alive.append(child)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
        return alive

    forced = False
    try:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            status = process.wait(timeout=240)
        except subprocess.TimeoutExpired:
            forced = True
            status = None
        alive = running_children()
        if status is not None and alive:
            psutil.wait_procs(alive, timeout=15)
            alive = running_children()
        if alive or status is None:
            forced = True
            for method in ("terminate", "kill"):
                for child in running_children():
                    try:
                        getattr(child, method)()
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        pass
                psutil.wait_procs(running_children(), timeout=15)
            if process.poll() is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            status = process.wait(timeout=15)
        remaining = [child.pid for child in running_children()]
        if forced or remaining:
            raise RuntimeError(
                f"native shared role required scoped cleanup; remaining={remaining}"
            )
        return status
    finally:
        write_json(
            output / "cleanup.json",
            {
                "exit_code": process.poll(),
                "forced": forced,
                "recorded_descendants": [child.pid for child in children],
            },
        )


def validate_reports(role, reports):
    assert reports and all(item["moe"]["backend"] == "grouped" and item["moe"]["precision"] == "w4a8" for item in reports), reports
    assert any(item["moe"]["grouped_gemm_calls"] > 0 for item in reports), reports
    for item in reports:
        if item["idle"]:
            assert len(item["routed_active_rows"]) == 40 and not any(item["routed_active_rows"].values()), item
            assert not item["native_writes"] and not item["native_pool_bindings"], item
    real = [item for item in reports if not item["idle"]]
    if not real:
        raise AssertionError("no actual shared model forward evidence")
    for item in real:
        assert item["model_layers"] == 40, item
        assert item["yoco"]["executed_layers"] == list(range(40)), item
        assert item["profile"] == {"mode": "compact", "keep_tokens": 140, "window": 128, "max_rollback_tokens": 0, "encoder_replay": False}, item
        assert all(pool["native_alias"] for pool in item["native_pool_bindings"]), item
        assert item["native_writes"], item
        assert item["output_finite"] is True, item
        assert item["normalization"]["attention_norm_calls"] == 40, item
        assert item["normalization"]["ffn_calls"] == 40, item
        assert item["normalization"]["ffn_norm_calls"] + item["normalization"]["mega_transitions"] == 40, item
    if role == "prefill":
        compact = [item for item in real if item["prefill"] and item["yoco"]["input_lengths"] == [257]]
        assert {item["rank"] for item in compact} == set(range(8)), compact
        actual_positions = set()
        for item in compact:
            yoco = item["yoco"]
            assert yoco["logical_retained_rows"] == 140 and yoco["cp_padding_rows"] == 4, item
            assert yoco["compacted"] and yoco["restored"] and yoco["cp_size"] == 8, item
            assert yoco["plan_reason"] == "compact140", item
            assert yoco["exact_short_fallback_observed"] is True, item
            actual_positions.update(yoco["local_positions"])
            assert item["mega_mhc"]["fallbacks"].get("ced:unqualified_phase", 0) > 0, item
        assert actual_positions == set(range(117, 257)), actual_positions
        return {"CP_ranks": list(range(8)), "logical_rows": 140, "padding_rows": 4, "positions": [117, 256]}
    assert not any(item["prefill"] for item in real), real
    owners = {item["rank"] for item in real}
    assert len(owners) == 1 and len(real) >= 3, real
    assert {item["rank"] for item in reports} == set(range(8)), reports
    positions = sorted({p for item in real for p in item["yoco"]["local_positions"]})
    assert positions[:3] == [257, 258, 259], positions
    assert any(item["mega_mhc"]["calls"].get("decode", 0) > 0 for item in real), real
    return {"destination_owners": sorted(owners), "prefill_executed": False, "decode_positions": positions}


def main(*, sharded=False, eplb=False):
    role = os.environ["RTP_V41_SHARED_ROLE"]
    if role not in ("prefill", "decode"):
        raise ValueError("shared smoke role must be prefill or decode")
    ced_profile = os.environ.get("RTP_V41_TEST_SHARED_CED_SUITE")
    eplb_graph = os.environ.get("RTP_V41_TEST_EPLB_GRAPH") == "1"
    perf_profile = os.environ.get("RTP_V41_TEST_SHARED_PERF")
    if perf_profile:
        from rtp_llm.models_py.modules.dsv41.ppu.test import rdma_perf as perf

        if not sharded or eplb or ced_profile:
            raise ValueError("clean performance requires the sharded native fixture")
    if eplb_graph:
        from rtp_llm.models_py.modules.dsv41.ppu.test import ced_rdma_suite as eplb_ced
        from rtp_llm.models_py.modules.dsv41.ppu.test import dspark_graph_batch as eplb_http

        if not eplb or not sharded or ced_profile:
            raise ValueError("EPLB Graph needs the isolated native EPLB role")
    if ced_profile:
        from rtp_llm.models_py.modules.dsv41.ppu.test import ced_rdma_suite as ced_suite

        parallel = os.environ["RTP_V41_TEST_CED_PARALLEL"]
        if not sharded or eplb or ced_profile not in ced_suite.PROFILES or parallel not in ("cp", "dp"):
            raise ValueError("invalid isolated shared CED suite")
        sharded = parallel == "cp"
        lengths = tuple(map(int, os.environ["RTP_V41_TEST_CED_INPUTS"].split(",")))
        capacity = int(os.environ["RTP_V41_TEST_CED_CAPACITY"])
        if not lengths or min(lengths) < 1 or max(lengths) + 4 > capacity:
            raise ValueError("shared CED workload exceeds admission capacity")
    affinity = os.environ.get("ACCL_NIC_GPU_AFFINITY")
    if affinity is None or json.loads(affinity) != AFFINITY:
        raise ValueError("explicit qualified eight-rank RDMA affinity is required")
    source = Path(os.environ["RTP_V41_TEST_SOURCE_ROOT"]).resolve()
    runtime = Path(os.environ["RTP_V41_TEST_RUNTIME"]).resolve()
    output = Path(os.environ["RTP_V41_SHARED_OUTPUT"]).resolve()
    control = Path(os.environ["RTP_V41_SHARED_CONTROL"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    control.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((source / ".v41_snapshot.json").read_text())
    bound = json.loads((runtime / ".v41_snapshot.json").read_text())
    for key in ("snapshot_id", "source_sha256", "public_commit", "internal_commit"):
        if snapshot[key] != bound[key]:
            raise ValueError(f"source/runtime mismatch: {key}")
    p_address = os.environ["RTP_V41_SHARED_PREFILL_ADDRESS"]
    d_address = os.environ["RTP_V41_SHARED_DECODE_ADDRESS"]
    address = p_address if role == "prefill" else d_address
    port = int(address.rsplit(":", 1)[1])
    env = dict(os.environ, PYTHONPATH=str(runtime) + ":" + os.environ.get("PYTHONPATH", ""),
               LOG_PATH=str(output / "logs"), RTP_V41_SHARED_EVIDENCE_DIR=str(output / "forwards"),
               RTP_V41_PPU_YOCO_KEEP_TOKENS="140", RTP_V41_PPU_MAX_ROLLBACK_TOKENS="0",
               RTP_V41_SHARED_VALIDATE_STATE="1",
               DSV41_CED="1", DSV41_SWA_BOUNDED_REPLAY="1", DSV41_ENCODER_SWA_REPLAY="0")
    argv = [sys.executable, "-B", "-m", "rtp_llm.tools.v41.launch", role,
            "--checkpoint-path", os.environ["RTP_V41_TEST_CHECKPOINT"],
            "--prefill-address", p_address, "--decode-address", d_address,
            "--start-port", str(port), "--rdma",
            "--ced-mode", "bounded", "--yoco-keep-tokens", "140", "--max-rollback-tokens", "0",
            "--token-map-cache", str(output / "token-map.json")]
    if sharded:
        argv.append("--kv-cache-sharded")
    if perf_profile:
        selected = perf.configure(argv, env, role, output, perf_profile)
    if ced_profile:
        ced_suite.configure(argv, env, ced_profile, parallel, role, output, capacity)
    if eplb:
        env.update(RTP_V41_N8_EPLB_FIXTURE="1", RTP_V41_N8_EPLB_REPORT_DIR=str(output / "eplb"),
                   RTP_V41_TOKEN_MAP_CACHE=str(output / "token-map.json"))
        argv = [sys.executable, "-B", "-m",
                "rtp_llm.models_py.modules.dsv41.ppu.test.eplb_rdma_role", "--server"]
        if eplb_graph:
            env.update(RTP_V41_SHARED_COMPACT_IDLE_EVIDENCE="1", DISABLE_ACCESS_LOG="0")
            if role == "decode":
                from rtp_llm.models_py.modules.dsv41.ppu.test.dspark_workload import configure_contract_logging

                retention = configure_contract_logging(env, "contracts", output, (257,), 64, 0)
                env.pop("RTP_V41_DSPARK_OBSERVE_DIR", None)
                write_json(output / "model-input-retention.json", retention)
    deadline = time.monotonic() + (14400 if ced_profile or perf_profile else 6300)
    if (ced_profile or eplb_graph or perf_profile) and (control / "abort").exists():
        raise RuntimeError("L0 closed this attempt before service launch")
    with (output / "process.log").open("x") as log:
        process = subprocess.Popen(argv, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(output / "launch.json", {"pid": process.pid, "argv": argv, "role": role, "snapshot": snapshot,
                                            "transport": "native RDMA CacheStore", "affinity": AFFINITY})
        stop_marker = None
        try:
            listen_all = role == "decode" or (ced_profile and parallel == "dp")
            urls = [f"http://127.0.0.1:{port + rank * 9}" for rank in range(8 if listen_all else 1)]
            wait_for(lambda: all(map(healthy, urls)), process, deadline, "native shared role HTTP readiness")
            write_json(output / "ready.json", {"role": role, "listeners": urls})
            if role == "prefill" and (ced_profile or eplb_graph or perf_profile):
                host, base = d_address.rsplit(":", 1)
                peers = [f"http://{host}:{int(base) + rank * 9}" for rank in range(8)]
                wait_for(lambda: all(map(healthy, peers)), process, deadline, "native D owner readiness")
                if perf_profile:
                    performance = perf.exercise(output, p_address)
                    write_json(output / "requests-done.json", {"requests": performance["requests"], "suite": "performance"})
                else:
                    if eplb_graph:
                        cases = eplb_http.exercise(output, p_address, process, deadline)
                    else:
                        cases = ced_suite.exercise(output, p_address, ced_profile, parallel, lengths)
                    write_json(output / "requests-done.json", {"requests": len(cases), "suite": ced_profile or "eplb-graph"})
            elif role == "prefill":
                host, base = d_address.rsplit(":", 1)
                peers = [f"http://{host}:{int(base) + rank * 9}" for rank in range(8)]
                wait_for(lambda: all(map(healthy, peers)), process, deadline, "native D owner readiness")
                subprocess.run([sys.executable, "-B", "-m", "rtp_llm.tools.v41.native_smoke",
                                "--url", "http://" + p_address, "--prompt", "hello " * 256,
                                "--expected-input-tokens", "257", "--requests", "1", "--max-new-tokens", "6" if eplb else "4",
                                *(["--trace-prefix", "n8-trigger"] if eplb else []),
                                "--timeout", "1200", "--output", str(output / "http-257")],
                               cwd=runtime, env=env, check=True, timeout=deadline - time.monotonic())
                if eplb:
                    # The trigger response completes only after its migrated
                    # same-instance continuation. A distinct short HTTP call
                    # must then enter the already loaded model with fresh KV.
                    subprocess.run([sys.executable, "-B", "-m", "rtp_llm.tools.v41.native_smoke",
                                    "--url", "http://" + p_address, "--prompt", "hello " * 16,
                                    "--expected-input-tokens", "17", "--requests", "1", "--max-new-tokens", "2",
                                    "--trace-prefix", "n8-postmigration",
                                    "--timeout", "1200", "--output", str(output / "http-17-after-migration")],
                                   cwd=runtime, env=env, check=True, timeout=deadline - time.monotonic())
                    write_json(output / "requests-done.json", {"requests": 2, "lineage": [
                        {"purpose": "trigger_real_migration", "input_tokens": 257, "output_tokens": 6, "http": "http-257"},
                        {"purpose": "fresh_request_after_migration_same_servers", "input_tokens": 17, "output_tokens": 2,
                         "http": "http-17-after-migration"}]})
                else:
                    write_json(output / "requests-done.json", {"requests": 1, "input_tokens": 257, "output_tokens": 4})
            wait_for(lambda: (control / "stop").exists(), process, deadline, "L0 reception and coordinated stop")
            with (control / "stop").open() as marker:
                marker_payload = json.load(marker)
                marker_stat = os.fstat(marker.fileno())
            stop_marker = {
                "path": str(control / "stop"),
                "inode": marker_stat.st_ino,
                "payload": marker_payload,
                "observed_at": datetime.now().isoformat(
                    sep=" ", timespec="microseconds"
                ),
            }
        finally:
            primary_error = sys.exc_info()[1]
            lifecycle = {
                "launcher_pid": process.pid,
                "snapshot_id": snapshot["snapshot_id"],
                "stop_marker": stop_marker,
                "launcher_running_at_start": process.poll() is None,
                "started_at": datetime.now().isoformat(
                    sep=" ", timespec="microseconds"
                ),
                "primary_error": (
                    repr(primary_error) if primary_error is not None else None
                ),
                "cleanup_error": None,
            }
            write_json(output / "teardown-start.json", lifecycle)
            try:
                status = stop_shared_owned(process, output)
            except Exception as cleanup_error:
                lifecycle["cleanup_error"] = repr(cleanup_error)
                write_json(
                    output / "server-exit.json",
                    {
                        "exit_code": process.poll(),
                        "cleanup_error": repr(cleanup_error),
                        "primary_error": (
                            repr(primary_error) if primary_error is not None else None
                        ),
                    },
                )
                if primary_error is None:
                    raise
            else:
                write_json(output / "server-exit.json", {"exit_code": status})
            finally:
                lifecycle["finished_at"] = datetime.now().isoformat(
                    sep=" ", timespec="microseconds"
                )
                lifecycle["exit_code"] = process.poll()
                write_json(output / "teardown-finished.json", lifecycle)
    if status != 0:
        raise RuntimeError(f"native shared server failed: {status}")
    logs = [output / "process.log", *(output / "logs").rglob("*.log")]
    text = "\n".join(path.read_text(errors="replace") for path in logs)
    if "tcp messager init success" in text.lower() or "TcpMessager.cpp" in text:
        raise AssertionError("TCP CacheStore data transport was instantiated")
    if "rdma messager init success" not in text.lower():
        raise AssertionError("missing actual RdmaMessager initialization")
    if role == "decode" and not re.search(r"rdma connection to .* connect success", text):
        raise AssertionError("no actual native RDMA connection for continuation")
    rdma_health = classify_rdma_failures(
        text,
        lifecycle,
        json.loads((output / "cleanup.json").read_text()),
        launcher_pid=process.pid,
        snapshot_id=snapshot["snapshot_id"],
        stop_path=str(control / "stop"),
        role=role,
        output_path=str(output),
    )
    write_json(output / "rdma-health.json", rdma_health)
    if not rdma_health["teardown_qualified"] or rdma_health["failures"]:
        raise AssertionError("native RDMA failed")
    if perf_profile:
        evidence = perf.summarize(role, output, selected, text, performance if role == "prefill" else None)
        if os.environ.get("RTP_V41_TEST_PERF_GEMM_REFERENCE") == "1":
            subprocess.run([sys.executable, "-B", "-m", perf.__name__, str(output),
                            os.environ["RTP_V41_TEST_CHECKPOINT"]], cwd=runtime, env=env,
                           check=True, timeout=300)
        write_json(output / "summary.json", {"native_role_ok": True, "role": role, "snapshot": snapshot,
                                            "transport": "native RDMA CacheStore", "evidence": evidence})
        return
    reports = [json.loads(path.read_text()) for path in sorted((output / "forwards").glob("rank*-call*.json"))]
    if eplb_graph:
        if role == "prefill":
            evidence = eplb_ced.validate_prefill(reports, cases, "compact", "cp")
        else:
            from rtp_llm.models_py.modules.dsv41.ppu.test.eplb_graph_evidence import validate

            assert not any(not row["idle"] for row in reports), "EPLB graph fell back to eager D"
            evidence = validate(output, eplb_ced.wait_peer(control, deadline), text)
        write_json(output / "summary.json", {"native_role_ok": True, "role": role, "snapshot": snapshot,
                                            "transport": "native RDMA CacheStore", "evidence": evidence})
        return
    if ced_profile:
        assert reports and all(not row["idle"] and row["output_finite"] for row in reports)
        assert all(row["yoco"]["executed_layers"] == list(range(40)) for row in reports)
        assert all((row["moe"]["backend"], row["moe"]["precision"]) ==
                   (env["RTP_V41_PPU_MOE_BACKEND"], env["RTP_V41_PPU_MOE_PRECISION"]) for row in reports)
        if role == "prefill":
            evidence = ced_suite.validate_prefill(reports, cases, ced_profile, parallel)
        else:
            evidence = ced_suite.validate_decode(output, ced_suite.wait_peer(control, deadline), reports)
        write_json(output / "summary.json", {"native_role_ok": True, "role": role, "snapshot": snapshot,
                                            "transport": "native RDMA CacheStore", "evidence": evidence})
        return
    if eplb:
        from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_eplb import validate_eplb_forward_reports

        grouped = validate_eplb_forward_reports(role, reports)
        evidence = {"trigger": validate_reports(role, grouped["trigger_reports"]),
                    "fresh": grouped["fresh"], "grouping": grouped["grouping"]}
    else:
        evidence = validate_reports(role, reports)
    protocols = {item["cache_protocol"]["sha256"] for item in reports}
    if len(protocols) != 1:
        raise AssertionError("runtime/cache profile identity changed during smoke")
    if sharded:
        actual = re.findall(r"\bkv_cache_sharded:\s*(\w+)", text)
        if not actual or set(actual) != {"1"}:
            raise AssertionError(f"actual native CP sharding config differs: {actual}")
        for item in reports:
            placement = item["cache_protocol"].get("placement", {})
            if placement.get("source_cp_size") != 8 or item["persistent_replicated_shadow"] is not False:
                raise AssertionError("missing explicit sharded source/destination identity")
        for item in [report for report in reports if not report["idle"]]:
            if not item["cache_placement"] or any(pool["sharded"] != (role == "prefill") for pool in item["cache_placement"]):
                raise AssertionError("actual native pool placement differs from role")
            if role == "prefill":
                # SWA exchanges plus source values and ratio-2 scores.
                expected = 40 + sum(pool["tag"] in ("global_kv_1", "global_kv_2") for pool in item["native_pool_bindings"])
                expected += sum(pool["tag"] == "csa_state" for pool in item["native_pool_bindings"])
                if len(item["current_row_gathers"]) != expected or not item["pool_gathers"]:
                    raise AssertionError("missing actual CP current-row/pool collective counts")
            elif item["current_row_gathers"] or item["pool_gathers"]:
                raise AssertionError("D full pools must not issue P CP collectives")
    live = [item for item in reports if not item["idle"] and (item["native_state_after"] if role == "prefill" else item["native_state_before"])]
    if not live or not all(any(pool["nonzero_bytes"] > 0 for pool in (item["native_state_after"] if role == "prefill" else item["native_state_before"]).values()) for item in live):
        raise AssertionError("missing actual live native-cache byte evidence")
    write_json(output / "summary.json", {"native_role_ok": True, "role": role, "snapshot": snapshot,
                                         "transport": "native RDMA CacheStore", "cache_protocol_sha256": next(iter(protocols)),
                                         "state_fingerprints": [{"rank": item["rank"], "call": item["call"], "prefix_lengths": item["yoco"]["prefix_lengths"],
                                                                 "state": item["native_state_after"] if role == "prefill" else item["native_state_before"]} for item in live],
                                         "sharded": sharded,
                                         "cache_placement": [{"rank": item["rank"], "call": item["call"], "pools": item["cache_placement"]} for item in reports if not item["idle"]],
                                         "cache_protocol": reports[0]["cache_protocol"],
                                         "collectives": [{"rank": item["rank"], "call": item["call"],
                                                          "current_rows": item["current_row_gathers"], "pools": item["pool_gathers"]} for item in reports if not item["idle"]],
                                         "evidence": evidence, "FULL_VALIDATION": "DEFERRED_TO_N10"})


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--sharded"]):
        raise SystemExit("expected no arguments or --sharded")
    main(sharded=bool(sys.argv[1:]))
