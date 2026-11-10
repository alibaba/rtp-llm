"""Native D runner smoke paired with the existing N6 CP8 Prefill role."""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from weakref import ref

from rtp_llm.models_py.modules.dsv41.ppu.test.service_fixture import healthy, wait_for, write_json
from rtp_llm.models_py.modules.dsv41.ppu.test.pd_rdma_role import (
    AFFINITY, classify_rdma_failures, stop_shared_owned,
)
from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_graph_modes import validate_native_receipts


def _now():
    return datetime.now().isoformat(sep=" ", timespec="microseconds")


def _read_stop_marker(path, output, snapshot_id):
    tag = path.parent.name.removeprefix("control-")
    if not tag or path.parent.name != f"control-{tag}" or output != path.parent.parent / f"native-{tag}":
        raise ValueError("stop marker and output must identify the same L0 run")
    with path.open() as stream:
        payload = json.load(stream)
        inode = os.fstat(stream.fileno()).st_ino
    if payload != {"owner": "L0", "role": "D", "tag": tag, "snapshot_id": snapshot_id} or inode <= 0:
        raise ValueError("stop marker does not authorize normal teardown of this D run")
    return {"path": str(path), "inode": inode, "payload": payload, "observed_at": _now()}


def main():
    async_mode = os.environ.get("RTP_V41_N9_ASYNC", "0")
    if async_mode not in ("0", "1"):
        raise ValueError("RTP_V41_N9_ASYNC must be 0 or 1")
    if json.loads(os.environ["ACCL_NIC_GPU_AFFINITY"]) != AFFINITY:
        raise ValueError("require the qualified N6 RDMA affinity")
    source = Path(os.environ["RTP_V41_TEST_SOURCE_ROOT"]).resolve()
    runtime = Path(os.environ["RTP_V41_TEST_RUNTIME"]).resolve()
    snapshot = json.loads((source / ".v41_snapshot.json").read_text())
    bound = json.loads((runtime / ".v41_snapshot.json").read_text())
    for key in ("snapshot_id", "source_sha256", "public_commit", "internal_commit"):
        if snapshot[key] != bound[key]:
            raise ValueError(f"source/runtime mismatch: {key}")
    output = Path(os.environ["RTP_V41_SHARED_OUTPUT"]).resolve()
    reports = output / "graph-replays"
    reports.mkdir(parents=True)
    control = Path(os.environ["RTP_V41_SHARED_CONTROL"]).resolve()
    control.mkdir(parents=True, exist_ok=True)
    stop_path = control / "stop"
    prefill = os.environ["RTP_V41_SHARED_PREFILL_ADDRESS"]
    decode = os.environ["RTP_V41_SHARED_DECODE_ADDRESS"]
    port = int(decode.rsplit(":", 1)[1])
    env = dict(
        os.environ,
        PYTHONPATH=str(runtime) + ":" + os.environ.get("PYTHONPATH", ""),
        LOG_PATH=str(output / "logs"),
        CONCURRENCY_LIMIT="1",
        MAX_CONTEXT_BATCH_SIZE="1",
        RTP_V41_N9_GRAPH_REPORT_DIR=str(reports),
        RTP_V41_N9_GRAPH_CHECK_RESTORE="1" if async_mode == "0" else "0",
        RTP_V41_SHARED_EVIDENCE_DIR=str(output / "forwards"),
        RTP_V41_SHARED_VALIDATE_STATE="0",
        RTP_LLM_MTP_ASYNC_PREPARE="0",
        RTP_LLM_DROP_BROAD_SYNC="0",
    )
    argv = [
        sys.executable, "-B", "-m", "rtp_llm.tools.v41.launch", "decode",
        "--checkpoint-path", os.environ["RTP_V41_TEST_CHECKPOINT"],
        "--prefill-address", prefill, "--decode-address", decode, "--start-port", str(port),
        "--rdma", "--kv-cache-sharded",
        "--ced-mode", "bounded", "--yoco-keep-tokens", "140", "--max-rollback-tokens", "0",
        "--max-seq-len", os.environ.get("MAX_SEQ_LEN", "512"),
        "--token-map-cache", str(output / "token-map.json"), "--shared-grouped-graph",
    ]
    if async_mode == "1":
        argv.append("--shared-graph-async")
    paths = [reports / f"rank{rank}-replay{step}.json" for rank in range(8) for step in (1, 2)]
    deadline = time.monotonic() + 6300
    with (output / "process.log").open("x") as log:
        process = subprocess.Popen(argv, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(output / "launch.json", {
            "pid": process.pid, "argv": argv, "role": "decode", "snapshot": snapshot, "async": async_mode == "1",
            "transport": "native RDMA CacheStore", "affinity": AFFINITY,
        })
        marker = None
        try:
            urls = [f"http://127.0.0.1:{port + rank * 9}" for rank in range(8)]
            wait_for(lambda: all(map(healthy, urls)), process, deadline, "native D readiness")
            write_json(output / "ready.json", {"listeners": urls})
            wait_for(lambda: all(p.exists() for p in paths), process, deadline, "two real runner replays")
            replay = validate_native_receipts(paths)
            if async_mode == "0":
                restored = reports / f"rank{replay['owner']}-restored-prefix.json"
                before = json.loads(restored.read_text())
                if (before["prefix_lengths"] != [257] or len(before["native_cache"]) != 51
                    or not any(item["nonzero_bytes"] for item in before["native_cache"].values())):
                    raise AssertionError("missing restored live prefix bytes before the first replay")
            write_json(output / "replays-verified.json", replay)
            wait_for(stop_path.exists, process, deadline, "L0 reception and coordinated stop")
            marker = _read_stop_marker(stop_path, output, snapshot["snapshot_id"])
        finally:
            primary_error = sys.exc_info()[1]
            lifecycle = {
                "launcher_pid": process.pid,
                "snapshot_id": snapshot["snapshot_id"],
                "stop_marker": marker,
                "launcher_running_at_start": process.poll() is None,
                "started_at": _now(),
                "primary_error": repr(primary_error) if primary_error is not None else None,
                "cleanup_error": None,
            }
            write_json(output / "teardown-start.json", lifecycle)
            try:
                status = stop_shared_owned(process, output)
            except Exception as cleanup_error:
                lifecycle["cleanup_error"] = repr(cleanup_error)
                if primary_error is None:
                    raise
            finally:
                lifecycle["finished_at"] = _now()
                lifecycle["exit_code"] = process.poll()
                write_json(output / "server-exit.json", {
                    "exit_code": process.poll(),
                    "cleanup_error": lifecycle["cleanup_error"],
                    "primary_error": lifecycle["primary_error"],
                })
                write_json(output / "teardown-finished.json", lifecycle)
    if status != 0:
        raise RuntimeError(f"N9 native D server failed: {status}")
    logs = [output / "process.log", *(output / "logs").rglob("*.log")]
    log = "\n".join(path.read_text(errors="replace") for path in logs)
    if "tcp messager init success" in log.lower() or "TcpMessager.cpp" in log:
        raise AssertionError("TCP CacheStore data transport was instantiated")
    if "rdma messager init success" not in log.lower():
        raise AssertionError("native D did not report RDMA CacheStore")
    if not re.search(r"rdma connection to .* connect success", log):
        raise AssertionError("missing actual D RDMA continuation connection")
    rdma_health = classify_rdma_failures(
        log, lifecycle, json.loads((output / "cleanup.json").read_text()),
        launcher_pid=process.pid, snapshot_id=snapshot["snapshot_id"],
        stop_path=str(stop_path), role="decode", output_path=str(output),
    )
    write_json(output / "rdma-health.json", rdma_health)
    if not rdma_health["teardown_qualified"] or rdma_health["failures"]:
        raise AssertionError("RDMA errors or unqualified D teardown")
    if not re.search(r"Capture for .* 1 begin\.", log):
        raise AssertionError("native runner did not log capture of bucket1")
    if f"-> useStreamAsync={async_mode}" not in log:
        raise AssertionError("public executor did not report the selected async mode")
    if f"RTP_LLM_DEVICE_INPUT={async_mode} -> enabled={async_mode}" not in log:
        raise AssertionError("public executor did not report the selected device-input mode")
    write_json(output / "summary.json", {
        **replay, "role": "decode", "native_exit": status, "async": async_mode == "1",
        "scope": "native grouped W4A8 Decode runner after native RDMA; P and resources checked by L0",
        "FULL_VALIDATION": "DEFERRED_TO_N10",
    })


_DIRECT_DRAFT_TYPE = "v41_ppu_shared_dspark_direct_fixture"
_DIRECT_MODULE = "rtp_llm.models_py.modules.dsv41.ppu.test.graph_modes_role"


if os.environ.get("RTP_V41_N9_DIRECT_FIXTURE") == "1":
    if (os.environ.get("SP_MODEL_TYPE") != _DIRECT_DRAFT_TYPE
        or os.environ.get("MODEL_TYPE") != "deepseek_v41"
        or os.environ.get("ROLE_TYPE") != "DECODE"
        or os.environ.get("ENABLE_CUDA_GRAPH") != "0"):
        raise ValueError("N9 direct fixture requires the explicit eager D-only test profile")
    # Like the existing N8 test subtype, this runs again in spawn children
    # and makes only a distinct name visible to the normal lazy factory.
    from rtp_llm.model_factory_register import register_lazy_model

    register_lazy_model(_DIRECT_DRAFT_TYPE, _DIRECT_MODULE)

    if __name__ == _DIRECT_MODULE:
        from rtp_llm.model_factory_register import register_model
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_model import DeepSeekV41PPUDSparkModel
        from rtp_llm.models.deepseek_v41 import DeepSeekV41DSpark

        class NativeDirectDraftModel(DeepSeekV41PPUDSparkModel):
            def __init__(self, *args, target, **kwargs):
                super().__init__(*args, **kwargs)
                self._fixture_target = ref(target)
                self._cache_backed_inits = 0

            def _initialize_impl(self, resources):
                result = super()._initialize_impl(resources)
                if resources.kv_cache is None:
                    return result
                self._cache_backed_inits += 1
                if self._cache_backed_inits == 1:
                    return result
                if self._cache_backed_inits != 2:
                    raise RuntimeError("expected one proposal and one commit wrapper for the draft")
                from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_graph_modes import (
                    run_direct_device_fixture,
                )

                # MtpExecutor constructs the cache-backed target first,
                # then both D wrappers over this same Python draft object.
                run_direct_device_fixture(
                    self._fixture_target(), self, _direct_modes(),
                    Path(os.environ["RTP_V41_N9_DIRECT_REPORT_DIR"]),
                )
                return result

        class NativeDirectDraft(DeepSeekV41DSpark):
            def _create_python_model(self):
                owner = self._weight_alias_owner
                if owner is None or owner.py_model is None:
                    raise ValueError("N9 direct fixture requires its native target vocab owner")
                self.py_model = NativeDirectDraftModel(
                    self.model_config, self.parallelism_config, self.weight, self.moe_config,
                    max_generate_batch_size=self.max_generate_batch_size,
                    fmha_config=self.fmha_config, py_hw_kernel_config=self.hw_kernel_config,
                    device_resource_config=self.device_resource_config, target=owner.py_model,
                )
                self.py_model.shared_vocab_owner = owner.weight

        register_model(_DIRECT_DRAFT_TYPE, NativeDirectDraft)


def _direct_modes():
    from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_graph_modes import DIRECT_MODES

    selected = os.environ.get("RTP_V41_N9_DIRECT_MODE", "all")
    if selected == "all":
        return DIRECT_MODES
    modes = tuple(selected.split(","))
    if len(set(modes)) != len(modes) or any(mode not in DIRECT_MODES for mode in modes):
        raise ValueError(f"invalid N9 direct fixture mode {selected!r}")
    return modes


def direct_server():
    if os.environ.get("RTP_V41_N9_DIRECT_FIXTURE") != "1":
        raise ValueError("N9 direct server requires its isolated fixture selector")
    sys.argv[1:] = []
    from rtp_llm.start_server import main as serve

    serve()


def direct_main():
    from rtp_llm.tools.v41.launch import preset_env
    from rtp_llm.tools.v41.preflight import checkpoint_metadata

    modes = _direct_modes()
    source = Path(os.environ["RTP_V41_TEST_SOURCE_ROOT"]).resolve()
    runtime = Path(os.environ["RTP_V41_TEST_RUNTIME"]).resolve()
    snapshot = json.loads((source / ".v41_snapshot.json").read_text())
    bound = json.loads((runtime / ".v41_snapshot.json").read_text())
    for key in ("snapshot_id", "source_sha256", "public_commit", "internal_commit"):
        if snapshot[key] != bound[key]:
            raise ValueError(f"source/runtime mismatch: {key}")
    checkpoint = checkpoint_metadata(os.environ["RTP_V41_TEST_CHECKPOINT"])
    if checkpoint["index_sha256"] != "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8":
        raise ValueError("N9 fixture requires the qualified V4.1 checkpoint index")
    port = int(os.environ["RTP_V41_N9_DIRECT_PORT"])
    if not 1024 <= port <= 65400:
        raise ValueError("N9 direct fixture requires an explicit valid D base port")
    output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "shared-dspark-direct"
    reports = output / "ranks"
    reports.mkdir(parents=True)
    env = preset_env(
        "decode", os.environ["RTP_V41_TEST_CHECKPOINT"], rdma=True,
        prefill_parallel="dp", max_seq_len=512, start_port=port,
        fake_balance_expert=False, dspark_gamma=3, dspark_confidence="bounded",
        dspark_verify_mode="packed", dspark_verify_budget=1,
        dspark_budget_mode="fixed", yoco_keep_tokens=140, max_rollback_tokens=4,
    )
    env.update(
        MODEL_TYPE="deepseek_v41", SP_MODEL_TYPE=_DIRECT_DRAFT_TYPE,
        ENABLE_CUDA_GRAPH="0", RTP_LLM_STREAM_ASYNC="0", RTP_LLM_DEVICE_INPUT="0",
        RTP_LLM_MTP_ASYNC_PREPARE="0", RTP_LLM_MTP_ASYNC_DEVICE_STATE="0", RTP_LLM_DROP_BROAD_SYNC="0",
        # The D aux allocation is concurrency * (gamma + 1). The 134-row
        # seed needs 34 * 4 slots; direct Graph buckets still use B1/B2.
        CONCURRENCY_LIMIT="34", MAX_CONTEXT_BATCH_SIZE="2",
        RTP_V41_N9_DIRECT_FIXTURE="1", RTP_V41_N9_DIRECT_REPORT_DIR=str(reports),
        RTP_V41_PPU_PACKED_ATTENTION="0", RTP_V41_PPU_FUSED_QK_ROPE="0", RTP_V41_PPU_QKV_STREAMS="0",
        LOG_PATH=str(output / "logs"),
        PYTHONPATH=str(runtime) + ":" + os.environ.get("PYTHONPATH", ""),
    )
    for key in ("MODEL_SERVICE_CONFIG", "REMOTE_RPC_SERVER_IP", "REMOTE_SERVER_PORT",
                "RTP_V41_SHARED_EVIDENCE_DIR", "RTP_V41_SHARED_VALIDATE_STATE",
                "RTP_V41_N9_GRAPH_REPORT_DIR", "RTP_V41_N9_GRAPH_CHECK_RESTORE"):
        env.pop(key, None)
    argv = [sys.executable, "-B", "-m", _DIRECT_MODULE, "--direct-server"]
    with (output / "process.log").open("x") as log:
        process = subprocess.Popen(argv, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
        write_json(output / "launch.json", {
            "pid": process.pid, "argv": argv, "role": "decode", "snapshot": snapshot,
            "checkpoint": checkpoint, "modes": modes, "native_graph": False, "peer_request": False,
            "environment": {key: env[key] for key in (
                "MODEL_TYPE", "SP_MODEL_TYPE", "ENABLE_CUDA_GRAPH", "RTP_LLM_STREAM_ASYNC",
                "SP_TYPE", "GEN_NUM_PER_CIRCLE", "SP_DSPARK_VERIFY_MODE", "SP_DSPARK_VERIFY_BUDGET",
                "MAX_SEQ_LEN", "START_PORT", "CACHE_STORE_RDMA_MODE", "CONCURRENCY_LIMIT",
                "MAX_CONTEXT_BATCH_SIZE",
            )},
        })
        try:
            deadline = time.monotonic() + 5400
            paths = [reports / f"rank{rank}.json" for rank in range(8)]
            wait_for(lambda: all(path.exists() for path in paths), process, deadline, "N9 direct rank reports")
            rows = [json.loads(path.read_text()) for path in paths]
            if not all(
                row["rank"] == rank and row["ok"] and row["fixture_pages_restored"]
                and tuple(row["modes"]) == modes
                and all(mode["captured"] == 1 and mode["replayed"] == 2 for mode in row["modes"].values())
                for rank, row in enumerate(rows)
            ):
                raise AssertionError("direct DSpark mode evidence is incomplete")
            urls = [f"http://127.0.0.1:{port + rank * 9}" for rank in range(8)]
            wait_for(lambda: all(map(healthy, urls)), process, deadline, "D readiness after direct captures")
            write_json(output / "ready.json", {"listeners": urls})
        finally:
            status = stop_shared_owned(process, output)
            write_json(output / "server-exit.json", {"exit_code": status})
    if status != 0:
        raise RuntimeError(f"N9 direct D server failed: {status}")
    cleanup = json.loads((output / "cleanup.json").read_text())
    if cleanup["forced"] or cleanup["exit_code"] != 0:
        raise AssertionError("direct fixture required forced cleanup")
    log_text = (output / "process.log").read_text(errors="replace")
    joins = re.findall(r"Process ([^\n]+) joined with exitcode=(-?\d+)", log_text)
    ranks = [int(match.group(1)) for name, code in joins
             if code == "0" and (match := re.match(r"rank-(\d+) \(pid=\d+\)$", name))]
    http = [int(code) for name, code in joins if name.startswith("frontend_server_")]
    other = [int(code) for name, code in joins if not name.startswith("frontend_server_")]
    if sorted(ranks) != list(range(8)) or http != [-15] * 8 or other != [0] * 17:
        raise AssertionError("missing normal owned D process joins")
    write_json(output / "summary.json", {
        "ok": True, "SMOKE": "PASS_DIRECT_MODEL_GRAPH_FIDELITY", "native_exit": status,
        "snapshot": snapshot, "modes": modes, "ranks": sorted(ranks),
        "non_http_joins": len(other), "http_sigterm_joins": len(http), "forced": False,
        "scope": "D-only direct model capture, public budget/served acceptance/async outside fixture scope",
        "FULL_VALIDATION": "DEFERRED_TO_N10",
    })


if __name__ == "__main__":
    if sys.argv[1:] == ["--direct"]:
        direct_main()
    elif sys.argv[1:] == ["--direct-server"]:
        direct_server()
    else:
        main()
