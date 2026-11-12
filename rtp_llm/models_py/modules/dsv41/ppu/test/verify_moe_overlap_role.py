"""D-only native checkpoint loader/controller for Verify FFN overlap.

The outer test wrapper already holds all eight GPU locks. Do not reacquire
DeviceResource here. This fixture issues no business request and needs no P.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


SELECTOR = "RTP_V41_TEST_VERIFY_MOE_OVERLAP"
REPORT_ENV = "RTP_V41_VERIFY_MOE_REPORT_DIR"
# Original Flash1017 identity, also pinned by graph_fidelity_role/test_model.
# PTPC here describes the loader's conversion, not a different checkpoint.
ORIGINAL_FLASH1017_INDEX_SHA256 = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"


def native_smoke_model_class():
    from rtp_llm.models_py.modules.dsv41.ppu.model import DeepSeekV41PPUModel

    class NativeVerifyMoEOverlapModel(DeepSeekV41PPUModel):
        def _initialize_impl(self, init_resource):
            result = super()._initialize_impl(init_resource)
            if init_resource.kv_cache is None:
                return result
            if getattr(self, "_verify_moe_fixture_ran", False):
                raise RuntimeError("native Verify MoE fixture must run exactly once")
            self._verify_moe_fixture_ran = True
            import torch
            import torch.distributed as dist
            from rtp_llm.models_py.modules.dsv41.ppu.test.verify_moe_overlap_native import run_native

            path = Path(os.environ[REPORT_ENV]) / f"rank{dist.get_rank()}.json"
            report = {"ok": False, "rank": dist.get_rank(), "SMOKE": "FAIL"}
            try:
                report.update(run_native(self))
                torch.cuda.synchronize()
                dist.barrier()
                report["SMOKE"] = "PASS_NATIVE_VERIFY_MOE_OVERLAP"
            except BaseException:
                report["ok"], report["SMOKE"] = False, "FAIL"
                report["error"] = traceback.format_exc()
                raise
            finally:
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(report, indent=2) + "\n")
                temporary.replace(path)
            return result

    return NativeVerifyMoEOverlapModel


def _selection(env):
    expected = {SELECTOR: "1", "RTP_V41_BACKEND": "ppu", "MODEL_TYPE": "deepseek_v41",
                "ROLE_TYPE": "DECODE", "RTP_V41_PPU_MOE_BACKEND": "megamoe",
                "RTP_V41_PPU_MOE_PRECISION": "w4a4"}
    if any(env.get(key) != value for key, value in expected.items()):
        raise ValueError("strict native Verify MoE PPU/DECODE/MegaMoE/W4A4 selector required")
    if any(key in env for key in ("RTP_V41_N4_NATIVE_GRAPH_SMOKE", "RTP_V41_N5_NATIVE_OPTIMIZED_SMOKE")):
        raise ValueError("native Verify overlap cannot be combined with N4/N5 selectors")


def server_main():
    _selection(os.environ)
    sys.argv[1:] = []
    from rtp_llm.start_server import main
    return main()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime(source, runtime):
    snapshot = json.loads((source / ".v41_snapshot.json").read_text())
    bound = json.loads((runtime / ".v41_snapshot.json").read_text())
    for key in ("snapshot_id", "source_sha256", "public_commit", "internal_commit"):
        if snapshot[key] != bound[key]:
            raise ValueError("native Verify source/runtime mismatch: " + key)
    manifest_path = runtime / "native-runtime.json"
    manifest = json.loads(manifest_path.read_text())
    libraries = manifest["libraries"]
    required = ("libth_transformer_config.so", "librtp_compute_ops.so", "libth_transformer.so",
                "libth_grammar_tokenizer_info.so")
    for name in required:
        record = libraries[name]
        relative = Path(record["runtime_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid native library runtime path")
        path = runtime / relative
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError("native library content mismatch: " + name)
    return snapshot, manifest


def main():
    if sys.argv[1:] == ["--server"]:
        return server_main()
    if sys.argv[1:]:
        raise ValueError("fixed native Verify MoE qualifier accepts only --server internally")
    from rtp_llm.models_py.modules.dsv41.ppu.test.pd_rdma_role import stop_shared_owned
    from rtp_llm.models_py.modules.dsv41.ppu.test.service_fixture import healthy, wait_for, write_json
    from rtp_llm.models_py.modules.dsv41.ppu.quant_mode import MXFP4_FP8_PTPC
    from rtp_llm.config.v41_runtime import MEGAMOE_DSPARK_ASYNC_FLAGS
    from rtp_llm.tools.v41.launch import preset_env
    from rtp_llm.tools.v41.preflight import CONFIG_SHA256, checkpoint_metadata

    source = Path(os.environ["RTP_V41_TEST_SOURCE_ROOT"]).resolve()
    runtime = Path(os.environ.get("RTP_V41_TEST_RUNTIME", str(source))).resolve()
    output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "verify-moe-overlap"
    output.mkdir(exist_ok=False)
    reports = output / "ranks"
    reports.mkdir()
    (output / "traces").mkdir()
    snapshot, manifest = _runtime(source, runtime)
    checkpoint = os.environ["RTP_V41_TEST_CHECKPOINT"]
    # Only metadata is hashed here. The normal framework loader owns weights.
    os.environ["RTP_V41_PPU_QUANT_MODE"] = MXFP4_FP8_PTPC
    identity = checkpoint_metadata(checkpoint, expected=CONFIG_SHA256)
    if identity["index_sha256"] != ORIGINAL_FLASH1017_INDEX_SHA256:
        raise ValueError("original Flash1017/PTPC index identity changed")
    timeout = int(os.environ.get("RTP_V41_VERIFY_MOE_TIMEOUT_SECONDS", "900"))
    if not 60 <= timeout <= 7200:
        raise ValueError("native Verify timeout must be between60 and7200 seconds")
    port = int(os.environ.get("RTP_V41_VERIFY_MOE_START_PORT", "19390"))
    env = preset_env("decode", checkpoint, rdma=True, prefill_parallel="dp", max_seq_len=512,
                     max_rollback_tokens=0, start_port=port, fake_balance_expert=False)
    env.pop("CP_ROTATE_METHOD", None)
    env.update({SELECTOR: "1", "RTP_V41_PPU_MOE_BACKEND": "megamoe",
                "RTP_V41_PPU_MOE_PRECISION": "w4a4", "RTP_V41_PPU_QUANT_MODE": MXFP4_FP8_PTPC,
                REPORT_ENV: str(reports), "RTP_V41_PPU_VERIFY_EXACT_SWIGLU": "1",
                "LOG_PATH": str(output / "logs"), "ENABLE_CUDA_GRAPH": "0",
                "FT_SERVER_TEST": "0", "CONCURRENCY_LIMIT": "8",
                "MAX_GENERATE_BATCH_SIZE": "8", "RTP_V41_PPU_MAX_ROLLBACK_TOKENS": "0",
                "PYTHONPATH": str(runtime) + ":" + os.environ.get("PYTHONPATH", "")})
    env.update({flag: "0" for flag in MEGAMOE_DSPARK_ASYNC_FLAGS})
    _selection(env)
    command = [sys.executable, "-B", "-m", "rtp_llm.models_py.modules.dsv41.ppu.test.verify_moe_overlap_role", "--server"]
    write_json(output / "launch.json", {"argv": command, "snapshot": snapshot,
        "native_runtime": manifest, "checkpoint": identity, "role": "D only; no P/no requests",
        "gpu_lock": "outer test wrapper holds eight devices; no nested lock",
        "effective_async_flags": {flag: env[flag] for flag in MEGAMOE_DSPARK_ASYNC_FLAGS},
        "timeout_seconds": timeout, "environment": {key: env.get(key) for key in (
            SELECTOR, "ROLE_TYPE", "TP_SIZE", "DP_SIZE", "EP_SIZE", "RTP_V41_PPU_QUANT_MODE",
            "RTP_V41_PPU_MOE_BACKEND", "RTP_V41_PPU_MOE_PRECISION", "ENABLE_CUDA_GRAPH",
            "CONCURRENCY_LIMIT", "MAX_GENERATE_BATCH_SIZE", "MAX_SEQ_LEN", "RTP_V41_PPU_MAX_ROLLBACK_TOKENS")}})
    rank_reports = None
    with (output / "process.log").open("x") as log:
        process = subprocess.Popen(command, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + timeout
            def all_reports():
                completed = []
                for rank in range(8):
                    path = reports / f"rank{rank}.json"
                    if path.exists():
                        value = json.loads(path.read_text())
                        if not value.get("ok") or value.get("SMOKE") == "FAIL":
                            raise AssertionError(f"native rank{rank} failed; receipt saved before owned teardown")
                        completed.append(value)
                return len(completed) == 8

            wait_for(all_reports,
                     process, deadline, "Verify FFN40 serial/overlap fidelity on all eight ranks")
            rank_reports = [json.loads((reports / f"rank{rank}.json").read_text()) for rank in range(8)]
            for rank, report in enumerate(rank_reports):
                if (not report.get("ok") or report.get("rank") != rank
                    or report.get("SMOKE") != "PASS_NATIVE_VERIFY_MOE_OVERLAP"
                    or report.get("graph_count") != 2 or not report.get("graph_cleanup") or not report.get("plan_closed")
                    or len(report.get("phases", [])) != 4
                    or report.get("sequential_profile_replays") != ["A", "B", "A"]):
                    raise AssertionError("native Verify rank receipt failure; inspect all saved reports")
                if not (output / "traces" / f"verify-moe-overlap-rank{rank}.trace.json").is_file():
                    raise AssertionError("missing native device trace")
                for phase in report["phases"]:
                    if not phase["raw_bytes_equal"] or not phase["finite"] or phase["world_received_assignments"] <= 0:
                        raise AssertionError("native Verify byte/finite/reception failure")
            if rank_reports[0]["phases"][2]["valid_rows"] != 0 or not any(r["phases"][2]["valid_rows"] > 0 for r in rank_reports[1:]):
                raise AssertionError("empty local origin/nonempty peer scenario was not exercised")
            urls = [f"http://127.0.0.1:{port + rank * 9}" for rank in range(8)]
            wait_for(lambda: all(map(healthy, urls)), process, deadline, "eight native ranks ready before normal teardown")
        finally:
            status = stop_shared_owned(process, output)
    if status != 0:
        raise RuntimeError(f"native Verify server failed normal teardown: {status}")
    write_json(output / "summary.json", {"ok": True, "native_exit": status, "ranks": list(range(8)),
        "graphs_per_rank": 2, "serial_byte_oracle": True, "actual_checkpoint_weights": True,
        "normal_native_shutdown": True, "trace_count": 8,
        "scope": "FFN-only full native WORLD8 progress/fidelity; fullmodel performance/attention/mHC NOT_RUN"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
