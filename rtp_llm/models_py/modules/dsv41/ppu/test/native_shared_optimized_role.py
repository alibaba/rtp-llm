"""Isolated N5 native fixture, reusing N4's owned server lifecycle."""

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


FUNCTIONAL_ENV = "RTP_V41_PPU_OPTIMIZED_FUNCTIONAL_SMOKE"


def functional_smoke_enabled():
    """Select the distinct functional contract before fixture side effects."""
    value = os.environ.get(FUNCTIONAL_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(FUNCTIONAL_ENV + " must be explicitly 0 or 1")
    if value == "0":
        return False
    for name in ("RTP_V41_PPU_OPTIMIZED_DIAGNOSTICS", "RTP_V41_PPU_OPTIMIZED_LATE_FOCUS"):
        other = os.environ.get(name, "0")
        if other not in ("0", "1"):
            raise ValueError(name + " must be explicitly 0 or 1")
        if other == "1":
            raise ValueError(FUNCTIONAL_ENV + "=1 requires " + name + "=0")
    return True


def native_smoke_model_class():
    from rtp_llm.models_py.modules.dsv41.ppu.shared_model import DeepSeekV41SharedPPUModel

    class NativeSharedOptimizedSmokeModel(DeepSeekV41SharedPPUModel):
        def _initialize_impl(self, init_resource):
            functional = functional_smoke_enabled()
            result = super()._initialize_impl(init_resource)
            if init_resource.kv_cache is None:
                return result
            if getattr(self, "_n5_fixture_ran", False):
                raise RuntimeError("N5 fixture must initialize exactly once")
            self._n5_fixture_ran = True
            import torch
            import torch.distributed as dist

            from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_shared_optimized import run_optimized

            path = Path(os.environ["RTP_V41_N5_REPORT_DIR"]) / f"rank{dist.get_rank()}.json"
            report = {"ok": False, "rank": dist.get_rank(), "SMOKE": "FAIL"}
            try:
                if functional:
                    from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_shared_optimized import run_functional

                    report.update(run_functional(self))
                else:
                    report.update(run_optimized(self))
                torch.cuda.synchronize()
                dist.barrier()
                report["SMOKE"] = "PASS_NATIVE_SHARED_OPTIMIZED"
                if functional:
                    report["SMOKE"] = "PASS_NATIVE_SHARED_OPTIMIZED_FUNCTIONAL"
            except Exception:
                report.update(ok=False, SMOKE="FAIL", error=traceback.format_exc())
                raise
            finally:
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                temporary.replace(path)
            return result

    return NativeSharedOptimizedSmokeModel


def server_main():
    from rtp_llm.models_py.modules.dsv41.ppu.backend import _native_optimized_smoke_selected

    if not _native_optimized_smoke_selected():
        raise ValueError("N5 fixture requires its explicit canonical PPU provider")
    sys.argv[1:] = []
    from rtp_llm.start_server import main

    main()


def main():
    functional = functional_smoke_enabled()
    if sys.argv[1:] == ["--server"]:
        return server_main()
    if sys.argv[1:]:
        raise ValueError("N5 fixed smoke accepts no matrix/step overrides")
    from rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_role import (
        healthy, stop_shared_owned, wait_for, write_json,
    )
    from rtp_llm.tools.v41.launch import preset_env
    from rtp_llm.tools.v41.preflight import checkpoint_metadata

    source = Path(os.environ["RTP_V41_TEST_SOURCE_ROOT"]).resolve()
    runtime = Path(os.environ["RTP_V41_TEST_RUNTIME"]).resolve()
    output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "shared-optimized"
    if functional:
        output = output.with_name("shared-optimized-functional")
    output.mkdir(exist_ok=False)
    reports_dir = output / "ranks"
    reports_dir.mkdir()
    snapshot = json.loads((source / ".v41_snapshot.json").read_text())
    bound = json.loads((runtime / ".v41_snapshot.json").read_text())
    for key in ("snapshot_id", "source_sha256", "public_commit", "internal_commit"):
        if snapshot[key] != bound[key]:
            raise ValueError(f"source/runtime mismatch: {key}")
    checkpoint = checkpoint_metadata(os.environ["RTP_V41_TEST_CHECKPOINT"])
    if checkpoint["index_sha256"] != "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8":
        raise ValueError("N5 checkpoint index identity changed")
    port = int(os.environ.get("RTP_V41_N5_START_PORT", "19190"))
    env = preset_env(
        "decode", os.environ["RTP_V41_TEST_CHECKPOINT"], rdma=True,
        prefill_parallel="dp", max_seq_len=512, start_port=port,
        ppu_model_path="shared", fake_balance_expert=False,
    )
    env.pop("CP_ROTATE_METHOD", None)
    env.update(
        MODEL_TYPE="deepseek_v41", RTP_V41_N5_NATIVE_OPTIMIZED_SMOKE="1",
        RTP_V41_PPU_MOE_BACKEND="megamoe", RTP_V41_PPU_MOE_PRECISION="w4a4",
        RTP_V41_PPU_PACKED_ATTENTION="1", RTP_V41_PPU_FUSED_QK_ROPE="1",
        RTP_V41_PPU_QKV_STREAMS="1", RTP_V41_N5_REPORT_DIR=str(reports_dir),
        LOG_PATH=str(output / "logs"), ENABLE_CUDA_GRAPH="0", RTP_LLM_STREAM_ASYNC="0",
        CONCURRENCY_LIMIT="1", RTP_V41_PPU_MAX_ROLLBACK_TOKENS="0",
        RTP_V41_PPU_YOCO_KEEP_TOKENS="140",
        RTP_V41_SHARED_EVIDENCE_DIR=str(output / "forwards"), RTP_V41_SHARED_VALIDATE_STATE="1",
        PYTHONPATH=str(runtime) + ":" + os.environ.get("PYTHONPATH", ""),
    )
    command = [sys.executable, "-B", "-m",
               "rtp_llm.models_py.modules.dsv41.ppu.test.native_shared_optimized_role", "--server"]
    write_json(output / "launch.json", {
        **({"fixture_mode": "FUNCTIONAL_CONTINUATION", "selection": {FUNCTIONAL_ENV: "1"},
            "numerical_alignment": "FAIL_SEPARATE_STRICT_CHECK"} if functional else {}),
        "argv": command, "snapshot": snapshot, "checkpoint": checkpoint,
        "role": "D single role", "native_runtime": str(runtime), "max_seq_len": 512,
        "forwards_by_rank": 5, "prompt_tokens": 32, "decode_tokens": 1,
        "environment": {key: env[key] for key in (
            "MODEL_TYPE", "RTP_V41_N5_NATIVE_OPTIMIZED_SMOKE", "RTP_V41_PPU_MODEL_PATH",
            "RTP_V41_PPU_MOE_BACKEND", "RTP_V41_PPU_MOE_PRECISION",
            "RTP_V41_PPU_PACKED_ATTENTION", "RTP_V41_PPU_FUSED_QK_ROPE", "RTP_V41_PPU_QKV_STREAMS",
            "ENABLE_CUDA_GRAPH", "RTP_LLM_STREAM_ASYNC", "DSV41_CED",
            "RTP_V41_PPU_MAX_ROLLBACK_TOKENS", "TP_SIZE", "DP_SIZE", "EP_SIZE",
            "RTP_V41_ENP_SIZE", "START_PORT", "PREFILL_CP_SIZE")},
    })
    pass_tag = "PASS_NATIVE_SHARED_OPTIMIZED"
    if functional:
        pass_tag = "PASS_NATIVE_SHARED_OPTIMIZED_FUNCTIONAL"
    reports = []
    with (output / "process.log").open("x") as log:
        process = subprocess.Popen(command, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 5400
            wait_for(lambda: all((reports_dir / f"rank{r}.json").exists() for r in range(8)),
                     process, deadline, "N5 full40 actual optimized routes on all8 ranks")
            reports = [json.loads((reports_dir / f"rank{r}.json").read_text()) for r in range(8)]
            if not all(r.get("ok") and r.get("SMOKE") == pass_tag
                       and r.get("model_forwards") == 5 for r in reports):
                raise AssertionError("N5 rank comparison failure; see per-rank reports")
            if functional and not all(
                r.get("fixture_mode") == "FUNCTIONAL_CONTINUATION"
                and r.get("numerical_alignment") == "FAIL_SEPARATE_STRICT_CHECK"
                and r.get("candidate_prefill_history_verified") is True for r in reports
            ):
                raise AssertionError("N5 functional contract/provenance failure")
            if not all(r["moe"]["backend"] == "megamoe" and r["moe"]["precision"] == "w4a4"
                       and r["streams"]["calls"] == 80 for r in reports):
                raise AssertionError("unexpected N5 compute backend/hits")
            urls = [f"http://127.0.0.1:{port + rank * 9}" for rank in range(8)]
            wait_for(lambda: all(map(healthy, urls)), process, deadline, "native readiness before normal stop")
        finally:
            status = stop_shared_owned(process, output)
    if status != 0:
        raise RuntimeError(f"N5 native server failed: {status}")
    write_json(output / "summary.json", {
        **({"SMOKE": "PASS_NATIVE_SHARED_OPTIMIZED_FUNCTIONAL", "fixture_mode": "FUNCTIONAL_CONTINUATION",
            "numerical_alignment": "FAIL_SEPARATE_STRICT_CHECK"} if functional else {}),
        "ok": True, "native_exit": status, "ranks": list(range(8)),
        "model_forwards_by_rank": [r["model_forwards"] for r in reports],
        "FULL_VALIDATION": "DEFERRED_TO_N10", "CP8_to_DP8_RDMA": "NOT_RUN",
        "resource_and_all_rank_join_reception": "L0 verifies final native logs/GPU/locks/owned processes",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
