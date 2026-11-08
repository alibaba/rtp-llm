"""Isolated N4 native initialization fixture; one D role, no PD claim.

The test model runs after the normal framework loader and native KVCache
initialization. Production shared classes do not invoke this fixture.
"""

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

def native_smoke_model_class():
    # The parent starts from immutable SOURCE; import native model bindings
    # only inside the registered server child running in matching RUNTIME.
    from rtp_llm.models_py.modules.dsv41.ppu.model import DeepSeekV41PPUModel

    class NativeSharedGraphSmokeModel(DeepSeekV41PPUModel):
        def _initialize_impl(self, init_resource):
            result = super()._initialize_impl(init_resource)
            if init_resource.kv_cache is None:
                return result
            if getattr(self, "_n4_fixture_ran", False):
                raise RuntimeError("N4 native fixture must initialize exactly once")
            self._n4_fixture_ran = True
            import torch
            import torch.distributed as dist

            from rtp_llm.models_py.modules.dsv41.ppu.tools.driver_graph_fidelity import run_fidelity

            directory = Path(os.environ["RTP_V41_N4_REPORT_DIR"])
            path = directory / f"rank{dist.get_rank()}.json"
            report = {"ok": False, "rank": dist.get_rank(), "SMOKE": "FAIL"}
            try:
                report.update(run_fidelity(self))
                torch.cuda.synchronize()
                dist.barrier()
                report["SMOKE"] = "PASS_NATIVE_SHARED_GRAPH_FIDELITY"
            except Exception:
                report["ok"] = False
                report["SMOKE"] = "FAIL"
                report["error"] = traceback.format_exc()
                raise
            finally:
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(report, indent=2) + "\n")
                temporary.replace(path)
            return result

    return NativeSharedGraphSmokeModel


def server_main():
    from rtp_llm.models_py.modules.dsv41.ppu.backend import _native_graph_smoke_selected

    if not _native_graph_smoke_selected():
        raise ValueError("N4 fixture requires its explicit canonical PPU test provider")
    # Consume our dispatch sentinel before the public server parses its argv.
    sys.argv[1:] = []
    from rtp_llm.start_server import main

    main()


def main():
    if sys.argv[1:] == ["--server"]:
        return server_main()
    if sys.argv[1:]:
        raise ValueError("N4 fixed smoke accepts no matrix/step overrides")
    from rtp_llm.models_py.modules.dsv41.ppu.test.pd_rdma_role import (
        healthy, stop_shared_owned, wait_for, write_json,
    )
    from rtp_llm.tools.v41.launch import preset_env
    from rtp_llm.tools.v41.preflight import checkpoint_metadata

    source = Path(os.environ["RTP_V41_TEST_SOURCE_ROOT"]).resolve()
    runtime = Path(os.environ["RTP_V41_TEST_RUNTIME"]).resolve()
    output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "shared-graph"
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
        raise ValueError("N4 checkpoint index identity changed")
    port = int(os.environ.get("RTP_V41_N4_START_PORT", "19090"))
    env = preset_env(
        "decode", os.environ["RTP_V41_TEST_CHECKPOINT"], rdma=True,
        prefill_parallel="dp", max_seq_len=512, start_port=port,
        fake_balance_expert=False,
    )
    # The public parser's disabled enum default has no accepted string alias.
    env.pop("CP_ROTATE_METHOD", None)
    env.update(
        MODEL_TYPE="deepseek_v41", RTP_V41_N4_NATIVE_GRAPH_SMOKE="1",
        RTP_V41_PPU_MOE_BACKEND="megamoe", RTP_V41_PPU_MOE_PRECISION="w4a4",
        RTP_V41_N4_REPORT_DIR=str(reports_dir), LOG_PATH=str(output / "logs"),
        ENABLE_CUDA_GRAPH="0", RTP_LLM_STREAM_ASYNC="0",
        CONCURRENCY_LIMIT="1",
        RTP_V41_PPU_MAX_ROLLBACK_TOKENS="0", RTP_V41_PPU_YOCO_KEEP_TOKENS="140",
        RTP_V41_SHARED_EVIDENCE_DIR=str(output / "eager-forwards"),
        RTP_V41_SHARED_VALIDATE_STATE="1",
        PYTHONPATH=str(runtime) + ":" + os.environ.get("PYTHONPATH", ""),
        # Native auto-capture is off: the fixture itself owns exactly one
        # torch CUDAGraph and two replays, with no hidden bucket warmups.
    )
    command = [sys.executable, "-B", "-m",
               "rtp_llm.models_py.modules.dsv41.ppu.test.graph_fidelity_role", "--server"]
    write_json(output / "launch.json", {"argv": command, "snapshot": snapshot, "checkpoint": checkpoint,
                                        "role": "D single role", "capture": 1, "replays": 2,
                                        "native_runtime": str(runtime), "max_seq_len": 512,
                                        "context_parallel": {"method_environment": env.get("CP_ROTATE_METHOD"),
                                                             "method_default": "CPRotateMethod.DISABLED",
                                                             "prefill_cp_size": int(env["PREFILL_CP_SIZE"])},
                                        "environment": {key: env[key] for key in (
                                            "MODEL_TYPE", "RTP_V41_N4_NATIVE_GRAPH_SMOKE", "RTP_V41_PPU_MOE_BACKEND",
                                            "RTP_V41_PPU_MOE_PRECISION", "ENABLE_CUDA_GRAPH", "RTP_LLM_STREAM_ASYNC",
                                            "DSV41_CED", "DSV41_SWA_BOUNDED_REPLAY", "DSV41_ENCODER_SWA_REPLAY",
                                            "RTP_V41_PPU_MAX_ROLLBACK_TOKENS", "TP_SIZE", "DP_SIZE", "EP_SIZE",
                                            "RTP_V41_ENP_SIZE", "CACHE_STORE_RDMA_MODE", "START_PORT", "CONCURRENCY_LIMIT", "PREFILL_CP_SIZE")}})
    with (output / "process.log").open("x") as log:
        process = subprocess.Popen(command, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 5400
            wait_for(lambda: all((reports_dir / f"rank{r}.json").exists() for r in range(8)),
                     process, deadline, "N4 full40 graph fidelity on all8 native ranks")
            reports = [json.loads((reports_dir / f"rank{r}.json").read_text()) for r in range(8)]
            if not all(r.get("ok") and r.get("SMOKE") == "PASS_NATIVE_SHARED_GRAPH_FIDELITY"
                       and r.get("replay_count") == 2 and r.get("capture_forwards") == 1 for r in reports):
                raise AssertionError("N4 rank fidelity failure; see per-rank reports")
            if any(sum(r["steps"][step]["received_assignments"] for r in reports) <= 0 for step in range(2)):
                raise AssertionError("actual Graph replay has no MegaMoE received assignments")
            if not all(r["moe"]["backend"] == "megamoe" and r["moe"]["precision"] == "w4a4"
                       and r["moe"]["grouped_gemm_calls"] == 0
                       and all(b["size"] == 8 for b in r["enp_bindings"].values()) for r in reports):
                raise AssertionError("unexpected routed precision/backend or ENP topology")
            urls = [f"http://127.0.0.1:{port + rank * 9}" for rank in range(8)]
            wait_for(lambda: all(map(healthy, urls)), process, deadline, "native rank readiness before normal teardown")
        finally:
            status = stop_shared_owned(process, output)
    if status != 0:
        raise RuntimeError(f"N4 native server failed: {status}")
    write_json(output / "summary.json", {
        "ok": True, "native_exit": status, "ranks": list(range(8)),
        "capture_count_by_rank": [r["capture_forwards"] for r in reports],
        "replay_count_by_rank": [r["replay_count"] for r in reports],
        "CP8_to_DP8_RDMA": "NOT_RUN", "FULL_VALIDATION": "DEFERRED_TO_N10",
        "resource_and_all_rank_join_reception": "L0 verifies final native logs/GPU/locks/owned processes",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
