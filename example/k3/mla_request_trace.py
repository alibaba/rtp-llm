"""Verify real request Graph replay scopes against their correlated GPU work."""
from __future__ import annotations

import json
from pathlib import Path
import re
import time


_REPLAY = re.compile(
    r"cuda_graph\.forward\((replayDecode|replayPrefill),B=(\d+),capture=(\d+),Q=(\d+),T=(\d+),fake=([01])\)"
)
_FILE = re.compile(r"mla_(small|large)_owner(\d+)_wr(\d+)_.*\.json$")
_PLAN = re.compile(r"\[FIA2A_PLAN\] world_rank=(\d+) TP=(\d+) B=(\d+) Q=(\d+) H=(\d+) S=(\d+) mode=(\w+) dtype=([\w.]+)")


def collect_plans(root: Path) -> dict:
    plans = {}
    for path in root.rglob("*.log"):
        with path.open(errors="replace") as stream:
            for line in stream:
                match = _PLAN.search(line)
                if match:
                    rank, tp, batch, queries, heads, splits = map(int, match.groups()[:6])
                    key = (rank, batch, queries, match[8])
                    value = dict(TP=tp, H=heads, S=splits, mode=match[7], dtype=match[8])
                    if key in plans and plans[key] != value:
                        raise ValueError(f"Conflicting FIA2A plan {key}")
                    plans[key] = value
    return plans


def graph_replays(path: Path) -> list[dict]:
    """Use CUDA launch correlation, never CPU-range containment for GPU work."""
    events = json.loads(path.read_text())["traceEvents"]
    gpu = {}
    for event in events:
        if event.get("ph") == "X" and event.get("cat") == "kernel":
            correlation = event.get("args", {}).get("correlation")
            gpu.setdefault(correlation, []).append(event)
    launches = [event for event in events if event.get("ph") == "X"
                and event.get("cat") == "cuda_runtime"
                and event.get("name", "").startswith("cudaGraphLaunch")]
    replays = []
    for event in events:
        match = _REPLAY.fullmatch(event.get("name", ""))
        if event.get("ph") != "X" or not match:
            continue
        begin, end = event["ts"], event["ts"] + event["dur"]
        calls = [call for call in launches if call["pid"] == event["pid"]
                 and call["tid"] == event["tid"] and begin <= call["ts"] < end]
        if len(calls) != 1:
            raise ValueError(f"{path.name}: Graph scope has {len(calls)} launch calls")
        correlation = calls[0]["args"]["correlation"]
        kernels = sorted(gpu.get(correlation, []), key=lambda item: item["ts"])
        # A queued launch may have no GPU activity in the recorded window.
        # Keep it as missing evidence; only correlated kernels can witness a
        # branch. One such launch must not erase other recorded replays.
        live, capture, queries, tokens, fake = map(int, match.groups()[1:])
        replays.append(dict(graph_kind=match[1], B_live=live, B_capture=capture, Q=queries, T_live=tokens,
                            fake=bool(fake), correlation=correlation, scope_ts_us=begin, kernels=kernels))
    if not replays:
        raise ValueError(f"{path.name}: no actual Decode Graph replay scopes")
    if not any(row["kernels"] for row in replays):
        raise ValueError(f"{path.name}: no GPU kernels correlated with Graph replays")
    return replays


def _unfused_transport(names: list[str]) -> str | None:
    combines = sum("_combine_a2a" in name for name in names)
    if not combines or not any("merge_local_splits" in name for name in names):
        return None
    pulls = sum("_a2a_pull" in name for name in names)
    if pulls >= combines and sum("PeerBarrier" in name for name in names) >= 2 * pulls:
        return "CUSTOM"
    if not pulls and any("ncclDev" in name and "SendRecv" in name for name in names):
        return "NCCL"
    return None


def _check_transport(actual: str, requested: str, plan: dict, row: dict) -> None:
    if requested == "AUTO":
        records = row["B_capture"] * row["Q"] * plan["H"]
        custom = (plan["TP"] in (4, 8) and 768 <= records <= 49152
                  and (records // plan["TP"]) % 4 == 0)
        expected = "CUSTOM" if custom else "NCCL"
    else:
        expected = requested
    if actual != expected:
        raise ValueError(f"GPU A2A transport {actual} disagrees with {requested}: expected {expected}")
    row["a2a_transport"] = actual


def verify_mla_traces(directory: Path, tp: int, dp: int, backend: str, plans: dict | None = None,
                      *, require_draft: bool = False, a2a_backend: str = "AUTO") -> dict:
    """Require independent small/large request-window evidence on every rank.

    Kernel assertions are kept separate from profile arming and capture logs.
    The caller waits for complete JSON files before running this verifier.
    """
    files = {}
    for path in directory.glob("mla_*_owner*_wr*.json"):
        match = _FILE.fullmatch(path.name)
        if match:
            stage, owner, rank = match.groups()
            owner, rank = int(owner), int(rank)
            if owner != rank // tp or rank >= tp * dp:
                raise ValueError(f"Trace owner/rank mapping mismatch: {path.name}")
            key = (stage, rank)
            if key in files:
                raise ValueError(f"Duplicate request profile window: {key}")
            files[key] = path
    expected = {(stage, rank) for stage in ("small", "large") for rank in range(tp * dp)}
    if set(files) != expected:
        raise ValueError(f"Request traces missing={sorted(expected-set(files))}, extra={sorted(set(files)-expected)}")
    observations = []
    incomplete_replays = []
    for (stage, rank), path in sorted(files.items()):
        replays = graph_replays(path)
        incomplete_replays.extend(
            dict(stage=stage, world_rank=rank, trace=str(path),
                 **{key: value for key, value in row.items() if key != "kernels"})
            for row in replays if not row["kernels"]
        )
        target = [row for row in replays if row["graph_kind"] == "replayDecode"
                  and row["Q"] == 4 and not row["fake"]]
        if not target:
            raise ValueError(f"{path.name}: no Q4 target Graph replay")
        witnessed = []
        for row in target:
            names = [event["name"] for event in row["kernels"]]
            fused = (any("_merge_splits_serial" in name for name in names)
                     and sum("PeerBarrier" in name for name in names) >= 2)
            transport = _unfused_transport(names)
            unfused = transport is not None
            if backend == "FIA2A":
                if not any("PageRRFusedMLAFP8" in name and "split_kv_kernel" in name for name in names):
                    continue
                if plans is not None:
                    plan = plans.get((rank, row["B_capture"], row["Q"], "torch.float8_e4m3fn"))
                    if not plan or plan["TP"] != tp or plan["H"] != 96:
                        raise ValueError(f"{path.name}: missing/mismatched FIA2A preparation mapping")
                    if (fused and (plan["S"] != 1 or plan["mode"] != "fused")) or (
                        unfused and (plan["S"] <= 1 or plan["mode"] != "unfused")
                    ):
                        raise ValueError(f"{path.name}: GPU branch disagrees with prepared split plan")
                    row.update(plan)
                    if unfused:
                        _check_transport(transport, a2a_backend, plan, row)
                if stage == "large" and fused:
                    if row["B_capture"] < 16:
                        raise ValueError(f"{path.name}: fused Graph is outside the declared B16 acceptance bucket")
                    witnessed.append(row)
                if stage == "small" and unfused:
                    witnessed.append(row)
            elif backend == "TOKENSPEED":
                if fused or unfused:
                    raise ValueError(f"{path.name}: FIA2A merge in a TokenSpeed round")
                if any("split_kv" in name and "PageRRFusedMLA" not in name for name in names) and (
                    stage == "small" or row["B_capture"] >= 16
                ):
                    witnessed.append(row)
            else:
                raise ValueError(f"Unknown MLA backend {backend}")
        if not witnessed:
            raise ValueError(f"{path.name}: requested {stage} {backend} branch was not witnessed on GPU")
        row = witnessed[0]
        observations.append(dict(stage=stage, world_rank=rank, owner=rank // tp,
                                 trace=str(path), replay_count=len(replays),
                                 **{key: value for key, value in row.items() if key != "kernels"},
                                 kernel_names=[event["name"] for event in row["kernels"]]))
        if require_draft:
            for phase, kind, queries in (("proposal", "replayDecode", 1),
                                         ("update", "replayPrefill", 4)):
                candidates = []
                for draft in replays:
                    if draft["graph_kind"] != kind or draft["Q"] != queries or draft["fake"]:
                        continue
                    names = [event["name"] for event in draft["kernels"]]
                    if backend == "FIA2A":
                        if not any("PageRRFusedMLABF16" in name and "split_kv_kernel" in name for name in names):
                            continue
                        plan = plans.get((rank, draft["B_capture"], queries, "torch.bfloat16")) if plans else None
                        if not plan or plan["TP"] != tp or plan["H"] != 96:
                            raise ValueError(f"{path.name}: missing BF16 {phase} preparation mapping")
                        fused = (any("_merge_splits_serial" in name for name in names)
                                 and sum("PeerBarrier" in name for name in names) >= 2)
                        transport = _unfused_transport(names)
                        unfused = transport is not None
                        if not ((plan["S"] == 1 and plan["mode"] == "fused" and fused)
                                or (plan["S"] > 1 and plan["mode"] == "unfused" and unfused)):
                            raise ValueError(f"{path.name}: BF16 {phase} GPU branch disagrees with plan")
                        draft.update(plan)
                        if unfused:
                            _check_transport(transport, a2a_backend, plan, draft)
                    elif not any("split_kv" in name and "PageRRFusedMLA" not in name for name in names):
                        continue
                    candidates.append(draft)
                if not candidates:
                    raise ValueError(f"{path.name}: no nonfake native MTP {phase} MLA on GPU")
                draft = candidates[0]
                observations.append(dict(stage=stage, phase=phase, world_rank=rank, owner=rank // tp,
                                         trace=str(path),
                                         **{key: value for key, value in draft.items() if key != "kernels"},
                                         kernel_names=[event["name"] for event in draft["kernels"]]))
    return dict(passed=True, backend=backend, a2a_backend=a2a_backend, tp=tp, dp=dp,
                source="actual request Graph scopes and CUDA launch-correlated GPU kernels",
                observations=observations, incomplete_replays=incomplete_replays)


def wait_and_verify_mla_traces(root: Path, tp: int, dp: int, backend: str, timeout_s=120,
                             *, a2a_backend: str = "AUTO") -> dict:
    directory = root / "mla-profile"
    deadline = time.monotonic() + timeout_s
    # Kineto saves asynchronously. Wait only for files/JSON completion; a
    # completed trace with the wrong branch fails immediately.
    while True:
        paths = list(directory.glob("mla_*_owner*_wr*.json"))
        complete = len(paths) >= 2 * tp * dp
        if complete:
            try:
                for path in paths:
                    json.loads(path.read_text())
            except json.JSONDecodeError:
                complete = False
        if complete:
            break
        if time.monotonic() >= deadline:
            raise ValueError(f"Timed out waiting for {2 * tp * dp} complete request traces in {directory}")
        time.sleep(2)
    result = verify_mla_traces(directory, tp, dp, backend,
                               collect_plans(root) if backend == "FIA2A" else None,
                               require_draft=True, a2a_backend=a2a_backend)
    (root / "mla-request-evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
