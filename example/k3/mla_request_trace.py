"""Verify real request Graph replay scopes against their correlated GPU work."""
from __future__ import annotations

import json
from pathlib import Path
import re
import time


_REPLAY = re.compile(
    r"cuda_graph\.forward\((replayDecode|replayPrefill),B=(\d+),capture=(\d+),Q=(\d+),T=(\d+),fake=([01])\)"
)
_FILE = re.compile(r"mla_(small|mid|large)_owner(\d+)_wr(\d+)_.*\.json$")
_PLAN = re.compile(r"\[FIA2A_PLAN\] world_rank=(\d+) TP=(\d+) B=(\d+) Q=(\d+) H=(\d+) S=(\d+) mode=(\w+) dtype=([\w.]+).* q_layout=(\w+)")


def collect_plans(root: Path) -> dict:
    plans = {}
    for path in root.rglob("*.log"):
        with path.open(errors="replace") as stream:
            for line in stream:
                match = _PLAN.search(line)
                if match:
                    rank, tp, batch, queries, heads, splits = map(int, match.groups()[:6])
                    key = (rank, batch, queries, match[8])
                    value = dict(TP=tp, H=heads, S=splits, mode=match[7], dtype=match[8], q_layout=match[9])
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


def _fia2a_replay(row: dict, rank: int, tp: int, plans: dict, dtype: str,
                   a2a_backend: str, q_replicated: bool) -> bool:
    """Verify each MLA call, not a union of unrelated work across the model."""
    names = [event["name"] for event in row["kernels"]]
    producer = "PageRRFusedMLAFP8" if dtype == "torch.float8_e4m3fn" else "PageRRFusedMLABF16"
    indices = [i for i, name in enumerate(names) if producer in name and "split_kv_kernel" in name]
    if not indices:
        return False
    plan = plans.get((rank, row["B_capture"], row["Q"], dtype))
    if not plan or plan["TP"] != tp or plan["H"] != 96:
        raise ValueError("Missing/mismatched FIA2A preparation mapping")
    if (plan["S"] == 1) != (plan["mode"] == "fused"):
        raise ValueError("FIA2A AUTO split plan disagrees with fusion mode")
    layout = "token_major" if q_replicated else "head_major"
    if plan.get("q_layout") != layout:
        raise ValueError(f"FIA2A Q layout disagrees with qrep={int(q_replicated)}")
    row.update(plan)
    for index in indices:
        # Query AllGather and _pack are adjacent to their own producer. Other
        # K3 projections may AllGather even with replicated Q enabled.
        packed_q = index > 0 and names[index - 1] == "_pack"
        if packed_q == q_replicated:
            raise ValueError("FIA2A Q reorder disagrees with Q layout")
        if not q_replicated and not (index >= 2 and "AllGather" in names[index - 2]):
            raise ValueError("FIA2A head-major Q has no preceding query AllGather")
        tail = names[index + 1:]
        if plan["mode"] == "fused":
            if not tail or tail[0] != "_merge_splits_serial":
                raise ValueError("FIA2A fused producer has no destination merge")
        else:
            if plan["S"] > 1:
                if not tail or tail[0] != "merge_local_splits":
                    raise ValueError("FIA2A split producer has no local merge")
                tail = tail[1:]
            if tail and "PeerPullMerge" in tail[0]:
                transport = "CUSTOM"
            elif (len(tail) >= 3 and tail[0] == "_pack_a2a"
                  and "ncclDev" in tail[1] and "SendRecv" in tail[1]
                  and tail[2] == "_combine_a2a"):
                transport = "NCCL"
            else:
                raise ValueError("FIA2A producer has no complete GPU A2A transport")
            _check_transport(transport, a2a_backend, plan, row)
    row["mla_calls"] = len(indices)
    return True


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
                      *, require_draft: bool = False, a2a_backend: str = "AUTO",
                      q_replicated: bool = False) -> dict:
    """Require independent request-window evidence on every rank.

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
    stages = ("small", "mid", "large") if backend == "FIA2A" else ("small", "large")
    expected = {(stage, rank) for stage in stages for rank in range(tp * dp)}
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
            if backend == "FIA2A":
                if not _fia2a_replay(row, rank, tp, plans or {}, "torch.float8_e4m3fn",
                                      a2a_backend, q_replicated):
                    continue
                if stage == "large" and row["mode"] == "fused":
                    if row["B_capture"] < 16:
                        raise ValueError(f"{path.name}: fused Graph is outside the declared B16 acceptance bucket")
                    witnessed.append(row)
                if stage == "small" and row["mode"] == "unfused":
                    witnessed.append(row)
                if stage == "mid" and row["mode"] == "unfused" and row["B_capture"] in (2, 4, 8):
                    witnessed.append(row)
            elif backend == "TOKENSPEED":
                if any("PageRRFusedMLA" in name or "PeerPullMerge" in name for name in names):
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
        if require_draft and stage != "mid":
            for phase, kind, queries in (("proposal", "replayDecode", 1),
                                         ("update", "replayPrefill", 4)):
                candidates = []
                for draft in replays:
                    if draft["graph_kind"] != kind or draft["Q"] != queries or draft["fake"]:
                        continue
                    names = [event["name"] for event in draft["kernels"]]
                    if backend == "FIA2A":
                        if not _fia2a_replay(draft, rank, tp, plans or {}, "torch.bfloat16",
                                              a2a_backend, q_replicated):
                            continue
                        # Match the planned MTP branches, even when a profile
                        # window also contains legal replays from smaller batches.
                        if phase == "proposal" and (
                            draft["mode"] != "unfused"
                            or (stage == "large" and draft["B_capture"] < 16)
                        ):
                            continue
                        if phase == "update" and draft["mode"] != "fused":
                            continue
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
    return dict(passed=True, backend=backend, a2a_backend=a2a_backend,
                q_replicated=q_replicated, tp=tp, dp=dp,
                source="actual request Graph scopes and CUDA launch-correlated GPU kernels",
                observations=observations, incomplete_replays=incomplete_replays)


def wait_and_verify_mla_traces(root: Path, tp: int, dp: int, backend: str, timeout_s=120,
                             *, a2a_backend: str = "AUTO", q_replicated: bool = False) -> dict:
    directory = root / "mla-profile"
    deadline = time.monotonic() + timeout_s
    # Kineto saves asynchronously. Wait only for files/JSON completion; a
    # completed trace with the wrong branch fails immediately.
    expected_files = (3 if backend == "FIA2A" else 2) * tp * dp
    while True:
        paths = list(directory.glob("mla_*_owner*_wr*.json"))
        complete = len(paths) >= expected_files
        if complete:
            try:
                for path in paths:
                    json.loads(path.read_text())
            except json.JSONDecodeError:
                complete = False
        if complete:
            break
        if time.monotonic() >= deadline:
            raise ValueError(f"Timed out waiting for {expected_files} complete request traces in {directory}")
        time.sleep(2)
    result = verify_mla_traces(directory, tp, dp, backend,
                               collect_plans(root) if backend == "FIA2A" else None,
                               require_draft=True, a2a_backend=a2a_backend, q_replicated=q_replicated)
    (root / "mla-request-evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
