"""Measure only GPU activities correlated with model CUDA Graph launches."""

import argparse
import json
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


def trace_events(path):
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        buffer = stream.read(65536)
        offset = buffer.index("[", buffer.index('"traceEvents"')) + 1
        while True:
            while offset < len(buffer) and buffer[offset] in " \n\r\t,":
                offset += 1
            if offset < len(buffer) and buffer[offset] == "]":
                return
            try:
                event, offset = decoder.raw_decode(buffer, offset)
            except json.JSONDecodeError:
                tail = stream.read(1048576)
                if not tail:
                    raise ValueError("Truncated traceEvents JSON")
                buffer = buffer[offset:] + tail
                offset = 0
                continue
            yield event
            if offset > 1048576:
                buffer = buffer[offset:]
                offset = 0


def analyze(events, actual_batch=32, verify_width=4, require_contract=False):
    scopes = defaultdict(list)
    graph_scopes = defaultdict(list)
    launches = []
    modeling_scopes = defaultdict(list)
    gpu = defaultdict(list)
    eager = []
    for event in events:
        name, cat = event.get("name", ""), event.get("cat", "")
        if event.get("ph") != "X":
            continue
        thread = (event.get("pid"), event.get("tid"))
        if re.match(r"executor\.mtp\.decode_step\(decode_stream_size=\d+\)", name):
            scopes[thread].append(event)
        if name.startswith("cuda_graph.forward(replay"):
            graph_scopes[thread].append(event)
        elif name.startswith("cuda_graph.modeling("):
            modeling_scopes[thread].append(event)
        elif name == "cudaGraphLaunch":
            launches.append(event)
        elif name == "py_model.forward(normal)":
            eager.append(event)
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                gpu[correlation].append(event)
    # Chrome trace arrays are not necessarily ordered by timestamp.
    # Proposal ordinals must follow submission time on the model thread.
    launches.sort(key=lambda event: event["ts"])
    for mapping in (scopes, graph_scopes, modeling_scopes):
        for values in mapping.values():
            values.sort(key=lambda e: e["ts"])

    def containing(mapping, event):
        values = mapping.get((event.get("pid"), event.get("tid")), [])
        starts = [v["ts"] for v in values]
        index = bisect_right(starts, event["ts"]) - 1
        if (
            index >= 0
            and values[index]["ts"]
            <= event["ts"]
            < values[index]["ts"] + values[index]["dur"]
        ):
            return index, values[index]
        return None

    rounds = defaultdict(
        lambda: {
            "proposal": [],
            "verify": [],
            "update": [],
            "eager": 0,
            "invalid_graphs": 0,
        }
    )
    missing_gpu = 0
    for launch in launches:
        graph = containing(graph_scopes, launch)
        round_scope = containing(scopes, launch)
        if graph is None or round_scope is None:
            continue
        graph = graph[1]
        index, parent = round_scope
        key = (parent["pid"], parent["tid"], index)
        name = graph["name"]
        if "fake=1" in name:
            rounds[key]["invalid_graphs"] += 1
            continue
        evidence = containing(modeling_scopes, launch)
        geometry = (
            re.search(
                r"logical_b=(\d+),physical_b=(\d+),q=(\d+),bucket=(\d+)",
                evidence[1]["name"],
            )
            if evidence
            else None
        )
        captured = (
            re.search(r"capture_b=(\d+),capture_t=(\d+)", evidence[1]["name"])
            if evidence
            else None
        )
        if require_contract:
            if geometry is None or captured is None:
                rounds[key]["invalid_graphs"] += 1
                continue
            logical_b, physical_b, q, _ = map(int, geometry.groups())
            capture_b, capture_t = map(int, captured.groups())
            if (
                logical_b != actual_batch
                or physical_b != actual_batch
                or capture_b != actual_batch
                or capture_t != actual_batch * q
            ):
                rounds[key]["invalid_graphs"] += 1
                continue
        match = re.search(r"B=(\d+),capture=(\d+),Q=(\d+),T=(\d+)", name)
        if match:
            batch, bucket, width, tokens = map(int, match.groups())
            if batch != actual_batch or bucket != batch or tokens != batch * width:
                continue
        elif name == "cuda_graph.forward(replayPrefill)":
            if geometry is None:
                # An unlabeled historical replay is not proof of Q4 update.
                continue
            logical, batch, width, bucket = map(int, geometry.groups())
            tokens = batch * width
            if captured is None:
                continue
            capture_batch, capture_tokens = map(int, captured.groups())
            if (
                logical != actual_batch
                or batch != actual_batch
                or capture_batch != batch
                or capture_tokens != tokens
            ):
                continue
        else:
            continue
        phase = (
            "update"
            if "replayPrefill" in name
            else ("proposal" if width == 1 else "verify")
        )
        if require_contract and (width != q or tokens != physical_b * q):
            rounds[key]["invalid_graphs"] += 1
            continue
        if width not in (1, verify_width):
            continue
        correlated = gpu.get(launch.get("args", {}).get("correlation"), [])
        if not correlated:
            missing_gpu += 1
            continue
        begin = min(e["ts"] for e in correlated)
        end = max(e["ts"] + e["dur"] for e in correlated)
        rounds[key][phase].append(
            {
                "gpu_span_us": end - begin,
                "gpu_start_us": begin,
                "gpu_end_us": end,
                "activities": len(correlated),
                "launch_correlation": launch["args"]["correlation"],
                "actual_batch": batch,
                "q": width,
                "tokens": tokens,
                "bucket": bucket,
            }
        )
    for event in eager:
        parent = containing(scopes, event)
        if parent:
            index, scope = parent
            rounds[(scope["pid"], scope["tid"], index)]["eager"] += 1
    complete = []
    rejected = []
    for key, record in sorted(rounds.items()):
        reason = []
        if record["eager"]:
            reason.append("eager forward in Decode round")
        if record["invalid_graphs"]:
            reason.append(
                "fake request or unproven logical/capture geometry in Decode round"
            )
        expected = (("proposal", verify_width - 2), ("verify", 1), ("update", 1))
        for phase, count in expected:
            if len(record[phase]) != count:
                reason.append(f"{phase} count {len(record[phase])}, expected {count}")
        if reason:
            rejected.append({"round": key[-1], "reasons": reason})
            continue
        record["round"] = key[-1]
        parent = scopes[(key[0], key[1])][key[-1]]
        record["cpu_round_start_us"] = parent["ts"]
        record["cpu_round_end_us"] = parent["ts"] + parent["dur"]
        record["target_verify_gpu_us"] = record["verify"][0]["gpu_span_us"]
        record["mtp_modeling_gpu_us"] = sum(
            v["gpu_span_us"] for v in record["proposal"] + record["update"]
        )
        complete.append(record)
    return {
        "boundary": "CUDA Graph launch correlation, GPU span across all graph activities",
        "actual_batch": actual_batch,
        "verify_width": verify_width,
        "complete_rounds": complete,
        "rejected_rounds": rejected,
        "graph_launches_missing_gpu_activities": missing_gpu,
        "timing_excludes": [
            "input preparation",
            "framework broadcast",
            "sampling",
            "rejection",
            "bookkeeping",
            "output postprocessing",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--actual-batch", type=int, default=32)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument(
        "--require-contract",
        action="store_true",
        help="Require logical/physical batch and actual capture geometry for every model launch",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        trace_events(args.trace),
        args.actual_batch,
        args.n_step + 1,
        args.require_contract,
    )
    result["trace"] = str(args.trace.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "trace": args.trace.name,
                "complete_rounds": len(result["complete_rounds"]),
                "rejected_rounds": len(result["rejected_rounds"]),
                "missing_gpu": result["graph_launches_missing_gpu_activities"],
            }
        )
    )
