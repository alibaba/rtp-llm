"""Check speculative penalty histories against committed and candidate tokens.

This checks causal prefixes, independent of the numerical penalty kernel. A
matching token set alone is insufficient: duplicate counts affect frequency
penalties and uncommitted future candidates must never enter an earlier row.
"""

import argparse
import json
from pathlib import Path

import torch
from kimi_k3_trace_audit import AuditError, audit_recorder, load_frame, require


def check_history(history, lengths, model_tokens, streams, propose_step):
    score_len = propose_step + 1
    require(propose_step > 0, "expected a speculative verify batch")
    require(history.ndim == 2, "history must be batch-major")
    require(lengths.ndim == 1, "lengths must be one-dimensional")
    require(lengths.numel() == history.size(0), "history/length row mismatch")
    require(
        model_tokens.numel() == len(streams) * score_len, "candidate layout mismatch"
    )
    model_tokens = model_tokens.reshape(len(streams), score_len)
    checked = set()
    failures = []
    for stream_index, (details, committed) in enumerate(streams):
        start, count = details["first_row"], details["row_count"]
        n = details["committed_length"]
        require(count == score_len, "stream score length mismatch")
        require(
            committed.ndim == 2 and committed.size(0) == 1, "expected untiled stream"
        )
        require(0 < n <= committed.size(1), "invalid committed length")
        require(start == stream_index * score_len, "unexpected stream row ordering")
        prefix = committed[0, :n]
        require(
            model_tokens[stream_index, 0].item() == prefix[-1].item(),
            "verify input does not begin with the last committed token",
        )
        for position in range(score_len):
            row = start + position
            require(
                row < history.size(0) and row not in checked, "invalid/duplicate row"
            )
            checked.add(row)
            expected = torch.cat((prefix, model_tokens[stream_index, 1 : position + 1]))
            observed_length = int(lengths[row])
            require(
                0 <= observed_length <= history.size(1), "history length out of bounds"
            )
            observed = history[row, :observed_length]
            shared = min(expected.numel(), observed.numel())
            differing = torch.nonzero(expected[:shared] != observed[:shared]).flatten()
            if observed_length != expected.numel() or differing.numel():
                first = int(differing[0]) if differing.numel() else shared
                failures.append(
                    {
                        "stream_id": details["stream_id"],
                        "row": row,
                        "spec_position": position,
                        "expected_length": expected.numel(),
                        "observed_length": observed_length,
                        "first_difference": first,
                        "expected_token": (
                            int(expected[first]) if first < expected.numel() else None
                        ),
                        "observed_token": (
                            int(observed[first]) if first < observed.numel() else None
                        ),
                    }
                )
    require(len(checked) == history.size(0), "unaccounted verify rows")
    return {
        "rows_checked": len(checked),
        "causal_history_matches": not failures,
        "failures": failures,
    }


def observations(directory):
    """Read complete logical observations after the integrity audit has passed."""
    current = None
    values = {}
    for line in (directory / "index.jsonl").read_text().splitlines():
        row = json.loads(line)
        frame = load_frame(directory / row["path"])
        metadata = frame["metadata"]
        if current != metadata["observation_id"]:
            current = metadata["observation_id"]
            values = {}
        for item in frame["tensors"]:
            require(
                item["name"] not in values, "duplicate tensor name within observation"
            )
            values[item["name"]] = item["value"]
        if metadata["trace_fragment"]["final"]:
            yield metadata, values


def audit_history(directory):
    directory = Path(directory)
    integrity = audit_recorder(directory)
    streams = {}
    reports = []
    for metadata, values in observations(directory):
        event = metadata.get("event")
        if event not in ("mtp.verify_history_rows", "mtp.verify_sampler_gathered"):
            continue
        scopes = [
            s for s in metadata["scopes"] if s["name"] == "mtp.gather_verify_sampler"
        ]
        require(len(scopes) == 1, "missing or ambiguous verify gather scope")
        scope = scopes[0]["id"]
        if event == "mtp.verify_history_rows":
            streams.setdefault(scope, []).append(
                (metadata["details"], values["committed_token_ids"])
            )
        else:
            require(scope in streams, "verify batch has no committed-history records")
            result = check_history(
                values["history"],
                values["sequence_lengths"],
                values["model_input_tokens"],
                streams.pop(scope),
                metadata["details"]["propose_step"],
            )
            reports.append(
                {"observation_id": metadata["observation_id"], "scope": scope, **result}
            )
    require(not streams, "unmatched committed-history records")
    require(bool(reports), "no speculative verify batches found")
    return {
        "identity": integrity["identity"],
        "integrity_verified": True,
        "causal_history_matches": all(r["causal_history_matches"] for r in reports),
        "coverage_verified": False,
        "note": "Checks gather histories only; kernel use, model accuracy and online causality are separate checks.",
        "batches": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recorder", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = audit_history(args.recorder)
    except (AuditError, KeyError, ValueError) as exc:
        report = {"audit_error": str(exc), "causal_history_matches": False}
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "batches"}))
    return 0 if report["causal_history_matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
