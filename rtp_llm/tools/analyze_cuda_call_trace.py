"""Correlate CUDA call-recorder JSONL without inferring a deadlock from missing data.

Usage: python analyze_cuda_call_trace.py trace*.jsonl --limit 100
Use --open-snapshot for the recorder's independent, point-in-time open-call file.
SQLite bounds RAM usage when processing rolling traces from multiple ranks.
"""

import argparse
import heapq
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path


def _integer(record, key, default=0):
    value = record.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, not bool")
    return int(value)


def _call_key(record):
    call_id = str(record.get("call_id", "0"))
    if call_id == "0":
        call_id = "correlation:" + str(record.get("correlation_id", 0))
    return ":".join((str(record.get("domain", 0)), str(record.get("tid", 0)), call_id))


def analyze(paths, snapshots=(), limit=50):
    if limit < 1:
        raise ValueError("limit must be positive")
    counts = Counter()
    issues = []
    issue_count = 0
    issue_counts = Counter()
    trace_ids = set()
    metadata = []
    metadata_seen = set()

    def issue(reason, **details):
        nonlocal issue_count
        issue_count += 1
        issue_counts[reason] += 1
        if len(issues) < limit:
            issues.append({"reason": reason, **details})

    with tempfile.TemporaryDirectory(prefix="cuda-call-trace-") as directory:
        db = sqlite3.connect(str(Path(directory) / "records.sqlite"))
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA cache_size=-8192")
        db.execute(
            "CREATE TABLE records (trace TEXT, seq INTEGER, type TEXT, "
            "call_key TEXT, correlation INTEGER, runtime_correlation INTEGER, "
            "context INTEGER, event INTEGER, generation TEXT, activity_kind INTEGER, "
            "producer_seq INTEGER, payload TEXT, "
            "PRIMARY KEY (trace, seq))"
        )
        for source in paths:
            with Path(source).open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        trace = record["trace_id"]
                        if not isinstance(trace, str) or not trace:
                            raise ValueError("missing trace_id")
                        kind = record["type"]
                        # Segment headers describe a file generation, not a
                        # queued record, and deliberately have no writer_seq.
                        if kind == "metadata":
                            payload = json.dumps(record, sort_keys=True)
                            if payload not in metadata_seen:
                                metadata_seen.add(payload)
                                metadata.append(record)
                                trace_ids.add(trace)
                                counts[kind] += 1
                            continue
                        seq = _integer(record, "writer_seq", -1)
                        if seq < 0:
                            raise ValueError("missing writer_seq")
                        if not isinstance(kind, str):
                            raise ValueError("type must be a string")
                        payload = json.dumps(record, sort_keys=True)
                        values = (
                            trace,
                            seq,
                            kind,
                            _call_key(record),
                            _integer(record, "correlation_id"),
                            _integer(record, "runtime_correlation_id"),
                            _integer(record, "context_uid"),
                            _integer(record, "event_id"),
                            str(record.get("event_sync_id", "0")),
                            _integer(record, "activity_kind"),
                            _integer(record, "record_seq"),
                            payload,
                        )
                        previous = db.execute(
                            "SELECT payload FROM records WHERE trace=? AND seq=?",
                            (trace, seq),
                        ).fetchone()
                        if previous is not None:
                            if previous[0] != payload:
                                issue(
                                    "conflicting_record_identity", trace=trace, seq=seq
                                )
                            continue
                        db.execute(
                            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            values,
                        )
                        counts[kind] += 1
                        trace_ids.add(trace)
                        if kind in ("drop", "warning"):
                            issue("recorder_" + kind, record=record)
                        if kind == "callback_unavailable":
                            issue("recorder_callback_unavailable", record=record)
                        if kind == "api_enter" and not record.get(
                            "params_known", False
                        ):
                            counts["api_enter_parameters_unknown"] += 1
                    except (ValueError, KeyError, TypeError, OverflowError) as error:
                        issue(
                            "invalid_or_truncated_record",
                            source=str(source),
                            line=line_number,
                            error=str(error),
                        )
                    if line_number % 4096 == 0:
                        db.commit()
        db.commit()
        db.execute("CREATE INDEX call_lookup ON records(trace,type,call_key)")
        db.execute("CREATE INDEX correlation_lookup ON records(trace,type,correlation)")
        db.execute(
            "CREATE INDEX runtime_lookup ON records(trace,type,runtime_correlation)"
        )
        db.execute(
            "CREATE INDEX event_lookup ON records(trace,type,context,event,generation)"
        )
        db.execute("CREATE INDEX producer_lookup ON records(trace,producer_seq)")
        previous_trace, previous_seq = None, None
        for trace, seq in db.execute(
            "SELECT trace,seq FROM records ORDER BY trace,seq"
        ):
            if trace == previous_trace and seq != previous_seq + 1:
                issue(
                    "retained_sequence_gap",
                    trace=trace,
                    first_missing=previous_seq + 1,
                    last_missing=seq - 1,
                )
            if trace != previous_trace:
                first_retained = min(
                    (
                        _integer(header, "first_writer_seq")
                        for header in metadata
                        if header["trace_id"] == trace and "first_writer_seq" in header
                    ),
                    default=seq,
                )
                if first_retained > 1:
                    issue(
                        "trace_prefix_not_retained",
                        trace=trace,
                        first_retained=first_retained,
                    )
                elif seq > first_retained:
                    issue(
                        "retained_sequence_gap",
                        trace=trace,
                        first_missing=first_retained,
                        last_missing=seq - 1,
                    )
            previous_trace, previous_seq = trace, seq

        # Producers reserve sequence numbers before publication. Priority queues
        # and concurrent callbacks can reorder their writes, so a disk-order
        # comparison would invent loss. Sort/deduplicate the producer sequence,
        # and report only holes *between* supplied records, never a guessed tail.
        previous_trace, previous_seq = None, None
        producer_gap_count = 0
        producer_missing_count = 0
        largest_producer_gaps = []
        for trace, seq in db.execute(
            "SELECT DISTINCT trace,producer_seq FROM records "
            "WHERE producer_seq>0 ORDER BY trace,producer_seq"
        ):
            if trace == previous_trace and seq > previous_seq + 1:
                gap = {
                    "trace_id": trace,
                    "first_missing": str(previous_seq + 1),
                    "last_missing": str(seq - 1),
                    "missing_count": seq - previous_seq - 1,
                }
                producer_gap_count += 1
                producer_missing_count += gap["missing_count"]
                ranked_gap = (gap["missing_count"], trace, previous_seq + 1, seq - 1)
                if len(largest_producer_gaps) < limit:
                    heapq.heappush(largest_producer_gaps, ranked_gap)
                elif ranked_gap > largest_producer_gaps[0]:
                    heapq.heapreplace(largest_producer_gaps, ranked_gap)
                issue("producer_records_not_retained", **gap)
            previous_trace, previous_seq = trace, seq
        producer_gaps = [
            {
                "trace_id": trace,
                "first_missing": str(first),
                "last_missing": str(last),
                "missing_count": missing,
            }
            for missing, trace, first, last in sorted(
                largest_producer_gaps, reverse=True
            )
        ]

        def records(query, args=()):
            return [json.loads(row[0]) for row in db.execute(query, args)]

        def activities_for(call):
            correlation = _integer(call, "correlation_id")
            if not correlation:
                return []
            return records(
                "SELECT payload FROM records WHERE trace=? AND type='activity' "
                "AND (correlation=? OR runtime_correlation=?) ORDER BY seq LIMIT ?",
                (call["trace_id"], correlation, correlation, limit),
            )

        # Aggregate once: a lost/reused call identity can have many ENTERs.
        # Counting them again for every ENTER makes real lossy traces quadratic.
        db.execute(
            "CREATE TEMP TABLE call_summary AS SELECT trace,call_key, "
            "sum(CASE WHEN type='api_enter' THEN 1 ELSE 0 END) AS enter_count, "
            "max(CASE WHEN type='api_exit' THEN seq END) AS last_exit "
            "FROM records WHERE type IN ('api_enter','api_exit') "
            "GROUP BY trace,call_key"
        )
        db.execute(
            "CREATE UNIQUE INDEX call_summary_key ON call_summary(trace,call_key)"
        )
        open_query = (
            "FROM records a JOIN call_summary b "
            "ON b.trace=a.trace AND b.call_key=a.call_key "
            "WHERE a.type='api_enter' AND "
            "(b.last_exit IS NULL OR b.last_exit<=a.seq OR b.enter_count>1)"
        )
        open_count = db.execute("SELECT count(*) " + open_query).fetchone()[0]
        open_calls = records(
            "SELECT a.payload " + open_query + " ORDER BY a.trace,a.seq LIMIT ?",
            (limit,),
        )
        for call in open_calls:
            call["activity_candidates"] = activities_for(call)
        # A repeated key may be correlation-ID reuse or lost records. Never
        # silently pretend that the surviving EXIT proves every ENTER completed.
        repeated = db.execute(
            "SELECT trace,call_key,enter_count FROM call_summary "
            "WHERE enter_count>1 ORDER BY trace,call_key LIMIT ?",
            (limit,),
        ).fetchall()
        for trace, call_key, count in repeated:
            issue(
                "ambiguous_call_identity", trace=trace, call_key=call_key, count=count
            )

        correlations = []
        for activity in records(
            "SELECT payload FROM records WHERE type='activity' "
            # CUPTI MEMCPY=1, MEMSET=2, CONCURRENT_KERNEL=10. API activities
            # are host timings and must not crowd GPU work out of this report.
            "AND activity_kind IN (1,2,10) "
            "AND (correlation>0 OR runtime_correlation>0) ORDER BY trace,seq LIMIT ?",
            (limit,),
        ):
            ids = {
                _integer(activity, "correlation_id"),
                _integer(activity, "runtime_correlation_id"),
            } - {0}
            placeholders = ",".join("?" for _ in ids)
            calls = records(
                "SELECT payload FROM records WHERE trace=? AND type='api_enter' "
                f"AND correlation IN ({placeholders}) ORDER BY seq LIMIT ?",
                (activity["trace_id"], *sorted(ids), limit),
            )
            correlations.append({"activity": activity, "api_candidates": calls})

        generations = []
        for trace, context, event, generation in db.execute(
            "SELECT DISTINCT trace,context,event,generation FROM records "
            # CUDA_EVENT=36, SYNCHRONIZATION=38. Non-event synchronizations
            # use CUPTI_SYNCHRONIZATION_INVALID_VALUE (UINT32_MAX), not an
            # event generation shared by unrelated stream/context waits.
            "WHERE type='activity' AND activity_kind IN (36,38) "
            "AND event!=4294967295 AND generation!='4294967295' "
            "ORDER BY trace,context,event LIMIT ?",
            (limit,),
        ):
            generations.append(
                {
                    "trace_id": trace,
                    "context_uid": context,
                    "event_id": event,
                    "event_sync_id": generation,
                    "records": records(
                        "SELECT payload FROM records WHERE trace=? AND type='activity' "
                        "AND activity_kind IN (36,38) "
                        "AND context=? AND event=? AND generation=? ORDER BY seq LIMIT ?",
                        (trace, context, event, generation, limit),
                    ),
                }
            )

        for trace in sorted(trace_ids):
            if not any(record["trace_id"] == trace for record in metadata):
                issue("startup_metadata_not_retained", trace=trace)
        snapshots_out = []
        for source in snapshots:
            try:
                with Path(source).open(encoding="utf-8") as stream:
                    snapshot = json.load(stream)
                if not isinstance(snapshot, dict) or not snapshot.get("trace_id"):
                    raise ValueError("snapshot must contain a trace_id")
                unavailable_count = _integer(snapshot, "optional_callback_unavailable")
                unavailable = snapshot.get("unavailable_callbacks", [])
                if unavailable_count < 0 or not isinstance(unavailable, list):
                    raise ValueError("invalid optional callback subscription coverage")
                for callback in unavailable:
                    if not isinstance(callback, dict) or not isinstance(
                        callback.get("name"), str
                    ):
                        raise ValueError("unavailable callback must have a name")
                    for key in ("domain", "cbid", "result"):
                        _integer(callback, key)
                subscription_limited = unavailable_count > 0 or bool(unavailable)
                if subscription_limited:
                    issue(
                        "snapshot_subscription_coverage_limited",
                        source=str(source),
                        trace=snapshot["trace_id"],
                        optional_callback_unavailable=unavailable_count,
                        unavailable_callbacks=unavailable,
                    )
                loss = {
                    key: _integer(snapshot, key)
                    for key in (
                        "host_dropped",
                        "activity_dropped",
                        "pool_exhausted",
                        "callback_errors",
                        "open_overflow",
                        "orphan_exits",
                        "critical_dropped",
                        "bulk_dropped",
                        "active_state_overflow",
                        "untracked_api_calls",
                        "close_mismatches",
                        "snapshot_missed_slots",
                    )
                    if _integer(snapshot, key) > 0
                }
                if loss:
                    issue(
                        "snapshot_reports_loss",
                        source=str(source),
                        trace=snapshot["trace_id"],
                        counters=loss,
                    )
                state_source = snapshot.get("active_state_source", "queued_records")
                if state_source == "callback_slots":
                    state_limited = any(
                        _integer(snapshot, key) > 0
                        for key in (
                            "active_state_overflow",
                            "untracked_api_calls",
                            "snapshot_missed_slots",
                        )
                    )
                    # An EXIT that failed to close its active slot can leave
                    # an old ENTER behind. This is stronger than incomplete
                    # coverage and must take precedence over a missing slot.
                    if _integer(snapshot, "close_mismatches") > 0:
                        state_coverage = "callback_state_may_be_stale"
                        issue(
                            "callback_active_state_may_be_stale",
                            source=str(source),
                            trace=snapshot["trace_id"],
                            close_mismatches=_integer(snapshot, "close_mismatches"),
                        )
                    else:
                        state_coverage = (
                            "incomplete_callback_sample"
                            if state_limited
                            else "bounded_non_atomic_callback_sample"
                        )
                    if state_limited:
                        issue(
                            "active_state_coverage_limited",
                            source=str(source),
                            trace=snapshot["trace_id"],
                        )
                else:
                    state_coverage = "queue_derived_may_be_stale"
                    if (
                        snapshot.get("open_calls") or snapshot.get("open_scopes")
                    ) and any(
                        _integer(snapshot, key) > 0
                        for key in ("host_dropped", "open_overflow", "orphan_exits")
                    ):
                        issue(
                            "queue_derived_active_state_incomplete",
                            source=str(source),
                            trace=snapshot["trace_id"],
                        )
                # Keep the recorder's timestamp, coverage and process identity.
                # A later snapshot is not atomic with an earlier GPU/GDB sample.
                snapshots_out.append(
                    {
                        "source": str(source),
                        "snapshot": snapshot,
                        "loss_counters": loss,
                        "active_state_source": state_source,
                        "active_state_coverage": state_coverage,
                        "subscription_coverage": (
                            "optional_callbacks_unavailable"
                            if subscription_limited
                            else "no_optional_callback_gap_reported"
                        ),
                    }
                )
            except (ValueError, TypeError, OSError) as error:
                issue("invalid_open_snapshot", source=str(source), error=str(error))
        db.close()

    return {
        "schema": 2,
        "trace_ids": sorted(trace_ids),
        "record_counts": dict(counts),
        "metadata": metadata,
        "coverage_issue_count": issue_count,
        "coverage_issue_counts": dict(issue_counts),
        "coverage_issues": issues,
        "producer_sequence_gap_count": producer_gap_count,
        "producer_records_not_retained_count": producer_missing_count,
        "producer_sequence_gaps": producer_gaps,
        "producer_sequence_gaps_order": "largest missing_count first; at most limit entries",
        "open_in_retained_records_count": open_count,
        "open_in_retained_records": open_calls,
        "independent_open_snapshots": snapshots_out,
        "gpu_correlations": correlations,
        "event_generation_groups": generations,
        "output_limit_per_section": limit,
        "limitations": [
            "Missing EXIT or GPU activity alone does not prove a hang or an unsubmitted operation.",
            "Rolling files, buffer loss and unsupported API parameters can leave causal edges unknown.",
            "Producer sequence holes are missing from the supplied files, not by themselves proof of queue loss.",
            "Snapshot loss counters are cumulative; they do not date loss to a particular API or request.",
            "Unavailable optional callbacks are subscription coverage gaps, not queue loss "
            "or evidence that captured live slots are stale.",
            "Callback-slot snapshots are bounded non-atomic samples; disk-unmatched calls are a separate history view.",
            "A callback-slot record with writer_seq=0 is an independent state sample, "
            "not a retained JSONL ENTER; failed closes can leave stale entries.",
            "API EXIT is a host return, not GPU completion; activity timestamps use the CUPTI clock.",
            "Correlation matches are process-local candidates; repeated IDs must be disambiguated.",
            "Event generations are grouped by context/event/sync ID, never by wall-clock proximity.",
            "Raw stream 0 across processes does not establish the same stream or default-stream mode.",
            "Cross-rank local phase ordinals are not NCCL sequence numbers or a proven collective match.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "traces",
        nargs="+",
        type=Path,
        help="Recorder JSONL files, including retained rotations",
    )
    parser.add_argument("--open-snapshot", action="append", default=[], type=Path)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    result = analyze(args.traces, args.open_snapshot, args.limit)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
