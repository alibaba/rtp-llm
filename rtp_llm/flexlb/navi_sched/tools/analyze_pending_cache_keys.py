#!/usr/bin/env python3

import argparse
import bisect
import datetime
import json
import pathlib
import re
from collections import defaultdict


# Master records end with a logging bracket immediately after the final value.
# Keep that delimiter out of numeric fields while preserving bracketed hash lists.
SCALAR_PATTERN = re.compile(r"([a-z][a-z0-9_]*)=([^ \]]+)")
HASH_PATTERN = re.compile(r"block_hashes=\[([^]]*)\]")
AUDIT_MARKER = "[navi-cache-key-audit]"
LIFECYCLE_AUDIT_MARKER = "[navi-cache-lifecycle-audit]"
MASTER_DECISION_AUDIT_MARKER = "[navi-master-cache-decision-audit]"
MASTER_LIFECYCLE_MARKER = "[navi-master-pending-cache-lifecycle]"
LOG_TIME_PATTERN = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\]")
LIFECYCLE_HASH_FIELDS = (
    "event_hashes",
    "added",
    "removed",
    "added_full_ready",
    "removed_full_ready",
    "added_tail_ready",
    "removed_tail_ready",
)
LIFECYCLE_REQUIRED_FIELDS = {
    "event_time_us",
    "sequence",
    "event_type",
    "group_id",
    "cause",
    "cause_request_id",
    "event_hashes",
    "added",
    "removed",
}
REQUIRED_FIELDS = {
    "request_id",
    "schedule_batch_id",
    "local_batch_id",
    "arrival_time_us",
    "ready_time_us",
    "start_time_us",
    "finish_time_us",
    "cache_revision",
    "hybrid_tail_required",
    "total_tokens",
    "block_size",
    "master_hit_tokens",
    "agent_hit_tokens",
    "rtp_hit_tokens",
    "hybrid_candidate_blocks",
    "hybrid_flat_matched_blocks",
    "hybrid_min_full_reuse_blocks",
    "hybrid_selected_reuse_blocks",
    "outcome",
}
OPTIONAL_INTEGER_FIELDS = {
    "agent_pending_aware_hit_tokens",
}


def parse_int(value, field, path, line_number):
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            "invalid {} at {}:{}: {}".format(field, path, line_number, value)
        ) from error


def parse_log(path):
    rows = []
    host = path.name.split(".agent.log", 1)[0].replace("_", ".")
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if AUDIT_MARKER not in line:
            continue
        hash_match = HASH_PATTERN.search(line)
        if not hash_match:
            raise ValueError("missing block hashes at {}:{}".format(path, line_number))
        values = dict(SCALAR_PATTERN.findall(line))
        missing = REQUIRED_FIELDS - values.keys()
        if missing:
            raise ValueError(
                "missing {} at {}:{}".format(sorted(missing), path, line_number)
            )
        hashes = []
        if hash_match.group(1):
            hashes = [
                parse_int(value, "block_hash", path, line_number)
                for value in hash_match.group(1).split(",")
            ]
        integer_fields = REQUIRED_FIELDS - {"outcome"}
        row = {
            field: parse_int(values[field], field, path, line_number)
            for field in integer_fields
        }
        row.update(
            {
                field: parse_int(values[field], field, path, line_number)
                if field in values
                else None
                for field in OPTIONAL_INTEGER_FIELDS
            }
        )
        row.update(
            {
                "outcome": values["outcome"],
                "block_hashes": hashes,
                "host": host,
                "source_path": str(path),
                "line_number": line_number,
            }
        )
        validate_row(row)
        rows.append(row)
    return rows


def parse_lifecycle_log(path):
    events = []
    host = path.name.split(".agent.log", 1)[0].replace("_", ".")
    decoder = json.JSONDecoder()
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if LIFECYCLE_AUDIT_MARKER not in line:
            continue
        payload = line.split(LIFECYCLE_AUDIT_MARKER, 1)[1].lstrip()
        try:
            event, _ = decoder.raw_decode(payload)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "invalid lifecycle JSON at {}:{}".format(path, line_number)
            ) from error
        if not isinstance(event, dict):
            raise ValueError(
                "lifecycle payload is not an object at {}:{}".format(
                    path, line_number
                )
            )
        missing = LIFECYCLE_REQUIRED_FIELDS - event.keys()
        if missing:
            raise ValueError(
                "missing lifecycle {} at {}:{}".format(
                    sorted(missing), path, line_number
                )
            )
        for field in ("event_time_us", "sequence"):
            if not isinstance(event[field], int) or event[field] <= 0:
                raise ValueError(
                    "invalid lifecycle {} at {}:{}".format(
                        field, path, line_number
                    )
                )
        if event["event_type"] not in ("stored", "removed", "cleared"):
            raise ValueError(
                "invalid lifecycle event_type at {}:{}".format(path, line_number)
            )
        if event["group_id"] is not None and not isinstance(event["group_id"], int):
            raise ValueError(
                "invalid lifecycle group_id at {}:{}".format(path, line_number)
            )
        if not isinstance(event["cause"], str) or not event["cause"]:
            raise ValueError(
                "invalid lifecycle cause at {}:{}".format(path, line_number)
            )
        if event["cause_request_id"] is not None and not isinstance(
            event["cause_request_id"], str
        ):
            raise ValueError(
                "invalid lifecycle cause_request_id at {}:{}".format(
                    path, line_number
                )
            )
        for field in LIFECYCLE_HASH_FIELDS:
            values = event.setdefault(field, [])
            if not isinstance(values, list) or any(
                not isinstance(value, int) or not 0 <= value < (1 << 64)
                for value in values
            ):
                raise ValueError(
                    "invalid lifecycle {} at {}:{}".format(
                        field, path, line_number
                    )
                )
        for field in ("cause_batch_id", "cause_schedule_batch_id"):
            value = event.get(field)
            if value is not None and not isinstance(value, int):
                raise ValueError(
                    "invalid lifecycle {} at {}:{}".format(
                        field, path, line_number
                    )
                )
            event.setdefault(field, None)
        event.update(
            {
                "host": host,
                "source_path": str(path),
                "line_number": line_number,
            }
        )
        events.append(event)
    return sorted(events, key=lifecycle_event_sort_key)


def lifecycle_event_sort_key(event):
    return (
        event["event_time_us"],
        event["sequence"],
        event["source_path"],
        event["line_number"],
    )


def validate_row(row):
    block_size = row["block_size"]
    if block_size <= 0:
        raise ValueError("invalid block size in {}".format(row))
    for field in (
        "master_hit_tokens",
        "agent_hit_tokens",
        "agent_pending_aware_hit_tokens",
        "rtp_hit_tokens",
    ):
        value = row[field]
        if value is None:
            continue
        if value >= 0 and value % block_size != 0:
            raise ValueError("unaligned {} in {}".format(field, row))
        if value > len(row["block_hashes"]) * block_size:
            raise ValueError("{} exceeds block hashes in {}".format(field, row))
    if row["outcome"] == "finished" and row["rtp_hit_tokens"] < 0:
        raise ValueError("finished request has no actual hit in {}".format(row))
    if row["start_time_us"] > 0 and row["start_time_us"] < row["arrival_time_us"]:
        raise ValueError("request starts before arrival in {}".format(row))
    if row["finish_time_us"] < row["arrival_time_us"]:
        raise ValueError("request finishes before arrival in {}".format(row))


def build_candidate_source_index(rows):
    sources = sorted(rows, key=lambda row: row["finish_time_us"])
    return [row["finish_time_us"] for row in sources], sources


def candidate_sources(current, source_index, exclude_same_schedule=False):
    finish_times, sources = source_index
    begin = bisect.bisect_right(finish_times, current["arrival_time_us"])
    end = bisect.bisect_right(finish_times, current["start_time_us"])
    candidates = []
    for source in sources[begin:end]:
        if source["local_batch_id"] == current["local_batch_id"]:
            continue
        if (
            exclude_same_schedule
            and source["schedule_batch_id"] == current["schedule_batch_id"]
        ):
            continue
        if source["arrival_time_us"] > current["arrival_time_us"]:
            continue
        candidates.append(source)
    return candidates


def pending_candidate_sources(current, rows):
    candidates = []
    for source in rows:
        if source["local_batch_id"] == current["local_batch_id"]:
            continue
        if source["schedule_batch_id"] == current["schedule_batch_id"]:
            continue
        if not 0 < source["ready_time_us"] <= current["arrival_time_us"]:
            continue
        if source["finish_time_us"] < current["arrival_time_us"]:
            continue
        candidates.append(source)
    return candidates


def source_generated_hashes(source):
    actual_blocks = source["rtp_hit_tokens"] // source["block_size"]
    return source["block_hashes"][actual_blocks:]


def source_pending_hashes(source, linear_step, reservation_scope):
    block_size = source["block_size"]
    if reservation_scope == "produced":
        first_reserved_block = source["master_hit_tokens"] // block_size
    elif reservation_scope == "referenced":
        first_reserved_block = 0
    else:
        raise ValueError("invalid reservation scope: {}".format(reservation_scope))
    block_hashes = source["block_hashes"]
    block_count = len(block_hashes)
    block_aligned = source["total_tokens"] % block_size == 0
    full_ready = set()
    tail_ready = set()
    for block_index in range(first_reserved_block, block_count):
        block_hash = block_hashes[block_index]
        full_ready.add(block_hash)
        if not source["hybrid_tail_required"]:
            continue
        periodic_tail = (block_index + 1) % linear_step == 0
        final_block = block_index + 1 == block_count
        aligned_previous_block = (
            block_aligned
            and block_count >= 2
            and block_index + 2 == block_count
        )
        if periodic_tail or final_block or aligned_previous_block:
            tail_ready.add(block_hash)
    return full_ready, tail_ready


def matchable_block_count(row):
    block_count = len(row["block_hashes"])
    if (
        row["total_tokens"] > 0
        and row["total_tokens"] % row["block_size"] == 0
        and block_count == row["total_tokens"] // row["block_size"]
        and block_count > 0
    ):
        return block_count - 1
    return block_count


def match_pending_blocks(current, pending_full, pending_tail):
    block_size = current["block_size"]
    agent_blocks = current["agent_hit_tokens"] // block_size
    limit = matchable_block_count(current)
    matched_blocks = min(agent_blocks, limit)
    while (
        matched_blocks < limit
        and current["block_hashes"][matched_blocks] in pending_full
    ):
        matched_blocks += 1
    if current["hybrid_tail_required"]:
        # The Agent-selected prefix already ends at a real tail-ready block.
        while (
            matched_blocks > agent_blocks
            and current["block_hashes"][matched_blocks - 1] not in pending_tail
        ):
            matched_blocks -= 1
    return matched_blocks


def validate_host_policy(rows, linear_step):
    if linear_step <= 0:
        raise ValueError("linear_step must be positive")
    block_sizes = {row["block_size"] for row in rows}
    hybrid_modes = {row["hybrid_tail_required"] for row in rows}
    if len(block_sizes) > 1:
        raise ValueError("host contains mixed block sizes: {}".format(block_sizes))
    if not hybrid_modes.issubset({0, 1}) or len(hybrid_modes) > 1:
        raise ValueError("host contains invalid or mixed hybrid modes: {}".format(
            hybrid_modes
        ))


def build_lifecycle_event_index(events):
    ordered = sorted(events, key=lifecycle_event_sort_key)
    return [event["event_time_us"] for event in ordered], ordered


def lifecycle_events_between(begin_time_us, end_time_us, lifecycle_index):
    event_times, events = lifecycle_index
    begin = bisect.bisect_left(event_times, begin_time_us)
    end = bisect.bisect_right(event_times, end_time_us)
    return events[begin:end]


def lifecycle_events_in_window(current, lifecycle_index):
    return lifecycle_events_between(
        current["arrival_time_us"], current["start_time_us"], lifecycle_index
    )


def lifecycle_events_during_execution(current, lifecycle_index):
    return lifecycle_events_between(
        current["arrival_time_us"], current["finish_time_us"], lifecycle_index
    )


def _lifecycle_transitions(event, block_hash):
    transitions = []
    for field in LIFECYCLE_HASH_FIELDS[1:]:
        if block_hash in event[field]:
            transitions.append(field)
    return transitions


def _compact_lifecycle_event(event, block_hash):
    return {
        "event_time_us": event["event_time_us"],
        "sequence": event["sequence"],
        "event_type": event["event_type"],
        "group_id": event["group_id"],
        "cause": event["cause"],
        "cause_request_id": event["cause_request_id"],
        "cause_batch_id": event["cause_batch_id"],
        "cause_schedule_batch_id": event["cause_schedule_batch_id"],
        "event_hash_match": block_hash in event["event_hashes"],
        "logical_transitions": _lifecycle_transitions(event, block_hash),
        "source_path": event["source_path"],
        "line_number": event["line_number"],
    }


def attribute_lifecycle_hashes(
    current, window_events, begin_block, end_block, expected_direction
):
    evidence = []
    for block_index in range(begin_block, end_block):
        block_hash = current["block_hashes"][block_index]
        history = [
            _compact_lifecycle_event(event, block_hash)
            for event in window_events
            if block_hash in event["event_hashes"]
            or any(block_hash in event[field] for field in LIFECYCLE_HASH_FIELDS[1:])
        ]
        last_effective = None
        replayed_state = "unknown"
        for event in history:
            transitions = event["logical_transitions"]
            if any(value.startswith("added") for value in transitions):
                replayed_state = "present"
                last_effective = event
            if any(value.startswith("removed") for value in transitions):
                replayed_state = "absent"
                last_effective = event
        attributed_event = None
        if last_effective is not None and any(
            value.startswith(expected_direction)
            for value in last_effective["logical_transitions"]
        ):
            attributed_event = last_effective
        evidence.append(
            {
                "block_index": block_index,
                "block_hash": block_hash,
                "expected_direction": expected_direction,
                "exactly_explained": attributed_event is not None,
                "replayed_state_after_window": replayed_state,
                "cause_known": (
                    attributed_event is not None
                    and attributed_event["cause"] != "unknown"
                ),
                "attributed_event": attributed_event,
                "events": history,
            }
        )
    return evidence


def is_same_batch_store_attribution(current, evidence):
    event = evidence.get("attributed_event")
    return (
        evidence.get("exactly_explained", False)
        and event is not None
        and event["cause"] == "request_store"
        and event["cause_schedule_batch_id"] == current["schedule_batch_id"]
        and event["cause_request_id"] != str(current["request_id"])
    )


def analyze_host(
    rows, linear_step, reservation_scope="referenced", lifecycle_events=()
):
    validate_host_policy(rows, linear_step)
    finished = [row for row in rows if row["outcome"] == "finished"]
    source_index = build_candidate_source_index(finished)
    lifecycle_index = build_lifecycle_event_index(lifecycle_events)
    generated_sources = {}
    pending_sources = {}
    sources_by_hash = defaultdict(list)
    for row in finished:
        generated = source_generated_hashes(row)
        generated_sources[id(row)] = set(generated)
        pending_sources[id(row)] = source_pending_hashes(
            row, linear_step, reservation_scope
        )
        for block_hash in generated:
            sources_by_hash[block_hash].append(row)

    details = []
    for current in finished:
        block_size = current["block_size"]
        agent_blocks = current["agent_hit_tokens"] // block_size
        pending_aware_tokens = current["agent_pending_aware_hit_tokens"]
        pending_aware_blocks = (
            pending_aware_tokens // block_size
            if pending_aware_tokens is not None
            else None
        )
        actual_blocks = current["rtp_hit_tokens"] // block_size
        physical_sources = candidate_sources(current, source_index)
        overlay_sources = pending_candidate_sources(current, finished)
        physical_source_ids = {id(source) for source in physical_sources}
        overlay_full_hashes = set()
        overlay_tail_hashes = set()
        for source in overlay_sources:
            full_ready, tail_ready = pending_sources[id(source)]
            overlay_full_hashes.update(full_ready)
            overlay_tail_hashes.update(tail_ready)

        overlay_blocks = match_pending_blocks(
            current, overlay_full_hashes, overlay_tail_hashes
        )
        lifecycle_window = lifecycle_events_in_window(current, lifecycle_index)
        execution_lifecycle_window = lifecycle_events_during_execution(
            current, lifecycle_index
        )
        multi_hit_lifecycle = attribute_lifecycle_hashes(
            current,
            lifecycle_window,
            agent_blocks,
            actual_blocks,
            "added",
        )
        less_hit_lifecycle = attribute_lifecycle_hashes(
            current,
            lifecycle_window,
            actual_blocks,
            agent_blocks,
            "removed",
        )
        pending_multi_hit_lifecycle = attribute_lifecycle_hashes(
            current,
            execution_lifecycle_window,
            pending_aware_blocks if pending_aware_blocks is not None else actual_blocks,
            actual_blocks,
            "added",
        )
        pending_less_hit_lifecycle = attribute_lifecycle_hashes(
            current,
            execution_lifecycle_window,
            actual_blocks,
            pending_aware_blocks if pending_aware_blocks is not None else actual_blocks,
            "removed",
        )

        extra_hashes = current["block_hashes"][agent_blocks:actual_blocks]
        evidence = []
        for index, block_hash in enumerate(extra_hashes, agent_blocks):
            physical_matching = [
                source
                for source in sources_by_hash.get(block_hash, [])
                if id(source) in physical_source_ids
            ]
            overlay_matching = [
                source
                for source in overlay_sources
                if block_hash in pending_sources[id(source)][0]
            ]
            overlay_tail_matching = [
                source
                for source in overlay_matching
                if block_hash in pending_sources[id(source)][1]
            ]
            evidence.append(
                {
                    "block_index": index,
                    "block_hash": block_hash,
                    "physical_source_count": len(physical_matching),
                    "overlay_source_count": len(overlay_matching),
                    "overlay_tail_source_count": len(overlay_tail_matching),
                    "sources": [
                        {
                            "request_id": source["request_id"],
                            "schedule_batch_id": source["schedule_batch_id"],
                            "local_batch_id": source["local_batch_id"],
                            "arrival_time_us": source["arrival_time_us"],
                            "finish_time_us": source["finish_time_us"],
                            "actual_hit_blocks": source["rtp_hit_tokens"]
                            // source["block_size"],
                            "hash_index": source["block_hashes"].index(block_hash),
                            "same_schedule_batch": source["schedule_batch_id"]
                            == current["schedule_batch_id"],
                        }
                        for source in physical_matching[:8]
                    ],
                    "pending_sources": [
                        {
                            "request_id": source["request_id"],
                            "schedule_batch_id": source["schedule_batch_id"],
                            "local_batch_id": source["local_batch_id"],
                            "arrival_time_us": source["arrival_time_us"],
                            "ready_time_us": source["ready_time_us"],
                            "start_time_us": source["start_time_us"],
                            "finish_time_us": source["finish_time_us"],
                            "hash_index": source["block_hashes"].index(block_hash),
                        }
                        for source in overlay_matching[:8]
                    ],
                }
            )

        overlay_error_blocks = overlay_blocks - actual_blocks
        details.append(
            {
                **current,
                "queue_from_arrival_ms": max(
                    0, current["start_time_us"] - current["arrival_time_us"]
                )
                / 1000.0,
                "agent_hit_blocks": agent_blocks,
                "agent_physical_hit_blocks": agent_blocks,
                "agent_pending_aware_hit_blocks": pending_aware_blocks,
                "actual_hit_blocks": actual_blocks,
                "extra_hit_blocks": actual_blocks - agent_blocks,
                "physical_source_count": len(physical_sources),
                "overlay_source_count": len(overlay_sources),
                "overlay_full_ready_hash_count": len(overlay_full_hashes),
                "overlay_tail_ready_hash_count": len(overlay_tail_hashes),
                "matchable_block_count": matchable_block_count(current),
                "overlay_hit_blocks": overlay_blocks,
                "overlay_error_blocks": overlay_error_blocks,
                "extra_block_evidence": evidence,
                "lifecycle_event_count_in_window": len(lifecycle_window),
                "lifecycle_event_count_during_execution": len(
                    execution_lifecycle_window
                ),
                "multi_hit_lifecycle_attribution": multi_hit_lifecycle,
                "less_hit_lifecycle_attribution": less_hit_lifecycle,
                "pending_multi_hit_lifecycle_attribution": (
                    pending_multi_hit_lifecycle
                ),
                "pending_less_hit_lifecycle_attribution": (
                    pending_less_hit_lifecycle
                ),
                "all_multi_hit_hashes_lifecycle_explained": all(
                    item["exactly_explained"] for item in multi_hit_lifecycle
                ),
                "all_less_hit_hashes_lifecycle_explained": all(
                    item["exactly_explained"] for item in less_hit_lifecycle
                ),
                "all_pending_multi_hit_hashes_lifecycle_explained": all(
                    item["exactly_explained"]
                    for item in pending_multi_hit_lifecycle
                ),
                "all_pending_less_hit_hashes_lifecycle_explained": all(
                    item["exactly_explained"]
                    for item in pending_less_hit_lifecycle
                ),
                "all_pending_multi_hit_hashes_same_batch_store_explained": all(
                    is_same_batch_store_attribution(current, item)
                    for item in pending_multi_hit_lifecycle
                ),
                "all_extra_blocks_physically_explained": all(
                    item["physical_source_count"] > 0 for item in evidence
                ),
                "all_extra_blocks_overlay_explained": all(
                    item["overlay_source_count"] > 0 for item in evidence
                ),
            }
        )
    return details


def summarize_prediction_accuracy(details, token_field):
    rows = [row for row in details if row.get(token_field) is not None]
    errors = [row[token_field] - row["rtp_hit_tokens"] for row in rows]
    return {
        "sample_count": len(rows),
        "exact_request_count": sum(error == 0 for error in errors),
        "exact_request_pct": sum(error == 0 for error in errors)
        / max(len(errors), 1)
        * 100.0,
        "underpredicted_request_count": sum(error < 0 for error in errors),
        "overpredicted_request_count": sum(error > 0 for error in errors),
        "bias_mean_tokens": sum(errors) / max(len(errors), 1),
        "mae_tokens": sum(abs(error) for error in errors) / max(len(errors), 1),
        "max_absolute_error_tokens": max((abs(error) for error in errors), default=0),
    }


def lifecycle_request_attribution(row):
    return {
        "host": row["host"],
        "request_id": row["request_id"],
        "schedule_batch_id": row["schedule_batch_id"],
        "local_batch_id": row["local_batch_id"],
        "arrival_time_us": row["arrival_time_us"],
        "start_time_us": row["start_time_us"],
        "agent_hit_blocks": row["agent_hit_blocks"],
        "actual_hit_blocks": row["actual_hit_blocks"],
        "lifecycle_event_count_in_window": row["lifecycle_event_count_in_window"],
        "lifecycle_event_count_during_execution": row[
            "lifecycle_event_count_during_execution"
        ],
        "multi_hit": row["multi_hit_lifecycle_attribution"],
        "less_hit": row["less_hit_lifecycle_attribution"],
        "pending_multi_hit": row["pending_multi_hit_lifecycle_attribution"],
        "pending_less_hit": row["pending_less_hit_lifecycle_attribution"],
        "all_multi_hit_hashes_explained": row[
            "all_multi_hit_hashes_lifecycle_explained"
        ],
        "all_less_hit_hashes_explained": row[
            "all_less_hit_hashes_lifecycle_explained"
        ],
        "all_pending_multi_hit_hashes_explained": row[
            "all_pending_multi_hit_hashes_lifecycle_explained"
        ],
        "all_pending_less_hit_hashes_explained": row[
            "all_pending_less_hit_hashes_lifecycle_explained"
        ],
        "all_pending_multi_hit_hashes_same_batch_store_explained": row[
            "all_pending_multi_hit_hashes_same_batch_store_explained"
        ],
        "source_path": row["source_path"],
        "line_number": row["line_number"],
    }


def summarize(details):
    mismatch = [row for row in details if row["extra_hit_blocks"] != 0]
    positive = [row for row in mismatch if row["extra_hit_blocks"] > 0]
    negative = [row for row in mismatch if row["extra_hit_blocks"] < 0]
    extra_blocks = sum(max(0, row["extra_hit_blocks"]) for row in positive)
    physically_explained_blocks = sum(
        sum(
            item["physical_source_count"] > 0
            for item in row["extra_block_evidence"]
        )
        for row in positive
    )
    overlay_explained_blocks = sum(
        sum(item["overlay_source_count"] > 0 for item in row["extra_block_evidence"])
        for row in positive
    )
    multi_hit_lifecycle = [
        item
        for row in details
        for item in row["multi_hit_lifecycle_attribution"]
    ]
    less_hit_lifecycle = [
        item
        for row in details
        for item in row["less_hit_lifecycle_attribution"]
    ]
    lifecycle_attributions = multi_hit_lifecycle + less_hit_lifecycle
    pending_multi_hit_lifecycle = [
        item
        for row in details
        for item in row["pending_multi_hit_lifecycle_attribution"]
    ]
    pending_less_hit_lifecycle = [
        item
        for row in details
        for item in row["pending_less_hit_lifecycle_attribution"]
    ]
    pending_lifecycle_attributions = (
        pending_multi_hit_lifecycle + pending_less_hit_lifecycle
    )
    overlay_errors = [row["overlay_error_blocks"] for row in details]
    return {
        "request_count": len(details),
        "prediction_accuracy": {
            "master": summarize_prediction_accuracy(details, "master_hit_tokens"),
            "agent_physical": summarize_prediction_accuracy(
                details, "agent_hit_tokens"
            ),
            "agent_pending_aware": summarize_prediction_accuracy(
                details, "agent_pending_aware_hit_tokens"
            ),
        },
        "mismatch_request_count": len(mismatch),
        "positive_mismatch_request_count": len(positive),
        "negative_mismatch_request_count": len(negative),
        "positive_extra_block_count": extra_blocks,
        "physically_explained_extra_block_count": physically_explained_blocks,
        "physically_explained_extra_block_pct": physically_explained_blocks
        / max(extra_blocks, 1)
        * 100.0,
        "overlay_explained_extra_block_count": overlay_explained_blocks,
        "overlay_explained_extra_block_pct": overlay_explained_blocks
        / max(extra_blocks, 1)
        * 100.0,
        "physically_explained_positive_request_count": sum(
            row["all_extra_blocks_physically_explained"] for row in positive
        ),
        "physically_explained_positive_request_pct": sum(
            row["all_extra_blocks_physically_explained"] for row in positive
        )
        / max(len(positive), 1)
        * 100.0,
        "overlay_explained_positive_request_count": sum(
            row["all_extra_blocks_overlay_explained"] for row in positive
        ),
        "overlay_explained_positive_request_pct": sum(
            row["all_extra_blocks_overlay_explained"] for row in positive
        )
        / max(len(positive), 1)
        * 100.0,
        "overlay_exact_request_count": sum(error == 0 for error in overlay_errors),
        "overlay_exact_request_pct": sum(error == 0 for error in overlay_errors)
        / max(len(overlay_errors), 1)
        * 100.0,
        "overlay_underpredicted_request_count": sum(
            error < 0 for error in overlay_errors
        ),
        "overlay_overpredicted_request_count": sum(
            error > 0 for error in overlay_errors
        ),
        "overlay_block_mae": sum(abs(error) for error in overlay_errors)
        / max(len(overlay_errors), 1),
        "lifecycle_multi_hit_hash_count": len(multi_hit_lifecycle),
        "lifecycle_explained_multi_hit_hash_count": sum(
            item["exactly_explained"] for item in multi_hit_lifecycle
        ),
        "lifecycle_less_hit_hash_count": len(less_hit_lifecycle),
        "lifecycle_explained_less_hit_hash_count": sum(
            item["exactly_explained"] for item in less_hit_lifecycle
        ),
        "lifecycle_attributed_unknown_cause_hash_count": sum(
            item["exactly_explained"] and not item["cause_known"]
            for item in lifecycle_attributions
        ),
        "pending_lifecycle_multi_hit_hash_count": len(
            pending_multi_hit_lifecycle
        ),
        "pending_lifecycle_explained_multi_hit_hash_count": sum(
            item["exactly_explained"] for item in pending_multi_hit_lifecycle
        ),
        "pending_same_batch_store_explained_multi_hit_hash_count": sum(
            is_same_batch_store_attribution(row, item)
            for row in details
            for item in row["pending_multi_hit_lifecycle_attribution"]
        ),
        "pending_lifecycle_less_hit_hash_count": len(pending_less_hit_lifecycle),
        "pending_lifecycle_explained_less_hit_hash_count": sum(
            item["exactly_explained"] for item in pending_less_hit_lifecycle
        ),
        "pending_lifecycle_attributed_unknown_cause_hash_count": sum(
            item["exactly_explained"] and not item["cause_known"]
            for item in pending_lifecycle_attributions
        ),
    }


def parse_log_time_us(line, path, line_number):
    match = LOG_TIME_PATTERN.search(line)
    if not match:
        raise ValueError("missing log timestamp at {}:{}".format(path, line_number))
    try:
        return int(
            datetime.datetime.strptime(
                match.group(1), "%Y-%m-%d %H:%M:%S.%f"
            ).timestamp()
            * 1_000_000
        )
    except ValueError as error:
        raise ValueError(
            "invalid log timestamp at {}:{}".format(path, line_number)
        ) from error


def parse_master_logs(paths):
    decisions = []
    events = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(errors="replace").splitlines(), 1
        ):
            if (
                MASTER_DECISION_AUDIT_MARKER not in line
                and MASTER_LIFECYCLE_MARKER not in line
            ):
                continue
            values = dict(SCALAR_PATTERN.findall(line))
            event_time_us = parse_log_time_us(line, path, line_number)
            location = {"source_path": str(path), "line_number": line_number}
            if MASTER_DECISION_AUDIT_MARKER in line:
                required = {
                    "request_id",
                    "schedule_time_us",
                    "batch_id",
                    "agent",
                    "pending_first_source_batch_id",
                    "pending_tail_source_batch_id",
                }
                missing = required - values.keys()
                if missing:
                    raise ValueError(
                        "missing {} at {}:{}".format(
                            sorted(missing), path, line_number
                        )
                    )
                decisions.append(
                    {
                        "request_id": parse_int(
                            values["request_id"], "request_id", path, line_number
                        ),
                        "schedule_time_us": parse_int(
                            values["schedule_time_us"],
                            "schedule_time_us",
                            path,
                            line_number,
                        ),
                        "batch_id": parse_int(
                            values["batch_id"], "batch_id", path, line_number
                        ),
                        "agent": values["agent"],
                        "pending_first_source_batch_id": parse_int(
                            values["pending_first_source_batch_id"],
                            "pending_first_source_batch_id",
                            path,
                            line_number,
                        ),
                        "pending_tail_source_batch_id": parse_int(
                            values["pending_tail_source_batch_id"],
                            "pending_tail_source_batch_id",
                            path,
                            line_number,
                        ),
                        **location,
                    }
                )
            else:
                required = {"phase", "agent"}
                missing = required - values.keys()
                if missing:
                    raise ValueError(
                        "missing {} at {}:{}".format(
                            sorted(missing), path, line_number
                        )
                    )
                event = {
                    "phase": values["phase"],
                    "agent": values["agent"],
                    "event_time_us": event_time_us,
                    **location,
                }
                for field in (
                    "batch_id",
                    "schedule_time_us",
                    "retired",
                    "expired_batches",
                ):
                    if field in values:
                        event[field] = parse_int(values[field], field, path, line_number)
                events.append(event)
    return decisions, sorted(events, key=lambda event: event["event_time_us"])


def normalize_agent_id(agent):
    agent = agent.replace("_", ".")
    host, separator, port = agent.rpartition(":")
    if separator and port.isdigit():
        return host
    return agent


def build_agent_schedule_index(rows_by_host):
    index = defaultdict(list)
    for host, rows in rows_by_host.items():
        agent = normalize_agent_id(host)
        for row in rows:
            if row["local_batch_id"] > 0:
                index[(agent, row["local_batch_id"])].append(row)
            if row["schedule_batch_id"] > 0:
                key = (agent, row["schedule_batch_id"])
                if row not in index[key]:
                    index[key].append(row)
    return index


def summarize_agent_schedule(rows):
    if not rows:
        return {
            "request_count": 0,
            "finished_request_count": 0,
            "first_arrival_time_us": 0,
            "first_start_time_us": 0,
            "last_finish_time_us": 0,
        }
    return {
        "request_count": len(rows),
        "finished_request_count": sum(row["outcome"] == "finished" for row in rows),
        "first_arrival_time_us": min(row["arrival_time_us"] for row in rows),
        "first_start_time_us": min(row["start_time_us"] for row in rows),
        "last_finish_time_us": max(row["finish_time_us"] for row in rows),
    }


def analyze_master_lifecycle(
    decisions, events, keep_time_ms, agent_schedule_index=None
):
    keep_time_us = int(keep_time_ms * 1000)
    active = defaultdict(dict)
    records = {}
    for event in events:
        agent = event["agent"]
        phase = event["phase"]
        if phase == "reserved":
            batch_id = event["batch_id"]
            record = {
                "agent": agent,
                "batch_id": batch_id,
                "schedule_time_us": event.get("schedule_time_us", 0),
                "status": "reserved",
                "events": [event],
            }
            active[agent][batch_id] = record
            records[(agent, batch_id)] = record
        elif phase == "all_requests_received":
            record = active[agent].get(event["batch_id"])
            if record:
                record["status"] = "all_requests_received"
                record["events"].append(event)
        elif phase == "batch_finished":
            batch_id = event["batch_id"]
            record = active[agent].pop(batch_id, None)
            if record:
                record["status"] = (
                    "batch_finished_retired"
                    if event.get("retired", 0)
                    else "batch_finished_unmatched"
                )
                record["events"].append(event)
            elif not event.get("retired", 0):
                records.setdefault(
                    (agent, batch_id),
                    {
                        "agent": agent,
                        "batch_id": batch_id,
                        "schedule_time_us": 0,
                        "status": "batch_finished_unmatched",
                        "events": [event],
                    },
                )
        elif phase in ("agent_handshake", "agent_stopping"):
            for record in active[agent].values():
                record["status"] = "cleared_" + phase
                record["events"].append(event)
            active[agent].clear()
        elif phase == "expired":
            cutoff_us = event["event_time_us"] - keep_time_us
            for batch_id, record in list(active[agent].items()):
                if (
                    record["schedule_time_us"] > 0
                    and record["schedule_time_us"] <= cutoff_us
                ):
                    record["status"] = "expired"
                    record["events"].append(event)
                    del active[agent][batch_id]

    for agent_records in active.values():
        for record in agent_records.values():
            record["status"] = "still_pending_at_log_end"

    source_rows = []
    agent_schedule_index = agent_schedule_index or {}
    for decision in decisions:
        source_ids = sorted(
            {
                source_id
                for source_id in (
                    decision["pending_first_source_batch_id"],
                    decision["pending_tail_source_batch_id"],
                )
                if source_id > 0
            }
        )
        for source_id in source_ids:
            record = records.get((decision["agent"], source_id))
            agent = normalize_agent_id(decision["agent"])
            source_execution = summarize_agent_schedule(
                agent_schedule_index.get((agent, source_id), [])
            )
            target_execution = summarize_agent_schedule(
                agent_schedule_index.get((agent, decision["batch_id"]), [])
            )
            source_finish_time_us = source_execution["last_finish_time_us"]
            target_start_time_us = target_execution["first_start_time_us"]
            source_rows.append(
                {
                    "request_id": decision["request_id"],
                    "decision_batch_id": decision["batch_id"],
                    "agent": decision["agent"],
                    "source_batch_id": source_id,
                    "source_status": record["status"] if record else "not_seen_in_log",
                    "source_schedule_time_us": (
                        record["schedule_time_us"] if record else 0
                    ),
                    "source_agent_execution": source_execution,
                    "target_agent_execution": target_execution,
                    "source_completed_before_target_started": (
                        source_finish_time_us > 0
                        and target_start_time_us > 0
                        and source_finish_time_us <= target_start_time_us
                    ),
                    "decision_source_path": decision["source_path"],
                    "decision_line_number": decision["line_number"],
                }
            )
    status_counts = defaultdict(int)
    for row in source_rows:
        status_counts[row["source_status"]] += 1
    return {
        "keep_time_ms": keep_time_ms,
        "reservation_count": len(records),
        "reservation_status_counts": {
            status: sum(record["status"] == status for record in records.values())
            for status in sorted({record["status"] for record in records.values()})
        },
        "decision_source_count": len(source_rows),
        "decision_source_status_counts": dict(sorted(status_counts.items())),
        "unresolved_decision_sources": [
            row
            for row in source_rows
            if row["source_status"]
            in ("not_seen_in_log", "still_pending_at_log_end", "all_requests_received")
        ],
        "decision_sources": source_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--linear-step", type=int, required=True)
    parser.add_argument(
        "--master-log", nargs="+", type=pathlib.Path,
        help="master log files containing pending-cache decision and lifecycle audits",
    )
    parser.add_argument("--pending-keep-time-ms", type=float, default=60000.0)
    parser.add_argument(
        "--reservation-scope",
        choices=("produced", "referenced"),
        default="referenced",
    )
    args = parser.parse_args()

    rows_by_host = defaultdict(list)
    lifecycle_by_host = defaultdict(list)
    for path in args.logs:
        for row in parse_log(path):
            rows_by_host[row["host"]].append(row)
        for event in parse_lifecycle_log(path):
            lifecycle_by_host[event["host"]].append(event)

    details = []
    for host, rows in rows_by_host.items():
        details.extend(
            analyze_host(
                rows,
                args.linear_step,
                args.reservation_scope,
                lifecycle_by_host.get(host, ()),
            )
        )
    positive = [row for row in details if row["extra_hit_blocks"] > 0]
    negative = [row for row in details if row["extra_hit_blocks"] < 0]
    report = {
        "effective_config": {
            "linear_step": args.linear_step,
            "reservation_scope": args.reservation_scope,
        },
        "overall": summarize(details),
        "by_host": {
            host: summarize([row for row in details if row["host"] == host])
            for host in sorted(rows_by_host)
        },
        "lifecycle_attribution": [
            lifecycle_request_attribution(row)
            for row in sorted(
                (
                    row
                    for row in details
                    if row["agent_hit_blocks"] != row["actual_hit_blocks"]
                ),
                key=lambda row: (
                    row["host"],
                    row["arrival_time_us"],
                    row["request_id"],
                ),
            )
        ],
        "unexplained_positive": sorted(
            [
                row
                for row in positive
                if not row["all_extra_blocks_overlay_explained"]
            ],
            key=lambda row: row["extra_hit_blocks"],
            reverse=True,
        )[: args.top],
        "overlay_overprediction": sorted(
            [row for row in details if row["overlay_error_blocks"] > 0],
            key=lambda row: row["overlay_error_blocks"],
            reverse=True,
        )[: args.top],
        "explained_positive": sorted(
            [row for row in positive if row["all_extra_blocks_overlay_explained"]],
            key=lambda row: row["extra_hit_blocks"],
            reverse=True,
        )[: args.top],
        "lifecycle_multi_hit": sorted(
            positive,
            key=lambda row: row["extra_hit_blocks"],
            reverse=True,
        )[: args.top],
        "lifecycle_less_hit": sorted(
            negative,
            key=lambda row: row["extra_hit_blocks"],
        )[: args.top],
        "same_batch_execution_positive": sorted(
            [
                row
                for row in details
                if row["agent_pending_aware_hit_blocks"] is not None
                and row["agent_pending_aware_hit_blocks"]
                < row["actual_hit_blocks"]
                and row[
                    "all_pending_multi_hit_hashes_same_batch_store_explained"
                ]
            ],
            key=lambda row: (
                row["actual_hit_blocks"]
                - row["agent_pending_aware_hit_blocks"]
            ),
            reverse=True,
        )[: args.top],
        "unexplained_pending_aware_positive": sorted(
            [
                row
                for row in details
                if row["agent_pending_aware_hit_blocks"] is not None
                and row["agent_pending_aware_hit_blocks"]
                < row["actual_hit_blocks"]
                and not row[
                    "all_pending_multi_hit_hashes_same_batch_store_explained"
                ]
            ],
            key=lambda row: (
                row["actual_hit_blocks"]
                - row["agent_pending_aware_hit_blocks"]
            ),
            reverse=True,
        )[: args.top],
    }
    if args.master_log:
        decisions, events = parse_master_logs(args.master_log)
        report["master_pending_lifecycle"] = analyze_master_lifecycle(
            decisions,
            events,
            args.pending_keep_time_ms,
            build_agent_schedule_index(rows_by_host),
        )
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
