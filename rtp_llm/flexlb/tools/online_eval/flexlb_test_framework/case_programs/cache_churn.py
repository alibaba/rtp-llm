"""Pre-request hit rate and windowed holder masks under four-family churn; bounded LRU replay retry and capacity eviction."""

from ..case_config import output

METADATA = {
    "id": "cache_churn",
    "description": "Pre-request hit rate and windowed holder masks under four-family churn; bounded "
    "LRU replay retry and capacity eviction.",
    "category": "kv",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def hot_churn(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "w0_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w0_r0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w0_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w0_r0", "requests")},
    )
    case.step(
        "w0_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w0_r0_before", "snapshot"),
            "requests": output("w0_r0", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w0_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w0_r1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w0_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w0_r1", "requests")},
    )
    case.step(
        "w0_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w0_r1_before", "snapshot"),
            "requests": output("w0_r1", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w0_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w0_r2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w0_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w0_r2", "requests")},
    )
    case.step(
        "w0_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w0_r2_before", "snapshot"),
            "requests": output("w0_r2", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w0_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w0_r3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w0_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w0_r3", "requests")},
    )
    case.step(
        "w0_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w0_r3_before", "snapshot"),
            "requests": output("w0_r3", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w0_r4_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w0_r4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w0_r4_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w0_r4", "requests")},
    )
    case.step(
        "w0_r4_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w0_r4_before", "snapshot"),
            "requests": output("w0_r4", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w0_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w1_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w1_r0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w1_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w1_r0", "requests")},
    )
    case.step(
        "w1_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w1_r0_before", "snapshot"),
            "requests": output("w1_r0", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w1_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w1_r1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w1_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w1_r1", "requests")},
    )
    case.step(
        "w1_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w1_r1_before", "snapshot"),
            "requests": output("w1_r1", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w1_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w1_r2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w1_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w1_r2", "requests")},
    )
    case.step(
        "w1_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w1_r2_before", "snapshot"),
            "requests": output("w1_r2", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w1_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w1_r3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w1_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w1_r3", "requests")},
    )
    case.step(
        "w1_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w1_r3_before", "snapshot"),
            "requests": output("w1_r3", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w1_r4_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w1_r4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w1_r4_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w1_r4", "requests")},
    )
    case.step(
        "w1_r4_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w1_r4_before", "snapshot"),
            "requests": output("w1_r4", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w1_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w2_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w2_r0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w2_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w2_r0", "requests")},
    )
    case.step(
        "w2_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w2_r0_before", "snapshot"),
            "requests": output("w2_r0", "requests"),
            "keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w2_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w2_r1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w2_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w2_r1", "requests")},
    )
    case.step(
        "w2_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w2_r1_before", "snapshot"),
            "requests": output("w2_r1", "requests"),
            "keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w2_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w2_r2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w2_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w2_r2", "requests")},
    )
    case.step(
        "w2_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w2_r2_before", "snapshot"),
            "requests": output("w2_r2", "requests"),
            "keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w2_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w2_r3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w2_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w2_r3", "requests")},
    )
    case.step(
        "w2_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w2_r3_before", "snapshot"),
            "requests": output("w2_r3", "requests"),
            "keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w2_r4_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w2_r4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w2_r4_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w2_r4", "requests")},
    )
    case.step(
        "w2_r4_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w2_r4_before", "snapshot"),
            "requests": output("w2_r4", "requests"),
            "keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w2_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w3_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w3_r0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w3_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w3_r0", "requests")},
    )
    case.step(
        "w3_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w3_r0_before", "snapshot"),
            "requests": output("w3_r0", "requests"),
            "keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w3_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w3_r1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w3_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w3_r1", "requests")},
    )
    case.step(
        "w3_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w3_r1_before", "snapshot"),
            "requests": output("w3_r1", "requests"),
            "keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w3_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w3_r2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w3_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w3_r2", "requests")},
    )
    case.step(
        "w3_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w3_r2_before", "snapshot"),
            "requests": output("w3_r2", "requests"),
            "keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w3_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w3_r3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w3_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w3_r3", "requests")},
    )
    case.step(
        "w3_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w3_r3_before", "snapshot"),
            "requests": output("w3_r3", "requests"),
            "keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w3_r4_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w3_r4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w3_r4_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w3_r4", "requests")},
    )
    case.step(
        "w3_r4_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w3_r4_before", "snapshot"),
            "requests": output("w3_r4", "requests"),
            "keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w3_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w4_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w4_r0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w4_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w4_r0", "requests")},
    )
    case.step(
        "w4_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w4_r0_before", "snapshot"),
            "requests": output("w4_r0", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w4_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w4_r1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w4_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w4_r1", "requests")},
    )
    case.step(
        "w4_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w4_r1_before", "snapshot"),
            "requests": output("w4_r1", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w4_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w4_r2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w4_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w4_r2", "requests")},
    )
    case.step(
        "w4_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w4_r2_before", "snapshot"),
            "requests": output("w4_r2", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w4_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w4_r3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w4_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w4_r3", "requests")},
    )
    case.step(
        "w4_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w4_r3_before", "snapshot"),
            "requests": output("w4_r3", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w4_r4_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w4_r4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w4_r4_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w4_r4", "requests")},
    )
    case.step(
        "w4_r4_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w4_r4_before", "snapshot"),
            "requests": output("w4_r4", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w4_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w5_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w5_r0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w5_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w5_r0", "requests")},
    )
    case.step(
        "w5_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w5_r0_before", "snapshot"),
            "requests": output("w5_r0", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w5_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w5_r1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w5_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w5_r1", "requests")},
    )
    case.step(
        "w5_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w5_r1_before", "snapshot"),
            "requests": output("w5_r1", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w5_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w5_r2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w5_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w5_r2", "requests")},
    )
    case.step(
        "w5_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w5_r2_before", "snapshot"),
            "requests": output("w5_r2", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w5_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w5_r3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w5_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w5_r3", "requests")},
    )
    case.step(
        "w5_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w5_r3_before", "snapshot"),
            "requests": output("w5_r3", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w5_r4_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w5_r4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w5_r4_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w5_r4", "requests")},
    )
    case.step(
        "w5_r4_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w5_r4_before", "snapshot"),
            "requests": output("w5_r4", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w5_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w6_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w6_r0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w6_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w6_r0", "requests")},
    )
    case.step(
        "w6_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w6_r0_before", "snapshot"),
            "requests": output("w6_r0", "requests"),
            "keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w6_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w6_r1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w6_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w6_r1", "requests")},
    )
    case.step(
        "w6_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w6_r1_before", "snapshot"),
            "requests": output("w6_r1", "requests"),
            "keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w6_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w6_r2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w6_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w6_r2", "requests")},
    )
    case.step(
        "w6_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w6_r2_before", "snapshot"),
            "requests": output("w6_r2", "requests"),
            "keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w6_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w6_r3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w6_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w6_r3", "requests")},
    )
    case.step(
        "w6_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w6_r3_before", "snapshot"),
            "requests": output("w6_r3", "requests"),
            "keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w6_r4_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w6_r4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w6_r4_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w6_r4", "requests")},
    )
    case.step(
        "w6_r4_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w6_r4_before", "snapshot"),
            "requests": output("w6_r4", "requests"),
            "keys": [
                812000,
                812001,
                812002,
                812003,
                812004,
                812005,
                812006,
                812007,
                812008,
                812009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w6_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w7_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w7_r0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w7_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w7_r0", "requests")},
    )
    case.step(
        "w7_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w7_r0_before", "snapshot"),
            "requests": output("w7_r0", "requests"),
            "keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w7_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w7_r1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w7_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w7_r1", "requests")},
    )
    case.step(
        "w7_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w7_r1_before", "snapshot"),
            "requests": output("w7_r1", "requests"),
            "keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w7_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w7_r2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w7_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w7_r2", "requests")},
    )
    case.step(
        "w7_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w7_r2_before", "snapshot"),
            "requests": output("w7_r2", "requests"),
            "keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w7_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w7_r3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w7_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w7_r3", "requests")},
    )
    case.step(
        "w7_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w7_r3_before", "snapshot"),
            "requests": output("w7_r3", "requests"),
            "keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w7_r4_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w7_r4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w7_r4_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w7_r4", "requests")},
    )
    case.step(
        "w7_r4_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w7_r4_before", "snapshot"),
            "requests": output("w7_r4", "requests"),
            "keys": [
                813000,
                813001,
                813002,
                813003,
                813004,
                813005,
                813006,
                813007,
                813008,
                813009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w7_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w8_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w8_r0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w8_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w8_r0", "requests")},
    )
    case.step(
        "w8_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w8_r0_before", "snapshot"),
            "requests": output("w8_r0", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w8_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w8_r1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w8_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w8_r1", "requests")},
    )
    case.step(
        "w8_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w8_r1_before", "snapshot"),
            "requests": output("w8_r1", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w8_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w8_r2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w8_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w8_r2", "requests")},
    )
    case.step(
        "w8_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w8_r2_before", "snapshot"),
            "requests": output("w8_r2", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w8_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w8_r3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w8_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w8_r3", "requests")},
    )
    case.step(
        "w8_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w8_r3_before", "snapshot"),
            "requests": output("w8_r3", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w8_r4_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w8_r4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w8_r4_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w8_r4", "requests")},
    )
    case.step(
        "w8_r4_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w8_r4_before", "snapshot"),
            "requests": output("w8_r4", "requests"),
            "keys": [
                810000,
                810001,
                810002,
                810003,
                810004,
                810005,
                810006,
                810007,
                810008,
                810009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w8_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w9_r0_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w9_r0",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w9_r0_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w9_r0", "requests")},
    )
    case.step(
        "w9_r0_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w9_r0_before", "snapshot"),
            "requests": output("w9_r0", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w9_r1_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w9_r1",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w9_r1_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w9_r1", "requests")},
    )
    case.step(
        "w9_r1_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w9_r1_before", "snapshot"),
            "requests": output("w9_r1", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w9_r2_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w9_r2",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w9_r2_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w9_r2", "requests")},
    )
    case.step(
        "w9_r2_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w9_r2_before", "snapshot"),
            "requests": output("w9_r2", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w9_r3_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w9_r3",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w9_r3_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w9_r3", "requests")},
    )
    case.step(
        "w9_r3_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w9_r3_before", "snapshot"),
            "requests": output("w9_r3", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w9_r4_before",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "w9_r4",
        "request",
        params={
            "count": 1,
            "input_len": 10240,
            "output_len": 2,
            "block_keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "w9_r4_done",
        "wait",
        timeout_s=15,
        params={"requests": output("w9_r4", "requests")},
    )
    case.step(
        "w9_r4_hit",
        "kv_hit_observe",
        params={
            "snapshot": output("w9_r4_before", "snapshot"),
            "requests": output("w9_r4", "requests"),
            "keys": [
                811000,
                811001,
                811002,
                811003,
                811004,
                811005,
                811006,
                811007,
                811008,
                811009,
            ],
            "min_contiguous": 8,
        },
    )
    case.step(
        "w9_end",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "flips",
        "kv_window_transitions",
        params={
            "snapshots": [
                output("w0_end", "snapshot"),
                output("w1_end", "snapshot"),
                output("w2_end", "snapshot"),
                output("w3_end", "snapshot"),
                output("w4_end", "snapshot"),
                output("w5_end", "snapshot"),
                output("w6_end", "snapshot"),
                output("w7_end", "snapshot"),
                output("w8_end", "snapshot"),
                output("w9_end", "snapshot"),
            ],
            "families": [
                [
                    810000,
                    810001,
                    810002,
                    810003,
                    810004,
                    810005,
                    810006,
                    810007,
                    810008,
                    810009,
                ],
                [
                    811000,
                    811001,
                    811002,
                    811003,
                    811004,
                    811005,
                    811006,
                    811007,
                    811008,
                    811009,
                ],
                [
                    812000,
                    812001,
                    812002,
                    812003,
                    812004,
                    812005,
                    812006,
                    812007,
                    812008,
                    812009,
                ],
                [
                    813000,
                    813001,
                    813002,
                    813003,
                    813004,
                    813005,
                    813006,
                    813007,
                    813008,
                    813009,
                ],
            ],
        },
    )
    case.step(
        "flip_bound",
        "check",
        params={"actual": output("flips", "flips"), "op": "le", "expected": 80},
    )
    case.step(
        "hit_rate",
        "kv_hit_rate_check",
        params={
            "samples": [
                output("w0_r0_hit", "sample"),
                output("w0_r1_hit", "sample"),
                output("w0_r2_hit", "sample"),
                output("w0_r3_hit", "sample"),
                output("w0_r4_hit", "sample"),
                output("w1_r0_hit", "sample"),
                output("w1_r1_hit", "sample"),
                output("w1_r2_hit", "sample"),
                output("w1_r3_hit", "sample"),
                output("w1_r4_hit", "sample"),
                output("w2_r0_hit", "sample"),
                output("w2_r1_hit", "sample"),
                output("w2_r2_hit", "sample"),
                output("w2_r3_hit", "sample"),
                output("w2_r4_hit", "sample"),
                output("w3_r0_hit", "sample"),
                output("w3_r1_hit", "sample"),
                output("w3_r2_hit", "sample"),
                output("w3_r3_hit", "sample"),
                output("w3_r4_hit", "sample"),
                output("w4_r0_hit", "sample"),
                output("w4_r1_hit", "sample"),
                output("w4_r2_hit", "sample"),
                output("w4_r3_hit", "sample"),
                output("w4_r4_hit", "sample"),
                output("w5_r0_hit", "sample"),
                output("w5_r1_hit", "sample"),
                output("w5_r2_hit", "sample"),
                output("w5_r3_hit", "sample"),
                output("w5_r4_hit", "sample"),
                output("w6_r0_hit", "sample"),
                output("w6_r1_hit", "sample"),
                output("w6_r2_hit", "sample"),
                output("w6_r3_hit", "sample"),
                output("w6_r4_hit", "sample"),
                output("w7_r0_hit", "sample"),
                output("w7_r1_hit", "sample"),
                output("w7_r2_hit", "sample"),
                output("w7_r3_hit", "sample"),
                output("w7_r4_hit", "sample"),
                output("w8_r0_hit", "sample"),
                output("w8_r1_hit", "sample"),
                output("w8_r2_hit", "sample"),
                output("w8_r3_hit", "sample"),
                output("w8_r4_hit", "sample"),
                output("w9_r0_hit", "sample"),
                output("w9_r1_hit", "sample"),
                output("w9_r2_hit", "sample"),
                output("w9_r3_hit", "sample"),
                output("w9_r4_hit", "sample"),
            ],
            "min_samples": 50,
            "bands": {"strict": 0.72, "normal": 0.5, "loose": 0.4},
        },
    )
    case.step(
        "replication",
        "kv_replication_check",
        params={
            "snapshot": output("w9_end", "snapshot"),
            "families": [
                [
                    810000,
                    810001,
                    810002,
                    810003,
                    810004,
                    810005,
                    810006,
                    810007,
                    810008,
                    810009,
                ],
                [
                    811000,
                    811001,
                    811002,
                    811003,
                    811004,
                    811005,
                    811006,
                    811007,
                    811008,
                    811009,
                ],
                [
                    812000,
                    812001,
                    812002,
                    812003,
                    812004,
                    812005,
                    812006,
                    812007,
                    812008,
                    812009,
                ],
                [
                    813000,
                    813001,
                    813002,
                    813003,
                    813004,
                    813005,
                    813006,
                    813007,
                    813008,
                    813009,
                ],
            ],
            "bands": {"strict": 1.5, "normal": 1.75, "loose": 2},
            "max_holders": 2,
        },
    )
    case.step("cleanup", "teardown")


def lru_affinity(case):
    # Capacity 4 includes one reserved block. Three requested blocks fit;
    # sharing only the first key forces eviction of a cold non-prefix block.
    if case.environment.get("prefill_cache_blocks") != 4:
        raise ValueError("lru_affinity requires four Prefill cache blocks")
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prime",
        "request",
        params={
            "count": 1,
            "input_len": 3056,
            "output_len": 2,
            "block_keys": [810001, 810002, 810003],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "prime_done",
        "wait",
        timeout_s=15,
        params={"requests": output("prime", "requests")},
    )
    case.step(
        "prime_holder", "kv_landing", params={"requests": output("prime", "requests")}
    )
    case.step("prime_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "replay",
        "request",
        params={
            "count": 1,
            "input_len": 3056,
            "output_len": 2,
            "block_keys": [810001, 810002, 810003],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "replay_done",
        "wait",
        timeout_s=15,
        params={"requests": output("replay", "requests")},
    )
    case.step(
        "retry",
        "kv_retry_misdirected",
        timeout_s=60,
        params={
            "anchor": output("prime", "requests"),
            "candidate": output("replay", "requests"),
            "keys": [810001, 810002, 810003],
            "input_len": 3056,
            "output_len": 2,
            "settle_s": 2,
            "request_timeout_s": 15,
        },
    )
    case.step(
        "replay_holder", "kv_landing", params={"requests": output("retry", "requests")}
    )
    case.step(
        "replay_affinity",
        "kv_same",
        params={
            "first": output("prime_holder", "engine"),
            "second": output("replay_holder", "engine"),
        },
    )
    case.step(
        "prime_snapshot",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "prime_counters",
        "kv_cache_statistics",
        params={
            "snapshot": output("prime_snapshot", "snapshot"),
            "engine": output("prime_holder", "engine"),
        },
    )
    case.step(
        "prime_keys",
        "check",
        params={"actual": output("prime_counters", "keys"), "op": "ge", "expected": 2},
    )
    case.step(
        "prime_evictions",
        "check",
        params={
            "actual": output("prime_counters", "evictions"),
            "op": "eq",
            "expected": 0,
        },
    )
    case.step(
        "cold_blocks",
        "kv_capacity_observe",
        params={
            "targets": [output("prime_holder", "engine")],
            "fields": ["referenced_blocks", "held_blocks", "cache_key_set"],
        },
    )
    for field, expected in (
        ("referenced_blocks", 0),
        ("held_blocks", 0),
        ("key_count", 3),
    ):
        case.step(
            "cold_" + field,
            "kv_capacity_counter",
            params={
                "observations": [output("cold_blocks", "observation")],
                "field": field,
                "stat": "latest",
                "op": "eq",
                "expected": expected,
            },
        )
    case.step(
        "pressure",
        "request",
        params={
            "count": 1,
            "input_len": 3056,
            "output_len": 2,
            "block_keys": [810001, 810004, 810005],
            "stream_timeout_s": 15,
        },
    )
    case.step(
        "pressure_done",
        "wait",
        timeout_s=15,
        params={"requests": output("pressure", "requests")},
    )
    case.step("pressure_settle", "balance_pause", params={"seconds": 0.5})
    case.step(
        "pressure_holder",
        "kv_landing",
        params={"requests": output("pressure", "requests")},
    )
    case.step(
        "pressure_snapshot",
        "kv_snapshot",
        timeout_s=10,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 0},
    )
    case.step(
        "pressure_counters",
        "kv_cache_statistics",
        params={
            "snapshot": output("pressure_snapshot", "snapshot"),
            "engine": output("pressure_holder", "engine"),
        },
    )
    case.step(
        "pressure_keys",
        "check",
        params={
            "actual": output("pressure_counters", "keys"),
            "op": "le",
            "expected": 4,
        },
    )
    case.step(
        "pressure_evictions",
        "check",
        params={
            "actual": output("pressure_counters", "evictions"),
            "op": "ge",
            "expected": 1,
        },
    )
    case.step(
        "prefix_affinity",
        "kv_same",
        params={
            "first": output("prime_holder", "engine"),
            "second": output("pressure_holder", "engine"),
        },
    )
    case.step("cleanup", "teardown")


def referenced_occupancy(case):
    # Same bounded request shape, but the warm prefix stays referenced by a
    # running request. A fresh request must use the other available Prefill.
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prime",
        "kv_capacity_request",
        timeout_s=40,
        params={
            "input_len": 3056,
            "output_len": 2,
            "block_keys": [820001, 820002, 820003],
        },
    )
    case.step("holder", "kv_landing", params={"requests": output("prime", "requests")})
    case.step("cache_sync", "balance_pause", params={"seconds": 2})
    case.step(
        "slow_holder",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("holder", "engine")],
            "perf": {"prefill_fixed_ms": 15000},
        },
    )
    case.step(
        "pin",
        "kv_capacity_request",
        timeout_s=40,
        params={
            "input_len": 3056,
            "output_len": 2,
            "block_keys": [820001, 820002, 820003],
            "mode": "fire",
            "stream_timeout_s": 40,
        },
    )
    case.step(
        "pinned_holder",
        "kv_landing",
        params={"requests": output("pin", "requests"), "phase": "scheduled"},
    )
    case.step(
        "same_holder",
        "kv_same",
        params={
            "first": output("holder", "engine"),
            "second": output("pinned_holder", "engine"),
        },
    )
    fields = ["referenced_blocks", "held_blocks", "cache_key_set", "cache_evictions"]
    case.step(
        "pinned",
        "kv_capacity_observe",
        timeout_s=6,
        params={
            "targets": [output("holder", "engine")],
            "fields": fields,
            "duration_s": 4,
            "until_field": "referenced_blocks",
            "until_op": "ge",
            "until_value": 3,
        },
    )
    case.step(
        "pin_proven",
        "kv_capacity_counter",
        params={
            "observations": [output("pinned", "observation")],
            "field": "referenced_blocks",
            "stat": "latest",
            "op": "eq",
            "expected": 3,
        },
    )
    case.step(
        "probe",
        "kv_capacity_request",
        timeout_s=6,
        params={
            "input_len": 1008,
            "output_len": 2,
            "block_keys": [820010],
            "schedule_timeout_s": 3,
            "stream_timeout_s": 3,
        },
    )
    case.step(
        "probe_success",
        "kv_capacity_outcome",
        params={"requests": [output("probe", "requests")], "metric": "success"},
    )
    case.step(
        "probe_holder", "kv_landing", params={"requests": output("probe", "requests")}
    )
    case.step(
        "overflow",
        "kv_distinct",
        params={
            "first": output("holder", "engine"),
            "second": output("probe_holder", "engine"),
        },
    )
    case.step(
        "protected",
        "kv_capacity_observe",
        params={"targets": [output("holder", "engine")], "fields": fields},
    )
    for name, field, expected, baseline in (
        ("still_pinned", "referenced_blocks", 3, False),
        ("not_evicted", "cache_evictions", 0, True),
    ):
        params = {
            "observations": [output("protected", "observation")],
            "field": field,
            "stat": "latest",
            "op": "eq",
            "expected": expected,
        }
        if baseline:
            params["baseline"] = output("pinned", "observation")
        case.step(name, "kv_capacity_counter", params=params)
    case.step(
        "pin_finished",
        "kv_capacity_wait",
        timeout_s=45,
        params={"requests": [output("pin", "requests")]},
    )
    case.step(
        "pin_success",
        "kv_capacity_outcome",
        params={"requests": [output("pin", "requests")], "metric": "success"},
    )
    case.step(
        "released",
        "kv_capacity_observe",
        timeout_s=8,
        params={
            "targets": [output("holder", "engine")],
            "fields": fields,
            "duration_s": 5,
            "until_field": "referenced_blocks",
            "until_op": "eq",
            "until_value": 0,
        },
    )
    case.step(
        "references_released",
        "kv_capacity_counter",
        params={
            "observations": [output("released", "observation")],
            "field": "referenced_blocks",
            "stat": "latest",
            "op": "eq",
            "expected": 0,
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "referenced_occupancy": {
        "build": referenced_occupancy,
        "profiles": PROFILES,
        "metadata": {},
    },
    "hot_churn": {
        "build": hot_churn,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "lru_affinity": {
        "build": lru_affinity,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
}
