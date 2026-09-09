"""Explicit P/D pool ownership, capacity WAIT versus terminal rejection, pressure eviction and affinity overflow."""

from ..case_config import output

METADATA = {
    "id": "cache_capacity_recovery",
    "description": "Explicit P/D pool ownership, capacity WAIT versus terminal rejection, pressure "
    "eviction and affinity overflow.",
    "category": "kv",
}

PROFILES = ["batch-window", "single-nonbatch", "single-batch", "window-nonbatch"]


def decode_pool_exhaustion_terminal(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "d_base",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["decode-0"],
            "fields": ["lack_mem_rejects", "kv_admission_fails"],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "p_base",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0", "prefill-1"],
            "fields": ["lack_mem_rejects", "held_blocks"],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "probe",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2560,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": ["UNKNOWN", "INTERNAL", "RESOURCE_EXHAUSTED"],
            "block_keys": [1400000, 1400001],
        },
    )
    case.step(
        "decode_error",
        "kv_capacity_outcome",
        params={
            "requests": [output("probe", "requests")],
            "metric": "typed_error",
            "tokens": ["lack_mem", "decode-side"],
            "max_s": 3,
        },
    )
    case.step(
        "d_grew",
        "kv_capacity_observe",
        timeout_s=15,
        params={
            "targets": ["decode-0"],
            "fields": ["lack_mem_rejects"],
            "duration_s": 10,
            "interval_s": 0.5,
            "until_field": "lack_mem_rejects",
            "until_op": "eq",
            "until_value": 1,
            "baseline": output("d_base", "observation"),
        },
    )
    case.step(
        "exact_once",
        "kv_capacity_counter",
        params={
            "observations": [output("d_grew", "observation")],
            "field": "lack_mem_rejects",
            "op": "eq",
            "expected": 1,
            "stat": "latest",
            "baseline": output("d_base", "observation"),
        },
    )
    case.step("counter_sync", "master_mark", params={"wait_s": 0.5})
    case.step(
        "d_after",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["decode-0"],
            "fields": ["kv_admission_fails"],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "retry_counter_flat",
        "kv_capacity_counter",
        params={
            "observations": [output("d_after", "observation")],
            "field": "kv_admission_fails",
            "op": "eq",
            "expected": 0,
            "stat": "all",
            "baseline": output("d_base", "observation"),
        },
    )
    case.step(
        "p_after",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0", "prefill-1"],
            "fields": ["lack_mem_rejects"],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "prefill_rejects_flat",
        "kv_capacity_counter",
        params={
            "observations": [output("p_after", "observation")],
            "field": "lack_mem_rejects",
            "op": "eq",
            "expected": 0,
            "stat": "all",
            "baseline": output("p_base", "observation"),
        },
    )
    case.step(
        "released",
        "kv_capacity_observe",
        timeout_s=15,
        params={
            "targets": ["prefill-0", "prefill-1"],
            "fields": ["held_blocks"],
            "duration_s": 10,
            "interval_s": 0.5,
            "until_field": "held_blocks",
            "until_op": "eq",
            "until_value": 0,
            "baseline": output("p_base", "observation"),
        },
    )
    case.step(
        "p_leases_released",
        "kv_capacity_counter",
        params={
            "observations": [output("released", "observation")],
            "field": "held_blocks",
            "op": "eq",
            "expected": 0,
            "stat": "latest",
            "baseline": output("p_base", "observation"),
        },
    )
    case.step(
        "d_recovery_base",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["decode-0"],
            "fields": ["lack_mem_rejects", "kv_admission_fails"],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "headroom",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 1024,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [1410000, 1410001],
        },
    )
    case.step(
        "headroom_succeeded",
        "kv_capacity_outcome",
        params={"requests": [output("headroom", "requests")], "metric": "success"},
    )
    case.step("recovery_sync", "master_mark", params={"wait_s": 0.5})
    case.step(
        "d_stable",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["decode-0"],
            "fields": ["lack_mem_rejects", "kv_admission_fails"],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "terminal_counter_stable",
        "kv_capacity_counter",
        params={
            "observations": [output("d_stable", "observation")],
            "field": "lack_mem_rejects",
            "op": "eq",
            "expected": 0,
            "stat": "all",
            "baseline": output("d_recovery_base", "observation"),
        },
    )
    case.step(
        "retry_counter_stable",
        "kv_capacity_counter",
        params={
            "observations": [output("d_stable", "observation")],
            "field": "kv_admission_fails",
            "op": "eq",
            "expected": 0,
            "stat": "all",
            "baseline": output("d_base", "observation"),
        },
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=30,
        params={"target": "single", "inflight_zero": True},
    )
    case.step(
        "engine_clean",
        "admission_engine_clean",
        timeout_s=30,
        params={"targets": ["prefill-0", "prefill-1", "decode-0"]},
    )
    case.step(
        "recovery",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 30,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
        },
    )
    case.step(
        "recovered",
        "kv_capacity_outcome",
        params={"requests": [output("recovery", "requests")], "metric": "success"},
    )
    case.step("cleanup", "teardown")


def decode_capacity_park(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "pressure",
        "kv_capacity_pressure",
        params={"targets": ["decode-0", "decode-1", "decode-2", "decode-3"]},
    )
    case.step("status_sync", "master_mark", params={"wait_s": 1.5})
    case.step("before", "kv_capacity_watermark")
    case.step(
        "probe",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 10,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 5,
            "expected_rpc_statuses": ["DEADLINE_EXCEEDED"],
        },
    )
    case.step(
        "schedule_stayed_pending",
        "kv_capacity_outcome",
        params={
            "requests": [output("probe", "requests")],
            "metric": "schedule_deadline",
        },
    )
    case.step("delivery_sync", "master_mark", params={"wait_s": 0.5})
    case.step(
        "delivery",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": [
                "prefill-0",
                "prefill-1",
                "decode-0",
                "decode-1",
                "decode-2",
                "decode-3",
            ],
            "fields": ["request_lifecycle"],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "not_delivered",
        "kv_capacity_not_delivered",
        params={
            "requests": output("probe", "requests"),
            "observation": output("delivery", "observation"),
        },
    )
    case.step(
        "cancel", "kv_capacity_cancel", params={"requests": output("probe", "requests")}
    )
    case.step("cancel_sync", "master_mark", params={"wait_s": 0.5})
    case.step(
        "watermark_restored",
        "kv_capacity_watermark_wait",
        timeout_s=10,
        params={"baseline": output("before", "watermark")},
    )
    case.step(
        "clear",
        "kv_capacity_clear",
        params={"pressure": output("pressure", "pressure")},
    )
    case.step("recovery_status_sync", "master_mark", params={"wait_s": 2})
    case.step(
        "fresh",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
        },
    )
    case.step(
        "fresh_succeeded",
        "kv_capacity_outcome",
        params={"requests": [output("fresh", "requests")], "metric": "success"},
    )
    case.step(
        "recovery",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 30,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
        },
    )
    case.step(
        "recovered",
        "kv_capacity_outcome",
        params={"requests": [output("recovery", "requests")], "metric": "success"},
    )
    case.step("cleanup", "teardown")


def pool_saturation_evict_reject_recover(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "base",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": ["cache_evictions", "lack_mem_rejects"],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "evict_wave_0",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                500000,
                500001,
                500002,
                500003,
                500004,
                500005,
                500006,
                500007,
            ],
        },
    )
    case.step(
        "evict_wave_0_ok",
        "kv_capacity_outcome",
        params={"requests": [output("evict_wave_0", "requests")], "metric": "success"},
    )
    case.step(
        "pool_0",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": [
                "cache_blocks",
                "held_blocks",
                "referenced_blocks",
                "available_blocks",
            ],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "evict_wave_1",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                501000,
                501001,
                501002,
                501003,
                501004,
                501005,
                501006,
                501007,
            ],
        },
    )
    case.step(
        "evict_wave_1_ok",
        "kv_capacity_outcome",
        params={"requests": [output("evict_wave_1", "requests")], "metric": "success"},
    )
    case.step(
        "pool_1",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": [
                "cache_blocks",
                "held_blocks",
                "referenced_blocks",
                "available_blocks",
            ],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "evict_wave_2",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                502000,
                502001,
                502002,
                502003,
                502004,
                502005,
                502006,
                502007,
            ],
        },
    )
    case.step(
        "evict_wave_2_ok",
        "kv_capacity_outcome",
        params={"requests": [output("evict_wave_2", "requests")], "metric": "success"},
    )
    case.step(
        "pool_2",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": [
                "cache_blocks",
                "held_blocks",
                "referenced_blocks",
                "available_blocks",
            ],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "evict_wave_3",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                503000,
                503001,
                503002,
                503003,
                503004,
                503005,
                503006,
                503007,
            ],
        },
    )
    case.step(
        "evict_wave_3_ok",
        "kv_capacity_outcome",
        params={"requests": [output("evict_wave_3", "requests")], "metric": "success"},
    )
    case.step(
        "pool_3",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": [
                "cache_blocks",
                "held_blocks",
                "referenced_blocks",
                "available_blocks",
            ],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "evict_wave_4",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                504000,
                504001,
                504002,
                504003,
                504004,
                504005,
                504006,
                504007,
            ],
        },
    )
    case.step(
        "evict_wave_4_ok",
        "kv_capacity_outcome",
        params={"requests": [output("evict_wave_4", "requests")], "metric": "success"},
    )
    case.step(
        "pool_4",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": [
                "cache_blocks",
                "held_blocks",
                "referenced_blocks",
                "available_blocks",
            ],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "evict_wave_5",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                505000,
                505001,
                505002,
                505003,
                505004,
                505005,
                505006,
                505007,
            ],
        },
    )
    case.step(
        "evict_wave_5_ok",
        "kv_capacity_outcome",
        params={"requests": [output("evict_wave_5", "requests")], "metric": "success"},
    )
    case.step(
        "pool_5",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": [
                "cache_blocks",
                "held_blocks",
                "referenced_blocks",
                "available_blocks",
            ],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "evict_wave_6",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                506000,
                506001,
                506002,
                506003,
                506004,
                506005,
                506006,
                506007,
            ],
        },
    )
    case.step(
        "evict_wave_6_ok",
        "kv_capacity_outcome",
        params={"requests": [output("evict_wave_6", "requests")], "metric": "success"},
    )
    case.step(
        "pool_6",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": [
                "cache_blocks",
                "held_blocks",
                "referenced_blocks",
                "available_blocks",
            ],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "evict_wave_7",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                507000,
                507001,
                507002,
                507003,
                507004,
                507005,
                507006,
                507007,
            ],
        },
    )
    case.step(
        "evict_wave_7_ok",
        "kv_capacity_outcome",
        params={"requests": [output("evict_wave_7", "requests")], "metric": "success"},
    )
    case.step(
        "pool_7",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": [
                "cache_blocks",
                "held_blocks",
                "referenced_blocks",
                "available_blocks",
            ],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "evict_wave_8",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                508000,
                508001,
                508002,
                508003,
                508004,
                508005,
                508006,
                508007,
            ],
        },
    )
    case.step(
        "evict_wave_8_ok",
        "kv_capacity_outcome",
        params={"requests": [output("evict_wave_8", "requests")], "metric": "success"},
    )
    case.step(
        "pool_8",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": [
                "cache_blocks",
                "held_blocks",
                "referenced_blocks",
                "available_blocks",
            ],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "evict_wave_9",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                509000,
                509001,
                509002,
                509003,
                509004,
                509005,
                509006,
                509007,
            ],
        },
    )
    case.step(
        "evict_wave_9_ok",
        "kv_capacity_outcome",
        params={"requests": [output("evict_wave_9", "requests")], "metric": "success"},
    )
    case.step(
        "pool_9",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": [
                "cache_blocks",
                "held_blocks",
                "referenced_blocks",
                "available_blocks",
            ],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "evicted",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": ["cache_evictions", "cache_key_set"],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "capacity_evictions",
        "kv_capacity_counter",
        params={
            "observations": [output("evicted", "observation")],
            "field": "cache_evictions",
            "op": "ge",
            "expected": 1,
            "stat": "all",
            "baseline": output("base", "observation"),
        },
    )
    case.step(
        "key_cap",
        "kv_capacity_counter",
        params={
            "observations": [output("evicted", "observation")],
            "field": "key_count",
            "op": "le",
            "expected": 27,
            "stat": "all",
        },
    )
    case.step(
        "eviction_conservation",
        "kv_capacity_counter",
        params={
            "observations": [
                output("pool_0", "observation"),
                output("pool_1", "observation"),
                output("pool_2", "observation"),
                output("pool_3", "observation"),
                output("pool_4", "observation"),
                output("pool_5", "observation"),
                output("pool_6", "observation"),
                output("pool_7", "observation"),
                output("pool_8", "observation"),
                output("pool_9", "observation"),
            ],
            "field": "conservation",
            "op": "eq",
            "expected": 0,
            "stat": "all",
        },
    )
    case.step(
        "slow",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 3000},
        },
    )
    case.step("perf_sync", "master_mark", params={"wait_s": 1.5})
    case.step(
        "occupant_0",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "fire",
            "stream_timeout_s": 30,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                600000,
                600001,
                600002,
                600003,
                600004,
                600005,
                600006,
                600007,
            ],
        },
    )
    case.step(
        "occupant_0_admitted",
        "kv_capacity_outcome",
        params={"requests": [output("occupant_0", "requests")], "metric": "admitted"},
    )
    case.step("occupant_spacing_0", "master_mark", params={"wait_s": 0.4})
    case.step(
        "occupant_1",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "fire",
            "stream_timeout_s": 30,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                601000,
                601001,
                601002,
                601003,
                601004,
                601005,
                601006,
                601007,
            ],
        },
    )
    case.step(
        "occupant_1_admitted",
        "kv_capacity_outcome",
        params={"requests": [output("occupant_1", "requests")], "metric": "admitted"},
    )
    case.step("occupant_spacing_1", "master_mark", params={"wait_s": 0.4})
    case.step(
        "occupant_2",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "fire",
            "stream_timeout_s": 30,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                602000,
                602001,
                602002,
                602003,
                602004,
                602005,
                602006,
                602007,
            ],
        },
    )
    case.step(
        "occupant_2_admitted",
        "kv_capacity_outcome",
        params={"requests": [output("occupant_2", "requests")], "metric": "admitted"},
    )
    case.step("occupant_spacing_2", "master_mark", params={"wait_s": 0.4})
    case.step("enqueues_landed", "master_mark", params={"wait_s": 0.3})
    case.step(
        "await_saturation",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": ["held_blocks"],
            "duration_s": 2,
            "interval_s": 0.05,
            "until_field": "held_blocks",
            "until_op": "ge",
            "until_value": 24,
        },
    )
    case.step(
        "leading_pool",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": [
                "cache_blocks",
                "held_blocks",
                "referenced_blocks",
                "available_blocks",
            ],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "probe",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": ["UNKNOWN", "INTERNAL", "RESOURCE_EXHAUSTED"],
            "block_keys": [
                690000,
                690001,
                690002,
                690003,
                690004,
                690005,
                690006,
                690007,
            ],
        },
    )
    case.step(
        "probe_typed_fast",
        "kv_capacity_outcome",
        params={
            "requests": [output("probe", "requests")],
            "metric": "typed_error",
            "tokens": ["lack_mem", "insufficient kv cache"],
            "max_s": 3,
        },
    )
    case.step(
        "burst_0",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": ["UNKNOWN", "INTERNAL", "RESOURCE_EXHAUSTED"],
            "block_keys": [
                680000,
                680001,
                680002,
                680003,
                680004,
                680005,
                680006,
                680007,
            ],
        },
    )
    case.step(
        "burst_1",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": ["UNKNOWN", "INTERNAL", "RESOURCE_EXHAUSTED"],
            "block_keys": [
                681000,
                681001,
                681002,
                681003,
                681004,
                681005,
                681006,
                681007,
            ],
        },
    )
    case.step(
        "burst_typed_fast",
        "kv_capacity_outcome",
        params={
            "requests": [output("burst_0", "requests"), output("burst_1", "requests")],
            "metric": "bounded_errors",
            "tokens": ["lack_mem"],
            "max_s": 3,
        },
    )
    case.step(
        "failure_share",
        "kv_capacity_failure_bound",
        params={
            "requests": [
                output("probe", "requests"),
                output("burst_0", "requests"),
                output("burst_1", "requests"),
            ],
            "minimum": 1,
            "maximum": 3,
        },
    )
    case.step(
        "saturation_window",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": [
                "cache_blocks",
                "held_blocks",
                "referenced_blocks",
                "available_blocks",
            ],
            "duration_s": 3,
            "interval_s": 0.1,
        },
    )
    case.step(
        "held_peak",
        "kv_capacity_counter",
        params={
            "observations": [
                output("leading_pool", "observation"),
                output("saturation_window", "observation"),
            ],
            "field": "held_blocks",
            "op": "ge",
            "expected": 24,
            "stat": "max",
        },
    )
    case.step(
        "available_floor",
        "kv_capacity_counter",
        params={
            "observations": [
                output("leading_pool", "observation"),
                output("saturation_window", "observation"),
            ],
            "field": "available_blocks",
            "op": "le",
            "expected": 3,
            "stat": "min",
        },
    )
    case.step(
        "saturation_conservation",
        "kv_capacity_counter",
        params={
            "observations": [
                output("leading_pool", "observation"),
                output("saturation_window", "observation"),
            ],
            "field": "conservation",
            "op": "eq",
            "expected": 0,
            "stat": "all",
        },
    )
    case.step(
        "lack_after",
        "kv_capacity_observe",
        timeout_s=10,
        params={
            "targets": ["prefill-0"],
            "fields": ["lack_mem_rejects"],
            "duration_s": 0,
            "interval_s": 0.5,
        },
    )
    case.step(
        "rejections_accounted",
        "kv_capacity_counter",
        params={
            "observations": [output("lack_after", "observation")],
            "field": "lack_mem_rejects",
            "op": "ge",
            "expected": 1,
            "stat": "all",
            "baseline": output("base", "observation"),
        },
    )
    case.step(
        "occupants_done",
        "kv_capacity_wait",
        timeout_s=90,
        params={
            "requests": [
                output("occupant_0", "requests"),
                output("occupant_1", "requests"),
                output("occupant_2", "requests"),
            ]
        },
    )
    case.step(
        "occupants_complete",
        "kv_capacity_outcome",
        params={
            "requests": [
                output("occupant_0", "requests"),
                output("occupant_1", "requests"),
                output("occupant_2", "requests"),
            ],
            "metric": "success",
        },
    )
    case.step(
        "available_again",
        "kv_capacity_observe",
        timeout_s=15,
        params={
            "targets": ["prefill-0"],
            "fields": ["available_blocks"],
            "duration_s": 10,
            "interval_s": 0.5,
            "until_field": "available_blocks",
            "until_op": "ge",
            "until_value": 8,
        },
    )
    case.step(
        "capacity_recovered",
        "kv_capacity_counter",
        params={
            "observations": [output("available_again", "observation")],
            "field": "available_blocks",
            "op": "ge",
            "expected": 8,
            "stat": "latest",
        },
    )
    case.step(
        "fresh",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                1000000,
                1000001,
                1000002,
                1000003,
                1000004,
                1000005,
                1000006,
                1000007,
            ],
        },
    )
    case.step(
        "fresh_succeeded",
        "kv_capacity_outcome",
        params={"requests": [output("fresh", "requests")], "metric": "success"},
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=30,
        params={"target": "single", "inflight_zero": True},
    )
    case.step(
        "engine_clean",
        "admission_engine_clean",
        timeout_s=30,
        params={"targets": ["prefill-0", "decode-0", "decode-1"]},
    )
    case.step(
        "recovery",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 30,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
        },
    )
    case.step(
        "recovered",
        "kv_capacity_outcome",
        params={"requests": [output("recovery", "requests")], "metric": "success"},
    )
    case.step(
        "restore_perf",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("cleanup", "teardown")


def capacity_conflict_overflow(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "prime",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 40960,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                500000,
                500001,
                500002,
                500003,
                500004,
                500005,
                500006,
                500007,
                500008,
                500009,
                500010,
                500011,
                500012,
                500013,
                500014,
                500015,
                500016,
                500017,
                500018,
                500019,
                500020,
                500021,
                500022,
                500023,
                500024,
                500025,
                500026,
                500027,
                500028,
                500029,
                500030,
                500031,
                500032,
                500033,
                500034,
                500035,
                500036,
                500037,
                500038,
                500039,
            ],
        },
    )
    case.step(
        "prime_succeeded",
        "kv_capacity_outcome",
        params={"requests": [output("prime", "requests")], "metric": "success"},
    )
    case.step(
        "holder",
        "kv_landing",
        params={"requests": output("prime", "requests"), "phase": "terminal"},
    )
    case.step(
        "cool",
        "kv_capacity_other",
        params={
            "targets": ["prefill-0", "prefill-1"],
            "exclude": output("holder", "engine"),
        },
    )
    case.step(
        "cache_sync",
        "kv_snapshot",
        timeout_s=8,
        params={"targets": ["prefill-0", "prefill-1"], "quiet_s": 3.5},
    )
    case.step(
        "slow_all",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 5000},
        },
    )
    case.step("perf_sync", "master_mark", params={"wait_s": 1.5})
    case.step(
        "seed",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 147456,
            "output_len": 2,
            "mode": "fire",
            "stream_timeout_s": 30,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                500000,
                500001,
                500002,
                500003,
                500004,
                500005,
                500006,
                500007,
                500008,
                500009,
                500010,
                500011,
                500012,
                500013,
                500014,
                500015,
                500016,
                500017,
                500018,
                500019,
                500020,
                500021,
                500022,
                500023,
                500024,
                500025,
                500026,
                500027,
                500028,
                500029,
                500030,
                500031,
                500032,
                500033,
                500034,
                500035,
                500036,
                500037,
                500038,
                500039,
                509000,
                509001,
                509002,
                509003,
                509004,
                509005,
                509006,
                509007,
                509008,
                509009,
                509010,
                509011,
                509012,
                509013,
                509014,
                509015,
                509016,
                509017,
                509018,
                509019,
                509020,
                509021,
                509022,
                509023,
                509024,
                509025,
                509026,
                509027,
                509028,
                509029,
                509030,
                509031,
                509032,
                509033,
                509034,
                509035,
                509036,
                509037,
                509038,
                509039,
                509040,
                509041,
                509042,
                509043,
                509044,
                509045,
                509046,
                509047,
                509048,
                509049,
                509050,
                509051,
                509052,
                509053,
                509054,
                509055,
                509056,
                509057,
                509058,
                509059,
                509060,
                509061,
                509062,
                509063,
                509064,
                509065,
                509066,
                509067,
                509068,
                509069,
                509070,
                509071,
                509072,
                509073,
                509074,
                509075,
                509076,
                509077,
                509078,
                509079,
                509080,
                509081,
                509082,
                509083,
                509084,
                509085,
                509086,
                509087,
                509088,
                509089,
                509090,
                509091,
                509092,
                509093,
                509094,
                509095,
                509096,
                509097,
                509098,
                509099,
                509100,
                509101,
                509102,
                509103,
            ],
        },
    )
    case.step(
        "seed_admitted",
        "kv_capacity_outcome",
        params={"requests": [output("seed", "requests")], "metric": "admitted"},
    )
    case.step(
        "seed_holder",
        "kv_landing",
        params={"requests": output("seed", "requests"), "phase": "scheduled"},
    )
    case.step(
        "seed_pinned",
        "kv_same",
        params={
            "first": output("holder", "engine"),
            "second": output("seed_holder", "engine"),
        },
    )
    case.step(
        "seed_pending",
        "kv_capacity_observe",
        timeout_s=11,
        params={
            "targets": [output("holder", "engine")],
            "fields": ["pending"],
            "duration_s": 6,
            "interval_s": 0.1,
            "until_field": "pending",
            "until_op": "ge",
            "until_value": 1,
        },
    )
    case.step(
        "seed_seen",
        "kv_capacity_counter",
        params={
            "observations": [output("seed_pending", "observation")],
            "field": "pending",
            "op": "ge",
            "expected": 1,
            "stat": "latest",
        },
    )
    case.step(
        "cool_fast",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("cool", "engine")],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("cool_sync", "master_mark", params={"wait_s": 0.3})
    case.step(
        "baseline",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 2048,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
        },
    )
    case.step(
        "baseline_ok",
        "kv_capacity_outcome",
        params={"requests": [output("baseline", "requests")], "metric": "success"},
    )
    case.step(
        "wave_0",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 40960,
            "output_len": 2,
            "mode": "fire",
            "stream_timeout_s": 30,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                500000,
                500001,
                500002,
                500003,
                500004,
                500005,
                500006,
                500007,
                500008,
                500009,
                500010,
                500011,
                500012,
                500013,
                500014,
                500015,
                500016,
                500017,
                500018,
                500019,
                500020,
                500021,
                500022,
                500023,
                500024,
                500025,
                500026,
                500027,
                500028,
                500029,
                500030,
                500031,
                500032,
                500033,
                500034,
                500035,
                500036,
                500037,
                500038,
                500039,
            ],
        },
    )
    case.step("wave_spacing_0", "master_mark", params={"wait_s": 0.12})
    case.step(
        "wave_1",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 40960,
            "output_len": 2,
            "mode": "fire",
            "stream_timeout_s": 30,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                500000,
                500001,
                500002,
                500003,
                500004,
                500005,
                500006,
                500007,
                500008,
                500009,
                500010,
                500011,
                500012,
                500013,
                500014,
                500015,
                500016,
                500017,
                500018,
                500019,
                500020,
                500021,
                500022,
                500023,
                500024,
                500025,
                500026,
                500027,
                500028,
                500029,
                500030,
                500031,
                500032,
                500033,
                500034,
                500035,
                500036,
                500037,
                500038,
                500039,
            ],
        },
    )
    case.step("wave_spacing_1", "master_mark", params={"wait_s": 0.12})
    case.step(
        "wave_2",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 40960,
            "output_len": 2,
            "mode": "fire",
            "stream_timeout_s": 30,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                500000,
                500001,
                500002,
                500003,
                500004,
                500005,
                500006,
                500007,
                500008,
                500009,
                500010,
                500011,
                500012,
                500013,
                500014,
                500015,
                500016,
                500017,
                500018,
                500019,
                500020,
                500021,
                500022,
                500023,
                500024,
                500025,
                500026,
                500027,
                500028,
                500029,
                500030,
                500031,
                500032,
                500033,
                500034,
                500035,
                500036,
                500037,
                500038,
                500039,
            ],
        },
    )
    case.step("wave_spacing_2", "master_mark", params={"wait_s": 0.12})
    case.step(
        "wave_3",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 40960,
            "output_len": 2,
            "mode": "fire",
            "stream_timeout_s": 30,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                500000,
                500001,
                500002,
                500003,
                500004,
                500005,
                500006,
                500007,
                500008,
                500009,
                500010,
                500011,
                500012,
                500013,
                500014,
                500015,
                500016,
                500017,
                500018,
                500019,
                500020,
                500021,
                500022,
                500023,
                500024,
                500025,
                500026,
                500027,
                500028,
                500029,
                500030,
                500031,
                500032,
                500033,
                500034,
                500035,
                500036,
                500037,
                500038,
                500039,
            ],
        },
    )
    case.step("wave_spacing_3", "master_mark", params={"wait_s": 0.12})
    case.step(
        "wave_4",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 40960,
            "output_len": 2,
            "mode": "fire",
            "stream_timeout_s": 30,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                500000,
                500001,
                500002,
                500003,
                500004,
                500005,
                500006,
                500007,
                500008,
                500009,
                500010,
                500011,
                500012,
                500013,
                500014,
                500015,
                500016,
                500017,
                500018,
                500019,
                500020,
                500021,
                500022,
                500023,
                500024,
                500025,
                500026,
                500027,
                500028,
                500029,
                500030,
                500031,
                500032,
                500033,
                500034,
                500035,
                500036,
                500037,
                500038,
                500039,
            ],
        },
    )
    case.step(
        "drained",
        "kv_capacity_wait",
        timeout_s=180,
        params={
            "requests": [
                output("seed", "requests"),
                output("wave_0", "requests"),
                output("wave_1", "requests"),
                output("wave_2", "requests"),
                output("wave_3", "requests"),
                output("wave_4", "requests"),
            ]
        },
    )
    case.step(
        "probe",
        "kv_capacity_request",
        timeout_s=70,
        params={
            "input_len": 40960,
            "output_len": 2,
            "mode": "complete",
            "stream_timeout_s": 15,
            "schedule_timeout_s": 30,
            "expected_rpc_statuses": [],
            "block_keys": [
                500000,
                500001,
                500002,
                500003,
                500004,
                500005,
                500006,
                500007,
                500008,
                500009,
                500010,
                500011,
                500012,
                500013,
                500014,
                500015,
                500016,
                500017,
                500018,
                500019,
                500020,
                500021,
                500022,
                500023,
                500024,
                500025,
                500026,
                500027,
                500028,
                500029,
                500030,
                500031,
                500032,
                500033,
                500034,
                500035,
                500036,
                500037,
                500038,
                500039,
            ],
        },
    )
    case.step(
        "protection",
        "kv_capacity_protection",
        params={
            "baseline": output("baseline", "requests"),
            "probe": output("probe", "requests"),
            "holder": output("holder", "engine"),
            "wave": [
                output("wave_0", "requests"),
                output("wave_1", "requests"),
                output("wave_2", "requests"),
                output("wave_3", "requests"),
                output("wave_4", "requests"),
            ],
            "share_bands": {"strict": 0, "normal": 0.05, "loose": 0.1},
            "ratio_bands": {"strict": 2, "normal": 3, "loose": 5},
        },
    )
    case.step(
        "restore_perf",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["prefill-0", "prefill-1"],
            "perf": {"prefill_fixed_ms": 100},
        },
    )
    case.step("cleanup", "teardown")


VARIANTS = {
    "decode_pool_exhaustion_terminal": {
        "build": decode_pool_exhaustion_terminal,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "decode_capacity_park": {
        "build": decode_capacity_park,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "pool_saturation_evict_reject_recover": {
        "build": pool_saturation_evict_reject_recover,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
    "capacity_conflict_overflow": {
        "build": capacity_conflict_overflow,
        "profiles": [
            "batch-window",
            "single-nonbatch",
            "single-batch",
            "window-nonbatch",
        ],
        "metadata": {},
    },
}
