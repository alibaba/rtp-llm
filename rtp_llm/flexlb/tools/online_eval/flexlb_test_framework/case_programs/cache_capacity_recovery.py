"""Explicit P/D pool ownership, capacity WAIT versus terminal rejection, pressure eviction and affinity overflow."""

from ..case_config import output


def decode_pool_exhaustion_terminal(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("decode_pool_exhaustion_terminal.setup_timeout_s"),
    )
    case.step(
        "d_base",
        "kv_capacity_observe",
        timeout_s=case.value("decode_pool_exhaustion_terminal.d_base_timeout_s"),
        params=case.value("decode_pool_exhaustion_terminal.d_base"),
    )
    case.step(
        "p_base",
        "kv_capacity_observe",
        timeout_s=case.value("decode_pool_exhaustion_terminal.p_base_timeout_s"),
        params=case.value("decode_pool_exhaustion_terminal.p_base"),
    )
    case.step(
        "probe",
        "kv_capacity_request",
        timeout_s=case.value("decode_pool_exhaustion_terminal.probe_timeout_s"),
        params=case.value("decode_pool_exhaustion_terminal.probe"),
    )
    case.step(
        "decode_error",
        "kv_capacity_outcome",
        params=case.params(
            "decode_pool_exhaustion_terminal.decode_error",
            {"requests": [output("probe", "requests")]},
        ),
    )
    case.step(
        "d_grew",
        "kv_capacity_observe",
        timeout_s=case.value("decode_pool_exhaustion_terminal.d_grew_timeout_s"),
        params=case.params(
            "decode_pool_exhaustion_terminal.d_grew",
            {"baseline": output("d_base", "observation")},
        ),
    )
    case.step(
        "exact_once",
        "kv_capacity_counter",
        params=case.params(
            "decode_pool_exhaustion_terminal.exact_once",
            {
                "observations": [output("d_grew", "observation")],
                "baseline": output("d_base", "observation"),
            },
        ),
    )
    case.step(
        "counter_sync",
        "master_mark",
        params=case.value("decode_pool_exhaustion_terminal.counter_sync"),
    )
    case.step(
        "d_after",
        "kv_capacity_observe",
        timeout_s=case.value("decode_pool_exhaustion_terminal.d_after_timeout_s"),
        params=case.value("decode_pool_exhaustion_terminal.d_after"),
    )
    case.step(
        "retry_counter_flat",
        "kv_capacity_counter",
        params=case.params(
            "decode_pool_exhaustion_terminal.retry_counter_flat",
            {
                "observations": [output("d_after", "observation")],
                "baseline": output("d_base", "observation"),
            },
        ),
    )
    case.step(
        "p_after",
        "kv_capacity_observe",
        timeout_s=case.value("decode_pool_exhaustion_terminal.p_after_timeout_s"),
        params=case.value("decode_pool_exhaustion_terminal.p_after"),
    )
    case.step(
        "prefill_rejects_flat",
        "kv_capacity_counter",
        params=case.params(
            "decode_pool_exhaustion_terminal.prefill_rejects_flat",
            {
                "observations": [output("p_after", "observation")],
                "baseline": output("p_base", "observation"),
            },
        ),
    )
    case.step(
        "released",
        "kv_capacity_observe",
        timeout_s=case.value("decode_pool_exhaustion_terminal.released_timeout_s"),
        params=case.params(
            "decode_pool_exhaustion_terminal.released",
            {"baseline": output("p_base", "observation")},
        ),
    )
    case.step(
        "p_leases_released",
        "kv_capacity_counter",
        params=case.params(
            "decode_pool_exhaustion_terminal.p_leases_released",
            {
                "observations": [output("released", "observation")],
                "baseline": output("p_base", "observation"),
            },
        ),
    )
    case.step(
        "d_recovery_base",
        "kv_capacity_observe",
        timeout_s=case.value(
            "decode_pool_exhaustion_terminal.d_recovery_base_timeout_s"
        ),
        params=case.value("decode_pool_exhaustion_terminal.d_recovery_base"),
    )
    case.step(
        "fence_released",
        "master_ready",
        timeout_s=case.value(
            "decode_pool_exhaustion_terminal.fence_released_timeout_s"
        ),
        params=case.value("decode_pool_exhaustion_terminal.fence_released"),
    )
    case.step(
        "headroom",
        "kv_capacity_request",
        timeout_s=case.value("decode_pool_exhaustion_terminal.headroom_timeout_s"),
        params=case.value("decode_pool_exhaustion_terminal.headroom"),
    )
    case.step(
        "headroom_succeeded",
        "kv_capacity_outcome",
        params=case.params(
            "decode_pool_exhaustion_terminal.headroom_succeeded",
            {"requests": [output("headroom", "requests")]},
        ),
    )
    case.step(
        "recovery_sync",
        "master_mark",
        params=case.value("decode_pool_exhaustion_terminal.recovery_sync"),
    )
    case.step(
        "d_stable",
        "kv_capacity_observe",
        timeout_s=case.value("decode_pool_exhaustion_terminal.d_stable_timeout_s"),
        params=case.value("decode_pool_exhaustion_terminal.d_stable"),
    )
    case.step(
        "terminal_counter_stable",
        "kv_capacity_counter",
        params=case.params(
            "decode_pool_exhaustion_terminal.terminal_counter_stable",
            {
                "observations": [output("d_stable", "observation")],
                "baseline": output("d_recovery_base", "observation"),
            },
        ),
    )
    case.step(
        "retry_counter_stable",
        "kv_capacity_counter",
        params=case.params(
            "decode_pool_exhaustion_terminal.retry_counter_stable",
            {
                "observations": [output("d_stable", "observation")],
                "baseline": output("d_base", "observation"),
            },
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value("decode_pool_exhaustion_terminal.master_clean_timeout_s"),
        params=case.value("decode_pool_exhaustion_terminal.master_clean"),
    )
    case.step(
        "engine_clean",
        "admission_engine_clean",
        timeout_s=case.value("decode_pool_exhaustion_terminal.engine_clean_timeout_s"),
        params=case.value("decode_pool_exhaustion_terminal.engine_clean"),
    )
    case.step(
        "recovery",
        "kv_capacity_request",
        timeout_s=case.value("decode_pool_exhaustion_terminal.recovery_timeout_s"),
        params=case.value("decode_pool_exhaustion_terminal.recovery"),
    )
    case.step(
        "recovered",
        "kv_capacity_outcome",
        params=case.params(
            "decode_pool_exhaustion_terminal.recovered",
            {"requests": [output("recovery", "requests")]},
        ),
    )
    case.step("cleanup", "teardown")


def decode_capacity_park(case):
    case.step(
        "setup", "setup", timeout_s=case.value("decode_capacity_park.setup_timeout_s")
    )
    case.step(
        "pressure",
        "kv_capacity_pressure",
        params=case.value("decode_capacity_park.pressure"),
    )
    case.step(
        "status_sync",
        "master_mark",
        params=case.value("decode_capacity_park.status_sync"),
    )
    case.step("before", "kv_capacity_watermark")
    case.step(
        "probe",
        "kv_capacity_request",
        timeout_s=case.value("decode_capacity_park.probe_timeout_s"),
        params=case.value("decode_capacity_park.probe"),
    )
    case.step(
        "schedule_stayed_pending",
        "kv_capacity_outcome",
        params=case.params(
            "decode_capacity_park.schedule_stayed_pending",
            {"requests": [output("probe", "requests")]},
        ),
    )
    case.step(
        "delivery_sync",
        "master_mark",
        params=case.value("decode_capacity_park.delivery_sync"),
    )
    case.step(
        "delivery",
        "kv_capacity_observe",
        timeout_s=case.value("decode_capacity_park.delivery_timeout_s"),
        params=case.value("decode_capacity_park.delivery"),
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
    case.step(
        "cancel_sync",
        "master_mark",
        params=case.value("decode_capacity_park.cancel_sync"),
    )
    case.step(
        "watermark_restored",
        "kv_capacity_watermark_wait",
        timeout_s=case.value("decode_capacity_park.watermark_restored_timeout_s"),
        params={"baseline": output("before", "watermark")},
    )
    case.step(
        "clear",
        "kv_capacity_clear",
        params={"pressure": output("pressure", "pressure")},
    )
    case.step(
        "recovery_status_sync",
        "master_mark",
        params=case.value("decode_capacity_park.recovery_status_sync"),
    )
    case.step(
        "fresh",
        "kv_capacity_request",
        timeout_s=case.value("decode_capacity_park.fresh_timeout_s"),
        params=case.value("decode_capacity_park.fresh"),
    )
    case.step(
        "fresh_succeeded",
        "kv_capacity_outcome",
        params=case.params(
            "decode_capacity_park.fresh_succeeded",
            {"requests": [output("fresh", "requests")]},
        ),
    )
    case.step(
        "recovery",
        "kv_capacity_request",
        timeout_s=case.value("decode_capacity_park.recovery_timeout_s"),
        params=case.value("decode_capacity_park.recovery"),
    )
    case.step(
        "recovered",
        "kv_capacity_outcome",
        params=case.params(
            "decode_capacity_park.recovered",
            {"requests": [output("recovery", "requests")]},
        ),
    )
    case.step("cleanup", "teardown")


def pool_saturation_evict_reject_recover(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("pool_saturation_evict_reject_recover.setup_timeout_s"),
    )
    case.step(
        "base",
        "kv_capacity_observe",
        timeout_s=case.value("pool_saturation_evict_reject_recover.base_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.base"),
    )
    case.step(
        "evict_wave_0",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.evict_wave_0_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.evict_wave_0"),
    )
    case.step(
        "evict_wave_0_ok",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.evict_wave_0_ok",
            {"requests": [output("evict_wave_0", "requests")]},
        ),
    )
    case.step(
        "pool_0",
        "kv_capacity_observe",
        timeout_s=case.value("pool_saturation_evict_reject_recover.pool_0_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.pool_0"),
    )
    case.step(
        "evict_wave_1",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.evict_wave_1_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.evict_wave_1"),
    )
    case.step(
        "evict_wave_1_ok",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.evict_wave_1_ok",
            {"requests": [output("evict_wave_1", "requests")]},
        ),
    )
    case.step(
        "pool_1",
        "kv_capacity_observe",
        timeout_s=case.value("pool_saturation_evict_reject_recover.pool_1_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.pool_1"),
    )
    case.step(
        "evict_wave_2",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.evict_wave_2_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.evict_wave_2"),
    )
    case.step(
        "evict_wave_2_ok",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.evict_wave_2_ok",
            {"requests": [output("evict_wave_2", "requests")]},
        ),
    )
    case.step(
        "pool_2",
        "kv_capacity_observe",
        timeout_s=case.value("pool_saturation_evict_reject_recover.pool_2_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.pool_2"),
    )
    case.step(
        "evict_wave_3",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.evict_wave_3_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.evict_wave_3"),
    )
    case.step(
        "evict_wave_3_ok",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.evict_wave_3_ok",
            {"requests": [output("evict_wave_3", "requests")]},
        ),
    )
    case.step(
        "pool_3",
        "kv_capacity_observe",
        timeout_s=case.value("pool_saturation_evict_reject_recover.pool_3_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.pool_3"),
    )
    case.step(
        "evict_wave_4",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.evict_wave_4_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.evict_wave_4"),
    )
    case.step(
        "evict_wave_4_ok",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.evict_wave_4_ok",
            {"requests": [output("evict_wave_4", "requests")]},
        ),
    )
    case.step(
        "pool_4",
        "kv_capacity_observe",
        timeout_s=case.value("pool_saturation_evict_reject_recover.pool_4_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.pool_4"),
    )
    case.step(
        "evict_wave_5",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.evict_wave_5_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.evict_wave_5"),
    )
    case.step(
        "evict_wave_5_ok",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.evict_wave_5_ok",
            {"requests": [output("evict_wave_5", "requests")]},
        ),
    )
    case.step(
        "pool_5",
        "kv_capacity_observe",
        timeout_s=case.value("pool_saturation_evict_reject_recover.pool_5_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.pool_5"),
    )
    case.step(
        "evict_wave_6",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.evict_wave_6_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.evict_wave_6"),
    )
    case.step(
        "evict_wave_6_ok",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.evict_wave_6_ok",
            {"requests": [output("evict_wave_6", "requests")]},
        ),
    )
    case.step(
        "pool_6",
        "kv_capacity_observe",
        timeout_s=case.value("pool_saturation_evict_reject_recover.pool_6_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.pool_6"),
    )
    case.step(
        "evict_wave_7",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.evict_wave_7_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.evict_wave_7"),
    )
    case.step(
        "evict_wave_7_ok",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.evict_wave_7_ok",
            {"requests": [output("evict_wave_7", "requests")]},
        ),
    )
    case.step(
        "pool_7",
        "kv_capacity_observe",
        timeout_s=case.value("pool_saturation_evict_reject_recover.pool_7_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.pool_7"),
    )
    case.step(
        "evict_wave_8",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.evict_wave_8_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.evict_wave_8"),
    )
    case.step(
        "evict_wave_8_ok",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.evict_wave_8_ok",
            {"requests": [output("evict_wave_8", "requests")]},
        ),
    )
    case.step(
        "pool_8",
        "kv_capacity_observe",
        timeout_s=case.value("pool_saturation_evict_reject_recover.pool_8_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.pool_8"),
    )
    case.step(
        "evict_wave_9",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.evict_wave_9_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.evict_wave_9"),
    )
    case.step(
        "evict_wave_9_ok",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.evict_wave_9_ok",
            {"requests": [output("evict_wave_9", "requests")]},
        ),
    )
    case.step(
        "pool_9",
        "kv_capacity_observe",
        timeout_s=case.value("pool_saturation_evict_reject_recover.pool_9_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.pool_9"),
    )
    case.step(
        "evicted",
        "kv_capacity_observe",
        timeout_s=case.value("pool_saturation_evict_reject_recover.evicted_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.evicted"),
    )
    case.step(
        "capacity_evictions",
        "kv_capacity_counter",
        params=case.params(
            "pool_saturation_evict_reject_recover.capacity_evictions",
            {
                "observations": [output("evicted", "observation")],
                "baseline": output("base", "observation"),
            },
        ),
    )
    case.step(
        "key_cap",
        "kv_capacity_counter",
        params=case.params(
            "pool_saturation_evict_reject_recover.key_cap",
            {"observations": [output("evicted", "observation")]},
        ),
    )
    case.step(
        "eviction_conservation",
        "kv_capacity_counter",
        params=case.params(
            "pool_saturation_evict_reject_recover.eviction_conservation",
            {
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
                ]
            },
        ),
    )
    case.step(
        "slow",
        "engine_control",
        params=case.value("pool_saturation_evict_reject_recover.slow"),
    )
    case.step(
        "perf_sync",
        "master_mark",
        params=case.value("pool_saturation_evict_reject_recover.perf_sync"),
    )
    case.step(
        "occupant_0",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.occupant_0_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.occupant_0"),
    )
    case.step(
        "occupant_0_admitted",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.occupant_0_admitted",
            {"requests": [output("occupant_0", "requests")]},
        ),
    )
    case.step(
        "occupant_spacing_0",
        "master_mark",
        params=case.value("pool_saturation_evict_reject_recover.occupant_spacing_0"),
    )
    case.step(
        "occupant_1",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.occupant_1_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.occupant_1"),
    )
    case.step(
        "occupant_1_admitted",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.occupant_1_admitted",
            {"requests": [output("occupant_1", "requests")]},
        ),
    )
    case.step(
        "occupant_spacing_1",
        "master_mark",
        params=case.value("pool_saturation_evict_reject_recover.occupant_spacing_1"),
    )
    case.step(
        "occupant_2",
        "kv_capacity_request",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.occupant_2_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.occupant_2"),
    )
    case.step(
        "occupant_2_admitted",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.occupant_2_admitted",
            {"requests": [output("occupant_2", "requests")]},
        ),
    )
    case.step(
        "occupant_spacing_2",
        "master_mark",
        params=case.value("pool_saturation_evict_reject_recover.occupant_spacing_2"),
    )
    case.step(
        "enqueues_landed",
        "master_mark",
        params=case.value("pool_saturation_evict_reject_recover.enqueues_landed"),
    )
    case.step(
        "await_saturation",
        "kv_capacity_observe",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.await_saturation_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.await_saturation"),
    )
    case.step(
        "leading_pool",
        "kv_capacity_observe",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.leading_pool_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.leading_pool"),
    )
    case.step(
        "probe",
        "kv_capacity_request",
        timeout_s=case.value("pool_saturation_evict_reject_recover.probe_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.probe"),
    )
    case.step(
        "probe_typed_fast",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.probe_typed_fast",
            {"requests": [output("probe", "requests")]},
        ),
    )
    case.step(
        "burst_0",
        "kv_capacity_request",
        timeout_s=case.value("pool_saturation_evict_reject_recover.burst_0_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.burst_0"),
    )
    case.step(
        "burst_1",
        "kv_capacity_request",
        timeout_s=case.value("pool_saturation_evict_reject_recover.burst_1_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.burst_1"),
    )
    case.step(
        "burst_typed_fast",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.burst_typed_fast",
            {
                "requests": [
                    output("burst_0", "requests"),
                    output("burst_1", "requests"),
                ]
            },
        ),
    )
    case.step(
        "failure_share",
        "kv_capacity_failure_bound",
        params=case.params(
            "pool_saturation_evict_reject_recover.failure_share",
            {
                "requests": [
                    output("probe", "requests"),
                    output("burst_0", "requests"),
                    output("burst_1", "requests"),
                ]
            },
        ),
    )
    case.step(
        "saturation_window",
        "kv_capacity_observe",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.saturation_window_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.saturation_window"),
    )
    case.step(
        "held_peak",
        "kv_capacity_counter",
        params=case.params(
            "pool_saturation_evict_reject_recover.held_peak",
            {
                "observations": [
                    output("leading_pool", "observation"),
                    output("saturation_window", "observation"),
                ]
            },
        ),
    )
    case.step(
        "available_floor",
        "kv_capacity_counter",
        params=case.params(
            "pool_saturation_evict_reject_recover.available_floor",
            {
                "observations": [
                    output("leading_pool", "observation"),
                    output("saturation_window", "observation"),
                ]
            },
        ),
    )
    case.step(
        "saturation_conservation",
        "kv_capacity_counter",
        params=case.params(
            "pool_saturation_evict_reject_recover.saturation_conservation",
            {
                "observations": [
                    output("leading_pool", "observation"),
                    output("saturation_window", "observation"),
                ]
            },
        ),
    )
    case.step(
        "lack_after",
        "kv_capacity_observe",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.lack_after_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.lack_after"),
    )
    case.step(
        "rejections_accounted",
        "kv_capacity_counter",
        params=case.params(
            "pool_saturation_evict_reject_recover.rejections_accounted",
            {
                "observations": [output("lack_after", "observation")],
                "baseline": output("base", "observation"),
            },
        ),
    )
    case.step(
        "occupants_done",
        "kv_capacity_wait",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.occupants_done_timeout_s"
        ),
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
        params=case.params(
            "pool_saturation_evict_reject_recover.occupants_complete",
            {
                "requests": [
                    output("occupant_0", "requests"),
                    output("occupant_1", "requests"),
                    output("occupant_2", "requests"),
                ]
            },
        ),
    )
    case.step(
        "available_again",
        "kv_capacity_observe",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.available_again_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.available_again"),
    )
    case.step(
        "capacity_recovered",
        "kv_capacity_counter",
        params=case.params(
            "pool_saturation_evict_reject_recover.capacity_recovered",
            {"observations": [output("available_again", "observation")]},
        ),
    )
    case.step(
        "fresh",
        "kv_capacity_request",
        timeout_s=case.value("pool_saturation_evict_reject_recover.fresh_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.fresh"),
    )
    case.step(
        "fresh_succeeded",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.fresh_succeeded",
            {"requests": [output("fresh", "requests")]},
        ),
    )
    case.step(
        "master_clean",
        "master_ready",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.master_clean_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.master_clean"),
    )
    case.step(
        "engine_clean",
        "admission_engine_clean",
        timeout_s=case.value(
            "pool_saturation_evict_reject_recover.engine_clean_timeout_s"
        ),
        params=case.value("pool_saturation_evict_reject_recover.engine_clean"),
    )
    case.step(
        "recovery",
        "kv_capacity_request",
        timeout_s=case.value("pool_saturation_evict_reject_recover.recovery_timeout_s"),
        params=case.value("pool_saturation_evict_reject_recover.recovery"),
    )
    case.step(
        "recovered",
        "kv_capacity_outcome",
        params=case.params(
            "pool_saturation_evict_reject_recover.recovered",
            {"requests": [output("recovery", "requests")]},
        ),
    )
    case.step(
        "restore_perf",
        "engine_control",
        params=case.value("pool_saturation_evict_reject_recover.restore_perf"),
    )
    case.step("cleanup", "teardown")


def capacity_conflict_overflow(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("capacity_conflict_overflow.setup_timeout_s"),
    )
    case.step(
        "prime",
        "kv_capacity_request",
        timeout_s=case.value("capacity_conflict_overflow.prime_timeout_s"),
        params=case.value("capacity_conflict_overflow.prime"),
    )
    case.step(
        "prime_succeeded",
        "kv_capacity_outcome",
        params=case.params(
            "capacity_conflict_overflow.prime_succeeded",
            {"requests": [output("prime", "requests")]},
        ),
    )
    case.step(
        "holder",
        "kv_landing",
        params=case.params(
            "capacity_conflict_overflow.holder",
            {"requests": output("prime", "requests")},
        ),
    )
    case.step(
        "cool",
        "kv_capacity_other",
        params=case.params(
            "capacity_conflict_overflow.cool", {"exclude": output("holder", "engine")}
        ),
    )
    case.step(
        "cache_sync",
        "kv_snapshot",
        timeout_s=case.value("capacity_conflict_overflow.cache_sync_timeout_s"),
        params=case.value("capacity_conflict_overflow.cache_sync"),
    )
    case.step(
        "perf_sync",
        "master_mark",
        params=case.value("capacity_conflict_overflow.perf_sync"),
    )
    case.step(
        "seed",
        "kv_capacity_request",
        timeout_s=case.value("capacity_conflict_overflow.seed_timeout_s"),
        params=case.value("capacity_conflict_overflow.seed"),
    )
    case.step(
        "seed_admitted",
        "kv_capacity_outcome",
        params=case.params(
            "capacity_conflict_overflow.seed_admitted",
            {"requests": [output("seed", "requests")]},
        ),
    )
    case.step(
        "seed_holder",
        "kv_landing",
        params=case.params(
            "capacity_conflict_overflow.seed_holder",
            {"requests": output("seed", "requests")},
        ),
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
        timeout_s=case.value("capacity_conflict_overflow.seed_pending_timeout_s"),
        params=case.params(
            "capacity_conflict_overflow.seed_pending",
            {"targets": [output("holder", "engine")]},
        ),
    )
    case.step(
        "seed_seen",
        "kv_capacity_counter",
        params=case.params(
            "capacity_conflict_overflow.seed_seen",
            {"observations": [output("seed_pending", "observation")]},
        ),
    )
    case.step(
        "cool_sync",
        "master_mark",
        params=case.value("capacity_conflict_overflow.cool_sync"),
    )
    case.step(
        "baseline",
        "kv_capacity_request",
        timeout_s=case.value("capacity_conflict_overflow.baseline_timeout_s"),
        params=case.value("capacity_conflict_overflow.baseline"),
    )
    case.step(
        "baseline_ok",
        "kv_capacity_outcome",
        params=case.params(
            "capacity_conflict_overflow.baseline_ok",
            {"requests": [output("baseline", "requests")]},
        ),
    )
    case.step(
        "wave_0",
        "kv_capacity_request",
        timeout_s=case.value("capacity_conflict_overflow.wave_0_timeout_s"),
        params=case.value("capacity_conflict_overflow.wave_0"),
    )
    case.step(
        "wave_spacing_0",
        "kv_capacity_spacing",
        params=case.params(
            "capacity_conflict_overflow.wave_spacing_0",
            {"requests": output("wave_0", "requests")},
        ),
    )
    case.step(
        "wave_1",
        "kv_capacity_request",
        timeout_s=case.value("capacity_conflict_overflow.wave_1_timeout_s"),
        params=case.value("capacity_conflict_overflow.wave_1"),
    )
    case.step(
        "wave_spacing_1",
        "kv_capacity_spacing",
        params=case.params(
            "capacity_conflict_overflow.wave_spacing_1",
            {"requests": output("wave_1", "requests")},
        ),
    )
    case.step(
        "wave_2",
        "kv_capacity_request",
        timeout_s=case.value("capacity_conflict_overflow.wave_2_timeout_s"),
        params=case.value("capacity_conflict_overflow.wave_2"),
    )
    case.step(
        "wave_spacing_2",
        "kv_capacity_spacing",
        params=case.params(
            "capacity_conflict_overflow.wave_spacing_2",
            {"requests": output("wave_2", "requests")},
        ),
    )
    case.step(
        "wave_3",
        "kv_capacity_request",
        timeout_s=case.value("capacity_conflict_overflow.wave_3_timeout_s"),
        params=case.value("capacity_conflict_overflow.wave_3"),
    )
    case.step(
        "wave_spacing_3",
        "kv_capacity_spacing",
        params=case.params(
            "capacity_conflict_overflow.wave_spacing_3",
            {"requests": output("wave_3", "requests")},
        ),
    )
    case.step(
        "wave_4",
        "kv_capacity_request",
        timeout_s=case.value("capacity_conflict_overflow.wave_4_timeout_s"),
        params=case.value("capacity_conflict_overflow.wave_4"),
    )
    case.step(
        "drained",
        "kv_capacity_wait",
        timeout_s=case.value("capacity_conflict_overflow.drained_timeout_s"),
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
        timeout_s=case.value("capacity_conflict_overflow.probe_timeout_s"),
        params=case.value("capacity_conflict_overflow.probe"),
    )
    case.step(
        "protection",
        "kv_capacity_protection",
        params=case.params(
            "capacity_conflict_overflow.protection",
            {
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
            },
        ),
    )
    case.step("cleanup", "teardown")
