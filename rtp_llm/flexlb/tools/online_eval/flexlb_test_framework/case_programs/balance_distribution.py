"""Explicit balance stages preserving client landing, token share, Decode deltas and latency contracts."""

from ..case_config import output


def uniform_serial(case):
    case.step("setup", "setup", timeout_s=case.value("uniform_serial.setup_timeout_s"))
    case.step(
        "fleet",
        "balance_snapshot",
        timeout_s=case.value("uniform_serial.fleet_timeout_s"),
        params=case.value("uniform_serial.fleet"),
    )
    case.step(
        "plain",
        "balance_start",
        timeout_s=case.value("uniform_serial.plain_timeout_s"),
        params=case.value("uniform_serial.plain"),
    )
    case.step(
        "plain_terminal",
        "balance_wait",
        timeout_s=case.value("uniform_serial.plain_terminal_timeout_s"),
        params={"requests": output("plain", "requests")},
    )
    case.step(
        "plain_p6",
        "balance_check",
        timeout_s=case.value("uniform_serial.plain_p6_timeout_s"),
        params=case.params(
            "uniform_serial.plain_p6",
            {
                "requests": [output("plain", "requests")],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "plain_p1",
        "balance_check",
        timeout_s=case.value("uniform_serial.plain_p1_timeout_s"),
        params=case.params(
            "uniform_serial.plain_p1",
            {
                "requests": [output("plain", "requests")],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "plain_p2",
        "balance_check",
        timeout_s=case.value("uniform_serial.plain_p2_timeout_s"),
        params=case.params(
            "uniform_serial.plain_p2",
            {
                "requests": [output("plain", "requests")],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "perf_sync",
        "balance_pause",
        timeout_s=case.value("uniform_serial.perf_sync_timeout_s"),
        params=case.value("uniform_serial.perf_sync"),
    )
    case.step(
        "idle_replay",
        "balance_start",
        timeout_s=case.value("uniform_serial.idle_replay_timeout_s"),
        params=case.value("uniform_serial.idle_replay"),
    )
    case.step(
        "idle_replay_terminal",
        "balance_wait",
        timeout_s=case.value("uniform_serial.idle_replay_terminal_timeout_s"),
        params={"requests": output("idle_replay", "requests")},
    )
    case.observe(
        "idle_replay_p6",
        "balance_check",
        timeout_s=case.value("uniform_serial.idle_replay_p6_timeout_s"),
        params=case.params(
            "uniform_serial.idle_replay_p6",
            {
                "requests": [output("idle_replay", "requests")],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.observe(
        "idle_replay_p1",
        "balance_check",
        timeout_s=case.value("uniform_serial.idle_replay_p1_timeout_s"),
        params=case.params(
            "uniform_serial.idle_replay_p1",
            {
                "requests": [output("idle_replay", "requests")],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.observe(
        "idle_replay_p2",
        "balance_check",
        timeout_s=case.value("uniform_serial.idle_replay_p2_timeout_s"),
        params=case.params(
            "uniform_serial.idle_replay_p2",
            {
                "requests": [output("idle_replay", "requests")],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("uniform_serial.teardown_timeout_s"),
    )


def concurrent_mix(case):
    case.step("setup", "setup", timeout_s=case.value("concurrent_mix.setup_timeout_s"))
    case.step(
        "fleet",
        "balance_snapshot",
        timeout_s=case.value("concurrent_mix.fleet_timeout_s"),
        params=case.value("concurrent_mix.fleet"),
    )
    case.step(
        "burst",
        "balance_start",
        timeout_s=case.value("concurrent_mix.burst_timeout_s"),
        params=case.value("concurrent_mix.burst"),
    )
    case.step(
        "terminal",
        "balance_wait",
        timeout_s=case.value("concurrent_mix.terminal_timeout_s"),
        params={"requests": output("burst", "requests")},
    )
    case.observe(
        "p6",
        "balance_check",
        timeout_s=case.value("concurrent_mix.p6_timeout_s"),
        params=case.params(
            "concurrent_mix.p6",
            {
                "requests": [output("burst", "requests")],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.observe(
        "p1",
        "balance_check",
        timeout_s=case.value("concurrent_mix.p1_timeout_s"),
        params=case.params(
            "concurrent_mix.p1",
            {
                "requests": [output("burst", "requests")],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.observe(
        "p2",
        "balance_check",
        timeout_s=case.value("concurrent_mix.p2_timeout_s"),
        params=case.params(
            "concurrent_mix.p2",
            {
                "requests": [output("burst", "requests")],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("concurrent_mix.teardown_timeout_s"),
    )


def decode_spread(case):
    case.step("setup", "setup", timeout_s=case.value("decode_spread.setup_timeout_s"))
    case.step(
        "n10_before",
        "balance_snapshot",
        timeout_s=case.value("decode_spread.n10_before_timeout_s"),
        params=case.value("decode_spread.n10_before"),
    )
    case.step(
        "n10",
        "balance_start",
        timeout_s=case.value("decode_spread.n10_timeout_s"),
        params=case.value("decode_spread.n10"),
    )
    case.step(
        "n10_terminal",
        "balance_wait",
        timeout_s=case.value("decode_spread.n10_terminal_timeout_s"),
        params={"requests": output("n10", "requests")},
    )
    case.step(
        "n10_after",
        "balance_snapshot",
        timeout_s=case.value("decode_spread.n10_after_timeout_s"),
        params=case.value("decode_spread.n10_after"),
    )
    case.step(
        "n10_p6",
        "balance_check",
        timeout_s=case.value("decode_spread.n10_p6_timeout_s"),
        params=case.params(
            "decode_spread.n10_p6",
            {
                "requests": [output("n10", "requests")],
                "fleet": output("n10_before", "snapshot"),
                "after": output("n10_after", "snapshot"),
            },
        ),
    )
    case.step(
        "n10_p2",
        "balance_check",
        timeout_s=case.value("decode_spread.n10_p2_timeout_s"),
        params=case.params(
            "decode_spread.n10_p2",
            {
                "requests": [output("n10", "requests")],
                "fleet": output("n10_before", "snapshot"),
                "after": output("n10_after", "snapshot"),
            },
        ),
    )
    case.step(
        "n10_p1",
        "balance_check",
        timeout_s=case.value("decode_spread.n10_p1_timeout_s"),
        params=case.params(
            "decode_spread.n10_p1",
            {
                "requests": [output("n10", "requests")],
                "fleet": output("n10_before", "snapshot"),
                "after": output("n10_after", "snapshot"),
            },
        ),
    )
    case.step(
        "n50_before",
        "balance_snapshot",
        timeout_s=case.value("decode_spread.n50_before_timeout_s"),
        params=case.value("decode_spread.n50_before"),
    )
    case.step(
        "n50",
        "balance_start",
        timeout_s=case.value("decode_spread.n50_timeout_s"),
        params=case.value("decode_spread.n50"),
    )
    case.step(
        "n50_terminal",
        "balance_wait",
        timeout_s=case.value("decode_spread.n50_terminal_timeout_s"),
        params={"requests": output("n50", "requests")},
    )
    case.step(
        "n50_after",
        "balance_snapshot",
        timeout_s=case.value("decode_spread.n50_after_timeout_s"),
        params=case.value("decode_spread.n50_after"),
    )
    case.observe(
        "n50_p6",
        "balance_check",
        timeout_s=case.value("decode_spread.n50_p6_timeout_s"),
        params=case.params(
            "decode_spread.n50_p6",
            {
                "requests": [output("n50", "requests")],
                "fleet": output("n50_before", "snapshot"),
                "after": output("n50_after", "snapshot"),
            },
        ),
    )
    case.observe(
        "n50_p2",
        "balance_check",
        timeout_s=case.value("decode_spread.n50_p2_timeout_s"),
        params=case.params(
            "decode_spread.n50_p2",
            {
                "requests": [output("n50", "requests")],
                "fleet": output("n50_before", "snapshot"),
                "after": output("n50_after", "snapshot"),
            },
        ),
    )
    case.observe(
        "n50_p1",
        "balance_check",
        timeout_s=case.value("decode_spread.n50_p1_timeout_s"),
        params=case.params(
            "decode_spread.n50_p1",
            {
                "requests": [output("n50", "requests")],
                "fleet": output("n50_before", "snapshot"),
                "after": output("n50_after", "snapshot"),
            },
        ),
    )
    case.step(
        "teardown", "teardown", timeout_s=case.value("decode_spread.teardown_timeout_s")
    )


def length_mixed(case):
    case.step("setup", "setup", timeout_s=case.value("length_mixed.setup_timeout_s"))
    case.step(
        "fleet",
        "balance_snapshot",
        timeout_s=case.value("length_mixed.fleet_timeout_s"),
        params=case.value("length_mixed.fleet"),
    )
    case.step(
        "wave1_long1",
        "balance_start",
        timeout_s=case.value("length_mixed.wave1_long1_timeout_s"),
        params=case.value("length_mixed.wave1_long1"),
    )
    case.step(
        "wave1_long1_pending",
        "balance_pending",
        timeout_s=case.value("length_mixed.wave1_long1_pending_timeout_s"),
        params={
            "requests": output("wave1_long1", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave1_long2",
        "balance_start",
        timeout_s=case.value("length_mixed.wave1_long2_timeout_s"),
        params=case.value("length_mixed.wave1_long2"),
    )
    case.step(
        "wave1_long2_pending",
        "balance_pending",
        timeout_s=case.value("length_mixed.wave1_long2_pending_timeout_s"),
        params={
            "requests": output("wave1_long2", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave1_short",
        "balance_start",
        timeout_s=case.value("length_mixed.wave1_short_timeout_s"),
        params=case.value("length_mixed.wave1_short"),
    )
    case.step(
        "wave1_long1_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave1_long1_terminal_timeout_s"),
        params={"requests": output("wave1_long1", "requests")},
    )
    case.step(
        "wave1_long2_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave1_long2_terminal_timeout_s"),
        params={"requests": output("wave1_long2", "requests")},
    )
    case.step(
        "wave1_short_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave1_short_terminal_timeout_s"),
        params={"requests": output("wave1_short", "requests")},
    )
    case.step(
        "wave1_p6",
        "balance_check",
        timeout_s=case.value("length_mixed.wave1_p6_timeout_s"),
        params=case.params(
            "length_mixed.wave1_p6",
            {
                "requests": [
                    output("wave1_long1", "requests"),
                    output("wave1_long2", "requests"),
                    output("wave1_short", "requests"),
                ],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "wave1_master_clean",
        "balance_clean",
        timeout_s=case.value("length_mixed.wave1_master_clean_timeout_s"),
        params=case.value("length_mixed.wave1_master_clean"),
    )
    case.step(
        "wave2_long1",
        "balance_start",
        timeout_s=case.value("length_mixed.wave2_long1_timeout_s"),
        params=case.value("length_mixed.wave2_long1"),
    )
    case.step(
        "wave2_long1_pending",
        "balance_pending",
        timeout_s=case.value("length_mixed.wave2_long1_pending_timeout_s"),
        params={
            "requests": output("wave2_long1", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave2_long2",
        "balance_start",
        timeout_s=case.value("length_mixed.wave2_long2_timeout_s"),
        params=case.value("length_mixed.wave2_long2"),
    )
    case.step(
        "wave2_long2_pending",
        "balance_pending",
        timeout_s=case.value("length_mixed.wave2_long2_pending_timeout_s"),
        params={
            "requests": output("wave2_long2", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave2_short",
        "balance_start",
        timeout_s=case.value("length_mixed.wave2_short_timeout_s"),
        params=case.value("length_mixed.wave2_short"),
    )
    case.step(
        "wave2_long1_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave2_long1_terminal_timeout_s"),
        params={"requests": output("wave2_long1", "requests")},
    )
    case.step(
        "wave2_long2_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave2_long2_terminal_timeout_s"),
        params={"requests": output("wave2_long2", "requests")},
    )
    case.step(
        "wave2_short_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave2_short_terminal_timeout_s"),
        params={"requests": output("wave2_short", "requests")},
    )
    case.step(
        "wave2_p6",
        "balance_check",
        timeout_s=case.value("length_mixed.wave2_p6_timeout_s"),
        params=case.params(
            "length_mixed.wave2_p6",
            {
                "requests": [
                    output("wave2_long1", "requests"),
                    output("wave2_long2", "requests"),
                    output("wave2_short", "requests"),
                ],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "wave2_master_clean",
        "balance_clean",
        timeout_s=case.value("length_mixed.wave2_master_clean_timeout_s"),
        params=case.value("length_mixed.wave2_master_clean"),
    )
    case.step(
        "wave3_long1",
        "balance_start",
        timeout_s=case.value("length_mixed.wave3_long1_timeout_s"),
        params=case.value("length_mixed.wave3_long1"),
    )
    case.step(
        "wave3_long1_pending",
        "balance_pending",
        timeout_s=case.value("length_mixed.wave3_long1_pending_timeout_s"),
        params={
            "requests": output("wave3_long1", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave3_long2",
        "balance_start",
        timeout_s=case.value("length_mixed.wave3_long2_timeout_s"),
        params=case.value("length_mixed.wave3_long2"),
    )
    case.step(
        "wave3_long2_pending",
        "balance_pending",
        timeout_s=case.value("length_mixed.wave3_long2_pending_timeout_s"),
        params={
            "requests": output("wave3_long2", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave3_short",
        "balance_start",
        timeout_s=case.value("length_mixed.wave3_short_timeout_s"),
        params=case.value("length_mixed.wave3_short"),
    )
    case.step(
        "wave3_long1_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave3_long1_terminal_timeout_s"),
        params={"requests": output("wave3_long1", "requests")},
    )
    case.step(
        "wave3_long2_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave3_long2_terminal_timeout_s"),
        params={"requests": output("wave3_long2", "requests")},
    )
    case.step(
        "wave3_short_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave3_short_terminal_timeout_s"),
        params={"requests": output("wave3_short", "requests")},
    )
    case.step(
        "wave3_p6",
        "balance_check",
        timeout_s=case.value("length_mixed.wave3_p6_timeout_s"),
        params=case.params(
            "length_mixed.wave3_p6",
            {
                "requests": [
                    output("wave3_long1", "requests"),
                    output("wave3_long2", "requests"),
                    output("wave3_short", "requests"),
                ],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "wave3_master_clean",
        "balance_clean",
        timeout_s=case.value("length_mixed.wave3_master_clean_timeout_s"),
        params=case.value("length_mixed.wave3_master_clean"),
    )
    case.step(
        "wave4_long1",
        "balance_start",
        timeout_s=case.value("length_mixed.wave4_long1_timeout_s"),
        params=case.value("length_mixed.wave4_long1"),
    )
    case.step(
        "wave4_long1_pending",
        "balance_pending",
        timeout_s=case.value("length_mixed.wave4_long1_pending_timeout_s"),
        params={
            "requests": output("wave4_long1", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave4_long2",
        "balance_start",
        timeout_s=case.value("length_mixed.wave4_long2_timeout_s"),
        params=case.value("length_mixed.wave4_long2"),
    )
    case.step(
        "wave4_long2_pending",
        "balance_pending",
        timeout_s=case.value("length_mixed.wave4_long2_pending_timeout_s"),
        params={
            "requests": output("wave4_long2", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave4_short",
        "balance_start",
        timeout_s=case.value("length_mixed.wave4_short_timeout_s"),
        params=case.value("length_mixed.wave4_short"),
    )
    case.step(
        "wave4_long1_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave4_long1_terminal_timeout_s"),
        params={"requests": output("wave4_long1", "requests")},
    )
    case.step(
        "wave4_long2_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave4_long2_terminal_timeout_s"),
        params={"requests": output("wave4_long2", "requests")},
    )
    case.step(
        "wave4_short_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave4_short_terminal_timeout_s"),
        params={"requests": output("wave4_short", "requests")},
    )
    case.step(
        "wave4_p6",
        "balance_check",
        timeout_s=case.value("length_mixed.wave4_p6_timeout_s"),
        params=case.params(
            "length_mixed.wave4_p6",
            {
                "requests": [
                    output("wave4_long1", "requests"),
                    output("wave4_long2", "requests"),
                    output("wave4_short", "requests"),
                ],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "wave4_master_clean",
        "balance_clean",
        timeout_s=case.value("length_mixed.wave4_master_clean_timeout_s"),
        params=case.value("length_mixed.wave4_master_clean"),
    )
    case.step(
        "wave5_long1",
        "balance_start",
        timeout_s=case.value("length_mixed.wave5_long1_timeout_s"),
        params=case.value("length_mixed.wave5_long1"),
    )
    case.step(
        "wave5_long1_pending",
        "balance_pending",
        timeout_s=case.value("length_mixed.wave5_long1_pending_timeout_s"),
        params={
            "requests": output("wave5_long1", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave5_long2",
        "balance_start",
        timeout_s=case.value("length_mixed.wave5_long2_timeout_s"),
        params=case.value("length_mixed.wave5_long2"),
    )
    case.step(
        "wave5_long2_pending",
        "balance_pending",
        timeout_s=case.value("length_mixed.wave5_long2_pending_timeout_s"),
        params={
            "requests": output("wave5_long2", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "wave5_short",
        "balance_start",
        timeout_s=case.value("length_mixed.wave5_short_timeout_s"),
        params=case.value("length_mixed.wave5_short"),
    )
    case.step(
        "wave5_long1_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave5_long1_terminal_timeout_s"),
        params={"requests": output("wave5_long1", "requests")},
    )
    case.step(
        "wave5_long2_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave5_long2_terminal_timeout_s"),
        params={"requests": output("wave5_long2", "requests")},
    )
    case.step(
        "wave5_short_terminal",
        "balance_wait",
        timeout_s=case.value("length_mixed.wave5_short_terminal_timeout_s"),
        params={"requests": output("wave5_short", "requests")},
    )
    case.observe(
        "wave5_p6",
        "balance_check",
        timeout_s=case.value("length_mixed.wave5_p6_timeout_s"),
        params=case.params(
            "length_mixed.wave5_p6",
            {
                "requests": [
                    output("wave5_long1", "requests"),
                    output("wave5_long2", "requests"),
                    output("wave5_short", "requests"),
                ],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "wave5_master_clean",
        "balance_clean",
        timeout_s=case.value("length_mixed.wave5_master_clean_timeout_s"),
        params=case.value("length_mixed.wave5_master_clean"),
    )
    case.observe(
        "token_p3",
        "balance_check",
        timeout_s=case.value("length_mixed.token_p3_timeout_s"),
        params=case.params(
            "length_mixed.token_p3",
            {
                "requests": [
                    output("wave1_long1", "requests"),
                    output("wave1_long2", "requests"),
                    output("wave1_short", "requests"),
                    output("wave2_long1", "requests"),
                    output("wave2_long2", "requests"),
                    output("wave2_short", "requests"),
                    output("wave3_long1", "requests"),
                    output("wave3_long2", "requests"),
                    output("wave3_short", "requests"),
                    output("wave4_long1", "requests"),
                    output("wave4_long2", "requests"),
                    output("wave4_short", "requests"),
                    output("wave5_long1", "requests"),
                    output("wave5_long2", "requests"),
                    output("wave5_short", "requests"),
                ],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.observe(
        "short_p2",
        "balance_check",
        timeout_s=case.value("length_mixed.short_p2_timeout_s"),
        params=case.params(
            "length_mixed.short_p2",
            {
                "requests": [
                    output("wave1_long1", "requests"),
                    output("wave1_long2", "requests"),
                    output("wave1_short", "requests"),
                    output("wave2_long1", "requests"),
                    output("wave2_long2", "requests"),
                    output("wave2_short", "requests"),
                    output("wave3_long1", "requests"),
                    output("wave3_long2", "requests"),
                    output("wave3_short", "requests"),
                    output("wave4_long1", "requests"),
                    output("wave4_long2", "requests"),
                    output("wave4_short", "requests"),
                    output("wave5_long1", "requests"),
                    output("wave5_long2", "requests"),
                    output("wave5_short", "requests"),
                ],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "teardown", "teardown", timeout_s=case.value("length_mixed.teardown_timeout_s")
    )
