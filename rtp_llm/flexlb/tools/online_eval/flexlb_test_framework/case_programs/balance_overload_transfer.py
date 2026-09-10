"""Explicit balance stages preserving client landing, token share, Decode deltas and latency contracts."""

from ..case_config import output


def decode_pressure(case):
    case.step("setup", "setup", timeout_s=case.value("decode_pressure.setup_timeout_s"))
    case.step(
        "fleet",
        "balance_snapshot",
        timeout_s=case.value("decode_pressure.fleet_timeout_s"),
        params=case.value("decode_pressure.fleet"),
    )
    case.step(
        "pressure",
        "balance_pressure",
        timeout_s=case.value("decode_pressure.pressure_timeout_s"),
        params={
            "fleet": output("fleet", "snapshot"),
            "target": output("fleet", "first"),
        },
    )
    case.step(
        "status_sync",
        "balance_pause",
        timeout_s=case.value("decode_pressure.status_sync_timeout_s"),
        params=case.value("decode_pressure.status_sync"),
    )
    case.step(
        "before",
        "balance_snapshot",
        timeout_s=case.value("decode_pressure.before_timeout_s"),
        params=case.value("decode_pressure.before"),
    )
    case.step(
        "traffic",
        "balance_start",
        timeout_s=case.value("decode_pressure.traffic_timeout_s"),
        params=case.value("decode_pressure.traffic"),
    )
    case.step(
        "terminal",
        "balance_wait",
        timeout_s=case.value("decode_pressure.terminal_timeout_s"),
        params={"requests": output("traffic", "requests")},
    )
    case.step(
        "after",
        "balance_snapshot",
        timeout_s=case.value("decode_pressure.after_timeout_s"),
        params=case.value("decode_pressure.after"),
    )
    case.step(
        "p6",
        "balance_check",
        timeout_s=case.value("decode_pressure.p6_timeout_s"),
        params=case.params(
            "decode_pressure.p6",
            {
                "requests": [output("traffic", "requests")],
                "fleet": output("before", "snapshot"),
                "after": output("after", "snapshot"),
            },
        ),
    )
    case.step(
        "p5",
        "balance_check",
        timeout_s=case.value("decode_pressure.p5_timeout_s"),
        params=case.params(
            "decode_pressure.p5",
            {
                "requests": [output("traffic", "requests")],
                "fleet": output("before", "snapshot"),
                "after": output("after", "snapshot"),
                "target": output("pressure", "target"),
            },
        ),
    )
    case.step(
        "p2",
        "balance_check",
        timeout_s=case.value("decode_pressure.p2_timeout_s"),
        params=case.params(
            "decode_pressure.p2",
            {
                "requests": [output("traffic", "requests")],
                "fleet": output("before", "snapshot"),
                "after": output("after", "snapshot"),
                "target": output("pressure", "target"),
            },
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("decode_pressure.teardown_timeout_s"),
    )


def prefill_pressure(case):
    case.step(
        "setup", "setup", timeout_s=case.value("prefill_pressure.setup_timeout_s")
    )
    case.step(
        "fleet",
        "balance_snapshot",
        timeout_s=case.value("prefill_pressure.fleet_timeout_s"),
        params=case.value("prefill_pressure.fleet"),
    )
    case.step(
        "perf_sync",
        "balance_pause",
        timeout_s=case.value("prefill_pressure.perf_sync_timeout_s"),
        params=case.value("prefill_pressure.perf_sync"),
    )
    case.step(
        "seed",
        "balance_start",
        timeout_s=case.value("prefill_pressure.seed_timeout_s"),
        params=case.value("prefill_pressure.seed"),
    )
    case.step(
        "seed_pending",
        "balance_pending",
        timeout_s=case.value("prefill_pressure.seed_pending_timeout_s"),
        params={
            "requests": output("seed", "requests"),
            "fleet": output("fleet", "snapshot"),
        },
    )
    case.step(
        "cool_sync",
        "balance_pause",
        timeout_s=case.value("prefill_pressure.cool_sync_timeout_s"),
        params=case.value("prefill_pressure.cool_sync"),
    )
    case.step(
        "baseline",
        "balance_start",
        timeout_s=case.value("prefill_pressure.baseline_timeout_s"),
        params=case.value("prefill_pressure.baseline"),
    )
    case.step(
        "baseline_terminal",
        "balance_wait",
        timeout_s=case.value("prefill_pressure.baseline_terminal_timeout_s"),
        params={"requests": output("baseline", "requests")},
    )
    case.step(
        "baseline_p6",
        "balance_check",
        timeout_s=case.value("prefill_pressure.baseline_p6_timeout_s"),
        params=case.params(
            "prefill_pressure.baseline_p6",
            {
                "requests": [output("baseline", "requests")],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "wave",
        "balance_start",
        timeout_s=case.value("prefill_pressure.wave_timeout_s"),
        params=case.value("prefill_pressure.wave"),
    )
    case.step(
        "wave_terminal",
        "balance_wait",
        timeout_s=case.value("prefill_pressure.wave_terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "p6",
        "balance_check",
        timeout_s=case.value("prefill_pressure.p6_timeout_s"),
        params=case.params(
            "prefill_pressure.p6",
            {
                "requests": [output("wave", "requests")],
                "fleet": output("fleet", "snapshot"),
            },
        ),
    )
    case.step(
        "p5",
        "balance_check",
        timeout_s=case.value("prefill_pressure.p5_timeout_s"),
        params=case.params(
            "prefill_pressure.p5",
            {
                "requests": [output("wave", "requests")],
                "fleet": output("fleet", "snapshot"),
                "target": output("seed_pending", "hot"),
            },
        ),
    )
    case.step(
        "p7",
        "balance_check",
        timeout_s=case.value("prefill_pressure.p7_timeout_s"),
        params=case.params(
            "prefill_pressure.p7",
            {
                "requests": [output("wave", "requests")],
                "fleet": output("fleet", "snapshot"),
                "baseline": output("baseline", "requests"),
            },
        ),
    )
    case.step(
        "seed_terminal",
        "balance_wait",
        timeout_s=case.value("prefill_pressure.seed_terminal_timeout_s"),
        params={"requests": output("seed", "requests")},
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("prefill_pressure.teardown_timeout_s"),
    )
