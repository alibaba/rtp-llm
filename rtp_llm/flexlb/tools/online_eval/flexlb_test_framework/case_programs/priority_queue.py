"""Concurrent priority queue ordering with explicit terminal and owner checks."""

from ..case_config import output


def same_level_fifo(case):
    case.step("setup", "setup", timeout_s=case.value("same_level_fifo.setup_timeout_s"))
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "same_level_fifo.slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step("sync", "balance_pause", params=case.value("same_level_fifo.sync"))
    case.step(
        "wave",
        "priority_start",
        timeout_s=case.value("same_level_fifo.wave_timeout_s"),
        params=case.value("same_level_fifo.wave"),
    )
    case.step(
        "terminal",
        "priority_wait",
        timeout_s=case.value("same_level_fifo.terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step("fifo", "priority_fifo", params={"requests": output("wave", "requests")})
    case.step(
        "owner_clean",
        "balance_clean",
        timeout_s=case.value("same_level_fifo.owner_clean_timeout_s"),
    )
    case.step(
        "restore",
        "engine_control",
        params=case.params(
            "same_level_fifo.restore", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("same_level_fifo.teardown_timeout_s"),
    )


def low_no_starvation(case):
    case.step(
        "setup", "setup", timeout_s=case.value("low_no_starvation.setup_timeout_s")
    )
    case.step(
        "slow",
        "priority_prefill_perf",
        params=case.value("low_no_starvation.slow"),
    )
    case.step("sync", "balance_pause", params=case.value("low_no_starvation.sync"))
    case.step(
        "wave0",
        "priority_start",
        timeout_s=case.value("low_no_starvation.wave0_timeout_s"),
        params=case.value("low_no_starvation.wave0"),
    )
    case.step(
        "terminal0",
        "priority_wait",
        timeout_s=case.value("low_no_starvation.terminal0_timeout_s"),
        params={"requests": output("wave0", "requests")},
    )
    case.step(
        "clean0",
        "balance_clean",
        timeout_s=case.value("low_no_starvation.clean0_timeout_s"),
    )
    case.step("quiet0", "balance_pause", params=case.value("low_no_starvation.quiet0"))
    case.step(
        "wave1",
        "priority_start",
        timeout_s=case.value("low_no_starvation.wave1_timeout_s"),
        params=case.value("low_no_starvation.wave1"),
    )
    case.step(
        "terminal1",
        "priority_wait",
        timeout_s=case.value("low_no_starvation.terminal1_timeout_s"),
        params={"requests": output("wave1", "requests")},
    )
    case.step(
        "clean1",
        "balance_clean",
        timeout_s=case.value("low_no_starvation.clean1_timeout_s"),
    )
    case.step("quiet1", "balance_pause", params=case.value("low_no_starvation.quiet1"))
    case.step(
        "final_clean",
        "balance_clean",
        timeout_s=case.value("low_no_starvation.final_clean_timeout_s"),
    )
    case.step(
        "completion",
        "priority_completion",
        params={"requests": [output("wave0", "requests"), output("wave1", "requests")]},
    )
    case.step(
        "restore",
        "priority_prefill_perf",
        params=case.value("low_no_starvation.restore"),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("low_no_starvation.teardown_timeout_s"),
    )


def queue_timeout_terminal(case):
    case.step(
        "setup", "setup", timeout_s=case.value("queue_timeout_terminal.setup_timeout_s")
    )
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "queue_timeout_terminal.slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step("sync", "balance_pause", params=case.value("queue_timeout_terminal.sync"))
    case.step(
        "placeholder",
        "priority_start",
        timeout_s=case.value("queue_timeout_terminal.placeholder_timeout_s"),
        params=case.value("queue_timeout_terminal.placeholder"),
    )
    case.step(
        "pending",
        "priority_pending",
        timeout_s=case.value("queue_timeout_terminal.pending_timeout_s"),
        params={
            "prefill": output("fleet", "prefill"),
            "requests": output("placeholder", "requests"),
        },
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=case.value("queue_timeout_terminal.wave_timeout_s"),
        params=case.value("queue_timeout_terminal.wave"),
    )
    case.step(
        "wave_settled",
        "priority_settled",
        timeout_s=case.value("queue_timeout_terminal.wave_settled_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_terminal",
        "priority_wait",
        timeout_s=case.value("queue_timeout_terminal.placeholder_terminal_timeout_s"),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave_terminal",
        "priority_wait",
        timeout_s=case.value("queue_timeout_terminal.wave_terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "expiry",
        "priority_expiry",
        params={
            "placeholder": output("placeholder", "requests"),
            "wave": output("wave", "requests"),
        },
    )
    case.step(
        "restore",
        "engine_control",
        params=case.params(
            "queue_timeout_terminal.restore", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("queue_timeout_terminal.teardown_timeout_s"),
    )


def order_basic(case):
    case.step("setup", "setup", timeout_s=case.value("order_basic.setup_timeout_s"))
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "order_basic.slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step("sync", "balance_pause", params=case.value("order_basic.sync"))
    case.step(
        "placeholder",
        "priority_start",
        timeout_s=case.value("order_basic.placeholder_timeout_s"),
        params=case.value("order_basic.placeholder"),
    )
    case.step(
        "pending",
        "priority_pending",
        timeout_s=case.value("order_basic.pending_timeout_s"),
        params={
            "prefill": output("fleet", "prefill"),
            "requests": output("placeholder", "requests"),
        },
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=case.value("order_basic.wave_timeout_s"),
        params=case.value("order_basic.wave"),
    )
    case.step(
        "wave_settled",
        "priority_settled",
        timeout_s=case.value("order_basic.wave_settled_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_terminal",
        "priority_wait",
        timeout_s=case.value("order_basic.placeholder_terminal_timeout_s"),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave_terminal",
        "priority_wait",
        timeout_s=case.value("order_basic.wave_terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "order",
        "priority_order_basic",
        params=case.params(
            "order_basic.order",
            {
                "placeholder": output("placeholder", "requests"),
                "wave": output("wave", "requests"),
            },
        ),
    )
    case.step(
        "owner_clean",
        "balance_clean",
        timeout_s=case.value("order_basic.owner_clean_timeout_s"),
    )
    case.step(
        "restore",
        "engine_control",
        params=case.params(
            "order_basic.restore", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "teardown", "teardown", timeout_s=case.value("order_basic.teardown_timeout_s")
    )


def normalize_default50(case):
    case.step(
        "setup", "setup", timeout_s=case.value("normalize_default50.setup_timeout_s")
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=case.value("normalize_default50.wave_timeout_s"),
        params=case.value("normalize_default50.wave"),
    )
    case.step(
        "terminal",
        "priority_normalize_wait",
        timeout_s=case.value("normalize_default50.terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "normalization",
        "priority_normalize_order",
        params=case.params(
            "normalize_default50.normalization",
            {"requests": [output("wave", "requests")]},
        ),
    )
    case.step(
        "owner_clean",
        "balance_clean",
        timeout_s=case.value("normalize_default50.owner_clean_timeout_s"),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("normalize_default50.teardown_timeout_s"),
    )


def normalize_channels(case):
    case.step(
        "setup", "setup", timeout_s=case.value("normalize_channels.setup_timeout_s")
    )
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "normalize_channels.slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step("sync", "balance_pause", params=case.value("normalize_channels.sync"))
    case.step(
        "placeholder",
        "priority_start",
        timeout_s=case.value("normalize_channels.placeholder_timeout_s"),
        params=case.value("normalize_channels.placeholder"),
    )
    case.step(
        "pending",
        "priority_pending",
        timeout_s=case.value("normalize_channels.pending_timeout_s"),
        params={
            "prefill": output("fleet", "prefill"),
            "requests": output("placeholder", "requests"),
        },
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=case.value("normalize_channels.wave_timeout_s"),
        params=case.value("normalize_channels.wave"),
    )
    case.step(
        "wave_settled",
        "priority_settled",
        timeout_s=case.value("normalize_channels.wave_settled_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_terminal",
        "priority_wait",
        timeout_s=case.value("normalize_channels.placeholder_terminal_timeout_s"),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave_terminal",
        "priority_wait",
        timeout_s=case.value("normalize_channels.wave_terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "normalization",
        "priority_normalize_order",
        params=case.params(
            "normalize_channels.normalization",
            {
                "requests": [
                    output("placeholder", "requests"),
                    output("wave", "requests"),
                ]
            },
        ),
    )
    case.step(
        "owner_clean",
        "balance_clean",
        timeout_s=case.value("normalize_channels.owner_clean_timeout_s"),
    )
    case.step(
        "restore",
        "engine_control",
        params=case.params(
            "normalize_channels.restore", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("normalize_channels.teardown_timeout_s"),
    )


def normalize_default30(case):
    case.step(
        "setup", "setup", timeout_s=case.value("normalize_default30.setup_timeout_s")
    )
    case.step("fleet", "priority_fleet")
    case.step(
        "slow",
        "engine_control",
        params=case.params(
            "normalize_default30.slow", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step("sync", "balance_pause", params=case.value("normalize_default30.sync"))
    case.step(
        "placeholder",
        "priority_start",
        timeout_s=case.value("normalize_default30.placeholder_timeout_s"),
        params=case.value("normalize_default30.placeholder"),
    )
    case.step(
        "pending",
        "priority_pending",
        timeout_s=case.value("normalize_default30.pending_timeout_s"),
        params={
            "prefill": output("fleet", "prefill"),
            "requests": output("placeholder", "requests"),
        },
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=case.value("normalize_default30.wave_timeout_s"),
        params=case.value("normalize_default30.wave"),
    )
    case.step(
        "wave_settled",
        "priority_settled",
        timeout_s=case.value("normalize_default30.wave_settled_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "placeholder_terminal",
        "priority_wait",
        timeout_s=case.value("normalize_default30.placeholder_terminal_timeout_s"),
        params={"requests": output("placeholder", "requests")},
    )
    case.step(
        "wave_terminal",
        "priority_wait",
        timeout_s=case.value("normalize_default30.wave_terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "normalization",
        "priority_normalize_order",
        params=case.params(
            "normalize_default30.normalization",
            {
                "requests": [
                    output("placeholder", "requests"),
                    output("wave", "requests"),
                ]
            },
        ),
    )
    case.step(
        "owner_clean",
        "balance_clean",
        timeout_s=case.value("normalize_default30.owner_clean_timeout_s"),
    )
    case.step(
        "restore",
        "engine_control",
        params=case.params(
            "normalize_default30.restore", {"targets": [output("fleet", "prefill")]}
        ),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("normalize_default30.teardown_timeout_s"),
    )


def normalize_metrics(case):
    case.step(
        "setup", "setup", timeout_s=case.value("normalize_metrics.setup_timeout_s")
    )
    case.step(
        "wave",
        "priority_start",
        timeout_s=case.value("normalize_metrics.wave_timeout_s"),
        params=case.value("normalize_metrics.wave"),
    )
    case.step(
        "terminal",
        "priority_normalize_metrics_wait",
        timeout_s=case.value("normalize_metrics.terminal_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "metrics",
        "priority_normalize_metrics",
        timeout_s=case.value("normalize_metrics.metrics_timeout_s"),
        params={"requests": output("wave", "requests")},
    )
    case.step(
        "owner_clean",
        "balance_clean",
        timeout_s=case.value("normalize_metrics.owner_clean_timeout_s"),
    )
    case.step(
        "teardown",
        "teardown",
        timeout_s=case.value("normalize_metrics.teardown_timeout_s"),
    )
