"""Pre-request hit rate and windowed holder masks under four-family churn; bounded LRU replay retry and capacity eviction."""

from ..case_config import output


def hot_churn(case):
    case.step("setup", "setup", timeout_s=case.value("hot_churn.setup_timeout_s"))
    case.step(
        "w0_r0_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w0_r0_before_timeout_s"),
        params=case.value("hot_churn.w0_r0_before"),
    )
    case.step(
        "w0_r0",
        "request",
        params=case.value("hot_churn.w0_r0"),
    )
    case.step(
        "w0_r0_done",
        "wait",
        timeout_s=case.value("hot_churn.w0_r0_done_timeout_s"),
        params={"requests": output("w0_r0", "requests")},
    )
    case.step(
        "w0_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w0_r0_hit",
            {
                "snapshot": output("w0_r0_before", "snapshot"),
                "requests": output("w0_r0", "requests"),
            },
        ),
    )
    case.step(
        "w0_r1_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w0_r1_before_timeout_s"),
        params=case.value("hot_churn.w0_r1_before"),
    )
    case.step(
        "w0_r1",
        "request",
        params=case.value("hot_churn.w0_r1"),
    )
    case.step(
        "w0_r1_done",
        "wait",
        timeout_s=case.value("hot_churn.w0_r1_done_timeout_s"),
        params={"requests": output("w0_r1", "requests")},
    )
    case.step(
        "w0_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w0_r1_hit",
            {
                "snapshot": output("w0_r1_before", "snapshot"),
                "requests": output("w0_r1", "requests"),
            },
        ),
    )
    case.step(
        "w0_r2_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w0_r2_before_timeout_s"),
        params=case.value("hot_churn.w0_r2_before"),
    )
    case.step(
        "w0_r2",
        "request",
        params=case.value("hot_churn.w0_r2"),
    )
    case.step(
        "w0_r2_done",
        "wait",
        timeout_s=case.value("hot_churn.w0_r2_done_timeout_s"),
        params={"requests": output("w0_r2", "requests")},
    )
    case.step(
        "w0_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w0_r2_hit",
            {
                "snapshot": output("w0_r2_before", "snapshot"),
                "requests": output("w0_r2", "requests"),
            },
        ),
    )
    case.step(
        "w0_r3_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w0_r3_before_timeout_s"),
        params=case.value("hot_churn.w0_r3_before"),
    )
    case.step(
        "w0_r3",
        "request",
        params=case.value("hot_churn.w0_r3"),
    )
    case.step(
        "w0_r3_done",
        "wait",
        timeout_s=case.value("hot_churn.w0_r3_done_timeout_s"),
        params={"requests": output("w0_r3", "requests")},
    )
    case.step(
        "w0_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w0_r3_hit",
            {
                "snapshot": output("w0_r3_before", "snapshot"),
                "requests": output("w0_r3", "requests"),
            },
        ),
    )
    case.step(
        "w0_r4_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w0_r4_before_timeout_s"),
        params=case.value("hot_churn.w0_r4_before"),
    )
    case.step(
        "w0_r4",
        "request",
        params=case.value("hot_churn.w0_r4"),
    )
    case.step(
        "w0_r4_done",
        "wait",
        timeout_s=case.value("hot_churn.w0_r4_done_timeout_s"),
        params={"requests": output("w0_r4", "requests")},
    )
    case.step(
        "w0_r4_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w0_r4_hit",
            {
                "snapshot": output("w0_r4_before", "snapshot"),
                "requests": output("w0_r4", "requests"),
            },
        ),
    )
    case.step(
        "w0_end",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w0_end_timeout_s"),
        params=case.value("hot_churn.w0_end"),
    )
    case.step(
        "w1_r0_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w1_r0_before_timeout_s"),
        params=case.value("hot_churn.w1_r0_before"),
    )
    case.step(
        "w1_r0",
        "request",
        params=case.value("hot_churn.w1_r0"),
    )
    case.step(
        "w1_r0_done",
        "wait",
        timeout_s=case.value("hot_churn.w1_r0_done_timeout_s"),
        params={"requests": output("w1_r0", "requests")},
    )
    case.step(
        "w1_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w1_r0_hit",
            {
                "snapshot": output("w1_r0_before", "snapshot"),
                "requests": output("w1_r0", "requests"),
            },
        ),
    )
    case.step(
        "w1_r1_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w1_r1_before_timeout_s"),
        params=case.value("hot_churn.w1_r1_before"),
    )
    case.step(
        "w1_r1",
        "request",
        params=case.value("hot_churn.w1_r1"),
    )
    case.step(
        "w1_r1_done",
        "wait",
        timeout_s=case.value("hot_churn.w1_r1_done_timeout_s"),
        params={"requests": output("w1_r1", "requests")},
    )
    case.step(
        "w1_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w1_r1_hit",
            {
                "snapshot": output("w1_r1_before", "snapshot"),
                "requests": output("w1_r1", "requests"),
            },
        ),
    )
    case.step(
        "w1_r2_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w1_r2_before_timeout_s"),
        params=case.value("hot_churn.w1_r2_before"),
    )
    case.step(
        "w1_r2",
        "request",
        params=case.value("hot_churn.w1_r2"),
    )
    case.step(
        "w1_r2_done",
        "wait",
        timeout_s=case.value("hot_churn.w1_r2_done_timeout_s"),
        params={"requests": output("w1_r2", "requests")},
    )
    case.step(
        "w1_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w1_r2_hit",
            {
                "snapshot": output("w1_r2_before", "snapshot"),
                "requests": output("w1_r2", "requests"),
            },
        ),
    )
    case.step(
        "w1_r3_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w1_r3_before_timeout_s"),
        params=case.value("hot_churn.w1_r3_before"),
    )
    case.step(
        "w1_r3",
        "request",
        params=case.value("hot_churn.w1_r3"),
    )
    case.step(
        "w1_r3_done",
        "wait",
        timeout_s=case.value("hot_churn.w1_r3_done_timeout_s"),
        params={"requests": output("w1_r3", "requests")},
    )
    case.step(
        "w1_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w1_r3_hit",
            {
                "snapshot": output("w1_r3_before", "snapshot"),
                "requests": output("w1_r3", "requests"),
            },
        ),
    )
    case.step(
        "w1_r4_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w1_r4_before_timeout_s"),
        params=case.value("hot_churn.w1_r4_before"),
    )
    case.step(
        "w1_r4",
        "request",
        params=case.value("hot_churn.w1_r4"),
    )
    case.step(
        "w1_r4_done",
        "wait",
        timeout_s=case.value("hot_churn.w1_r4_done_timeout_s"),
        params={"requests": output("w1_r4", "requests")},
    )
    case.step(
        "w1_r4_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w1_r4_hit",
            {
                "snapshot": output("w1_r4_before", "snapshot"),
                "requests": output("w1_r4", "requests"),
            },
        ),
    )
    case.step(
        "w1_end",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w1_end_timeout_s"),
        params=case.value("hot_churn.w1_end"),
    )
    case.step(
        "w2_r0_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w2_r0_before_timeout_s"),
        params=case.value("hot_churn.w2_r0_before"),
    )
    case.step(
        "w2_r0",
        "request",
        params=case.value("hot_churn.w2_r0"),
    )
    case.step(
        "w2_r0_done",
        "wait",
        timeout_s=case.value("hot_churn.w2_r0_done_timeout_s"),
        params={"requests": output("w2_r0", "requests")},
    )
    case.step(
        "w2_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w2_r0_hit",
            {
                "snapshot": output("w2_r0_before", "snapshot"),
                "requests": output("w2_r0", "requests"),
            },
        ),
    )
    case.step(
        "w2_r1_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w2_r1_before_timeout_s"),
        params=case.value("hot_churn.w2_r1_before"),
    )
    case.step(
        "w2_r1",
        "request",
        params=case.value("hot_churn.w2_r1"),
    )
    case.step(
        "w2_r1_done",
        "wait",
        timeout_s=case.value("hot_churn.w2_r1_done_timeout_s"),
        params={"requests": output("w2_r1", "requests")},
    )
    case.step(
        "w2_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w2_r1_hit",
            {
                "snapshot": output("w2_r1_before", "snapshot"),
                "requests": output("w2_r1", "requests"),
            },
        ),
    )
    case.step(
        "w2_r2_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w2_r2_before_timeout_s"),
        params=case.value("hot_churn.w2_r2_before"),
    )
    case.step(
        "w2_r2",
        "request",
        params=case.value("hot_churn.w2_r2"),
    )
    case.step(
        "w2_r2_done",
        "wait",
        timeout_s=case.value("hot_churn.w2_r2_done_timeout_s"),
        params={"requests": output("w2_r2", "requests")},
    )
    case.step(
        "w2_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w2_r2_hit",
            {
                "snapshot": output("w2_r2_before", "snapshot"),
                "requests": output("w2_r2", "requests"),
            },
        ),
    )
    case.step(
        "w2_r3_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w2_r3_before_timeout_s"),
        params=case.value("hot_churn.w2_r3_before"),
    )
    case.step(
        "w2_r3",
        "request",
        params=case.value("hot_churn.w2_r3"),
    )
    case.step(
        "w2_r3_done",
        "wait",
        timeout_s=case.value("hot_churn.w2_r3_done_timeout_s"),
        params={"requests": output("w2_r3", "requests")},
    )
    case.step(
        "w2_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w2_r3_hit",
            {
                "snapshot": output("w2_r3_before", "snapshot"),
                "requests": output("w2_r3", "requests"),
            },
        ),
    )
    case.step(
        "w2_r4_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w2_r4_before_timeout_s"),
        params=case.value("hot_churn.w2_r4_before"),
    )
    case.step(
        "w2_r4",
        "request",
        params=case.value("hot_churn.w2_r4"),
    )
    case.step(
        "w2_r4_done",
        "wait",
        timeout_s=case.value("hot_churn.w2_r4_done_timeout_s"),
        params={"requests": output("w2_r4", "requests")},
    )
    case.step(
        "w2_r4_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w2_r4_hit",
            {
                "snapshot": output("w2_r4_before", "snapshot"),
                "requests": output("w2_r4", "requests"),
            },
        ),
    )
    case.step(
        "w2_end",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w2_end_timeout_s"),
        params=case.value("hot_churn.w2_end"),
    )
    case.step(
        "w3_r0_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w3_r0_before_timeout_s"),
        params=case.value("hot_churn.w3_r0_before"),
    )
    case.step(
        "w3_r0",
        "request",
        params=case.value("hot_churn.w3_r0"),
    )
    case.step(
        "w3_r0_done",
        "wait",
        timeout_s=case.value("hot_churn.w3_r0_done_timeout_s"),
        params={"requests": output("w3_r0", "requests")},
    )
    case.step(
        "w3_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w3_r0_hit",
            {
                "snapshot": output("w3_r0_before", "snapshot"),
                "requests": output("w3_r0", "requests"),
            },
        ),
    )
    case.step(
        "w3_r1_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w3_r1_before_timeout_s"),
        params=case.value("hot_churn.w3_r1_before"),
    )
    case.step(
        "w3_r1",
        "request",
        params=case.value("hot_churn.w3_r1"),
    )
    case.step(
        "w3_r1_done",
        "wait",
        timeout_s=case.value("hot_churn.w3_r1_done_timeout_s"),
        params={"requests": output("w3_r1", "requests")},
    )
    case.step(
        "w3_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w3_r1_hit",
            {
                "snapshot": output("w3_r1_before", "snapshot"),
                "requests": output("w3_r1", "requests"),
            },
        ),
    )
    case.step(
        "w3_r2_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w3_r2_before_timeout_s"),
        params=case.value("hot_churn.w3_r2_before"),
    )
    case.step(
        "w3_r2",
        "request",
        params=case.value("hot_churn.w3_r2"),
    )
    case.step(
        "w3_r2_done",
        "wait",
        timeout_s=case.value("hot_churn.w3_r2_done_timeout_s"),
        params={"requests": output("w3_r2", "requests")},
    )
    case.step(
        "w3_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w3_r2_hit",
            {
                "snapshot": output("w3_r2_before", "snapshot"),
                "requests": output("w3_r2", "requests"),
            },
        ),
    )
    case.step(
        "w3_r3_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w3_r3_before_timeout_s"),
        params=case.value("hot_churn.w3_r3_before"),
    )
    case.step(
        "w3_r3",
        "request",
        params=case.value("hot_churn.w3_r3"),
    )
    case.step(
        "w3_r3_done",
        "wait",
        timeout_s=case.value("hot_churn.w3_r3_done_timeout_s"),
        params={"requests": output("w3_r3", "requests")},
    )
    case.step(
        "w3_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w3_r3_hit",
            {
                "snapshot": output("w3_r3_before", "snapshot"),
                "requests": output("w3_r3", "requests"),
            },
        ),
    )
    case.step(
        "w3_r4_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w3_r4_before_timeout_s"),
        params=case.value("hot_churn.w3_r4_before"),
    )
    case.step(
        "w3_r4",
        "request",
        params=case.value("hot_churn.w3_r4"),
    )
    case.step(
        "w3_r4_done",
        "wait",
        timeout_s=case.value("hot_churn.w3_r4_done_timeout_s"),
        params={"requests": output("w3_r4", "requests")},
    )
    case.step(
        "w3_r4_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w3_r4_hit",
            {
                "snapshot": output("w3_r4_before", "snapshot"),
                "requests": output("w3_r4", "requests"),
            },
        ),
    )
    case.step(
        "w3_end",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w3_end_timeout_s"),
        params=case.value("hot_churn.w3_end"),
    )
    case.step(
        "w4_r0_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w4_r0_before_timeout_s"),
        params=case.value("hot_churn.w4_r0_before"),
    )
    case.step(
        "w4_r0",
        "request",
        params=case.value("hot_churn.w4_r0"),
    )
    case.step(
        "w4_r0_done",
        "wait",
        timeout_s=case.value("hot_churn.w4_r0_done_timeout_s"),
        params={"requests": output("w4_r0", "requests")},
    )
    case.step(
        "w4_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w4_r0_hit",
            {
                "snapshot": output("w4_r0_before", "snapshot"),
                "requests": output("w4_r0", "requests"),
            },
        ),
    )
    case.step(
        "w4_r1_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w4_r1_before_timeout_s"),
        params=case.value("hot_churn.w4_r1_before"),
    )
    case.step(
        "w4_r1",
        "request",
        params=case.value("hot_churn.w4_r1"),
    )
    case.step(
        "w4_r1_done",
        "wait",
        timeout_s=case.value("hot_churn.w4_r1_done_timeout_s"),
        params={"requests": output("w4_r1", "requests")},
    )
    case.step(
        "w4_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w4_r1_hit",
            {
                "snapshot": output("w4_r1_before", "snapshot"),
                "requests": output("w4_r1", "requests"),
            },
        ),
    )
    case.step(
        "w4_r2_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w4_r2_before_timeout_s"),
        params=case.value("hot_churn.w4_r2_before"),
    )
    case.step(
        "w4_r2",
        "request",
        params=case.value("hot_churn.w4_r2"),
    )
    case.step(
        "w4_r2_done",
        "wait",
        timeout_s=case.value("hot_churn.w4_r2_done_timeout_s"),
        params={"requests": output("w4_r2", "requests")},
    )
    case.step(
        "w4_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w4_r2_hit",
            {
                "snapshot": output("w4_r2_before", "snapshot"),
                "requests": output("w4_r2", "requests"),
            },
        ),
    )
    case.step(
        "w4_r3_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w4_r3_before_timeout_s"),
        params=case.value("hot_churn.w4_r3_before"),
    )
    case.step(
        "w4_r3",
        "request",
        params=case.value("hot_churn.w4_r3"),
    )
    case.step(
        "w4_r3_done",
        "wait",
        timeout_s=case.value("hot_churn.w4_r3_done_timeout_s"),
        params={"requests": output("w4_r3", "requests")},
    )
    case.step(
        "w4_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w4_r3_hit",
            {
                "snapshot": output("w4_r3_before", "snapshot"),
                "requests": output("w4_r3", "requests"),
            },
        ),
    )
    case.step(
        "w4_r4_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w4_r4_before_timeout_s"),
        params=case.value("hot_churn.w4_r4_before"),
    )
    case.step(
        "w4_r4",
        "request",
        params=case.value("hot_churn.w4_r4"),
    )
    case.step(
        "w4_r4_done",
        "wait",
        timeout_s=case.value("hot_churn.w4_r4_done_timeout_s"),
        params={"requests": output("w4_r4", "requests")},
    )
    case.step(
        "w4_r4_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w4_r4_hit",
            {
                "snapshot": output("w4_r4_before", "snapshot"),
                "requests": output("w4_r4", "requests"),
            },
        ),
    )
    case.step(
        "w4_end",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w4_end_timeout_s"),
        params=case.value("hot_churn.w4_end"),
    )
    case.step(
        "w5_r0_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w5_r0_before_timeout_s"),
        params=case.value("hot_churn.w5_r0_before"),
    )
    case.step(
        "w5_r0",
        "request",
        params=case.value("hot_churn.w5_r0"),
    )
    case.step(
        "w5_r0_done",
        "wait",
        timeout_s=case.value("hot_churn.w5_r0_done_timeout_s"),
        params={"requests": output("w5_r0", "requests")},
    )
    case.step(
        "w5_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w5_r0_hit",
            {
                "snapshot": output("w5_r0_before", "snapshot"),
                "requests": output("w5_r0", "requests"),
            },
        ),
    )
    case.step(
        "w5_r1_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w5_r1_before_timeout_s"),
        params=case.value("hot_churn.w5_r1_before"),
    )
    case.step(
        "w5_r1",
        "request",
        params=case.value("hot_churn.w5_r1"),
    )
    case.step(
        "w5_r1_done",
        "wait",
        timeout_s=case.value("hot_churn.w5_r1_done_timeout_s"),
        params={"requests": output("w5_r1", "requests")},
    )
    case.step(
        "w5_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w5_r1_hit",
            {
                "snapshot": output("w5_r1_before", "snapshot"),
                "requests": output("w5_r1", "requests"),
            },
        ),
    )
    case.step(
        "w5_r2_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w5_r2_before_timeout_s"),
        params=case.value("hot_churn.w5_r2_before"),
    )
    case.step(
        "w5_r2",
        "request",
        params=case.value("hot_churn.w5_r2"),
    )
    case.step(
        "w5_r2_done",
        "wait",
        timeout_s=case.value("hot_churn.w5_r2_done_timeout_s"),
        params={"requests": output("w5_r2", "requests")},
    )
    case.step(
        "w5_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w5_r2_hit",
            {
                "snapshot": output("w5_r2_before", "snapshot"),
                "requests": output("w5_r2", "requests"),
            },
        ),
    )
    case.step(
        "w5_r3_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w5_r3_before_timeout_s"),
        params=case.value("hot_churn.w5_r3_before"),
    )
    case.step(
        "w5_r3",
        "request",
        params=case.value("hot_churn.w5_r3"),
    )
    case.step(
        "w5_r3_done",
        "wait",
        timeout_s=case.value("hot_churn.w5_r3_done_timeout_s"),
        params={"requests": output("w5_r3", "requests")},
    )
    case.step(
        "w5_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w5_r3_hit",
            {
                "snapshot": output("w5_r3_before", "snapshot"),
                "requests": output("w5_r3", "requests"),
            },
        ),
    )
    case.step(
        "w5_r4_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w5_r4_before_timeout_s"),
        params=case.value("hot_churn.w5_r4_before"),
    )
    case.step(
        "w5_r4",
        "request",
        params=case.value("hot_churn.w5_r4"),
    )
    case.step(
        "w5_r4_done",
        "wait",
        timeout_s=case.value("hot_churn.w5_r4_done_timeout_s"),
        params={"requests": output("w5_r4", "requests")},
    )
    case.step(
        "w5_r4_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w5_r4_hit",
            {
                "snapshot": output("w5_r4_before", "snapshot"),
                "requests": output("w5_r4", "requests"),
            },
        ),
    )
    case.step(
        "w5_end",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w5_end_timeout_s"),
        params=case.value("hot_churn.w5_end"),
    )
    case.step(
        "w6_r0_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w6_r0_before_timeout_s"),
        params=case.value("hot_churn.w6_r0_before"),
    )
    case.step(
        "w6_r0",
        "request",
        params=case.value("hot_churn.w6_r0"),
    )
    case.step(
        "w6_r0_done",
        "wait",
        timeout_s=case.value("hot_churn.w6_r0_done_timeout_s"),
        params={"requests": output("w6_r0", "requests")},
    )
    case.step(
        "w6_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w6_r0_hit",
            {
                "snapshot": output("w6_r0_before", "snapshot"),
                "requests": output("w6_r0", "requests"),
            },
        ),
    )
    case.step(
        "w6_r1_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w6_r1_before_timeout_s"),
        params=case.value("hot_churn.w6_r1_before"),
    )
    case.step(
        "w6_r1",
        "request",
        params=case.value("hot_churn.w6_r1"),
    )
    case.step(
        "w6_r1_done",
        "wait",
        timeout_s=case.value("hot_churn.w6_r1_done_timeout_s"),
        params={"requests": output("w6_r1", "requests")},
    )
    case.step(
        "w6_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w6_r1_hit",
            {
                "snapshot": output("w6_r1_before", "snapshot"),
                "requests": output("w6_r1", "requests"),
            },
        ),
    )
    case.step(
        "w6_r2_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w6_r2_before_timeout_s"),
        params=case.value("hot_churn.w6_r2_before"),
    )
    case.step(
        "w6_r2",
        "request",
        params=case.value("hot_churn.w6_r2"),
    )
    case.step(
        "w6_r2_done",
        "wait",
        timeout_s=case.value("hot_churn.w6_r2_done_timeout_s"),
        params={"requests": output("w6_r2", "requests")},
    )
    case.step(
        "w6_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w6_r2_hit",
            {
                "snapshot": output("w6_r2_before", "snapshot"),
                "requests": output("w6_r2", "requests"),
            },
        ),
    )
    case.step(
        "w6_r3_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w6_r3_before_timeout_s"),
        params=case.value("hot_churn.w6_r3_before"),
    )
    case.step(
        "w6_r3",
        "request",
        params=case.value("hot_churn.w6_r3"),
    )
    case.step(
        "w6_r3_done",
        "wait",
        timeout_s=case.value("hot_churn.w6_r3_done_timeout_s"),
        params={"requests": output("w6_r3", "requests")},
    )
    case.step(
        "w6_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w6_r3_hit",
            {
                "snapshot": output("w6_r3_before", "snapshot"),
                "requests": output("w6_r3", "requests"),
            },
        ),
    )
    case.step(
        "w6_r4_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w6_r4_before_timeout_s"),
        params=case.value("hot_churn.w6_r4_before"),
    )
    case.step(
        "w6_r4",
        "request",
        params=case.value("hot_churn.w6_r4"),
    )
    case.step(
        "w6_r4_done",
        "wait",
        timeout_s=case.value("hot_churn.w6_r4_done_timeout_s"),
        params={"requests": output("w6_r4", "requests")},
    )
    case.step(
        "w6_r4_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w6_r4_hit",
            {
                "snapshot": output("w6_r4_before", "snapshot"),
                "requests": output("w6_r4", "requests"),
            },
        ),
    )
    case.step(
        "w6_end",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w6_end_timeout_s"),
        params=case.value("hot_churn.w6_end"),
    )
    case.step(
        "w7_r0_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w7_r0_before_timeout_s"),
        params=case.value("hot_churn.w7_r0_before"),
    )
    case.step(
        "w7_r0",
        "request",
        params=case.value("hot_churn.w7_r0"),
    )
    case.step(
        "w7_r0_done",
        "wait",
        timeout_s=case.value("hot_churn.w7_r0_done_timeout_s"),
        params={"requests": output("w7_r0", "requests")},
    )
    case.step(
        "w7_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w7_r0_hit",
            {
                "snapshot": output("w7_r0_before", "snapshot"),
                "requests": output("w7_r0", "requests"),
            },
        ),
    )
    case.step(
        "w7_r1_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w7_r1_before_timeout_s"),
        params=case.value("hot_churn.w7_r1_before"),
    )
    case.step(
        "w7_r1",
        "request",
        params=case.value("hot_churn.w7_r1"),
    )
    case.step(
        "w7_r1_done",
        "wait",
        timeout_s=case.value("hot_churn.w7_r1_done_timeout_s"),
        params={"requests": output("w7_r1", "requests")},
    )
    case.step(
        "w7_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w7_r1_hit",
            {
                "snapshot": output("w7_r1_before", "snapshot"),
                "requests": output("w7_r1", "requests"),
            },
        ),
    )
    case.step(
        "w7_r2_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w7_r2_before_timeout_s"),
        params=case.value("hot_churn.w7_r2_before"),
    )
    case.step(
        "w7_r2",
        "request",
        params=case.value("hot_churn.w7_r2"),
    )
    case.step(
        "w7_r2_done",
        "wait",
        timeout_s=case.value("hot_churn.w7_r2_done_timeout_s"),
        params={"requests": output("w7_r2", "requests")},
    )
    case.step(
        "w7_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w7_r2_hit",
            {
                "snapshot": output("w7_r2_before", "snapshot"),
                "requests": output("w7_r2", "requests"),
            },
        ),
    )
    case.step(
        "w7_r3_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w7_r3_before_timeout_s"),
        params=case.value("hot_churn.w7_r3_before"),
    )
    case.step(
        "w7_r3",
        "request",
        params=case.value("hot_churn.w7_r3"),
    )
    case.step(
        "w7_r3_done",
        "wait",
        timeout_s=case.value("hot_churn.w7_r3_done_timeout_s"),
        params={"requests": output("w7_r3", "requests")},
    )
    case.step(
        "w7_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w7_r3_hit",
            {
                "snapshot": output("w7_r3_before", "snapshot"),
                "requests": output("w7_r3", "requests"),
            },
        ),
    )
    case.step(
        "w7_r4_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w7_r4_before_timeout_s"),
        params=case.value("hot_churn.w7_r4_before"),
    )
    case.step(
        "w7_r4",
        "request",
        params=case.value("hot_churn.w7_r4"),
    )
    case.step(
        "w7_r4_done",
        "wait",
        timeout_s=case.value("hot_churn.w7_r4_done_timeout_s"),
        params={"requests": output("w7_r4", "requests")},
    )
    case.step(
        "w7_r4_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w7_r4_hit",
            {
                "snapshot": output("w7_r4_before", "snapshot"),
                "requests": output("w7_r4", "requests"),
            },
        ),
    )
    case.step(
        "w7_end",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w7_end_timeout_s"),
        params=case.value("hot_churn.w7_end"),
    )
    case.step(
        "w8_r0_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w8_r0_before_timeout_s"),
        params=case.value("hot_churn.w8_r0_before"),
    )
    case.step(
        "w8_r0",
        "request",
        params=case.value("hot_churn.w8_r0"),
    )
    case.step(
        "w8_r0_done",
        "wait",
        timeout_s=case.value("hot_churn.w8_r0_done_timeout_s"),
        params={"requests": output("w8_r0", "requests")},
    )
    case.step(
        "w8_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w8_r0_hit",
            {
                "snapshot": output("w8_r0_before", "snapshot"),
                "requests": output("w8_r0", "requests"),
            },
        ),
    )
    case.step(
        "w8_r1_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w8_r1_before_timeout_s"),
        params=case.value("hot_churn.w8_r1_before"),
    )
    case.step(
        "w8_r1",
        "request",
        params=case.value("hot_churn.w8_r1"),
    )
    case.step(
        "w8_r1_done",
        "wait",
        timeout_s=case.value("hot_churn.w8_r1_done_timeout_s"),
        params={"requests": output("w8_r1", "requests")},
    )
    case.step(
        "w8_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w8_r1_hit",
            {
                "snapshot": output("w8_r1_before", "snapshot"),
                "requests": output("w8_r1", "requests"),
            },
        ),
    )
    case.step(
        "w8_r2_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w8_r2_before_timeout_s"),
        params=case.value("hot_churn.w8_r2_before"),
    )
    case.step(
        "w8_r2",
        "request",
        params=case.value("hot_churn.w8_r2"),
    )
    case.step(
        "w8_r2_done",
        "wait",
        timeout_s=case.value("hot_churn.w8_r2_done_timeout_s"),
        params={"requests": output("w8_r2", "requests")},
    )
    case.step(
        "w8_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w8_r2_hit",
            {
                "snapshot": output("w8_r2_before", "snapshot"),
                "requests": output("w8_r2", "requests"),
            },
        ),
    )
    case.step(
        "w8_r3_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w8_r3_before_timeout_s"),
        params=case.value("hot_churn.w8_r3_before"),
    )
    case.step(
        "w8_r3",
        "request",
        params=case.value("hot_churn.w8_r3"),
    )
    case.step(
        "w8_r3_done",
        "wait",
        timeout_s=case.value("hot_churn.w8_r3_done_timeout_s"),
        params={"requests": output("w8_r3", "requests")},
    )
    case.step(
        "w8_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w8_r3_hit",
            {
                "snapshot": output("w8_r3_before", "snapshot"),
                "requests": output("w8_r3", "requests"),
            },
        ),
    )
    case.step(
        "w8_r4_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w8_r4_before_timeout_s"),
        params=case.value("hot_churn.w8_r4_before"),
    )
    case.step(
        "w8_r4",
        "request",
        params=case.value("hot_churn.w8_r4"),
    )
    case.step(
        "w8_r4_done",
        "wait",
        timeout_s=case.value("hot_churn.w8_r4_done_timeout_s"),
        params={"requests": output("w8_r4", "requests")},
    )
    case.step(
        "w8_r4_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w8_r4_hit",
            {
                "snapshot": output("w8_r4_before", "snapshot"),
                "requests": output("w8_r4", "requests"),
            },
        ),
    )
    case.step(
        "w8_end",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w8_end_timeout_s"),
        params=case.value("hot_churn.w8_end"),
    )
    case.step(
        "w9_r0_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w9_r0_before_timeout_s"),
        params=case.value("hot_churn.w9_r0_before"),
    )
    case.step(
        "w9_r0",
        "request",
        params=case.value("hot_churn.w9_r0"),
    )
    case.step(
        "w9_r0_done",
        "wait",
        timeout_s=case.value("hot_churn.w9_r0_done_timeout_s"),
        params={"requests": output("w9_r0", "requests")},
    )
    case.step(
        "w9_r0_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w9_r0_hit",
            {
                "snapshot": output("w9_r0_before", "snapshot"),
                "requests": output("w9_r0", "requests"),
            },
        ),
    )
    case.step(
        "w9_r1_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w9_r1_before_timeout_s"),
        params=case.value("hot_churn.w9_r1_before"),
    )
    case.step(
        "w9_r1",
        "request",
        params=case.value("hot_churn.w9_r1"),
    )
    case.step(
        "w9_r1_done",
        "wait",
        timeout_s=case.value("hot_churn.w9_r1_done_timeout_s"),
        params={"requests": output("w9_r1", "requests")},
    )
    case.step(
        "w9_r1_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w9_r1_hit",
            {
                "snapshot": output("w9_r1_before", "snapshot"),
                "requests": output("w9_r1", "requests"),
            },
        ),
    )
    case.step(
        "w9_r2_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w9_r2_before_timeout_s"),
        params=case.value("hot_churn.w9_r2_before"),
    )
    case.step(
        "w9_r2",
        "request",
        params=case.value("hot_churn.w9_r2"),
    )
    case.step(
        "w9_r2_done",
        "wait",
        timeout_s=case.value("hot_churn.w9_r2_done_timeout_s"),
        params={"requests": output("w9_r2", "requests")},
    )
    case.step(
        "w9_r2_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w9_r2_hit",
            {
                "snapshot": output("w9_r2_before", "snapshot"),
                "requests": output("w9_r2", "requests"),
            },
        ),
    )
    case.step(
        "w9_r3_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w9_r3_before_timeout_s"),
        params=case.value("hot_churn.w9_r3_before"),
    )
    case.step(
        "w9_r3",
        "request",
        params=case.value("hot_churn.w9_r3"),
    )
    case.step(
        "w9_r3_done",
        "wait",
        timeout_s=case.value("hot_churn.w9_r3_done_timeout_s"),
        params={"requests": output("w9_r3", "requests")},
    )
    case.step(
        "w9_r3_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w9_r3_hit",
            {
                "snapshot": output("w9_r3_before", "snapshot"),
                "requests": output("w9_r3", "requests"),
            },
        ),
    )
    case.step(
        "w9_r4_before",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w9_r4_before_timeout_s"),
        params=case.value("hot_churn.w9_r4_before"),
    )
    case.step(
        "w9_r4",
        "request",
        params=case.value("hot_churn.w9_r4"),
    )
    case.step(
        "w9_r4_done",
        "wait",
        timeout_s=case.value("hot_churn.w9_r4_done_timeout_s"),
        params={"requests": output("w9_r4", "requests")},
    )
    case.step(
        "w9_r4_hit",
        "kv_hit_observe",
        params=case.params(
            "hot_churn.w9_r4_hit",
            {
                "snapshot": output("w9_r4_before", "snapshot"),
                "requests": output("w9_r4", "requests"),
            },
        ),
    )
    case.step(
        "w9_end",
        "kv_snapshot",
        timeout_s=case.value("hot_churn.w9_end_timeout_s"),
        params=case.value("hot_churn.w9_end"),
    )
    case.step(
        "flips",
        "kv_window_transitions",
        params=case.params(
            "hot_churn.flips",
            {
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
                ]
            },
        ),
    )
    case.step(
        "flip_bound",
        "check",
        params=case.params(
            "hot_churn.flip_bound", {"actual": output("flips", "flips")}
        ),
    )
    case.step(
        "hit_rate",
        "kv_hit_rate_check",
        params=case.params(
            "hot_churn.hit_rate",
            {
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
                ]
            },
        ),
    )
    case.step(
        "replication",
        "kv_replication_check",
        params=case.params(
            "hot_churn.replication", {"snapshot": output("w9_end", "snapshot")}
        ),
    )
    case.step("cleanup", "teardown")


def lru_affinity(case):
    # Capacity 4 includes one reserved block. Three requested blocks fit;
    # sharing only the first key forces eviction of a cold non-prefix block.
    case.step("setup", "setup", timeout_s=case.value("lru_affinity.setup_timeout_s"))
    case.step(
        "prime",
        "request",
        params=case.value("lru_affinity.prime"),
    )
    case.step(
        "prime_done",
        "wait",
        timeout_s=case.value("lru_affinity.prime_done_timeout_s"),
        params={"requests": output("prime", "requests")},
    )
    case.step(
        "prime_holder", "kv_landing", params={"requests": output("prime", "requests")}
    )
    case.step(
        "prime_sync", "balance_pause", params=case.value("lru_affinity.prime_sync")
    )
    case.step(
        "replay",
        "request",
        params=case.value("lru_affinity.replay"),
    )
    case.step(
        "replay_done",
        "wait",
        timeout_s=case.value("lru_affinity.replay_done_timeout_s"),
        params={"requests": output("replay", "requests")},
    )
    case.step(
        "retry",
        "kv_retry_misdirected",
        timeout_s=case.value("lru_affinity.retry_timeout_s"),
        params=case.params(
            "lru_affinity.retry",
            {
                "anchor": output("prime", "requests"),
                "candidate": output("replay", "requests"),
            },
        ),
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
        timeout_s=case.value("lru_affinity.prime_snapshot_timeout_s"),
        params=case.value("lru_affinity.prime_snapshot"),
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
        params=case.params(
            "lru_affinity.prime_keys", {"actual": output("prime_counters", "keys")}
        ),
    )
    case.step(
        "prime_evictions",
        "check",
        params=case.params(
            "lru_affinity.prime_evictions",
            {"actual": output("prime_counters", "evictions")},
        ),
    )
    case.step(
        "cold_blocks",
        "kv_capacity_observe",
        params=case.params(
            "lru_affinity.cold_blocks", {"targets": [output("prime_holder", "engine")]}
        ),
    )
    for field in ("referenced_blocks", "held_blocks", "key_count"):
        expected = case.value(f"lru_affinity.expected_cold.{field}")
        case.step(
            "cold_" + field,
            "kv_capacity_counter",
            params=case.params(
                "lru_affinity.step_26",
                {
                    "observations": [output("cold_blocks", "observation")],
                    "field": field,
                    "expected": expected,
                },
            ),
        )
    case.step(
        "pressure",
        "request",
        params=case.value("lru_affinity.pressure"),
    )
    case.step(
        "pressure_done",
        "wait",
        timeout_s=case.value("lru_affinity.pressure_done_timeout_s"),
        params={"requests": output("pressure", "requests")},
    )
    case.step(
        "pressure_settle",
        "balance_pause",
        params=case.value("lru_affinity.pressure_settle"),
    )
    case.step(
        "pressure_holder",
        "kv_landing",
        params={"requests": output("pressure", "requests")},
    )
    case.step(
        "pressure_snapshot",
        "kv_snapshot",
        timeout_s=case.value("lru_affinity.pressure_snapshot_timeout_s"),
        params=case.value("lru_affinity.pressure_snapshot"),
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
        params=case.params(
            "lru_affinity.pressure_keys",
            {"actual": output("pressure_counters", "keys")},
        ),
    )
    case.step(
        "pressure_evictions",
        "check",
        params=case.params(
            "lru_affinity.pressure_evictions",
            {"actual": output("pressure_counters", "evictions")},
        ),
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
    case.step(
        "setup", "setup", timeout_s=case.value("referenced_occupancy.setup_timeout_s")
    )
    case.step(
        "prime",
        "kv_capacity_request",
        timeout_s=case.value("referenced_occupancy.prime_timeout_s"),
        params=case.value("referenced_occupancy.prime"),
    )
    case.step("holder", "kv_landing", params={"requests": output("prime", "requests")})
    case.step(
        "cache_sync",
        "balance_pause",
        params=case.value("referenced_occupancy.cache_sync"),
    )
    case.step(
        "slow_holder",
        "engine_control",
        params=case.params(
            "referenced_occupancy.slow_holder",
            {"targets": [output("holder", "engine")]},
        ),
    )
    case.step(
        "pin",
        "kv_capacity_request",
        timeout_s=case.value("referenced_occupancy.pin_timeout_s"),
        params=case.value("referenced_occupancy.pin"),
    )
    case.step(
        "pinned_holder",
        "kv_landing",
        params=case.params(
            "referenced_occupancy.pinned_holder",
            {"requests": output("pin", "requests")},
        ),
    )
    case.step(
        "same_holder",
        "kv_same",
        params={
            "first": output("holder", "engine"),
            "second": output("pinned_holder", "engine"),
        },
    )
    fields = case.value("referenced_occupancy.fields")
    case.step(
        "pinned",
        "kv_capacity_observe",
        timeout_s=case.value("referenced_occupancy.pinned_timeout_s"),
        params=case.params(
            "referenced_occupancy.pinned",
            {"targets": [output("holder", "engine")], "fields": fields},
        ),
    )
    case.step(
        "pin_proven",
        "kv_capacity_counter",
        params=case.params(
            "referenced_occupancy.pin_proven",
            {"observations": [output("pinned", "observation")]},
        ),
    )
    case.step(
        "probe",
        "kv_capacity_request",
        timeout_s=case.value("referenced_occupancy.probe_timeout_s"),
        params=case.value("referenced_occupancy.probe"),
    )
    case.step(
        "probe_success",
        "kv_capacity_outcome",
        params=case.params(
            "referenced_occupancy.probe_success",
            {"requests": [output("probe", "requests")]},
        ),
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
    for name, field, baseline in (
        ("still_pinned", "referenced_blocks", False),
        ("not_evicted", "cache_evictions", True),
    ):
        expected = case.value(f"referenced_occupancy.expected.{name}")
        params = case.params(
            "referenced_occupancy.params",
            {
                "observations": [output("protected", "observation")],
                "field": field,
                "expected": expected,
            },
        )
        if baseline:
            params["baseline"] = output("pinned", "observation")
        case.step(name, "kv_capacity_counter", params=params)
    case.step(
        "pin_finished",
        "kv_capacity_wait",
        timeout_s=case.value("referenced_occupancy.pin_finished_timeout_s"),
        params={"requests": [output("pin", "requests")]},
    )
    case.step(
        "pin_success",
        "kv_capacity_outcome",
        params=case.params(
            "referenced_occupancy.pin_success",
            {"requests": [output("pin", "requests")]},
        ),
    )
    case.step(
        "released",
        "kv_capacity_observe",
        timeout_s=case.value("referenced_occupancy.released_timeout_s"),
        params=case.params(
            "referenced_occupancy.released",
            {"targets": [output("holder", "engine")], "fields": fields},
        ),
    )
    case.step(
        "references_released",
        "kv_capacity_counter",
        params=case.params(
            "referenced_occupancy.references_released",
            {"observations": [output("released", "observation")]},
        ),
    )
    case.step("cleanup", "teardown")
