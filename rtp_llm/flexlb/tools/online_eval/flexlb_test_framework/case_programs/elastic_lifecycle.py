"""Ordered scale-out, measured preference and rebalance, graceful removal and three complete cycles."""

from ..case_config import output


def normal(case):
    case.step("setup", "setup", timeout_s=case.value("normal.setup_timeout_s"))
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("normal.initial_topology_timeout_s"),
        params=case.value("normal.initial_topology"),
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params=case.value("normal.ramp"))
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=case.value("normal.baseline_window_timeout_s"),
        params=case.value("normal.baseline_window"),
    )
    case.step("add", "elastic_add", params=case.value("normal.add"))
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=case.value("normal.first_traffic_timeout_s"),
        params=case.params("normal.first_traffic", {"engine": output("add", "engine")}),
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("normal.added_topology_timeout_s"),
        params=case.params("normal.added_topology", {"port": output("add", "port")}),
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=case.value("normal.post_window_timeout_s"),
        params=case.params(
            "normal.post_window",
            {
                "engines": [
                    case.value("normal.post_window.engines.item_0"),
                    case.value("normal.post_window.engines.item_1"),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal.preference_flow_stop_timeout_s"),
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal.preference_protocol", {"flow": output("preference_flow", "flow")}
        ),
    )
    case.step(
        "add_availability",
        "elastic_add_availability",
        params={
            "flow": output("preference_flow", "flow"),
            "mutation": output("add", "mutation"),
            "received_s": output("first_traffic", "observed_s"),
        },
    )
    case.step(
        "preference_received",
        "elastic_window_received",
        params={
            "engine": output("add", "engine"),
            "before": output("post_window", "first"),
            "after": output("post_window", "last"),
        },
    )
    case.step(
        "preference_shares",
        "elastic_share",
        params=case.params(
            "normal.preference_shares",
            {
                "series": output("post_window", "series"),
                "engine": output("add", "engine"),
            },
        ),
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params=case.params(
            "normal.preference_availability",
            {"result": output("preference_flow_stop", "result")},
        ),
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal.remove_traffic_timeout_s"),
        params=case.params(
            "normal.remove_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=case.value("normal.remove_timeout_s"),
        params=case.params("normal.remove", {"engine": output("add", "engine")}),
    )
    case.step("remove_hold", "elastic_pause", params=case.value("normal.remove_hold"))
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal.remove_flow_stop_timeout_s"),
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal.remove_protocol", {"flow": output("remove_flow", "flow")}
        ),
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=case.value("normal.removed_topology_timeout_s"),
        params=case.params("normal.removed_topology", {"port": output("add", "port")}),
    )
    case.step(
        "remove_accounting",
        "elastic_accounting",
        timeout_s=case.value("normal.remove_accounting_timeout_s"),
    )
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal.remove_zero_errors",
            {"result": output("remove_flow_stop", "result")},
        ),
    )
    case.step("cycle1_add", "elastic_add", params=case.value("normal.cycle1_add"))
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=case.value("normal.cycle1_added_topology_timeout_s"),
        params=case.params(
            "normal.cycle1_added_topology", {"port": output("cycle1_add", "port")}
        ),
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal.cycle1_traffic_timeout_s"),
        params=case.params(
            "normal.cycle1_traffic", {"engine": output("cycle1_add", "engine")}
        ),
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step("cycle1_ramp", "elastic_pause", params=case.value("normal.cycle1_ramp"))
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=case.value("normal.cycle1_remove_timeout_s"),
        params=case.params(
            "normal.cycle1_remove", {"engine": output("cycle1_add", "engine")}
        ),
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal.cycle1_flow_stop_timeout_s"),
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal.cycle1_protocol", {"flow": output("cycle1_flow", "flow")}
        ),
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=case.value("normal.cycle1_removed_topology_timeout_s"),
        params=case.params(
            "normal.cycle1_removed_topology", {"port": output("cycle1_add", "port")}
        ),
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal.cycle1_zero_errors",
            {"result": output("cycle1_flow_stop", "result")},
        ),
    )
    case.step("cycle2_add", "elastic_add", params=case.value("normal.cycle2_add"))
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=case.value("normal.cycle2_added_topology_timeout_s"),
        params=case.params(
            "normal.cycle2_added_topology", {"port": output("cycle2_add", "port")}
        ),
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal.cycle2_traffic_timeout_s"),
        params=case.params(
            "normal.cycle2_traffic", {"engine": output("cycle2_add", "engine")}
        ),
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step("cycle2_ramp", "elastic_pause", params=case.value("normal.cycle2_ramp"))
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=case.value("normal.cycle2_remove_timeout_s"),
        params=case.params(
            "normal.cycle2_remove", {"engine": output("cycle2_add", "engine")}
        ),
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal.cycle2_flow_stop_timeout_s"),
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal.cycle2_protocol", {"flow": output("cycle2_flow", "flow")}
        ),
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=case.value("normal.cycle2_removed_topology_timeout_s"),
        params=case.params(
            "normal.cycle2_removed_topology", {"port": output("cycle2_add", "port")}
        ),
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal.cycle2_zero_errors",
            {"result": output("cycle2_flow_stop", "result")},
        ),
    )
    case.step("cycle3_add", "elastic_add", params=case.value("normal.cycle3_add"))
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=case.value("normal.cycle3_added_topology_timeout_s"),
        params=case.params(
            "normal.cycle3_added_topology", {"port": output("cycle3_add", "port")}
        ),
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal.cycle3_traffic_timeout_s"),
        params=case.params(
            "normal.cycle3_traffic", {"engine": output("cycle3_add", "engine")}
        ),
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step("cycle3_ramp", "elastic_pause", params=case.value("normal.cycle3_ramp"))
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=case.value("normal.cycle3_remove_timeout_s"),
        params=case.params(
            "normal.cycle3_remove", {"engine": output("cycle3_add", "engine")}
        ),
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal.cycle3_flow_stop_timeout_s"),
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal.cycle3_protocol", {"flow": output("cycle3_flow", "flow")}
        ),
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=case.value("normal.cycle3_removed_topology_timeout_s"),
        params=case.params(
            "normal.cycle3_removed_topology", {"port": output("cycle3_add", "port")}
        ),
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal.cycle3_zero_errors",
            {"result": output("cycle3_flow_stop", "result")},
        ),
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=case.value("normal.recovery_timeout_s"),
        params=case.value("normal.recovery"),
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=case.value("normal.final_topology_timeout_s"),
        params=case.value("normal.final_topology"),
    )
    case.step("teardown", "teardown")


def strict(case):
    case.step("setup", "setup", timeout_s=case.value("strict.setup_timeout_s"))
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("strict.initial_topology_timeout_s"),
        params=case.value("strict.initial_topology"),
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params=case.value("strict.ramp"))
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=case.value("strict.baseline_window_timeout_s"),
        params=case.value("strict.baseline_window"),
    )
    case.step("add", "elastic_add", params=case.value("strict.add"))
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=case.value("strict.first_traffic_timeout_s"),
        params=case.params("strict.first_traffic", {"engine": output("add", "engine")}),
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("strict.added_topology_timeout_s"),
        params=case.params("strict.added_topology", {"port": output("add", "port")}),
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=case.value("strict.post_window_timeout_s"),
        params=case.params(
            "strict.post_window",
            {
                "engines": [
                    case.value("strict.post_window.engines.item_0"),
                    case.value("strict.post_window.engines.item_1"),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict.preference_flow_stop_timeout_s"),
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict.preference_protocol", {"flow": output("preference_flow", "flow")}
        ),
    )
    case.step(
        "add_availability",
        "elastic_add_availability",
        params={
            "flow": output("preference_flow", "flow"),
            "mutation": output("add", "mutation"),
            "received_s": output("first_traffic", "observed_s"),
        },
    )
    case.step(
        "preference_received",
        "elastic_window_received",
        params={
            "engine": output("add", "engine"),
            "before": output("post_window", "first"),
            "after": output("post_window", "last"),
        },
    )
    case.step(
        "preference_shares",
        "elastic_share",
        params=case.params(
            "strict.preference_shares",
            {
                "series": output("post_window", "series"),
                "engine": output("add", "engine"),
            },
        ),
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params=case.params(
            "strict.preference_availability",
            {"result": output("preference_flow_stop", "result")},
        ),
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict.remove_traffic_timeout_s"),
        params=case.params(
            "strict.remove_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=case.value("strict.remove_timeout_s"),
        params=case.params("strict.remove", {"engine": output("add", "engine")}),
    )
    case.step("remove_hold", "elastic_pause", params=case.value("strict.remove_hold"))
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict.remove_flow_stop_timeout_s"),
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict.remove_protocol", {"flow": output("remove_flow", "flow")}
        ),
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=case.value("strict.removed_topology_timeout_s"),
        params=case.params("strict.removed_topology", {"port": output("add", "port")}),
    )
    case.step(
        "remove_accounting",
        "elastic_accounting",
        timeout_s=case.value("strict.remove_accounting_timeout_s"),
    )
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict.remove_zero_errors",
            {"result": output("remove_flow_stop", "result")},
        ),
    )
    case.step("cycle1_add", "elastic_add", params=case.value("strict.cycle1_add"))
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=case.value("strict.cycle1_added_topology_timeout_s"),
        params=case.params(
            "strict.cycle1_added_topology", {"port": output("cycle1_add", "port")}
        ),
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict.cycle1_traffic_timeout_s"),
        params=case.params(
            "strict.cycle1_traffic", {"engine": output("cycle1_add", "engine")}
        ),
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step("cycle1_ramp", "elastic_pause", params=case.value("strict.cycle1_ramp"))
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=case.value("strict.cycle1_remove_timeout_s"),
        params=case.params(
            "strict.cycle1_remove", {"engine": output("cycle1_add", "engine")}
        ),
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict.cycle1_flow_stop_timeout_s"),
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict.cycle1_protocol", {"flow": output("cycle1_flow", "flow")}
        ),
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=case.value("strict.cycle1_removed_topology_timeout_s"),
        params=case.params(
            "strict.cycle1_removed_topology", {"port": output("cycle1_add", "port")}
        ),
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict.cycle1_zero_errors",
            {"result": output("cycle1_flow_stop", "result")},
        ),
    )
    case.step("cycle2_add", "elastic_add", params=case.value("strict.cycle2_add"))
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=case.value("strict.cycle2_added_topology_timeout_s"),
        params=case.params(
            "strict.cycle2_added_topology", {"port": output("cycle2_add", "port")}
        ),
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict.cycle2_traffic_timeout_s"),
        params=case.params(
            "strict.cycle2_traffic", {"engine": output("cycle2_add", "engine")}
        ),
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step("cycle2_ramp", "elastic_pause", params=case.value("strict.cycle2_ramp"))
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=case.value("strict.cycle2_remove_timeout_s"),
        params=case.params(
            "strict.cycle2_remove", {"engine": output("cycle2_add", "engine")}
        ),
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict.cycle2_flow_stop_timeout_s"),
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict.cycle2_protocol", {"flow": output("cycle2_flow", "flow")}
        ),
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=case.value("strict.cycle2_removed_topology_timeout_s"),
        params=case.params(
            "strict.cycle2_removed_topology", {"port": output("cycle2_add", "port")}
        ),
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict.cycle2_zero_errors",
            {"result": output("cycle2_flow_stop", "result")},
        ),
    )
    case.step("cycle3_add", "elastic_add", params=case.value("strict.cycle3_add"))
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=case.value("strict.cycle3_added_topology_timeout_s"),
        params=case.params(
            "strict.cycle3_added_topology", {"port": output("cycle3_add", "port")}
        ),
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict.cycle3_traffic_timeout_s"),
        params=case.params(
            "strict.cycle3_traffic", {"engine": output("cycle3_add", "engine")}
        ),
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step("cycle3_ramp", "elastic_pause", params=case.value("strict.cycle3_ramp"))
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=case.value("strict.cycle3_remove_timeout_s"),
        params=case.params(
            "strict.cycle3_remove", {"engine": output("cycle3_add", "engine")}
        ),
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict.cycle3_flow_stop_timeout_s"),
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict.cycle3_protocol", {"flow": output("cycle3_flow", "flow")}
        ),
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=case.value("strict.cycle3_removed_topology_timeout_s"),
        params=case.params(
            "strict.cycle3_removed_topology", {"port": output("cycle3_add", "port")}
        ),
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict.cycle3_zero_errors",
            {"result": output("cycle3_flow_stop", "result")},
        ),
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=case.value("strict.recovery_timeout_s"),
        params=case.value("strict.recovery"),
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=case.value("strict.final_topology_timeout_s"),
        params=case.value("strict.final_topology"),
    )
    case.step("teardown", "teardown")


def rebalance(case):
    case.step("setup", "setup", timeout_s=case.value("rebalance.setup_timeout_s"))
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("rebalance.initial_topology_timeout_s"),
        params=case.value("rebalance.initial_topology"),
    )
    case.step(
        "rebalance_baseline",
        "elastic_rebalance_batch",
        timeout_s=case.value("rebalance.rebalance_baseline_timeout_s"),
        params=case.value("rebalance.rebalance_baseline"),
    )
    case.step(
        "baseline_after",
        "elastic_timeline",
        params=case.value("rebalance.baseline_after"),
    )
    case.step("add", "elastic_add", params=case.value("rebalance.add"))
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("rebalance.added_topology_timeout_s"),
        params=case.params("rebalance.added_topology", {"port": output("add", "port")}),
    )
    case.step(
        "rebalance_before",
        "elastic_timeline",
        params=case.params(
            "rebalance.rebalance_before",
            {
                "engines": [
                    case.value("rebalance.rebalance_before.engines.item_0"),
                    case.value("rebalance.rebalance_before.engines.item_1"),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "rebalance_anchor",
        "elastic_rebalance_anchor",
        params={
            "old": output("baseline_after", "last"),
            "new": output("rebalance_before", "last"),
            "engine": output("add", "engine"),
        },
    )
    case.step(
        "rebalance_after_add",
        "elastic_rebalance_batch",
        timeout_s=case.value("rebalance.rebalance_after_add_timeout_s"),
        params=case.value("rebalance.rebalance_after_add"),
    )
    case.step(
        "rebalance_after",
        "elastic_timeline",
        params=case.params(
            "rebalance.rebalance_after",
            {
                "engines": [
                    case.value("rebalance.rebalance_after.engines.item_0"),
                    case.value("rebalance.rebalance_after.engines.item_1"),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "rebalance_share",
        "elastic_share",
        params=case.params(
            "rebalance.rebalance_share",
            {
                "before": output("rebalance_anchor", "before"),
                "after": output("rebalance_after", "last"),
                "engine": output("add", "engine"),
            },
        ),
    )
    case.step("teardown", "teardown")


def kv_skew_hot(case):
    case.step("setup", "setup", timeout_s=case.value("kv_skew_hot.setup_timeout_s"))
    case.step(
        "seed", "elastic_seed", timeout_s=case.value("kv_skew_hot.seed_timeout_s")
    )
    case.step("metrics", "elastic_metrics_start")
    case.step(
        "flow", "elastic_flow_start", params={"families": output("seed", "families")}
    )
    case.step(
        "baseline",
        "elastic_baseline",
        timeout_s=case.value("kv_skew_hot.baseline_timeout_s"),
        params=case.params(
            "kv_skew_hot.baseline", {"observation": output("metrics", "observation")}
        ),
    )
    case.step(
        "scale",
        "elastic_scale",
        timeout_s=case.value("kv_skew_hot.scale_timeout_s"),
        params=case.params(
            "kv_skew_hot.scale", {"families": output("seed", "families")}
        ),
    )
    case.step(
        "transient",
        "elastic_window",
        timeout_s=case.value("kv_skew_hot.transient_timeout_s"),
        params=case.params(
            "kv_skew_hot.transient",
            {
                "observation": output("metrics", "observation"),
                "baseline": output("baseline", "window"),
                "scale": output("scale", "scale"),
                "families": output("seed", "families"),
            },
        ),
    )
    case.step(
        "steady",
        "elastic_window",
        timeout_s=case.value("kv_skew_hot.steady_timeout_s"),
        params=case.params(
            "kv_skew_hot.steady",
            {
                "observation": output("metrics", "observation"),
                "baseline": output("baseline", "window"),
                "scale": output("scale", "scale"),
                "families": output("seed", "families"),
                "transient": output("transient", "window"),
            },
        ),
    )
    case.step(
        "flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("kv_skew_hot.flow_stop_timeout_s"),
        params={"flow": output("flow", "flow")},
    )
    case.step(
        "recovery",
        "elastic_recovery",
        timeout_s=case.value("kv_skew_hot.recovery_timeout_s"),
    )
    case.step(
        "verdict",
        "elastic_verdict",
        params=case.params(
            "kv_skew_hot.verdict",
            {
                "baseline": output("baseline", "window"),
                "transient": output("transient", "window"),
                "steady": output("steady", "window"),
                "scale": output("scale", "scale"),
                "flow_result": output("flow_stop", "result"),
                "recovery": output("recovery", "requests"),
            },
        ),
    )
    case.step("teardown", "teardown")


def kv_skew_cold(case):
    case.step("setup", "setup", timeout_s=case.value("kv_skew_cold.setup_timeout_s"))
    case.step(
        "seed", "elastic_seed", timeout_s=case.value("kv_skew_cold.seed_timeout_s")
    )
    case.step("metrics", "elastic_metrics_start")
    case.step(
        "flow", "elastic_flow_start", params={"families": output("seed", "families")}
    )
    case.step(
        "baseline",
        "elastic_baseline",
        timeout_s=case.value("kv_skew_cold.baseline_timeout_s"),
        params=case.params(
            "kv_skew_cold.baseline", {"observation": output("metrics", "observation")}
        ),
    )
    case.step(
        "scale",
        "elastic_scale",
        timeout_s=case.value("kv_skew_cold.scale_timeout_s"),
        params=case.params(
            "kv_skew_cold.scale", {"families": output("seed", "families")}
        ),
    )
    case.step(
        "transient",
        "elastic_window",
        timeout_s=case.value("kv_skew_cold.transient_timeout_s"),
        params=case.params(
            "kv_skew_cold.transient",
            {
                "observation": output("metrics", "observation"),
                "baseline": output("baseline", "window"),
                "scale": output("scale", "scale"),
                "families": output("seed", "families"),
            },
        ),
    )
    case.step(
        "steady",
        "elastic_window",
        timeout_s=case.value("kv_skew_cold.steady_timeout_s"),
        params=case.params(
            "kv_skew_cold.steady",
            {
                "observation": output("metrics", "observation"),
                "baseline": output("baseline", "window"),
                "scale": output("scale", "scale"),
                "families": output("seed", "families"),
                "transient": output("transient", "window"),
            },
        ),
    )
    case.step(
        "flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("kv_skew_cold.flow_stop_timeout_s"),
        params={"flow": output("flow", "flow")},
    )
    case.step(
        "recovery",
        "elastic_recovery",
        timeout_s=case.value("kv_skew_cold.recovery_timeout_s"),
    )
    case.step(
        "verdict",
        "elastic_verdict",
        params=case.params(
            "kv_skew_cold.verdict",
            {
                "baseline": output("baseline", "window"),
                "transient": output("transient", "window"),
                "steady": output("steady", "window"),
                "scale": output("scale", "scale"),
                "flow_result": output("flow_stop", "result"),
                "recovery": output("recovery", "requests"),
            },
        ),
    )
    case.step("teardown", "teardown")


def steady_recovery(case):
    case.step("setup", "setup", timeout_s=case.value("steady_recovery.setup_timeout_s"))
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("steady_recovery.initial_topology_timeout_s"),
        params=case.value("steady_recovery.initial_topology"),
    )
    case.step("observe", "elastic_balance_observe")
    case.step("flow", "elastic_balance_flow", params=case.value("steady_recovery.flow"))
    case.step(
        "baseline",
        "elastic_balance_window",
        timeout_s=case.value("steady_recovery.baseline_timeout_s"),
        params=case.params(
            "steady_recovery.baseline",
            {"observation": output("observe", "observation")},
        ),
    )
    case.step(
        "baseline_guard",
        "elastic_steady_baseline",
        params={"window": output("baseline", "window")},
    )
    case.step(
        "remove",
        "elastic_balance_remove",
        timeout_s=case.value("steady_recovery.remove_timeout_s"),
        params=case.value("steady_recovery.remove"),
    )
    case.step(
        "transient",
        "elastic_balance_window",
        timeout_s=case.value("steady_recovery.transient_timeout_s"),
        params=case.params(
            "steady_recovery.transient",
            {"observation": output("observe", "observation")},
        ),
    )
    case.step(
        "settled_topology",
        "elastic_topology",
        timeout_s=case.value("steady_recovery.settled_topology_timeout_s"),
        params=case.params(
            "steady_recovery.settled_topology", {"port": output("remove", "port")}
        ),
    )
    case.step(
        "steady",
        "elastic_balance_window",
        timeout_s=case.value("steady_recovery.steady_timeout_s"),
        params=case.params(
            "steady_recovery.steady", {"observation": output("observe", "observation")}
        ),
    )
    case.step(
        "recovery",
        "elastic_pending_recovery",
        timeout_s=case.value("steady_recovery.recovery_timeout_s"),
    )
    case.step(
        "flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("steady_recovery.flow_stop_timeout_s"),
        params={"flow": output("flow", "flow")},
    )
    case.step(
        "verdict",
        "elastic_steady_verdict",
        params={
            "baseline": output("baseline", "window"),
            "steady": output("steady", "window"),
        },
    )
    case.step("teardown", "teardown")


def kv_full_shrink(case):
    case.step("setup", "setup", timeout_s=case.value("kv_full_shrink.setup_timeout_s"))
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("kv_full_shrink.initial_topology_timeout_s"),
        params=case.value("kv_full_shrink.initial_topology"),
    )
    case.step("observe", "elastic_balance_observe")
    case.step(
        "baseline_flow",
        "elastic_balance_flow",
        params=case.value("kv_full_shrink.baseline_flow"),
    )
    case.step(
        "baseline_ramp",
        "elastic_pause",
        params=case.value("kv_full_shrink.baseline_ramp"),
    )
    case.step(
        "baseline",
        "elastic_balance_window",
        timeout_s=case.value("kv_full_shrink.baseline_timeout_s"),
        params=case.params(
            "kv_full_shrink.baseline",
            {
                "observation": output("observe", "observation"),
                "since": output("observe", "started_s"),
            },
        ),
    )
    case.step(
        "baseline_stop",
        "elastic_flow_stop",
        timeout_s=case.value("kv_full_shrink.baseline_stop_timeout_s"),
        params={"flow": output("baseline_flow", "flow")},
    )
    case.step(
        "slow_both",
        "engine_control",
        params=case.value("kv_full_shrink.slow_both"),
    )
    case.step(
        "slow_sync", "elastic_pause", params=case.value("kv_full_shrink.slow_sync")
    )
    case.step(
        "fill_ok",
        "elastic_decode_fill",
        timeout_s=case.value("kv_full_shrink.fill_ok_timeout_s"),
        params=case.value("kv_full_shrink.fill_ok"),
    )
    case.step(
        "remove_ok",
        "elastic_balance_remove",
        timeout_s=case.value("kv_full_shrink.remove_ok_timeout_s"),
        params=case.value("kv_full_shrink.remove_ok"),
    )
    case.step(
        "collect_ok",
        "elastic_full_collect",
        timeout_s=case.value("kv_full_shrink.collect_ok_timeout_s"),
        params={
            "requests": output("fill_ok", "requests"),
            "mutation": output("remove_ok", "mutation"),
        },
    )
    case.step(
        "accounting_ok",
        "elastic_pending_accounting",
        timeout_s=case.value("kv_full_shrink.accounting_ok_timeout_s"),
    )
    case.step(
        "add_decode", "elastic_add", params=case.value("kv_full_shrink.add_decode")
    )
    case.step(
        "restored_topology",
        "elastic_topology",
        timeout_s=case.value("kv_full_shrink.restored_topology_timeout_s"),
        params=case.params(
            "kv_full_shrink.restored_topology", {"port": output("add_decode", "port")}
        ),
    )
    case.step("settled", "elastic_balance_mark")
    case.step(
        "steady_flow",
        "elastic_balance_flow",
        params=case.value("kv_full_shrink.steady_flow"),
    )
    case.step(
        "steady_ramp", "elastic_pause", params=case.value("kv_full_shrink.steady_ramp")
    )
    case.step(
        "steady",
        "elastic_balance_window",
        timeout_s=case.value("kv_full_shrink.steady_timeout_s"),
        params=case.params(
            "kv_full_shrink.steady",
            {
                "observation": output("observe", "observation"),
                "since": output("settled", "time_s"),
            },
        ),
    )
    case.step(
        "steady_stop",
        "elastic_flow_stop",
        timeout_s=case.value("kv_full_shrink.steady_stop_timeout_s"),
        params={"flow": output("steady_flow", "flow")},
    )
    case.step(
        "steady_end",
        "elastic_balance_finish_window",
        params={
            "window": output("steady", "window"),
            "observation": output("observe", "observation"),
        },
    )
    case.step(
        "steady_bounds",
        "elastic_full_steady",
        params={
            "baseline": output("baseline", "window"),
            "steady": output("steady_end", "window"),
            "newcomer": output("add_decode", "engine"),
        },
    )
    case.step(
        "transient_ok",
        "elastic_full_transient",
        params={
            "observation": output("observe", "observation"),
            "fill": output("fill_ok", "fill"),
            "mutation": output("remove_ok", "mutation"),
        },
    )
    case.step(
        "terminal_ok",
        "elastic_full_terminal",
        params=case.params(
            "kv_full_shrink.terminal_ok", {"result": output("collect_ok", "result")}
        ),
    )
    case.step(
        "slow_victim",
        "engine_control",
        params=case.value("kv_full_shrink.slow_victim"),
    )
    case.step(
        "slow_survivor",
        "engine_control",
        params=case.params(
            "kv_full_shrink.slow_survivor",
            {"targets": [output("add_decode", "engine")]},
        ),
    )
    case.step(
        "timeout_sync",
        "elastic_pause",
        params=case.value("kv_full_shrink.timeout_sync"),
    )
    case.step(
        "fill_timeout",
        "elastic_decode_fill",
        timeout_s=case.value("kv_full_shrink.fill_timeout_timeout_s"),
        params=case.params(
            "kv_full_shrink.fill_timeout", {"survivor": output("add_decode", "engine")}
        ),
    )
    case.step(
        "remove_timeout",
        "elastic_balance_remove",
        timeout_s=case.value("kv_full_shrink.remove_timeout_timeout_s"),
        params=case.value("kv_full_shrink.remove_timeout"),
    )
    case.step(
        "collect_timeout",
        "elastic_full_collect",
        timeout_s=case.value("kv_full_shrink.collect_timeout_timeout_s"),
        params={
            "requests": output("fill_timeout", "requests"),
            "mutation": output("remove_timeout", "mutation"),
        },
    )
    case.step(
        "observe_full_transient",
        "elastic_balance_window",
        timeout_s=case.value("kv_full_shrink.observe_full_transient_timeout_s"),
        params=case.params(
            "kv_full_shrink.observe_full_transient",
            {"observation": output("observe", "observation")},
        ),
    )
    case.step(
        "transient_timeout",
        "elastic_full_transient",
        params={
            "observation": output("observe", "observation"),
            "fill": output("fill_timeout", "fill"),
            "mutation": output("remove_timeout", "mutation"),
        },
    )
    case.step(
        "terminal_timeout",
        "elastic_full_terminal",
        params=case.params(
            "kv_full_shrink.terminal_timeout",
            {"result": output("collect_timeout", "result")},
        ),
    )
    case.step(
        "restore_survivor",
        "engine_control",
        params=case.params(
            "kv_full_shrink.restore_survivor",
            {"targets": [output("add_decode", "engine")]},
        ),
    )
    case.step(
        "recovery",
        "elastic_pending_recovery",
        timeout_s=case.value("kv_full_shrink.recovery_timeout_s"),
    )
    case.step("teardown", "teardown")


def transient_imbalance(case):
    case.step(
        "setup", "setup", timeout_s=case.value("transient_imbalance.setup_timeout_s")
    )
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("transient_imbalance.initial_topology_timeout_s"),
        params=case.value("transient_imbalance.initial_topology"),
    )
    case.step("observe", "elastic_transient_observe")
    case.step(
        "flow", "elastic_balance_flow", params=case.value("transient_imbalance.flow")
    )
    case.step(
        "baseline",
        "elastic_balance_window",
        timeout_s=case.value("transient_imbalance.baseline_timeout_s"),
        params=case.params(
            "transient_imbalance.baseline",
            {
                "observation": output("observe", "observation"),
                "since": output("observe", "started_s"),
            },
        ),
    )
    case.step("pre_event", "elastic_transient_prepare")
    case.step(
        "burst",
        "elastic_transient_burst",
        timeout_s=case.value("transient_imbalance.burst_timeout_s"),
    )
    case.step(
        "remove",
        "elastic_transient_remove",
        timeout_s=case.value("transient_imbalance.remove_timeout_s"),
        params={"requests": output("burst", "requests")},
    )
    case.step(
        "burst_settled",
        "elastic_transient_collect",
        timeout_s=case.value("transient_imbalance.burst_settled_timeout_s"),
        params={"requests": output("burst", "requests")},
    )
    case.step(
        "transient",
        "elastic_balance_window",
        timeout_s=case.value("transient_imbalance.transient_timeout_s"),
        params=case.params(
            "transient_imbalance.transient",
            {
                "observation": output("observe", "observation"),
                "since": output("remove", "started_s"),
            },
        ),
    )
    case.step(
        "survivor_topology",
        "elastic_topology",
        timeout_s=case.value("transient_imbalance.survivor_topology_timeout_s"),
        params=case.value("transient_imbalance.survivor_topology"),
    )
    case.step("settled", "elastic_balance_mark")
    case.step(
        "steady",
        "elastic_balance_window",
        timeout_s=case.value("transient_imbalance.steady_timeout_s"),
        params=case.params(
            "transient_imbalance.steady",
            {
                "observation": output("observe", "observation"),
                "since": output("settled", "time_s"),
            },
        ),
    )
    case.step(
        "transient_bounds",
        "elastic_transient_bounds",
        params={
            "pre_event": output("pre_event", "snapshot"),
            "baseline": output("baseline", "window"),
            "transient": output("transient", "window"),
            "master_observation": output("observe", "master_observation"),
        },
    )
    case.step(
        "steady_bounds",
        "elastic_transient_steady",
        params={
            "baseline": output("baseline", "window"),
            "steady": output("steady", "window"),
        },
    )
    case.step(
        "recovery",
        "elastic_pending_recovery",
        timeout_s=case.value("transient_imbalance.recovery_timeout_s"),
    )
    case.step(
        "flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("transient_imbalance.flow_stop_timeout_s"),
        params={"flow": output("flow", "flow")},
    )
    case.step(
        "locality",
        "elastic_transient_locality",
        params={
            "pre_event": output("pre_event", "snapshot"),
            "flow": output("flow", "flow"),
            "requests": output("burst", "requests"),
        },
    )
    case.step("teardown", "teardown")


def rebalance_single_batch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("rebalance_single_batch.setup_timeout_s")
    )
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("rebalance_single_batch.initial_topology_timeout_s"),
        params=case.value("rebalance_single_batch.initial_topology"),
    )
    case.step(
        "rebalance_baseline",
        "elastic_rebalance_batch",
        timeout_s=case.value("rebalance_single_batch.rebalance_baseline_timeout_s"),
        params=case.value("rebalance_single_batch.rebalance_baseline"),
    )
    case.step(
        "baseline_after",
        "elastic_timeline",
        params=case.value("rebalance_single_batch.baseline_after"),
    )
    case.step("add", "elastic_add", params=case.value("rebalance_single_batch.add"))
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("rebalance_single_batch.added_topology_timeout_s"),
        params=case.params(
            "rebalance_single_batch.added_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "rebalance_before",
        "elastic_timeline",
        params=case.params(
            "rebalance_single_batch.rebalance_before",
            {
                "engines": [
                    case.value(
                        "rebalance_single_batch.rebalance_before.engines.item_0"
                    ),
                    case.value(
                        "rebalance_single_batch.rebalance_before.engines.item_1"
                    ),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "rebalance_anchor",
        "elastic_rebalance_anchor",
        params={
            "old": output("baseline_after", "last"),
            "new": output("rebalance_before", "last"),
            "engine": output("add", "engine"),
        },
    )
    case.step(
        "rebalance_after_add",
        "elastic_rebalance_batch",
        timeout_s=case.value("rebalance_single_batch.rebalance_after_add_timeout_s"),
        params=case.value("rebalance_single_batch.rebalance_after_add"),
    )
    case.step(
        "rebalance_after",
        "elastic_timeline",
        params=case.params(
            "rebalance_single_batch.rebalance_after",
            {
                "engines": [
                    case.value("rebalance_single_batch.rebalance_after.engines.item_0"),
                    case.value("rebalance_single_batch.rebalance_after.engines.item_1"),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "rebalance_share",
        "elastic_share",
        params=case.params(
            "rebalance_single_batch.rebalance_share",
            {
                "before": output("rebalance_anchor", "before"),
                "after": output("rebalance_after", "last"),
                "engine": output("add", "engine"),
            },
        ),
    )
    case.step("teardown", "teardown")


def rebalance_single_nonbatch(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("rebalance_single_nonbatch.setup_timeout_s"),
    )
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("rebalance_single_nonbatch.initial_topology_timeout_s"),
        params=case.value("rebalance_single_nonbatch.initial_topology"),
    )
    case.step(
        "rebalance_baseline",
        "elastic_rebalance_batch",
        timeout_s=case.value("rebalance_single_nonbatch.rebalance_baseline_timeout_s"),
        params=case.value("rebalance_single_nonbatch.rebalance_baseline"),
    )
    case.step(
        "baseline_after",
        "elastic_timeline",
        params=case.value("rebalance_single_nonbatch.baseline_after"),
    )
    case.step("add", "elastic_add", params=case.value("rebalance_single_nonbatch.add"))
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("rebalance_single_nonbatch.added_topology_timeout_s"),
        params=case.params(
            "rebalance_single_nonbatch.added_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "rebalance_before",
        "elastic_timeline",
        params=case.params(
            "rebalance_single_nonbatch.rebalance_before",
            {
                "engines": [
                    case.value(
                        "rebalance_single_nonbatch.rebalance_before.engines.item_0"
                    ),
                    case.value(
                        "rebalance_single_nonbatch.rebalance_before.engines.item_1"
                    ),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "rebalance_anchor",
        "elastic_rebalance_anchor",
        params={
            "old": output("baseline_after", "last"),
            "new": output("rebalance_before", "last"),
            "engine": output("add", "engine"),
        },
    )
    case.step(
        "rebalance_after_add",
        "elastic_rebalance_batch",
        timeout_s=case.value("rebalance_single_nonbatch.rebalance_after_add_timeout_s"),
        params=case.value("rebalance_single_nonbatch.rebalance_after_add"),
    )
    case.step(
        "rebalance_after",
        "elastic_timeline",
        params=case.params(
            "rebalance_single_nonbatch.rebalance_after",
            {
                "engines": [
                    case.value(
                        "rebalance_single_nonbatch.rebalance_after.engines.item_0"
                    ),
                    case.value(
                        "rebalance_single_nonbatch.rebalance_after.engines.item_1"
                    ),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "rebalance_share",
        "elastic_share",
        params=case.params(
            "rebalance_single_nonbatch.rebalance_share",
            {
                "before": output("rebalance_anchor", "before"),
                "after": output("rebalance_after", "last"),
                "engine": output("add", "engine"),
            },
        ),
    )
    case.step("teardown", "teardown")


def rebalance_window_nonbatch(case):
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("rebalance_window_nonbatch.setup_timeout_s"),
    )
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("rebalance_window_nonbatch.initial_topology_timeout_s"),
        params=case.value("rebalance_window_nonbatch.initial_topology"),
    )
    case.step(
        "rebalance_baseline",
        "elastic_rebalance_batch",
        timeout_s=case.value("rebalance_window_nonbatch.rebalance_baseline_timeout_s"),
        params=case.value("rebalance_window_nonbatch.rebalance_baseline"),
    )
    case.step(
        "baseline_after",
        "elastic_timeline",
        params=case.value("rebalance_window_nonbatch.baseline_after"),
    )
    case.step("add", "elastic_add", params=case.value("rebalance_window_nonbatch.add"))
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("rebalance_window_nonbatch.added_topology_timeout_s"),
        params=case.params(
            "rebalance_window_nonbatch.added_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "rebalance_before",
        "elastic_timeline",
        params=case.params(
            "rebalance_window_nonbatch.rebalance_before",
            {
                "engines": [
                    case.value(
                        "rebalance_window_nonbatch.rebalance_before.engines.item_0"
                    ),
                    case.value(
                        "rebalance_window_nonbatch.rebalance_before.engines.item_1"
                    ),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "rebalance_anchor",
        "elastic_rebalance_anchor",
        params={
            "old": output("baseline_after", "last"),
            "new": output("rebalance_before", "last"),
            "engine": output("add", "engine"),
        },
    )
    case.step(
        "rebalance_after_add",
        "elastic_rebalance_batch",
        timeout_s=case.value("rebalance_window_nonbatch.rebalance_after_add_timeout_s"),
        params=case.value("rebalance_window_nonbatch.rebalance_after_add"),
    )
    case.step(
        "rebalance_after",
        "elastic_timeline",
        params=case.params(
            "rebalance_window_nonbatch.rebalance_after",
            {
                "engines": [
                    case.value(
                        "rebalance_window_nonbatch.rebalance_after.engines.item_0"
                    ),
                    case.value(
                        "rebalance_window_nonbatch.rebalance_after.engines.item_1"
                    ),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "rebalance_share",
        "elastic_share",
        params=case.params(
            "rebalance_window_nonbatch.rebalance_share",
            {
                "before": output("rebalance_anchor", "before"),
                "after": output("rebalance_after", "last"),
                "engine": output("add", "engine"),
            },
        ),
    )
    case.step("teardown", "teardown")


def normal_single_batch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("normal_single_batch.setup_timeout_s")
    )
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_batch.initial_topology_timeout_s"),
        params=case.value("normal_single_batch.initial_topology"),
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params=case.value("normal_single_batch.ramp"))
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=case.value("normal_single_batch.baseline_window_timeout_s"),
        params=case.value("normal_single_batch.baseline_window"),
    )
    case.step("add", "elastic_add", params=case.value("normal_single_batch.add"))
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=case.value("normal_single_batch.first_traffic_timeout_s"),
        params=case.params(
            "normal_single_batch.first_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_batch.added_topology_timeout_s"),
        params=case.params(
            "normal_single_batch.added_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=case.value("normal_single_batch.post_window_timeout_s"),
        params=case.params(
            "normal_single_batch.post_window",
            {
                "engines": [
                    case.value("normal_single_batch.post_window.engines.item_0"),
                    case.value("normal_single_batch.post_window.engines.item_1"),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_single_batch.preference_flow_stop_timeout_s"),
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_single_batch.preference_protocol",
            {"flow": output("preference_flow", "flow")},
        ),
    )
    case.step(
        "add_availability",
        "elastic_add_availability",
        params={
            "flow": output("preference_flow", "flow"),
            "mutation": output("add", "mutation"),
            "received_s": output("first_traffic", "observed_s"),
        },
    )
    case.step(
        "preference_received",
        "elastic_window_received",
        params={
            "engine": output("add", "engine"),
            "before": output("post_window", "first"),
            "after": output("post_window", "last"),
        },
    )
    case.step(
        "preference_shares",
        "elastic_share",
        params=case.params(
            "normal_single_batch.preference_shares",
            {
                "series": output("post_window", "series"),
                "engine": output("add", "engine"),
            },
        ),
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params=case.params(
            "normal_single_batch.preference_availability",
            {"result": output("preference_flow_stop", "result")},
        ),
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal_single_batch.remove_traffic_timeout_s"),
        params=case.params(
            "normal_single_batch.remove_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=case.value("normal_single_batch.remove_timeout_s"),
        params=case.params(
            "normal_single_batch.remove", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove_hold",
        "elastic_pause",
        params=case.value("normal_single_batch.remove_hold"),
    )
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_single_batch.remove_flow_stop_timeout_s"),
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_single_batch.remove_protocol",
            {"flow": output("remove_flow", "flow")},
        ),
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_batch.removed_topology_timeout_s"),
        params=case.params(
            "normal_single_batch.removed_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "remove_accounting",
        "elastic_accounting",
        timeout_s=case.value("normal_single_batch.remove_accounting_timeout_s"),
    )
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal_single_batch.remove_zero_errors",
            {"result": output("remove_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle1_add", "elastic_add", params=case.value("normal_single_batch.cycle1_add")
    )
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_batch.cycle1_added_topology_timeout_s"),
        params=case.params(
            "normal_single_batch.cycle1_added_topology",
            {"port": output("cycle1_add", "port")},
        ),
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal_single_batch.cycle1_traffic_timeout_s"),
        params=case.params(
            "normal_single_batch.cycle1_traffic",
            {"engine": output("cycle1_add", "engine")},
        ),
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step(
        "cycle1_ramp",
        "elastic_pause",
        params=case.value("normal_single_batch.cycle1_ramp"),
    )
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=case.value("normal_single_batch.cycle1_remove_timeout_s"),
        params=case.params(
            "normal_single_batch.cycle1_remove",
            {"engine": output("cycle1_add", "engine")},
        ),
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_single_batch.cycle1_flow_stop_timeout_s"),
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_single_batch.cycle1_protocol",
            {"flow": output("cycle1_flow", "flow")},
        ),
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_batch.cycle1_removed_topology_timeout_s"),
        params=case.params(
            "normal_single_batch.cycle1_removed_topology",
            {"port": output("cycle1_add", "port")},
        ),
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal_single_batch.cycle1_zero_errors",
            {"result": output("cycle1_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle2_add", "elastic_add", params=case.value("normal_single_batch.cycle2_add")
    )
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_batch.cycle2_added_topology_timeout_s"),
        params=case.params(
            "normal_single_batch.cycle2_added_topology",
            {"port": output("cycle2_add", "port")},
        ),
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal_single_batch.cycle2_traffic_timeout_s"),
        params=case.params(
            "normal_single_batch.cycle2_traffic",
            {"engine": output("cycle2_add", "engine")},
        ),
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step(
        "cycle2_ramp",
        "elastic_pause",
        params=case.value("normal_single_batch.cycle2_ramp"),
    )
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=case.value("normal_single_batch.cycle2_remove_timeout_s"),
        params=case.params(
            "normal_single_batch.cycle2_remove",
            {"engine": output("cycle2_add", "engine")},
        ),
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_single_batch.cycle2_flow_stop_timeout_s"),
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_single_batch.cycle2_protocol",
            {"flow": output("cycle2_flow", "flow")},
        ),
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_batch.cycle2_removed_topology_timeout_s"),
        params=case.params(
            "normal_single_batch.cycle2_removed_topology",
            {"port": output("cycle2_add", "port")},
        ),
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal_single_batch.cycle2_zero_errors",
            {"result": output("cycle2_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle3_add", "elastic_add", params=case.value("normal_single_batch.cycle3_add")
    )
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_batch.cycle3_added_topology_timeout_s"),
        params=case.params(
            "normal_single_batch.cycle3_added_topology",
            {"port": output("cycle3_add", "port")},
        ),
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal_single_batch.cycle3_traffic_timeout_s"),
        params=case.params(
            "normal_single_batch.cycle3_traffic",
            {"engine": output("cycle3_add", "engine")},
        ),
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step(
        "cycle3_ramp",
        "elastic_pause",
        params=case.value("normal_single_batch.cycle3_ramp"),
    )
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=case.value("normal_single_batch.cycle3_remove_timeout_s"),
        params=case.params(
            "normal_single_batch.cycle3_remove",
            {"engine": output("cycle3_add", "engine")},
        ),
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_single_batch.cycle3_flow_stop_timeout_s"),
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_single_batch.cycle3_protocol",
            {"flow": output("cycle3_flow", "flow")},
        ),
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_batch.cycle3_removed_topology_timeout_s"),
        params=case.params(
            "normal_single_batch.cycle3_removed_topology",
            {"port": output("cycle3_add", "port")},
        ),
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal_single_batch.cycle3_zero_errors",
            {"result": output("cycle3_flow_stop", "result")},
        ),
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=case.value("normal_single_batch.recovery_timeout_s"),
        params=case.value("normal_single_batch.recovery"),
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_batch.final_topology_timeout_s"),
        params=case.value("normal_single_batch.final_topology"),
    )
    case.step("teardown", "teardown")


def strict_single_batch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("strict_single_batch.setup_timeout_s")
    )
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_batch.initial_topology_timeout_s"),
        params=case.value("strict_single_batch.initial_topology"),
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params=case.value("strict_single_batch.ramp"))
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=case.value("strict_single_batch.baseline_window_timeout_s"),
        params=case.value("strict_single_batch.baseline_window"),
    )
    case.step("add", "elastic_add", params=case.value("strict_single_batch.add"))
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=case.value("strict_single_batch.first_traffic_timeout_s"),
        params=case.params(
            "strict_single_batch.first_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_batch.added_topology_timeout_s"),
        params=case.params(
            "strict_single_batch.added_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=case.value("strict_single_batch.post_window_timeout_s"),
        params=case.params(
            "strict_single_batch.post_window",
            {
                "engines": [
                    case.value("strict_single_batch.post_window.engines.item_0"),
                    case.value("strict_single_batch.post_window.engines.item_1"),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_single_batch.preference_flow_stop_timeout_s"),
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_single_batch.preference_protocol",
            {"flow": output("preference_flow", "flow")},
        ),
    )
    case.step(
        "add_availability",
        "elastic_add_availability",
        params={
            "flow": output("preference_flow", "flow"),
            "mutation": output("add", "mutation"),
            "received_s": output("first_traffic", "observed_s"),
        },
    )
    case.step(
        "preference_received",
        "elastic_window_received",
        params={
            "engine": output("add", "engine"),
            "before": output("post_window", "first"),
            "after": output("post_window", "last"),
        },
    )
    case.step(
        "preference_shares",
        "elastic_share",
        params=case.params(
            "strict_single_batch.preference_shares",
            {
                "series": output("post_window", "series"),
                "engine": output("add", "engine"),
            },
        ),
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params=case.params(
            "strict_single_batch.preference_availability",
            {"result": output("preference_flow_stop", "result")},
        ),
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict_single_batch.remove_traffic_timeout_s"),
        params=case.params(
            "strict_single_batch.remove_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=case.value("strict_single_batch.remove_timeout_s"),
        params=case.params(
            "strict_single_batch.remove", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove_hold",
        "elastic_pause",
        params=case.value("strict_single_batch.remove_hold"),
    )
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_single_batch.remove_flow_stop_timeout_s"),
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_single_batch.remove_protocol",
            {"flow": output("remove_flow", "flow")},
        ),
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_batch.removed_topology_timeout_s"),
        params=case.params(
            "strict_single_batch.removed_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "remove_accounting",
        "elastic_accounting",
        timeout_s=case.value("strict_single_batch.remove_accounting_timeout_s"),
    )
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict_single_batch.remove_zero_errors",
            {"result": output("remove_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle1_add", "elastic_add", params=case.value("strict_single_batch.cycle1_add")
    )
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_batch.cycle1_added_topology_timeout_s"),
        params=case.params(
            "strict_single_batch.cycle1_added_topology",
            {"port": output("cycle1_add", "port")},
        ),
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict_single_batch.cycle1_traffic_timeout_s"),
        params=case.params(
            "strict_single_batch.cycle1_traffic",
            {"engine": output("cycle1_add", "engine")},
        ),
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step(
        "cycle1_ramp",
        "elastic_pause",
        params=case.value("strict_single_batch.cycle1_ramp"),
    )
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=case.value("strict_single_batch.cycle1_remove_timeout_s"),
        params=case.params(
            "strict_single_batch.cycle1_remove",
            {"engine": output("cycle1_add", "engine")},
        ),
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_single_batch.cycle1_flow_stop_timeout_s"),
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_single_batch.cycle1_protocol",
            {"flow": output("cycle1_flow", "flow")},
        ),
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_batch.cycle1_removed_topology_timeout_s"),
        params=case.params(
            "strict_single_batch.cycle1_removed_topology",
            {"port": output("cycle1_add", "port")},
        ),
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict_single_batch.cycle1_zero_errors",
            {"result": output("cycle1_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle2_add", "elastic_add", params=case.value("strict_single_batch.cycle2_add")
    )
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_batch.cycle2_added_topology_timeout_s"),
        params=case.params(
            "strict_single_batch.cycle2_added_topology",
            {"port": output("cycle2_add", "port")},
        ),
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict_single_batch.cycle2_traffic_timeout_s"),
        params=case.params(
            "strict_single_batch.cycle2_traffic",
            {"engine": output("cycle2_add", "engine")},
        ),
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step(
        "cycle2_ramp",
        "elastic_pause",
        params=case.value("strict_single_batch.cycle2_ramp"),
    )
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=case.value("strict_single_batch.cycle2_remove_timeout_s"),
        params=case.params(
            "strict_single_batch.cycle2_remove",
            {"engine": output("cycle2_add", "engine")},
        ),
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_single_batch.cycle2_flow_stop_timeout_s"),
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_single_batch.cycle2_protocol",
            {"flow": output("cycle2_flow", "flow")},
        ),
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_batch.cycle2_removed_topology_timeout_s"),
        params=case.params(
            "strict_single_batch.cycle2_removed_topology",
            {"port": output("cycle2_add", "port")},
        ),
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict_single_batch.cycle2_zero_errors",
            {"result": output("cycle2_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle3_add", "elastic_add", params=case.value("strict_single_batch.cycle3_add")
    )
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_batch.cycle3_added_topology_timeout_s"),
        params=case.params(
            "strict_single_batch.cycle3_added_topology",
            {"port": output("cycle3_add", "port")},
        ),
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict_single_batch.cycle3_traffic_timeout_s"),
        params=case.params(
            "strict_single_batch.cycle3_traffic",
            {"engine": output("cycle3_add", "engine")},
        ),
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step(
        "cycle3_ramp",
        "elastic_pause",
        params=case.value("strict_single_batch.cycle3_ramp"),
    )
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=case.value("strict_single_batch.cycle3_remove_timeout_s"),
        params=case.params(
            "strict_single_batch.cycle3_remove",
            {"engine": output("cycle3_add", "engine")},
        ),
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_single_batch.cycle3_flow_stop_timeout_s"),
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_single_batch.cycle3_protocol",
            {"flow": output("cycle3_flow", "flow")},
        ),
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_batch.cycle3_removed_topology_timeout_s"),
        params=case.params(
            "strict_single_batch.cycle3_removed_topology",
            {"port": output("cycle3_add", "port")},
        ),
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict_single_batch.cycle3_zero_errors",
            {"result": output("cycle3_flow_stop", "result")},
        ),
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=case.value("strict_single_batch.recovery_timeout_s"),
        params=case.value("strict_single_batch.recovery"),
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_batch.final_topology_timeout_s"),
        params=case.value("strict_single_batch.final_topology"),
    )
    case.step("teardown", "teardown")


def normal_single_nonbatch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("normal_single_nonbatch.setup_timeout_s")
    )
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_nonbatch.initial_topology_timeout_s"),
        params=case.value("normal_single_nonbatch.initial_topology"),
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params=case.value("normal_single_nonbatch.ramp"))
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=case.value("normal_single_nonbatch.baseline_window_timeout_s"),
        params=case.value("normal_single_nonbatch.baseline_window"),
    )
    case.step("add", "elastic_add", params=case.value("normal_single_nonbatch.add"))
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=case.value("normal_single_nonbatch.first_traffic_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.first_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_nonbatch.added_topology_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.added_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=case.value("normal_single_nonbatch.post_window_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.post_window",
            {
                "engines": [
                    case.value("normal_single_nonbatch.post_window.engines.item_0"),
                    case.value("normal_single_nonbatch.post_window.engines.item_1"),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_single_nonbatch.preference_flow_stop_timeout_s"),
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_single_nonbatch.preference_protocol",
            {"flow": output("preference_flow", "flow")},
        ),
    )
    case.step(
        "add_availability",
        "elastic_add_availability",
        params={
            "flow": output("preference_flow", "flow"),
            "mutation": output("add", "mutation"),
            "received_s": output("first_traffic", "observed_s"),
        },
    )
    case.step(
        "preference_received",
        "elastic_window_received",
        params={
            "engine": output("add", "engine"),
            "before": output("post_window", "first"),
            "after": output("post_window", "last"),
        },
    )
    case.step(
        "preference_shares",
        "elastic_share",
        params=case.params(
            "normal_single_nonbatch.preference_shares",
            {
                "series": output("post_window", "series"),
                "engine": output("add", "engine"),
            },
        ),
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params=case.params(
            "normal_single_nonbatch.preference_availability",
            {"result": output("preference_flow_stop", "result")},
        ),
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal_single_nonbatch.remove_traffic_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.remove_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=case.value("normal_single_nonbatch.remove_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.remove", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove_hold",
        "elastic_pause",
        params=case.value("normal_single_nonbatch.remove_hold"),
    )
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_single_nonbatch.remove_flow_stop_timeout_s"),
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_single_nonbatch.remove_protocol",
            {"flow": output("remove_flow", "flow")},
        ),
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_nonbatch.removed_topology_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.removed_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "remove_accounting",
        "elastic_literal_nonbatch_accounting",
        timeout_s=case.value("normal_single_nonbatch.remove_accounting_timeout_s"),
    )
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal_single_nonbatch.remove_zero_errors",
            {"result": output("remove_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle1_add",
        "elastic_add",
        params=case.value("normal_single_nonbatch.cycle1_add"),
    )
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_nonbatch.cycle1_added_topology_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.cycle1_added_topology",
            {"port": output("cycle1_add", "port")},
        ),
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal_single_nonbatch.cycle1_traffic_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.cycle1_traffic",
            {"engine": output("cycle1_add", "engine")},
        ),
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step(
        "cycle1_ramp",
        "elastic_pause",
        params=case.value("normal_single_nonbatch.cycle1_ramp"),
    )
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=case.value("normal_single_nonbatch.cycle1_remove_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.cycle1_remove",
            {"engine": output("cycle1_add", "engine")},
        ),
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_single_nonbatch.cycle1_flow_stop_timeout_s"),
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_single_nonbatch.cycle1_protocol",
            {"flow": output("cycle1_flow", "flow")},
        ),
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=case.value(
            "normal_single_nonbatch.cycle1_removed_topology_timeout_s"
        ),
        params=case.params(
            "normal_single_nonbatch.cycle1_removed_topology",
            {"port": output("cycle1_add", "port")},
        ),
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal_single_nonbatch.cycle1_zero_errors",
            {"result": output("cycle1_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle2_add",
        "elastic_add",
        params=case.value("normal_single_nonbatch.cycle2_add"),
    )
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_nonbatch.cycle2_added_topology_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.cycle2_added_topology",
            {"port": output("cycle2_add", "port")},
        ),
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal_single_nonbatch.cycle2_traffic_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.cycle2_traffic",
            {"engine": output("cycle2_add", "engine")},
        ),
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step(
        "cycle2_ramp",
        "elastic_pause",
        params=case.value("normal_single_nonbatch.cycle2_ramp"),
    )
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=case.value("normal_single_nonbatch.cycle2_remove_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.cycle2_remove",
            {"engine": output("cycle2_add", "engine")},
        ),
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_single_nonbatch.cycle2_flow_stop_timeout_s"),
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_single_nonbatch.cycle2_protocol",
            {"flow": output("cycle2_flow", "flow")},
        ),
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=case.value(
            "normal_single_nonbatch.cycle2_removed_topology_timeout_s"
        ),
        params=case.params(
            "normal_single_nonbatch.cycle2_removed_topology",
            {"port": output("cycle2_add", "port")},
        ),
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal_single_nonbatch.cycle2_zero_errors",
            {"result": output("cycle2_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle3_add",
        "elastic_add",
        params=case.value("normal_single_nonbatch.cycle3_add"),
    )
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_nonbatch.cycle3_added_topology_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.cycle3_added_topology",
            {"port": output("cycle3_add", "port")},
        ),
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal_single_nonbatch.cycle3_traffic_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.cycle3_traffic",
            {"engine": output("cycle3_add", "engine")},
        ),
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step(
        "cycle3_ramp",
        "elastic_pause",
        params=case.value("normal_single_nonbatch.cycle3_ramp"),
    )
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=case.value("normal_single_nonbatch.cycle3_remove_timeout_s"),
        params=case.params(
            "normal_single_nonbatch.cycle3_remove",
            {"engine": output("cycle3_add", "engine")},
        ),
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_single_nonbatch.cycle3_flow_stop_timeout_s"),
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_single_nonbatch.cycle3_protocol",
            {"flow": output("cycle3_flow", "flow")},
        ),
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=case.value(
            "normal_single_nonbatch.cycle3_removed_topology_timeout_s"
        ),
        params=case.params(
            "normal_single_nonbatch.cycle3_removed_topology",
            {"port": output("cycle3_add", "port")},
        ),
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal_single_nonbatch.cycle3_zero_errors",
            {"result": output("cycle3_flow_stop", "result")},
        ),
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=case.value("normal_single_nonbatch.recovery_timeout_s"),
        params=case.value("normal_single_nonbatch.recovery"),
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=case.value("normal_single_nonbatch.final_topology_timeout_s"),
        params=case.value("normal_single_nonbatch.final_topology"),
    )
    case.step("teardown", "teardown")


def strict_single_nonbatch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("strict_single_nonbatch.setup_timeout_s")
    )
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_nonbatch.initial_topology_timeout_s"),
        params=case.value("strict_single_nonbatch.initial_topology"),
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params=case.value("strict_single_nonbatch.ramp"))
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=case.value("strict_single_nonbatch.baseline_window_timeout_s"),
        params=case.value("strict_single_nonbatch.baseline_window"),
    )
    case.step("add", "elastic_add", params=case.value("strict_single_nonbatch.add"))
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=case.value("strict_single_nonbatch.first_traffic_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.first_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_nonbatch.added_topology_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.added_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=case.value("strict_single_nonbatch.post_window_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.post_window",
            {
                "engines": [
                    case.value("strict_single_nonbatch.post_window.engines.item_0"),
                    case.value("strict_single_nonbatch.post_window.engines.item_1"),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_single_nonbatch.preference_flow_stop_timeout_s"),
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_single_nonbatch.preference_protocol",
            {"flow": output("preference_flow", "flow")},
        ),
    )
    case.step(
        "add_availability",
        "elastic_add_availability",
        params={
            "flow": output("preference_flow", "flow"),
            "mutation": output("add", "mutation"),
            "received_s": output("first_traffic", "observed_s"),
        },
    )
    case.step(
        "preference_received",
        "elastic_window_received",
        params={
            "engine": output("add", "engine"),
            "before": output("post_window", "first"),
            "after": output("post_window", "last"),
        },
    )
    case.step(
        "preference_shares",
        "elastic_share",
        params=case.params(
            "strict_single_nonbatch.preference_shares",
            {
                "series": output("post_window", "series"),
                "engine": output("add", "engine"),
            },
        ),
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params=case.params(
            "strict_single_nonbatch.preference_availability",
            {"result": output("preference_flow_stop", "result")},
        ),
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict_single_nonbatch.remove_traffic_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.remove_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=case.value("strict_single_nonbatch.remove_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.remove", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove_hold",
        "elastic_pause",
        params=case.value("strict_single_nonbatch.remove_hold"),
    )
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_single_nonbatch.remove_flow_stop_timeout_s"),
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_single_nonbatch.remove_protocol",
            {"flow": output("remove_flow", "flow")},
        ),
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_nonbatch.removed_topology_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.removed_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "remove_accounting",
        "elastic_literal_nonbatch_accounting",
        timeout_s=case.value("strict_single_nonbatch.remove_accounting_timeout_s"),
    )
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict_single_nonbatch.remove_zero_errors",
            {"result": output("remove_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle1_add",
        "elastic_add",
        params=case.value("strict_single_nonbatch.cycle1_add"),
    )
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_nonbatch.cycle1_added_topology_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.cycle1_added_topology",
            {"port": output("cycle1_add", "port")},
        ),
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict_single_nonbatch.cycle1_traffic_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.cycle1_traffic",
            {"engine": output("cycle1_add", "engine")},
        ),
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step(
        "cycle1_ramp",
        "elastic_pause",
        params=case.value("strict_single_nonbatch.cycle1_ramp"),
    )
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=case.value("strict_single_nonbatch.cycle1_remove_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.cycle1_remove",
            {"engine": output("cycle1_add", "engine")},
        ),
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_single_nonbatch.cycle1_flow_stop_timeout_s"),
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_single_nonbatch.cycle1_protocol",
            {"flow": output("cycle1_flow", "flow")},
        ),
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=case.value(
            "strict_single_nonbatch.cycle1_removed_topology_timeout_s"
        ),
        params=case.params(
            "strict_single_nonbatch.cycle1_removed_topology",
            {"port": output("cycle1_add", "port")},
        ),
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict_single_nonbatch.cycle1_zero_errors",
            {"result": output("cycle1_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle2_add",
        "elastic_add",
        params=case.value("strict_single_nonbatch.cycle2_add"),
    )
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_nonbatch.cycle2_added_topology_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.cycle2_added_topology",
            {"port": output("cycle2_add", "port")},
        ),
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict_single_nonbatch.cycle2_traffic_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.cycle2_traffic",
            {"engine": output("cycle2_add", "engine")},
        ),
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step(
        "cycle2_ramp",
        "elastic_pause",
        params=case.value("strict_single_nonbatch.cycle2_ramp"),
    )
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=case.value("strict_single_nonbatch.cycle2_remove_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.cycle2_remove",
            {"engine": output("cycle2_add", "engine")},
        ),
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_single_nonbatch.cycle2_flow_stop_timeout_s"),
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_single_nonbatch.cycle2_protocol",
            {"flow": output("cycle2_flow", "flow")},
        ),
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=case.value(
            "strict_single_nonbatch.cycle2_removed_topology_timeout_s"
        ),
        params=case.params(
            "strict_single_nonbatch.cycle2_removed_topology",
            {"port": output("cycle2_add", "port")},
        ),
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict_single_nonbatch.cycle2_zero_errors",
            {"result": output("cycle2_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle3_add",
        "elastic_add",
        params=case.value("strict_single_nonbatch.cycle3_add"),
    )
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_nonbatch.cycle3_added_topology_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.cycle3_added_topology",
            {"port": output("cycle3_add", "port")},
        ),
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict_single_nonbatch.cycle3_traffic_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.cycle3_traffic",
            {"engine": output("cycle3_add", "engine")},
        ),
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step(
        "cycle3_ramp",
        "elastic_pause",
        params=case.value("strict_single_nonbatch.cycle3_ramp"),
    )
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=case.value("strict_single_nonbatch.cycle3_remove_timeout_s"),
        params=case.params(
            "strict_single_nonbatch.cycle3_remove",
            {"engine": output("cycle3_add", "engine")},
        ),
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_single_nonbatch.cycle3_flow_stop_timeout_s"),
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_single_nonbatch.cycle3_protocol",
            {"flow": output("cycle3_flow", "flow")},
        ),
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=case.value(
            "strict_single_nonbatch.cycle3_removed_topology_timeout_s"
        ),
        params=case.params(
            "strict_single_nonbatch.cycle3_removed_topology",
            {"port": output("cycle3_add", "port")},
        ),
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict_single_nonbatch.cycle3_zero_errors",
            {"result": output("cycle3_flow_stop", "result")},
        ),
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=case.value("strict_single_nonbatch.recovery_timeout_s"),
        params=case.value("strict_single_nonbatch.recovery"),
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=case.value("strict_single_nonbatch.final_topology_timeout_s"),
        params=case.value("strict_single_nonbatch.final_topology"),
    )
    case.step("teardown", "teardown")


def normal_window_nonbatch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("normal_window_nonbatch.setup_timeout_s")
    )
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("normal_window_nonbatch.initial_topology_timeout_s"),
        params=case.value("normal_window_nonbatch.initial_topology"),
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params=case.value("normal_window_nonbatch.ramp"))
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=case.value("normal_window_nonbatch.baseline_window_timeout_s"),
        params=case.value("normal_window_nonbatch.baseline_window"),
    )
    case.step("add", "elastic_add", params=case.value("normal_window_nonbatch.add"))
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=case.value("normal_window_nonbatch.first_traffic_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.first_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("normal_window_nonbatch.added_topology_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.added_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=case.value("normal_window_nonbatch.post_window_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.post_window",
            {
                "engines": [
                    case.value("normal_window_nonbatch.post_window.engines.item_0"),
                    case.value("normal_window_nonbatch.post_window.engines.item_1"),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_window_nonbatch.preference_flow_stop_timeout_s"),
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_window_nonbatch.preference_protocol",
            {"flow": output("preference_flow", "flow")},
        ),
    )
    case.step(
        "add_availability",
        "elastic_add_availability",
        params={
            "flow": output("preference_flow", "flow"),
            "mutation": output("add", "mutation"),
            "received_s": output("first_traffic", "observed_s"),
        },
    )
    case.step(
        "preference_received",
        "elastic_window_received",
        params={
            "engine": output("add", "engine"),
            "before": output("post_window", "first"),
            "after": output("post_window", "last"),
        },
    )
    case.step(
        "preference_shares",
        "elastic_share",
        params=case.params(
            "normal_window_nonbatch.preference_shares",
            {
                "series": output("post_window", "series"),
                "engine": output("add", "engine"),
            },
        ),
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params=case.params(
            "normal_window_nonbatch.preference_availability",
            {"result": output("preference_flow_stop", "result")},
        ),
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal_window_nonbatch.remove_traffic_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.remove_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=case.value("normal_window_nonbatch.remove_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.remove", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove_hold",
        "elastic_pause",
        params=case.value("normal_window_nonbatch.remove_hold"),
    )
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_window_nonbatch.remove_flow_stop_timeout_s"),
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_window_nonbatch.remove_protocol",
            {"flow": output("remove_flow", "flow")},
        ),
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=case.value("normal_window_nonbatch.removed_topology_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.removed_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "remove_accounting",
        "elastic_literal_nonbatch_accounting",
        timeout_s=case.value("normal_window_nonbatch.remove_accounting_timeout_s"),
    )
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal_window_nonbatch.remove_zero_errors",
            {"result": output("remove_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle1_add",
        "elastic_add",
        params=case.value("normal_window_nonbatch.cycle1_add"),
    )
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=case.value("normal_window_nonbatch.cycle1_added_topology_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.cycle1_added_topology",
            {"port": output("cycle1_add", "port")},
        ),
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal_window_nonbatch.cycle1_traffic_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.cycle1_traffic",
            {"engine": output("cycle1_add", "engine")},
        ),
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step(
        "cycle1_ramp",
        "elastic_pause",
        params=case.value("normal_window_nonbatch.cycle1_ramp"),
    )
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=case.value("normal_window_nonbatch.cycle1_remove_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.cycle1_remove",
            {"engine": output("cycle1_add", "engine")},
        ),
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_window_nonbatch.cycle1_flow_stop_timeout_s"),
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_window_nonbatch.cycle1_protocol",
            {"flow": output("cycle1_flow", "flow")},
        ),
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=case.value(
            "normal_window_nonbatch.cycle1_removed_topology_timeout_s"
        ),
        params=case.params(
            "normal_window_nonbatch.cycle1_removed_topology",
            {"port": output("cycle1_add", "port")},
        ),
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal_window_nonbatch.cycle1_zero_errors",
            {"result": output("cycle1_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle2_add",
        "elastic_add",
        params=case.value("normal_window_nonbatch.cycle2_add"),
    )
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=case.value("normal_window_nonbatch.cycle2_added_topology_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.cycle2_added_topology",
            {"port": output("cycle2_add", "port")},
        ),
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal_window_nonbatch.cycle2_traffic_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.cycle2_traffic",
            {"engine": output("cycle2_add", "engine")},
        ),
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step(
        "cycle2_ramp",
        "elastic_pause",
        params=case.value("normal_window_nonbatch.cycle2_ramp"),
    )
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=case.value("normal_window_nonbatch.cycle2_remove_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.cycle2_remove",
            {"engine": output("cycle2_add", "engine")},
        ),
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_window_nonbatch.cycle2_flow_stop_timeout_s"),
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_window_nonbatch.cycle2_protocol",
            {"flow": output("cycle2_flow", "flow")},
        ),
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=case.value(
            "normal_window_nonbatch.cycle2_removed_topology_timeout_s"
        ),
        params=case.params(
            "normal_window_nonbatch.cycle2_removed_topology",
            {"port": output("cycle2_add", "port")},
        ),
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal_window_nonbatch.cycle2_zero_errors",
            {"result": output("cycle2_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle3_add",
        "elastic_add",
        params=case.value("normal_window_nonbatch.cycle3_add"),
    )
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=case.value("normal_window_nonbatch.cycle3_added_topology_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.cycle3_added_topology",
            {"port": output("cycle3_add", "port")},
        ),
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("normal_window_nonbatch.cycle3_traffic_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.cycle3_traffic",
            {"engine": output("cycle3_add", "engine")},
        ),
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step(
        "cycle3_ramp",
        "elastic_pause",
        params=case.value("normal_window_nonbatch.cycle3_ramp"),
    )
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=case.value("normal_window_nonbatch.cycle3_remove_timeout_s"),
        params=case.params(
            "normal_window_nonbatch.cycle3_remove",
            {"engine": output("cycle3_add", "engine")},
        ),
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("normal_window_nonbatch.cycle3_flow_stop_timeout_s"),
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "normal_window_nonbatch.cycle3_protocol",
            {"flow": output("cycle3_flow", "flow")},
        ),
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=case.value(
            "normal_window_nonbatch.cycle3_removed_topology_timeout_s"
        ),
        params=case.params(
            "normal_window_nonbatch.cycle3_removed_topology",
            {"port": output("cycle3_add", "port")},
        ),
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "normal_window_nonbatch.cycle3_zero_errors",
            {"result": output("cycle3_flow_stop", "result")},
        ),
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=case.value("normal_window_nonbatch.recovery_timeout_s"),
        params=case.value("normal_window_nonbatch.recovery"),
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=case.value("normal_window_nonbatch.final_topology_timeout_s"),
        params=case.value("normal_window_nonbatch.final_topology"),
    )
    case.step("teardown", "teardown")


def strict_window_nonbatch(case):
    case.step(
        "setup", "setup", timeout_s=case.value("strict_window_nonbatch.setup_timeout_s")
    )
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("strict_window_nonbatch.initial_topology_timeout_s"),
        params=case.value("strict_window_nonbatch.initial_topology"),
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params=case.value("strict_window_nonbatch.ramp"))
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=case.value("strict_window_nonbatch.baseline_window_timeout_s"),
        params=case.value("strict_window_nonbatch.baseline_window"),
    )
    case.step("add", "elastic_add", params=case.value("strict_window_nonbatch.add"))
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=case.value("strict_window_nonbatch.first_traffic_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.first_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("strict_window_nonbatch.added_topology_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.added_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=case.value("strict_window_nonbatch.post_window_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.post_window",
            {
                "engines": [
                    case.value("strict_window_nonbatch.post_window.engines.item_0"),
                    case.value("strict_window_nonbatch.post_window.engines.item_1"),
                    output("add", "engine"),
                ]
            },
        ),
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_window_nonbatch.preference_flow_stop_timeout_s"),
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_window_nonbatch.preference_protocol",
            {"flow": output("preference_flow", "flow")},
        ),
    )
    case.step(
        "add_availability",
        "elastic_add_availability",
        params={
            "flow": output("preference_flow", "flow"),
            "mutation": output("add", "mutation"),
            "received_s": output("first_traffic", "observed_s"),
        },
    )
    case.step(
        "preference_received",
        "elastic_window_received",
        params={
            "engine": output("add", "engine"),
            "before": output("post_window", "first"),
            "after": output("post_window", "last"),
        },
    )
    case.step(
        "preference_shares",
        "elastic_share",
        params=case.params(
            "strict_window_nonbatch.preference_shares",
            {
                "series": output("post_window", "series"),
                "engine": output("add", "engine"),
            },
        ),
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params=case.params(
            "strict_window_nonbatch.preference_availability",
            {"result": output("preference_flow_stop", "result")},
        ),
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict_window_nonbatch.remove_traffic_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.remove_traffic", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=case.value("strict_window_nonbatch.remove_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.remove", {"engine": output("add", "engine")}
        ),
    )
    case.step(
        "remove_hold",
        "elastic_pause",
        params=case.value("strict_window_nonbatch.remove_hold"),
    )
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_window_nonbatch.remove_flow_stop_timeout_s"),
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_window_nonbatch.remove_protocol",
            {"flow": output("remove_flow", "flow")},
        ),
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=case.value("strict_window_nonbatch.removed_topology_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.removed_topology", {"port": output("add", "port")}
        ),
    )
    case.step(
        "remove_accounting",
        "elastic_literal_nonbatch_accounting",
        timeout_s=case.value("strict_window_nonbatch.remove_accounting_timeout_s"),
    )
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict_window_nonbatch.remove_zero_errors",
            {"result": output("remove_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle1_add",
        "elastic_add",
        params=case.value("strict_window_nonbatch.cycle1_add"),
    )
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=case.value("strict_window_nonbatch.cycle1_added_topology_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.cycle1_added_topology",
            {"port": output("cycle1_add", "port")},
        ),
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict_window_nonbatch.cycle1_traffic_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.cycle1_traffic",
            {"engine": output("cycle1_add", "engine")},
        ),
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step(
        "cycle1_ramp",
        "elastic_pause",
        params=case.value("strict_window_nonbatch.cycle1_ramp"),
    )
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=case.value("strict_window_nonbatch.cycle1_remove_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.cycle1_remove",
            {"engine": output("cycle1_add", "engine")},
        ),
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_window_nonbatch.cycle1_flow_stop_timeout_s"),
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_window_nonbatch.cycle1_protocol",
            {"flow": output("cycle1_flow", "flow")},
        ),
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=case.value(
            "strict_window_nonbatch.cycle1_removed_topology_timeout_s"
        ),
        params=case.params(
            "strict_window_nonbatch.cycle1_removed_topology",
            {"port": output("cycle1_add", "port")},
        ),
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict_window_nonbatch.cycle1_zero_errors",
            {"result": output("cycle1_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle2_add",
        "elastic_add",
        params=case.value("strict_window_nonbatch.cycle2_add"),
    )
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=case.value("strict_window_nonbatch.cycle2_added_topology_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.cycle2_added_topology",
            {"port": output("cycle2_add", "port")},
        ),
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict_window_nonbatch.cycle2_traffic_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.cycle2_traffic",
            {"engine": output("cycle2_add", "engine")},
        ),
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step(
        "cycle2_ramp",
        "elastic_pause",
        params=case.value("strict_window_nonbatch.cycle2_ramp"),
    )
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=case.value("strict_window_nonbatch.cycle2_remove_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.cycle2_remove",
            {"engine": output("cycle2_add", "engine")},
        ),
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_window_nonbatch.cycle2_flow_stop_timeout_s"),
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_window_nonbatch.cycle2_protocol",
            {"flow": output("cycle2_flow", "flow")},
        ),
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=case.value(
            "strict_window_nonbatch.cycle2_removed_topology_timeout_s"
        ),
        params=case.params(
            "strict_window_nonbatch.cycle2_removed_topology",
            {"port": output("cycle2_add", "port")},
        ),
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict_window_nonbatch.cycle2_zero_errors",
            {"result": output("cycle2_flow_stop", "result")},
        ),
    )
    case.step(
        "cycle3_add",
        "elastic_add",
        params=case.value("strict_window_nonbatch.cycle3_add"),
    )
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=case.value("strict_window_nonbatch.cycle3_added_topology_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.cycle3_added_topology",
            {"port": output("cycle3_add", "port")},
        ),
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=case.value("strict_window_nonbatch.cycle3_traffic_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.cycle3_traffic",
            {"engine": output("cycle3_add", "engine")},
        ),
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step(
        "cycle3_ramp",
        "elastic_pause",
        params=case.value("strict_window_nonbatch.cycle3_ramp"),
    )
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=case.value("strict_window_nonbatch.cycle3_remove_timeout_s"),
        params=case.params(
            "strict_window_nonbatch.cycle3_remove",
            {"engine": output("cycle3_add", "engine")},
        ),
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=case.value("strict_window_nonbatch.cycle3_flow_stop_timeout_s"),
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params=case.params(
            "strict_window_nonbatch.cycle3_protocol",
            {"flow": output("cycle3_flow", "flow")},
        ),
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=case.value(
            "strict_window_nonbatch.cycle3_removed_topology_timeout_s"
        ),
        params=case.params(
            "strict_window_nonbatch.cycle3_removed_topology",
            {"port": output("cycle3_add", "port")},
        ),
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params=case.params(
            "strict_window_nonbatch.cycle3_zero_errors",
            {"result": output("cycle3_flow_stop", "result")},
        ),
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=case.value("strict_window_nonbatch.recovery_timeout_s"),
        params=case.value("strict_window_nonbatch.recovery"),
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=case.value("strict_window_nonbatch.final_topology_timeout_s"),
        params=case.value("strict_window_nonbatch.final_topology"),
    )
    case.step("teardown", "teardown")


def decode_scale_out_protection(case):
    """Keep a newly discovered Decode alive and progressing under existing load."""
    env = case.environment
    limit = env.get("config_overrides", {}).get("decode_max_engine_requests")
    capacity = case.value("decode_scale_out_protection.capacity")
    if (
        type(limit) is not int
        or not capacity["minimum"] <= limit <= capacity["maximum"]
    ):
        raise ValueError("Decode protection capacity is outside the YAML range")
    old_count = env["n_decode"]
    concurrency = case.number("concurrency")
    output_len = case.number("output_len")
    window_s = case.number("window_s")
    if concurrency <= old_count * limit:
        raise ValueError("traffic concurrency must exceed the old Decode pool capacity")
    if env.get("discovery") != "discovery_file":
        raise ValueError("Decode scale-out requires dynamic discovery_file")
    case.step(
        "setup",
        "setup",
        timeout_s=case.value("decode_scale_out_protection.setup_timeout_s"),
    )
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=case.value("decode_scale_out_protection.initial_topology_timeout_s"),
        params=case.params(
            "decode_scale_out_protection.initial_topology",
            {"discovered": old_count, "alive": old_count},
        ),
    )
    case.step(
        "traffic",
        "decode_scale_flow",
        params={"concurrency": concurrency, "output_len": output_len},
    )
    flow = output("traffic", "flow")
    case.step(
        "old_decodes_loaded",
        "decode_scale_loaded",
        timeout_s=case.value(
            "decode_scale_out_protection.old_decodes_loaded_timeout_s"
        ),
        params={"flow": flow, "limit": limit},
    )
    case.step(
        "add_decode",
        "elastic_add",
        params=case.value("decode_scale_out_protection.add_decode"),
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=case.value("decode_scale_out_protection.added_topology_timeout_s"),
        params=case.params(
            "decode_scale_out_protection.added_topology",
            {
                "discovered": old_count + 1,
                "alive": old_count + 1,
                "port": output("add_decode", "port"),
            },
        ),
    )
    case.step(
        "loaded_window",
        "decode_scale_window",
        timeout_s=window_s
        + case.value("decode_scale_out_protection.window_timeout_margin_s"),
        params={"flow": flow, "seconds": window_s},
    )
    case.step(
        "stop_traffic",
        "elastic_flow_stop",
        timeout_s=case.value("decode_scale_out_protection.stop_traffic_timeout_s"),
        params={"flow": flow},
    )
    case.step(
        "all_requests_complete",
        "elastic_flow_assert",
        params=case.params(
            "decode_scale_out_protection.all_requests_complete",
            {"result": output("stop_traffic", "result")},
        ),
    )
    case.step(
        "master_drained",
        "master_inflight_clean",
        timeout_s=case.value("decode_scale_out_protection.master_drained_timeout_s"),
        params=case.value("decode_scale_out_protection.master_drained"),
    )
    case.step(
        "new_decode_protected",
        "decode_scale_check",
        params={
            "flow": flow,
            "engine": output("add_decode", "engine"),
            "limit": limit,
            "started_s": output("loaded_window", "started_s"),
            "window_s": window_s,
        },
    )
