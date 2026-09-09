"""Ordered scale-out, measured preference and rebalance, graceful removal and three complete cycles."""

from ..case_config import output

METADATA = {
    "id": "elastic_lifecycle",
    "description": "Ordered scale-out, measured preference and rebalance, graceful removal and three "
    "complete cycles.",
    "category": "elastic",
    "requires": ["queue"],
    "estimated_duration_s": 420,
}

PROFILES = ["batch-window", "single-batch", "single-nonbatch", "window-nonbatch"]


def normal(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params={"seconds": 1})
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=25,
        params={"engines": ["prefill-0", "prefill-1"], "offsets_s": [0, 15]},
    )
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=15,
        params={"engine": output("add", "engine"), "baseline": 0, "window_s": 10},
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=55,
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0, 5, 10, 17, 24, 31, 38, 45],
        },
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("preference_flow", "flow"), "method": "FetchResponse"},
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
        params={
            "series": output("post_window", "series"),
            "engine": output("add", "engine"),
            "max_share": 0.6,
            "old_floor": 0.1,
            "exclusive": False,
            "require_new": False,
        },
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params={
            "result": output("preference_flow_stop", "result"),
            "min_success_rate": 0.9,
        },
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("add", "engine"),
            "window_s": 10,
            "method": "FetchResponse",
        },
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step("remove_hold", "elastic_pause", params={"seconds": 3})
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("remove_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("add", "port"),
            "present": False,
        },
    )
    case.step("remove_accounting", "elastic_accounting", timeout_s=105)
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params={"result": output("remove_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle1_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle1_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle1_add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step("cycle1_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle1_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle1_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle1_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle1_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle2_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle2_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle2_add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step("cycle2_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle2_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle2_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle2_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle2_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle3_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle3_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle3_add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step("cycle3_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle3_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle3_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle3_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle3_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=65,
        params={"method": "FetchResponse"},
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("teardown", "teardown")


def strict(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params={"seconds": 1})
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=25,
        params={"engines": ["prefill-0", "prefill-1"], "offsets_s": [0, 15]},
    )
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=15,
        params={"engine": output("add", "engine"), "baseline": 0, "window_s": 10},
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=55,
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0, 5, 10, 17, 24, 31, 38, 45],
        },
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("preference_flow", "flow"), "method": "FetchResponse"},
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
        params={
            "series": output("post_window", "series"),
            "engine": output("add", "engine"),
            "max_share": 0.5,
            "old_floor": 0.1,
            "exclusive": False,
            "require_new": False,
        },
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params={
            "result": output("preference_flow_stop", "result"),
            "min_success_rate": 0.9,
        },
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("add", "engine"),
            "window_s": 10,
            "method": "FetchResponse",
        },
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step("remove_hold", "elastic_pause", params={"seconds": 3})
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("remove_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("add", "port"),
            "present": False,
        },
    )
    case.step("remove_accounting", "elastic_accounting", timeout_s=105)
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params={"result": output("remove_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle1_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle1_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle1_add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step("cycle1_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle1_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle1_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle1_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle1_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle2_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle2_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle2_add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step("cycle2_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle2_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle2_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle2_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle2_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle3_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle3_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle3_add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step("cycle3_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle3_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle3_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle3_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle3_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=65,
        params={"method": "FetchResponse"},
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("teardown", "teardown")


def rebalance(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step(
        "rebalance_baseline",
        "elastic_rebalance_batch",
        timeout_s=240,
        params={"method": "FetchResponse"},
    )
    case.step(
        "baseline_after",
        "elastic_timeline",
        params={"engines": ["prefill-0", "prefill-1"], "offsets_s": [0]},
    )
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "rebalance_before",
        "elastic_timeline",
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0],
        },
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
        timeout_s=240,
        params={"method": "FetchResponse"},
    )
    case.step(
        "rebalance_after",
        "elastic_timeline",
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0],
        },
    )
    case.step(
        "rebalance_share",
        "elastic_share",
        params={
            "before": output("rebalance_anchor", "before"),
            "after": output("rebalance_after", "last"),
            "engine": output("add", "engine"),
            "max_share": 0.6,
            "old_floor": 0,
            "exclusive": True,
            "require_new": True,
        },
    )
    case.step("teardown", "teardown")


def kv_skew_hot(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("seed", "elastic_seed", timeout_s=120)
    case.step("metrics", "elastic_metrics_start")
    case.step(
        "flow", "elastic_flow_start", params={"families": output("seed", "families")}
    )
    case.step(
        "baseline",
        "elastic_baseline",
        timeout_s=30,
        params={"observation": output("metrics", "observation"), "phase": "baseline"},
    )
    case.step(
        "scale",
        "elastic_scale",
        timeout_s=75,
        params={
            "families": output("seed", "families"),
            "victim": "hot",
            "drain_timeout_ms": 60000,
        },
    )
    case.step(
        "transient",
        "elastic_window",
        timeout_s=30,
        params={
            "observation": output("metrics", "observation"),
            "baseline": output("baseline", "window"),
            "scale": output("scale", "scale"),
            "families": output("seed", "families"),
            "victim": "hot",
            "phase": "transient",
        },
    )
    case.step(
        "steady",
        "elastic_window",
        timeout_s=95,
        params={
            "observation": output("metrics", "observation"),
            "baseline": output("baseline", "window"),
            "scale": output("scale", "scale"),
            "families": output("seed", "families"),
            "victim": "hot",
            "transient": output("transient", "window"),
            "phase": "steady",
        },
    )
    case.step(
        "flow_stop",
        "elastic_flow_stop",
        timeout_s=65,
        params={"flow": output("flow", "flow")},
    )
    case.step("recovery", "elastic_recovery", timeout_s=240)
    case.step(
        "verdict",
        "elastic_verdict",
        params={
            "baseline": output("baseline", "window"),
            "transient": output("transient", "window"),
            "steady": output("steady", "window"),
            "scale": output("scale", "scale"),
            "flow_result": output("flow_stop", "result"),
            "recovery": output("recovery", "requests"),
            "victim": "hot",
        },
    )
    case.step("teardown", "teardown")


def kv_skew_cold(case):
    case.step("setup", "setup", timeout_s=180)
    case.step("seed", "elastic_seed", timeout_s=120)
    case.step("metrics", "elastic_metrics_start")
    case.step(
        "flow", "elastic_flow_start", params={"families": output("seed", "families")}
    )
    case.step(
        "baseline",
        "elastic_baseline",
        timeout_s=30,
        params={"observation": output("metrics", "observation"), "phase": "baseline"},
    )
    case.step(
        "scale",
        "elastic_scale",
        timeout_s=75,
        params={
            "families": output("seed", "families"),
            "victim": "cold",
            "drain_timeout_ms": 60000,
        },
    )
    case.step(
        "transient",
        "elastic_window",
        timeout_s=30,
        params={
            "observation": output("metrics", "observation"),
            "baseline": output("baseline", "window"),
            "scale": output("scale", "scale"),
            "families": output("seed", "families"),
            "victim": "cold",
            "phase": "transient",
        },
    )
    case.step(
        "steady",
        "elastic_window",
        timeout_s=95,
        params={
            "observation": output("metrics", "observation"),
            "baseline": output("baseline", "window"),
            "scale": output("scale", "scale"),
            "families": output("seed", "families"),
            "victim": "cold",
            "transient": output("transient", "window"),
            "phase": "steady",
        },
    )
    case.step(
        "flow_stop",
        "elastic_flow_stop",
        timeout_s=65,
        params={"flow": output("flow", "flow")},
    )
    case.step("recovery", "elastic_recovery", timeout_s=240)
    case.step(
        "verdict",
        "elastic_verdict",
        params={
            "baseline": output("baseline", "window"),
            "transient": output("transient", "window"),
            "steady": output("steady", "window"),
            "scale": output("scale", "scale"),
            "flow_result": output("flow_stop", "result"),
            "recovery": output("recovery", "requests"),
            "victim": "cold",
        },
    )
    case.step("teardown", "teardown")


def steady_recovery(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "DECODE", "discovered": 4, "alive": 4},
    )
    case.step("observe", "elastic_balance_observe")
    case.step("flow", "elastic_balance_flow", params={"interval_s": 0.2})
    case.step(
        "baseline",
        "elastic_balance_window",
        timeout_s=25,
        params={"observation": output("observe", "observation"), "duration_s": 20},
    )
    case.step(
        "baseline_guard",
        "elastic_steady_baseline",
        params={"window": output("baseline", "window")},
    )
    case.step(
        "remove",
        "elastic_balance_remove",
        timeout_s=110,
        params={"engine": "decode-0", "drain_timeout_ms": 60000},
    )
    case.step(
        "transient",
        "elastic_balance_window",
        timeout_s=25,
        params={"observation": output("observe", "observation"), "duration_s": 20},
    )
    case.step(
        "settled_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "DECODE",
            "discovered": 3,
            "alive": 3,
            "port": output("remove", "port"),
            "present": False,
        },
    )
    case.step(
        "steady",
        "elastic_balance_window",
        timeout_s=65,
        params={"observation": output("observe", "observation"), "duration_s": 60},
    )
    case.step("recovery", "elastic_pending_recovery", timeout_s=100)
    case.step(
        "flow_stop",
        "elastic_flow_stop",
        timeout_s=65,
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
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "DECODE", "discovered": 2, "alive": 2},
    )
    case.step("observe", "elastic_balance_observe")
    case.step(
        "baseline_flow",
        "elastic_balance_flow",
        params={"interval_s": 0.5, "stream_timeout_s": 10},
    )
    case.step("baseline_ramp", "elastic_pause", params={"seconds": 1})
    case.step(
        "baseline",
        "elastic_balance_window",
        timeout_s=25,
        params={
            "observation": output("observe", "observation"),
            "since": output("observe", "started_s"),
            "duration_s": 20,
        },
    )
    case.step(
        "baseline_stop",
        "elastic_flow_stop",
        timeout_s=65,
        params={"flow": output("baseline_flow", "flow")},
    )
    case.step(
        "slow_both",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["decode-0", "decode-1"],
            "perf": {"decode_scale": 60},
        },
    )
    case.step("slow_sync", "elastic_pause", params={"seconds": 1})
    case.step(
        "fill_ok",
        "elastic_decode_fill",
        timeout_s=70,
        params={"victim": "decode-0", "survivor": "decode-1"},
    )
    case.step(
        "remove_ok",
        "elastic_balance_remove",
        timeout_s=110,
        params={"engine": "decode-0", "drain_timeout_ms": 60000},
    )
    case.step(
        "collect_ok",
        "elastic_full_collect",
        timeout_s=100,
        params={
            "requests": output("fill_ok", "requests"),
            "mutation": output("remove_ok", "mutation"),
        },
    )
    case.step("accounting_ok", "elastic_pending_accounting", timeout_s=60)
    case.step("add_decode", "elastic_add", params={"role": "decode"})
    case.step(
        "restored_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "DECODE",
            "discovered": 2,
            "alive": 2,
            "port": output("add_decode", "port"),
            "present": True,
        },
    )
    case.step("settled", "elastic_balance_mark")
    case.step(
        "steady_flow",
        "elastic_balance_flow",
        params={"interval_s": 0.5, "stream_timeout_s": 10},
    )
    case.step("steady_ramp", "elastic_pause", params={"seconds": 1})
    case.step(
        "steady",
        "elastic_balance_window",
        timeout_s=65,
        params={
            "observation": output("observe", "observation"),
            "since": output("settled", "time_s"),
            "duration_s": 60,
        },
    )
    case.step(
        "steady_stop",
        "elastic_flow_stop",
        timeout_s=65,
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
        params={"result": output("collect_ok", "result"), "branch": "drain_ok"},
    )
    case.step(
        "slow_victim",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": ["decode-1"],
            "perf": {"decode_scale": 1000},
        },
    )
    case.step(
        "slow_survivor",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("add_decode", "engine")],
            "perf": {"decode_scale": 60},
        },
    )
    case.step("timeout_sync", "elastic_pause", params={"seconds": 1})
    case.step(
        "fill_timeout",
        "elastic_decode_fill",
        timeout_s=70,
        params={"victim": "decode-1", "survivor": output("add_decode", "engine")},
    )
    case.step(
        "remove_timeout",
        "elastic_balance_remove",
        timeout_s=110,
        params={"engine": "decode-1", "drain_timeout_ms": 5000},
    )
    case.step(
        "collect_timeout",
        "elastic_full_collect",
        timeout_s=100,
        params={
            "requests": output("fill_timeout", "requests"),
            "mutation": output("remove_timeout", "mutation"),
        },
    )
    case.step(
        "observe_full_transient",
        "elastic_balance_window",
        timeout_s=25,
        params={"observation": output("observe", "observation"), "duration_s": 20},
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
        params={
            "result": output("collect_timeout", "result"),
            "branch": "drain_timeout",
        },
    )
    case.step(
        "restore_survivor",
        "engine_control",
        params={
            "operation": "set_perf",
            "targets": [output("add_decode", "engine")],
            "perf": {"decode_scale": 1},
        },
    )
    case.step("recovery", "elastic_pending_recovery", timeout_s=100)
    case.step("teardown", "teardown")


def transient_imbalance(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 3, "alive": 3},
    )
    case.step("observe", "elastic_transient_observe")
    case.step("flow", "elastic_balance_flow", params={"interval_s": 0.1})
    case.step(
        "baseline",
        "elastic_balance_window",
        timeout_s=30,
        params={
            "observation": output("observe", "observation"),
            "duration_s": 20,
            "since": output("observe", "started_s"),
        },
    )
    case.step("pre_event", "elastic_transient_prepare")
    case.step("burst", "elastic_transient_burst", timeout_s=10)
    case.step(
        "remove",
        "elastic_transient_remove",
        timeout_s=15,
        params={"requests": output("burst", "requests")},
    )
    case.step(
        "burst_settled",
        "elastic_transient_collect",
        timeout_s=160,
        params={"requests": output("burst", "requests")},
    )
    case.step(
        "transient",
        "elastic_balance_window",
        timeout_s=30,
        params={
            "observation": output("observe", "observation"),
            "duration_s": 20,
            "since": output("remove", "started_s"),
        },
    )
    case.step(
        "survivor_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("settled", "elastic_balance_mark")
    case.step(
        "steady",
        "elastic_balance_window",
        timeout_s=70,
        params={
            "observation": output("observe", "observation"),
            "duration_s": 60,
            "since": output("settled", "time_s"),
        },
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
    case.step("recovery", "elastic_pending_recovery", timeout_s=60)
    case.step(
        "flow_stop",
        "elastic_flow_stop",
        timeout_s=65,
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
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step(
        "rebalance_baseline",
        "elastic_rebalance_batch",
        timeout_s=240,
        params={"method": "FetchResponse"},
    )
    case.step(
        "baseline_after",
        "elastic_timeline",
        params={"engines": ["prefill-0", "prefill-1"], "offsets_s": [0]},
    )
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "rebalance_before",
        "elastic_timeline",
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0],
        },
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
        timeout_s=240,
        params={"method": "FetchResponse"},
    )
    case.step(
        "rebalance_after",
        "elastic_timeline",
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0],
        },
    )
    case.step(
        "rebalance_share",
        "elastic_share",
        params={
            "before": output("rebalance_anchor", "before"),
            "after": output("rebalance_after", "last"),
            "engine": output("add", "engine"),
            "max_share": 0.6,
            "old_floor": 0,
            "exclusive": True,
            "require_new": True,
        },
    )
    case.step("teardown", "teardown")


def rebalance_single_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step(
        "rebalance_baseline",
        "elastic_rebalance_batch",
        timeout_s=240,
        params={"method": "GenerateStreamCall"},
    )
    case.step(
        "baseline_after",
        "elastic_timeline",
        params={"engines": ["prefill-0", "prefill-1"], "offsets_s": [0]},
    )
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "rebalance_before",
        "elastic_timeline",
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0],
        },
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
        timeout_s=240,
        params={"method": "GenerateStreamCall"},
    )
    case.step(
        "rebalance_after",
        "elastic_timeline",
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0],
        },
    )
    case.step(
        "rebalance_share",
        "elastic_share",
        params={
            "before": output("rebalance_anchor", "before"),
            "after": output("rebalance_after", "last"),
            "engine": output("add", "engine"),
            "max_share": 0.6,
            "old_floor": 0,
            "exclusive": True,
            "require_new": True,
        },
    )
    case.step("teardown", "teardown")


def rebalance_window_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step(
        "rebalance_baseline",
        "elastic_rebalance_batch",
        timeout_s=240,
        params={"method": "GenerateStreamCall"},
    )
    case.step(
        "baseline_after",
        "elastic_timeline",
        params={"engines": ["prefill-0", "prefill-1"], "offsets_s": [0]},
    )
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "rebalance_before",
        "elastic_timeline",
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0],
        },
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
        timeout_s=240,
        params={"method": "GenerateStreamCall"},
    )
    case.step(
        "rebalance_after",
        "elastic_timeline",
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0],
        },
    )
    case.step(
        "rebalance_share",
        "elastic_share",
        params={
            "before": output("rebalance_anchor", "before"),
            "after": output("rebalance_after", "last"),
            "engine": output("add", "engine"),
            "max_share": 0.6,
            "old_floor": 0,
            "exclusive": True,
            "require_new": True,
        },
    )
    case.step("teardown", "teardown")


def normal_single_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params={"seconds": 1})
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=25,
        params={"engines": ["prefill-0", "prefill-1"], "offsets_s": [0, 15]},
    )
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=15,
        params={"engine": output("add", "engine"), "baseline": 0, "window_s": 10},
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=55,
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0, 5, 10, 17, 24, 31, 38, 45],
        },
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("preference_flow", "flow"), "method": "FetchResponse"},
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
        params={
            "series": output("post_window", "series"),
            "engine": output("add", "engine"),
            "max_share": 0.6,
            "old_floor": 0.1,
            "exclusive": False,
            "require_new": False,
        },
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params={
            "result": output("preference_flow_stop", "result"),
            "min_success_rate": 0.9,
        },
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("add", "engine"),
            "window_s": 10,
            "method": "FetchResponse",
        },
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step("remove_hold", "elastic_pause", params={"seconds": 3})
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("remove_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("add", "port"),
            "present": False,
        },
    )
    case.step("remove_accounting", "elastic_accounting", timeout_s=105)
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params={"result": output("remove_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle1_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle1_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle1_add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step("cycle1_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle1_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle1_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle1_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle1_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle2_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle2_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle2_add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step("cycle2_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle2_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle2_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle2_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle2_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle3_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle3_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle3_add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step("cycle3_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle3_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle3_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle3_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle3_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=65,
        params={"method": "FetchResponse"},
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("teardown", "teardown")


def strict_single_batch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params={"seconds": 1})
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=25,
        params={"engines": ["prefill-0", "prefill-1"], "offsets_s": [0, 15]},
    )
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=15,
        params={"engine": output("add", "engine"), "baseline": 0, "window_s": 10},
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=55,
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0, 5, 10, 17, 24, 31, 38, 45],
        },
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("preference_flow", "flow"), "method": "FetchResponse"},
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
        params={
            "series": output("post_window", "series"),
            "engine": output("add", "engine"),
            "max_share": 0.5,
            "old_floor": 0.1,
            "exclusive": False,
            "require_new": False,
        },
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params={
            "result": output("preference_flow_stop", "result"),
            "min_success_rate": 0.9,
        },
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("add", "engine"),
            "window_s": 10,
            "method": "FetchResponse",
        },
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step("remove_hold", "elastic_pause", params={"seconds": 3})
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("remove_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("add", "port"),
            "present": False,
        },
    )
    case.step("remove_accounting", "elastic_accounting", timeout_s=105)
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params={"result": output("remove_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle1_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle1_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle1_add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step("cycle1_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle1_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle1_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle1_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle1_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle2_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle2_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle2_add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step("cycle2_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle2_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle2_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle2_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle2_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle3_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle3_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle3_add", "engine"),
            "window_s": 15,
            "method": "FetchResponse",
        },
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step("cycle3_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle3_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle3_flow", "flow"), "method": "FetchResponse"},
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle3_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle3_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=65,
        params={"method": "FetchResponse"},
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("teardown", "teardown")


def normal_single_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params={"seconds": 1})
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=25,
        params={"engines": ["prefill-0", "prefill-1"], "offsets_s": [0, 15]},
    )
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=15,
        params={"engine": output("add", "engine"), "baseline": 0, "window_s": 10},
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=55,
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0, 5, 10, 17, 24, 31, 38, 45],
        },
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params={
            "flow": output("preference_flow", "flow"),
            "method": "GenerateStreamCall",
        },
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
        params={
            "series": output("post_window", "series"),
            "engine": output("add", "engine"),
            "max_share": 0.6,
            "old_floor": 0.1,
            "exclusive": False,
            "require_new": False,
        },
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params={
            "result": output("preference_flow_stop", "result"),
            "min_success_rate": 0.9,
        },
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("add", "engine"),
            "window_s": 10,
            "method": "GenerateStreamCall",
        },
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step("remove_hold", "elastic_pause", params={"seconds": 3})
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("remove_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("add", "port"),
            "present": False,
        },
    )
    case.step("remove_accounting", "elastic_literal_nonbatch_accounting", timeout_s=105)
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params={"result": output("remove_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle1_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle1_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle1_add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step("cycle1_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle1_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle1_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle1_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle1_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle2_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle2_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle2_add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step("cycle2_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle2_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle2_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle2_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle2_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle3_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle3_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle3_add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step("cycle3_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle3_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle3_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle3_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle3_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=65,
        params={"method": "GenerateStreamCall"},
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("teardown", "teardown")


def strict_single_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params={"seconds": 1})
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=25,
        params={"engines": ["prefill-0", "prefill-1"], "offsets_s": [0, 15]},
    )
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=15,
        params={"engine": output("add", "engine"), "baseline": 0, "window_s": 10},
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=55,
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0, 5, 10, 17, 24, 31, 38, 45],
        },
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params={
            "flow": output("preference_flow", "flow"),
            "method": "GenerateStreamCall",
        },
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
        params={
            "series": output("post_window", "series"),
            "engine": output("add", "engine"),
            "max_share": 0.5,
            "old_floor": 0.1,
            "exclusive": False,
            "require_new": False,
        },
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params={
            "result": output("preference_flow_stop", "result"),
            "min_success_rate": 0.9,
        },
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("add", "engine"),
            "window_s": 10,
            "method": "GenerateStreamCall",
        },
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step("remove_hold", "elastic_pause", params={"seconds": 3})
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("remove_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("add", "port"),
            "present": False,
        },
    )
    case.step("remove_accounting", "elastic_literal_nonbatch_accounting", timeout_s=105)
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params={"result": output("remove_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle1_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle1_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle1_add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step("cycle1_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle1_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle1_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle1_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle1_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle2_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle2_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle2_add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step("cycle2_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle2_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle2_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle2_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle2_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle3_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle3_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle3_add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step("cycle3_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle3_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle3_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle3_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle3_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=65,
        params={"method": "GenerateStreamCall"},
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("teardown", "teardown")


def normal_window_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params={"seconds": 1})
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=25,
        params={"engines": ["prefill-0", "prefill-1"], "offsets_s": [0, 15]},
    )
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=15,
        params={"engine": output("add", "engine"), "baseline": 0, "window_s": 10},
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=55,
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0, 5, 10, 17, 24, 31, 38, 45],
        },
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params={
            "flow": output("preference_flow", "flow"),
            "method": "GenerateStreamCall",
        },
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
        params={
            "series": output("post_window", "series"),
            "engine": output("add", "engine"),
            "max_share": 0.6,
            "old_floor": 0.1,
            "exclusive": False,
            "require_new": False,
        },
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params={
            "result": output("preference_flow_stop", "result"),
            "min_success_rate": 0.9,
        },
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("add", "engine"),
            "window_s": 10,
            "method": "GenerateStreamCall",
        },
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step("remove_hold", "elastic_pause", params={"seconds": 3})
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("remove_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("add", "port"),
            "present": False,
        },
    )
    case.step("remove_accounting", "elastic_literal_nonbatch_accounting", timeout_s=105)
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params={"result": output("remove_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle1_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle1_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle1_add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step("cycle1_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle1_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle1_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle1_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle1_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle2_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle2_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle2_add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step("cycle2_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle2_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle2_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle2_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle2_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle3_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle3_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle3_add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step("cycle3_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle3_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle3_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle3_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle3_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=65,
        params={"method": "GenerateStreamCall"},
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("teardown", "teardown")


def strict_window_nonbatch(case):
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("preference_flow", "elastic_cold_flow")
    case.step("ramp", "elastic_pause", params={"seconds": 1})
    case.step(
        "baseline_window",
        "elastic_timeline",
        timeout_s=25,
        params={"engines": ["prefill-0", "prefill-1"], "offsets_s": [0, 15]},
    )
    case.step("add", "elastic_add", params={"role": "prefill"})
    case.step(
        "first_traffic",
        "elastic_accepted_timed",
        timeout_s=15,
        params={"engine": output("add", "engine"), "baseline": 0, "window_s": 10},
    )
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("add", "port"),
            "present": True,
        },
    )
    case.step(
        "post_window",
        "elastic_timeline",
        timeout_s=55,
        params={
            "engines": ["prefill-0", "prefill-1", output("add", "engine")],
            "offsets_s": [0, 5, 10, 17, 24, 31, 38, 45],
        },
    )
    case.step(
        "preference_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("preference_flow", "flow")},
    )
    case.step(
        "preference_protocol",
        "elastic_lifecycle_flow_protocol",
        params={
            "flow": output("preference_flow", "flow"),
            "method": "GenerateStreamCall",
        },
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
        params={
            "series": output("post_window", "series"),
            "engine": output("add", "engine"),
            "max_share": 0.5,
            "old_floor": 0.1,
            "exclusive": False,
            "require_new": False,
        },
    )
    case.step(
        "preference_availability",
        "elastic_flow_assert",
        params={
            "result": output("preference_flow_stop", "result"),
            "min_success_rate": 0.9,
        },
    )
    case.step("remove_flow", "elastic_cold_flow")
    case.step(
        "remove_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("add", "engine"),
            "window_s": 10,
            "method": "GenerateStreamCall",
        },
    )
    case.step(
        "remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step("remove_hold", "elastic_pause", params={"seconds": 3})
    case.step(
        "remove_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("remove_flow", "flow")},
    )
    case.step(
        "remove_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("remove_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("add", "port"),
            "present": False,
        },
    )
    case.step("remove_accounting", "elastic_literal_nonbatch_accounting", timeout_s=105)
    case.step(
        "remove_zero_errors",
        "elastic_flow_assert",
        params={"result": output("remove_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle1_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle1_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle1_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle1_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle1_add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step("cycle1_flow", "elastic_cold_flow")
    case.step("cycle1_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle1_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle1_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle1_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle1_flow", "flow")},
    )
    case.step(
        "cycle1_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle1_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "cycle1_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle1_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle1_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle1_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle2_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle2_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle2_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle2_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle2_add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step("cycle2_flow", "elastic_cold_flow")
    case.step("cycle2_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle2_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle2_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle2_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle2_flow", "flow")},
    )
    case.step(
        "cycle2_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle2_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "cycle2_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle2_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle2_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle2_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step("cycle3_add", "elastic_add", params={"role": "prefill"})
    case.step(
        "cycle3_added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 3,
            "alive": 3,
            "port": output("cycle3_add", "port"),
            "present": True,
        },
    )
    case.step(
        "cycle3_traffic",
        "elastic_lifecycle_probe",
        timeout_s=60,
        params={
            "engine": output("cycle3_add", "engine"),
            "window_s": 15,
            "method": "GenerateStreamCall",
        },
    )
    case.step("cycle3_flow", "elastic_cold_flow")
    case.step("cycle3_ramp", "elastic_pause", params={"seconds": 0.5})
    case.step(
        "cycle3_remove",
        "elastic_remove",
        timeout_s=75,
        params={"engine": output("cycle3_add", "engine"), "drain_timeout_ms": 60000},
    )
    case.step(
        "cycle3_flow_stop",
        "elastic_flow_stop",
        timeout_s=45,
        params={"flow": output("cycle3_flow", "flow")},
    )
    case.step(
        "cycle3_protocol",
        "elastic_lifecycle_flow_protocol",
        params={"flow": output("cycle3_flow", "flow"), "method": "GenerateStreamCall"},
    )
    case.step(
        "cycle3_removed_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "PREFILL",
            "discovered": 2,
            "alive": 2,
            "port": output("cycle3_add", "port"),
            "present": False,
        },
    )
    case.step(
        "cycle3_zero_errors",
        "elastic_flow_assert",
        params={"result": output("cycle3_flow_stop", "result"), "min_success_rate": 1},
    )
    case.step(
        "recovery",
        "elastic_cycle_recovery",
        timeout_s=65,
        params={"method": "GenerateStreamCall"},
    )
    case.step(
        "final_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "PREFILL", "discovered": 2, "alive": 2},
    )
    case.step("teardown", "teardown")


def decode_scale_out_protection(case):
    """Keep a newly discovered Decode alive and progressing under existing load."""
    env = case.environment
    limit = env.get("config_overrides", {}).get("decode_max_engine_requests")
    if type(limit) is not int or not 2 <= limit <= 64:
        raise ValueError("Decode protection requires an explicit capacity of 2..64")
    old_count = env["n_decode"]
    concurrency = case.number("concurrency", 24, minimum=2, maximum=128)
    output_len = case.number("output_len", 128, minimum=64, maximum=512)
    window_s = case.number("window_s", 20, minimum=20, maximum=60)
    if concurrency <= old_count * limit:
        raise ValueError("traffic concurrency must exceed the old Decode pool capacity")
    if env.get("discovery") != "discovery_file":
        raise ValueError("Decode scale-out requires dynamic discovery_file")
    case.step("setup", "setup", timeout_s=180)
    case.step(
        "initial_topology",
        "elastic_topology",
        timeout_s=35,
        params={"role": "DECODE", "discovered": old_count, "alive": old_count},
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
        timeout_s=20,
        params={"flow": flow, "limit": limit},
    )
    case.step("add_decode", "elastic_add", params={"role": "decode"})
    case.step(
        "added_topology",
        "elastic_topology",
        timeout_s=35,
        params={
            "role": "DECODE",
            "discovered": old_count + 1,
            "alive": old_count + 1,
            "port": output("add_decode", "port"),
            "present": True,
        },
    )
    case.step(
        "loaded_window",
        "decode_scale_window",
        timeout_s=window_s + 5,
        params={"flow": flow, "seconds": window_s},
    )
    case.step("stop_traffic", "elastic_flow_stop", timeout_s=50, params={"flow": flow})
    case.step(
        "all_requests_complete",
        "elastic_flow_assert",
        params={
            "result": output("stop_traffic", "result"),
            "min_success_rate": 1,
        },
    )
    case.step(
        "master_drained",
        "master_inflight_clean",
        timeout_s=40,
        params={"target": "single"},
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


VARIANTS = {
    "decode_scale_out_protection": {
        "build": decode_scale_out_protection,
        "profiles": PROFILES,
        "metadata": {},
    },
    "normal": {
        "build": normal,
        "profiles": ["batch-window"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "strict": {
        "build": strict,
        "profiles": ["batch-window"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "rebalance": {
        "build": rebalance,
        "profiles": ["batch-window"],
        "metadata": {},
    },
    "kv_skew_hot": {
        "build": kv_skew_hot,
        "profiles": ["batch-window"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "kv_skew_cold": {
        "build": kv_skew_cold,
        "profiles": ["batch-window"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "steady_recovery": {
        "build": steady_recovery,
        "profiles": ["batch-window"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "kv_full_shrink": {
        "build": kv_full_shrink,
        "profiles": ["batch-window"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "transient_imbalance": {
        "build": transient_imbalance,
        "profiles": ["batch-window"],
        "metadata": {
            "requires": ["enqueue_batch"],
        },
    },
    "rebalance_single_batch": {
        "build": rebalance_single_batch,
        "profiles": ["single-batch"],
        "metadata": {},
    },
    "rebalance_single_nonbatch": {
        "build": rebalance_single_nonbatch,
        "profiles": ["single-nonbatch"],
        "metadata": {},
    },
    "rebalance_window_nonbatch": {
        "build": rebalance_window_nonbatch,
        "profiles": ["window-nonbatch"],
        "metadata": {},
    },
    "normal_single_batch": {
        "build": normal_single_batch,
        "profiles": ["single-batch"],
        "metadata": {
            "requires": ["queue"],
        },
    },
    "strict_single_batch": {
        "build": strict_single_batch,
        "profiles": ["single-batch"],
        "metadata": {
            "requires": ["queue"],
        },
    },
    "normal_single_nonbatch": {
        "build": normal_single_nonbatch,
        "profiles": ["single-nonbatch"],
        "metadata": {
            "requires": ["queue"],
        },
    },
    "strict_single_nonbatch": {
        "build": strict_single_nonbatch,
        "profiles": ["single-nonbatch"],
        "metadata": {
            "requires": ["queue"],
        },
    },
    "normal_window_nonbatch": {
        "build": normal_window_nonbatch,
        "profiles": ["window-nonbatch"],
        "metadata": {
            "requires": ["queue"],
        },
    },
    "strict_window_nonbatch": {
        "build": strict_window_nonbatch,
        "profiles": ["window-nonbatch"],
        "metadata": {
            "requires": ["queue"],
        },
    },
}
