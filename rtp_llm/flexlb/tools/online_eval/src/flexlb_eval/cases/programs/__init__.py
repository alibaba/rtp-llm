"""Registered Python case programs. YAML can only select entries in this registry."""

PROGRAMS = {
    "cache_scale_in": "flexlb_eval.cases.programs.cache_scale_in",
    "trace_scale_out": "flexlb_eval.cases.programs.trace_scale_out",
    "balance_distribution": "flexlb_eval.cases.programs.balance_distribution",
    "request_completion": "flexlb_eval.cases.programs.request_completion",
    "elastic_concurrent_mutation": "flexlb_eval.cases.programs.elastic_concurrent_mutation",
    "elastic_lifecycle": "flexlb_eval.cases.programs.elastic_lifecycle",
    "elastic_pending_drain": "flexlb_eval.cases.programs.elastic_pending_drain",
    "engine_fault_recovery": "flexlb_eval.cases.programs.engine_fault_recovery",
    "cache_affinity": "flexlb_eval.cases.programs.cache_affinity",
    "cache_capacity_recovery": "flexlb_eval.cases.programs.cache_capacity_recovery",
    "cache_churn": "flexlb_eval.cases.programs.cache_churn",
    "client_fallback_failback": "flexlb_eval.cases.programs.client_fallback_failback",
    "master_ha_failover": "flexlb_eval.cases.programs.master_ha_failover",
    "master_lifecycle": "flexlb_eval.cases.programs.master_lifecycle",
}
