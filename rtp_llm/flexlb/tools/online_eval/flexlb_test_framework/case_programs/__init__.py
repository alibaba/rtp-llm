"""Registered Python case programs. YAML can only select entries in this registry."""

PROGRAMS = {
    "cache_scale_in": "flexlb_test_framework.case_programs.cache_scale_in",
    "trace_scale_out": "flexlb_test_framework.case_programs.trace_scale_out",
    "balance_distribution": "flexlb_test_framework.case_programs.balance_distribution",
    "request_completion": "flexlb_test_framework.case_programs.request_completion",
    "elastic_concurrent_mutation": "flexlb_test_framework.case_programs.elastic_concurrent_mutation",
    "elastic_lifecycle": "flexlb_test_framework.case_programs.elastic_lifecycle",
    "elastic_pending_drain": "flexlb_test_framework.case_programs.elastic_pending_drain",
    "engine_fault_recovery": "flexlb_test_framework.case_programs.engine_fault_recovery",
    "cache_affinity": "flexlb_test_framework.case_programs.cache_affinity",
    "cache_capacity_recovery": "flexlb_test_framework.case_programs.cache_capacity_recovery",
    "cache_churn": "flexlb_test_framework.case_programs.cache_churn",
    "client_fallback_failback": "flexlb_test_framework.case_programs.client_fallback_failback",
    "master_ha_failover": "flexlb_test_framework.case_programs.master_ha_failover",
    "master_lifecycle": "flexlb_test_framework.case_programs.master_lifecycle",
}
