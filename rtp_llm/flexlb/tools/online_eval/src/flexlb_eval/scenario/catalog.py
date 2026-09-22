"""Explicit builtin adapter registration. Scenario data never imports Python."""

from flexlb_eval.scenario.actions.admission import HANDLERS as ADMISSION_HANDLERS
from flexlb_eval.scenario.actions.balance import HANDLERS as BALANCE_HANDLERS
from flexlb_eval.scenario.actions.cache_scale_in import HANDLERS as CACHE_SCALE_IN_HANDLERS
from flexlb_eval.scenario.actions.cache_storm import HANDLERS as CACHE_STORM_HANDLERS
from flexlb_eval.scenario.actions.cancel import HANDLERS as CANCEL_HANDLERS
from flexlb_eval.scenario.actions.client_fetch import HANDLERS as CLIENT_FETCH_HANDLERS
from flexlb_eval.scenario.actions.decode_scale_out import HANDLERS as DECODE_SCALE_OUT_HANDLERS
from flexlb_eval.scenario.actions.elastic import HANDLERS as ELASTIC_HANDLERS
from flexlb_eval.scenario.actions.elastic_added_worker import HANDLERS as ELASTIC_ADDED_WORKER_HANDLERS
from flexlb_eval.scenario.actions.engine_control import HANDLERS as ENGINE_CONTROL_HANDLERS
from flexlb_eval.scenario.actions.engine_fault import HANDLERS as ENGINE_FAULT_HANDLERS
from flexlb_eval.scenario.actions.engine_recovery import HANDLERS as ENGINE_RECOVERY_HANDLERS
from flexlb_eval.scenario.actions.environment import HANDLERS as ENVIRONMENT_HANDLERS
from flexlb_eval.scenario.actions.java_flow import HANDLERS as JAVA_FLOW_HANDLERS
from flexlb_eval.scenario.actions.kv import HANDLERS as KV_HANDLERS
from flexlb_eval.scenario.actions.kv_capacity import HANDLERS as KV_CAPACITY_HANDLERS
from flexlb_eval.scenario.actions.kv_measurement import HANDLERS as KV_MEASUREMENT_HANDLERS
from flexlb_eval.scenario.actions.late_completion import HANDLERS as LATE_COMPLETION_HANDLERS
from flexlb_eval.scenario.actions.master import HANDLERS as MASTER_HANDLERS
from flexlb_eval.scenario.actions.master_observation import HANDLERS as MASTER_OBSERVATION_HANDLERS
from flexlb_eval.scenario.actions.observation import HANDLERS as OBSERVATION_HANDLERS
from flexlb_eval.scenario.actions.priority import HANDLERS as PRIORITY_HANDLERS
from flexlb_eval.scenario.actions.priority_preemption import HANDLERS as PRIORITY_PREEMPTION_HANDLERS
from flexlb_eval.scenario.actions.rpc_measurement import HANDLERS as RPC_MEASUREMENT_HANDLERS
from flexlb_eval.scenario.actions.status_protocol import HANDLERS as STATUS_PROTOCOL_HANDLERS


def handlers():
    result = {}
    for descriptor in [
        *LATE_COMPLETION_HANDLERS,
        *ADMISSION_HANDLERS,
        *BALANCE_HANDLERS,
        *CANCEL_HANDLERS,
        *CLIENT_FETCH_HANDLERS,
        *CACHE_STORM_HANDLERS,
        *CACHE_SCALE_IN_HANDLERS,
        *DECODE_SCALE_OUT_HANDLERS,
        *ELASTIC_HANDLERS,
        *ELASTIC_ADDED_WORKER_HANDLERS,
        *ENGINE_CONTROL_HANDLERS,
        *ENGINE_FAULT_HANDLERS,
        *ENGINE_RECOVERY_HANDLERS,
        *ENVIRONMENT_HANDLERS,
        *JAVA_FLOW_HANDLERS,
        *KV_HANDLERS,
        *KV_CAPACITY_HANDLERS,
        *KV_MEASUREMENT_HANDLERS,
        *MASTER_HANDLERS,
        *MASTER_OBSERVATION_HANDLERS,
        *OBSERVATION_HANDLERS,
        *PRIORITY_HANDLERS,
        *PRIORITY_PREEMPTION_HANDLERS,
        *RPC_MEASUREMENT_HANDLERS,
        *STATUS_PROTOCOL_HANDLERS,
    ]:
        if descriptor.name in result:
            raise ValueError(f"duplicate action {descriptor.name}")
        result[descriptor.name] = descriptor
    return result
