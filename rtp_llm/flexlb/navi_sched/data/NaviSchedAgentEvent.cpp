#include "aios/navi_sched/data/NaviSchedAgentEvent.h"

namespace navi_sched {

NaviSchedAgentEvent::Type NaviSchedAgentEvent::fromName(const std::string &name)
{
    if ("agent_handshake" == name) {
        return AGENT_HANDSHAKE;
    }
    if ("agent_stopping" == name) {
        return AGENT_STOPPING;
    }
    if ("request_received" == name) {
        return REQUEST_RECEIVED;
    }
    if ("cache_updated" == name) {
        return CACHE_UPDATED;
    }
    if ("latency_model_updated" == name) {
        return LATENCY_MODEL_UPDATED;
    }
    if ("execution_load_updated" == name) {
        return EXECUTION_LOAD_UPDATED;
    }
    if ("batch_finished" == name) {
        return BATCH_FINISHED;
    }
    if ("pending_cache_updated" == name) {
        return PENDING_CACHE_UPDATED;
    }
    if ("workload_updated" == name) {
        return WORKLOAD_UPDATED;
    }
    if ("batch_requests_aborted" == name) {
        return BATCH_REQUESTS_ABORTED;
    }
    if ("scheduler_config_updated" == name) {
        return SCHEDULER_CONFIG_UPDATED;
    }
    return UNKNOWN;
}

std::string NaviSchedAgentEvent::getName(Type event)
{
    switch (event) {
        case AGENT_HANDSHAKE:
            return "agent_handshake";
        case AGENT_STOPPING:
            return "agent_stopping";
        case REQUEST_RECEIVED:
            return "request_received";
        case CACHE_UPDATED:
            return "cache_updated";
        case LATENCY_MODEL_UPDATED:
            return "latency_model_updated";
        case EXECUTION_LOAD_UPDATED:
            return "execution_load_updated";
        case BATCH_FINISHED:
            return "batch_finished";
        case PENDING_CACHE_UPDATED:
            return "pending_cache_updated";
        case WORKLOAD_UPDATED:
            return "workload_updated";
        case BATCH_REQUESTS_ABORTED:
            return "batch_requests_aborted";
        case SCHEDULER_CONFIG_UPDATED:
            return "scheduler_config_updated";
        default:
            return "unknown";
    }
}

}
