#pragma once

#include <stdint.h>
#include <string>

namespace navi_sched {

class NaviSchedAgentEvent {
public:
    enum Type : int32_t {
        UNKNOWN = 0,
        AGENT_HANDSHAKE = 1,
        AGENT_STOPPING = 2,
        REQUEST_RECEIVED = 3,
        CACHE_UPDATED = 4,
        LATENCY_MODEL_UPDATED = 5,
        EXECUTION_LOAD_UPDATED = 6,
        BATCH_FINISHED = 7,
        PENDING_CACHE_UPDATED = 8,
        WORKLOAD_UPDATED = 9,
        BATCH_REQUESTS_ABORTED = 10,
        SCHEDULER_CONFIG_UPDATED = 11,
        PEER_CACHE_READ_UPDATED = 12,
    };

public:
    static Type fromName(const std::string &name);
    static std::string getName(Type event);
};

}
