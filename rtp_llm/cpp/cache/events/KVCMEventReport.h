#pragma once

#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherConfig.h"

namespace rtp_llm::detail {

enum class ControlEventType {
    HOST_DOWN,
    NODE_REGISTER,
    HEARTBEAT
};

std::string buildRegisterInstanceRequest(const KVCacheEventPublisherContext& context, const std::string& trace_id);
std::string buildMutationReport(const KVCacheEventPublisherContext& context,
                                const std::string&                  trace_id,
                                const std::vector<KVCacheEvent>&    events);
std::vector<KVCacheEvent> coalesceMutations(const std::vector<KVCacheEvent>& events);
std::string               buildControlReport(const KVCacheEventPublisherContext& context,
                                             const std::string&                  trace_id,
                                             ControlEventType                    type,
                                             bool                                publish_cache_events = true);
std::string               buildSnapshotReport(const KVCacheEventPublisherContext& context,
                                              const std::string&                  trace_id,
                                              const KVCacheSnapshot&              snapshot);

}  // namespace rtp_llm::detail
