#pragma once

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/events/KVCacheEvent.h"
#include "rtp_llm/cpp/cache/events/KVCacheEventPublisherConfig.h"

namespace rtp_llm::detail {

struct JsonPayloadLimitExceeded {};
struct SnapshotBuildCancelled {};

enum class ControlEventType {
    HOST_DOWN,
    NODE_REGISTER,
    HEARTBEAT,
};

std::string               buildRegisterInstanceRequest(const KVCacheEventPublisherContext& context,
                                                       const std::string&                  trace_id,
                                                       size_t                              max_bytes);
std::string               buildMutationReport(const KVCacheEventPublisherContext& context,
                                              const std::string&                  trace_id,
                                              const std::vector<KVCacheEvent>&    events,
                                              size_t                              max_bytes);
std::vector<KVCacheEvent> coalesceMutations(const std::vector<KVCacheEvent>& events);
std::string               buildControlReport(const KVCacheEventPublisherContext& context,
                                             const std::string&                  trace_id,
                                             ControlEventType                    type,
                                             size_t                              max_bytes);
std::string               buildSnapshotReport(const KVCacheEventPublisherContext& context,
                                              const std::string&                  trace_id,
                                              const KVCacheSnapshot&              snapshot,
                                              size_t                              max_bytes,
                                              const std::atomic<bool>*            cancelled = nullptr);

}  // namespace rtp_llm::detail
