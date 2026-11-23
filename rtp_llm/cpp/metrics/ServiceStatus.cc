#include "rtp_llm/cpp/metrics/ServiceStatus.h"

#include <atomic>

namespace rtp_llm {
namespace {
std::atomic<bool> draining{false};
}

void markKmonServiceDraining() {
    draining.store(true, std::memory_order_relaxed);
}

kmonitor::MetricsTags kmonTagsWithServiceStatus(const kmonitor::MetricsTags* tags) {
    auto service_tags = tags ? *tags : kmonitor::MetricsTags();
    service_tags.DelTag("is_serving");
    service_tags.AddTag("is_serving", draining.load(std::memory_order_relaxed) ? "false" : "true");
    return service_tags;
}

}  // namespace rtp_llm
