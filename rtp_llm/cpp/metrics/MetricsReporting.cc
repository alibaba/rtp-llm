#include "rtp_llm/cpp/metrics/MetricsReporting.h"

#include "kmonitor/client/KMonitorFactory.h"
#include "kmonitor/client/KMonitorWorker.h"
#include "kmonitor/client/core/MetricsSystem.h"
#include <atomic>
#include <mutex>

namespace rtp_llm {
namespace {
std::atomic<uint64_t> reporting_epoch{0};
std::atomic<int64_t>  reporting_resume_ns{0};
std::mutex            reporting_mutex;
}  // namespace

uint64_t kmonitorReportingEpoch() {
    return reporting_epoch.load(std::memory_order_acquire);
}

KmonitorReportingState kmonitorReportingState() {
    // Resume time is written while the epoch is odd, before publishing the
    // next even epoch. A stable even epoch therefore owns this timestamp.
    for (;;) {
        const auto epoch = kmonitorReportingEpoch();
        const auto ns    = reporting_resume_ns.load(std::memory_order_acquire);
        if (epoch == kmonitorReportingEpoch()) {
            return {epoch, std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns))};
        }
    }
}

bool setKmonitorReportingEnabled(bool enabled) {
    std::lock_guard<std::mutex> lock(reporting_mutex);
    const auto                  epoch = kmonitorReportingEpoch();
    if ((epoch % 2 == 0) == enabled) {
        return true;
    }
    if (kmonitor::KMonitorFactory::IsStarted()) {
        auto* system = kmonitor::KMonitorFactory::GetWorker()->getMetricsSystem();
        if (!system->SetReportingEnabled(enabled)) {
            return false;
        }
    }
    if (enabled) {
        reporting_resume_ns.store(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count(),
            std::memory_order_release);
    }
    reporting_epoch.store(epoch + 1, std::memory_order_release);
    return true;
}

}  // namespace rtp_llm
