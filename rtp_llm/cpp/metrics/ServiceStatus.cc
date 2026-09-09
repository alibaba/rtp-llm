#include "rtp_llm/cpp/metrics/ServiceStatus.h"

#include "autil/LoopThread.h"
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unistd.h>

namespace rtp_llm {
namespace {

enum class ServiceState {
    STARTING,
    WAITING_FOR_WARMUP,
    SERVING,
    DRAINING,
    STOPPED
};

std::atomic<ServiceState>    g_state{ServiceState::STARTING};
std::atomic<uint32_t>        g_suppression_count{0};
std::atomic<int64_t>         g_reporting_start_ns{0};
std::mutex                   g_state_mutex;
std::string                  g_gate_file;
kmonitor::MetricsReporterPtr g_reporter;
// Destroy the loop before the state and reporter on process exit.
autil::LoopThreadPtr g_status_loop;

bool reportingEnabled(ServiceState state) {
    return (state == ServiceState::SERVING || state == ServiceState::DRAINING)
           && g_suppression_count.load(std::memory_order_acquire) == 0;
}

void reportStatusLocked() {
    auto state = g_state.load(std::memory_order_acquire);
    if (state == ServiceState::WAITING_FOR_WARMUP
        && (g_gate_file.empty() || ::access(g_gate_file.c_str(), F_OK) == 0)) {
        state = ServiceState::SERVING;
        g_reporting_start_ns.store(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count(),
            std::memory_order_release);
        g_state.store(state, std::memory_order_release);
    }
    if (g_reporter && reportingEnabled(state)) {
        const bool            serving = state == ServiceState::SERVING;
        kmonitor::MetricsTags tags(KMON_SERVICE_STATUS_TAG, serving ? "true" : "false");
        g_reporter->report(serving ? 1 : 0, KMON_SERVICE_STATUS_METRIC, kmonitor::GAUGE, &tags);
    }
}

}  // namespace

void startKmonServiceStatus(const kmonitor::MetricsReporterPtr& reporter) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_state.store(ServiceState::STARTING, std::memory_order_release);
    const char* gate_file = std::getenv("RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE");
    g_gate_file           = gate_file ? gate_file : "";
    g_reporter            = reporter;
    if (reporter && !g_status_loop) {
        g_status_loop = autil::LoopThread::createLoopThread(reportKmonServiceStatus, 1000 * 1000, "kmon_status");
    }
}

void stopKmonServiceStatus() {
    // Join outside the mutex: the callback also acquires it.
    g_status_loop.reset();
    std::lock_guard<std::mutex> lock(g_state_mutex);
    g_reporter.reset();
    g_state.store(ServiceState::STOPPED, std::memory_order_release);
}

void reportKmonServiceStatus() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    reportStatusLocked();
}

void setKmonServiceServing(bool serving) {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    const auto                  state = g_state.load(std::memory_order_acquire);
    if (state == ServiceState::DRAINING || state == ServiceState::STOPPED
        || (serving && state != ServiceState::STARTING)) {
        return;
    }
    g_state.store(serving ? ServiceState::WAITING_FOR_WARMUP :
                            (state == ServiceState::SERVING ? ServiceState::DRAINING : ServiceState::STOPPED),
                  std::memory_order_release);
    reportStatusLocked();
}

bool isKmonServiceServing() {
    return g_state.load(std::memory_order_acquire) == ServiceState::SERVING;
}

bool isKmonMetricReportingEnabled() {
    return reportingEnabled(g_state.load(std::memory_order_acquire));
}

std::chrono::steady_clock::time_point kmonMetricReportingStartTime() {
    return std::chrono::steady_clock::time_point(std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::nanoseconds(g_reporting_start_ns.load(std::memory_order_acquire))));
}

kmonitor::MetricsTags kmonTagsWithServiceStatus(const kmonitor::MetricsTags* tags) {
    kmonitor::MetricsTags service_tags;
    if (tags != nullptr) {
        service_tags = *tags;
    }
    service_tags.DelTag(KMON_SERVICE_STATUS_TAG);
    service_tags.AddTag(KMON_SERVICE_STATUS_TAG, isKmonServiceServing() ? "true" : "false");
    return service_tags;
}

ScopedKmonMetricsSuppression::ScopedKmonMetricsSuppression() {
    g_suppression_count.fetch_add(1, std::memory_order_acq_rel);
}

ScopedKmonMetricsSuppression::~ScopedKmonMetricsSuppression() {
    g_suppression_count.fetch_sub(1, std::memory_order_acq_rel);
}

}  // namespace rtp_llm
