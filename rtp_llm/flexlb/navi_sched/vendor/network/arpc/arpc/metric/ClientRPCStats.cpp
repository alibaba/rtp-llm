#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/metric/ClientRPCStats.h"

namespace arpc {

AUTIL_LOG_SETUP(arpc, ClientRPCStats);

ClientRPCStats::~ClientRPCStats() {
    auto metricReporter = _metricReporter.lock();
    if (metricReporter != nullptr) {
        metricReporter->reportRPCStats(this);
    }
}

void ClientRPCStats::markRequestDone(arpc::ErrorCode error) {
    _error = error;
    _doneTimeUs = autil::TimeUtility::currentTimeInMicroSeconds();
}

} // namespace arpc
