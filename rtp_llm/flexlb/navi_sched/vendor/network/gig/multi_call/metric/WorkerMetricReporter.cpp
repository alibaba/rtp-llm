#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/WorkerMetricReporter.h"

using namespace std;
using namespace kmonitor;
namespace multi_call {
AUTIL_LOG_SETUP(multi_call, WorkerMetricReporter);

WorkerMetricReporter::WorkerMetricReporter(KMonitor *kMonitor, const MetaEnv &metaEnv)
    : _kMonitor(kMonitor) {
    if (_kMonitor) {
        std::map<std::string, std::string> tagkv;
        _defaultTags = _kMonitor->GetMetricsTags(tagkv); // empty tagkv
    }
    DEFINE_METRIC(kMonitor, QueueSize, "buildin.queueSize", GAUGE, NORMAL, _defaultTags);
    DEFINE_METRIC(kMonitor, SubUpdateSuccQps, "buildin.subUpdateSuccQps", QPS, NORMAL,
                  _defaultTags);
    DEFINE_METRIC(kMonitor, SubUpdateFailedQps, "buildin.subUpdateFailedQps", QPS, NORMAL,
                  _defaultTags);
    DEFINE_METRIC(kMonitor, CreateSnapshotSuccQps, "buildin.createSnapshotSuccQps", QPS, NORMAL,
                  _defaultTags);
    DEFINE_METRIC(kMonitor, CreateSnapshotFailQps, "buildin.createSnapshotFailQps", QPS, NORMAL,
                  _defaultTags);
    DEFINE_METRIC(kMonitor, UpdateSnapshotQps, "buildin.updateSnapshotQps", QPS, NORMAL,
                  _defaultTags);
}

WorkerMetricReporter::~WorkerMetricReporter() {
}

} // namespace multi_call
