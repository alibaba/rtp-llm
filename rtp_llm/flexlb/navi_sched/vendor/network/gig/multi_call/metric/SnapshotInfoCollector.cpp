#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/metric/SnapshotInfoCollector.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, SnapshotInfoCollector);

SnapshotInfoCollector::SnapshotInfoCollector() {
}

SnapshotInfoCollector::~SnapshotInfoCollector() {
}

void SnapshotInfoCollector::addSnapshotBizInfo(const string &bizName,
                                               const SnapshotBizInfo &snapshotInfo) {
    _snapshotInfoMap[bizName] = snapshotInfo;
}

const SnapshotBizInfoMap &SnapshotInfoCollector::getSnapshotInfoMap() const {
    return _snapshotInfoMap;
}

} // namespace multi_call
