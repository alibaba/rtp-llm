#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/QueryResultStatistic.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigAgent.pb.h"

namespace multi_call {

AUTIL_LOG_SETUP(multi_call, QueryResultStatistic);

QueryResultStatistic::~QueryResultStatistic() {
}

void QueryResultStatistic::setAgentInfo(GigResponseInfo *pbInfo) {
    if (!pbInfo) {
        return;
    }
    auto &info = *pbInfo;
    if (GIG_RESPONSE_CHECKSUM != info.gig_response_checksum()) {
        return;
    }
    if (MULTI_CALL_ERROR_NONE == ec && info.has_ec()) {
        ec = (MultiCallErrorCode)info.ec();
    }
    if (MAX_WEIGHT == targetWeight && info.has_target_weight()) {
        setTargetWeight(info.target_weight());
    }
    agentInfo = pbInfo;
}

bool QueryResultStatistic::operator==(const QueryResultStatistic &rhs) const {
    auto ret = targetWeight == rhs.targetWeight && callBegTime == rhs.callBegTime &&
               rpcBeginTime == rhs.rpcBeginTime && callEndTime == rhs.callEndTime && ec == rhs.ec;
    if (!ret) {
        return ret;
    }
    if (agentInfo && rhs.agentInfo) {
        return agentInfo->SerializeAsString() == rhs.agentInfo->SerializeAsString();
    }
    return agentInfo == rhs.agentInfo;
}

bool QueryResultStatistic::validateMirror() const {
    if (unlikely((bool)heartbeatPropagationStat)) {
        return true;
    }
    if (statIndex < 0 || !agentInfo || !agentInfo->has_propagation_stats()) {
        return false;
    }
    const auto &propagationStats = agentInfo->propagation_stats();
    if (statIndex >= propagationStats.stats_size()) {
        return false;
    }
    return true;
}

const PropagationStatDef &QueryResultStatistic::getMirrorStat() const {
    if (unlikely((bool)heartbeatPropagationStat)) {
        return *heartbeatPropagationStat;
    }
    const auto &propagationStats = agentInfo->propagation_stats();
    assert(statIndex < propagationStats.stats_size());
    return propagationStats.stats(statIndex);
}

} // namespace multi_call
