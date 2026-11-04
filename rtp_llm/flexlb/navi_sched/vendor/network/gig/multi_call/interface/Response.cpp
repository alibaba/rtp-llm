#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Response.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigAgent.pb.h"

namespace multi_call {

Response::~Response() {
    freeProtoMessage(_agentInfo);
}

void Response::initAgentInfo(const std::string &agentInfoStr) {
    if (agentInfoStr.empty()) {
        return;
    }
    GigResponseInfo *responseInfo = NULL;
    if (_arena) {
        responseInfo = google::protobuf::Arena::CreateMessage<GigResponseInfo>(_arena.get());
    } else {
        responseInfo = new GigResponseInfo();
    }
    std::unique_ptr<GigResponseInfo, void (*)(google::protobuf::Message *)> pbInfo(
        responseInfo, freeProtoMessage);
    auto &info = *pbInfo;
    if (!info.ParseFromString(agentInfoStr)) {
        return;
    }
    freeProtoMessage(_agentInfo);
    _agentInfo = pbInfo.release();
}

bool Response::hasStatInAgentInfo() {
    if (!_agentInfo) {
        return false;
    }
    return _agentInfo->has_error_ratio() || _agentInfo->has_degrade_ratio() ||
           _agentInfo->has_avg_latency() || _agentInfo->has_target_weight() ||
           _agentInfo->has_warm_up_status() || _agentInfo->has_propagation_stats();
}

} // namespace multi_call
