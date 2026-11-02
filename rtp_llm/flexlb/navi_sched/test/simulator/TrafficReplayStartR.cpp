#include "aios/navi_sched/test/simulator/TrafficReplayStartR.h"

#include "autil/TimeUtility.h"
#include <algorithm>

namespace navi_sched::test {

const std::string TrafficReplayStartR::RESOURCE_ID = "navi_sched.traffic_replay_start_r";

void TrafficReplayStartR::def(navi::ResourceDefBuilder &builder) const
{
    builder.name(RESOURCE_ID, navi::RS_BIZ);
}

navi::ErrorCode TrafficReplayStartR::init(navi::ResourceInitContext &)
{
    return navi::EC_NONE;
}

bool TrafficReplayStartR::join(const TrafficReplayStartRequestData &request,
                               const navi::AsyncPipePtr &responsePipe)
{
    if (!responsePipe || request.clientCount == 0 ||
        request.clientIndex >= request.clientCount || request.firstRequestTimeMs < 0 ||
        request.startDelayMs <= 0) {
        return false;
    }
    std::vector<navi::AsyncPipePtr> responsePipes;
    auto response = std::make_shared<TrafficReplayStartResponseData>();
    {
        autil::ScopedLock lock(_lock);
        auto &group = _groups[request.replayGroup];
        if (group.started ||
            (!group.clients.empty() &&
             (group.clientCount != request.clientCount ||
              group.startDelayMs != request.startDelayMs))) {
            return false;
        }
        group.clientCount = request.clientCount;
        group.startDelayMs = request.startDelayMs;
        group.clients[request.clientIndex] =
            ClientState{request.firstRequestTimeMs, responsePipe};
        NAVI_KERNEL_LOG(INFO,
                        "traffic replay start joined, group [%s], clientIndex [%lu], "
                        "clientCount [%lu], joined [%lu]",
                        request.replayGroup.c_str(),
                        request.clientIndex, request.clientCount,
                        group.clients.size());
        if (group.clients.size() != group.clientCount) {
            return true;
        }
        group.started = true;
        response->firstRequestTimeMs =
            group.clients.begin()->second.firstRequestTimeMs;
        for (const auto &pair : group.clients) {
            response->firstRequestTimeMs =
                std::min(response->firstRequestTimeMs, pair.second.firstRequestTimeMs);
            responsePipes.emplace_back(pair.second.responsePipe);
        }
        response->startTimeMs =
            autil::TimeUtility::currentTime() / 1000 + group.startDelayMs;
    }
    NAVI_KERNEL_LOG(INFO,
                    "traffic replay start released, group [%s], clientCount [%lu]",
                    request.replayGroup.c_str(), responsePipes.size());
    for (const auto &pipe : responsePipes) {
        if (navi::EC_NONE != pipe->setData(response)) {
            return false;
        }
    }
    return true;
}

REGISTER_RESOURCE(TrafficReplayStartR);

}
