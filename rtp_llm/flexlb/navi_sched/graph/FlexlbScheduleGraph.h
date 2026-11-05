#pragma once

#include "navi/engine/Navi.h"
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>

namespace grpc {
class ChannelInterface;
class ClientContext;
}

namespace navi_sched {

class FlexlbScheduleGraph {
public:
    FlexlbScheduleGraph(navi::NaviPtr navi, std::string clientBizName);
    std::tuple<int32_t, std::string, std::string> schedule(
        uint64_t requestId, const std::string &address, const std::string &payload,
        int64_t timeoutMs, const std::vector<std::pair<std::string, std::string>> &metadata);
    bool abort(uint64_t requestId);
    void stop();

private:
    navi::NaviPtr _navi;
    std::string _clientBizName;
    std::mutex _mutex;
    bool _stopped = false;
    std::unordered_map<std::string, std::shared_ptr<grpc::ChannelInterface>> _channels;
    std::unordered_map<uint64_t, std::shared_ptr<grpc::ClientContext>> _activeRequests;
};

}
