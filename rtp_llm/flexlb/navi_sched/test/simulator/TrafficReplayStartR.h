#pragma once

#include "rtp_llm/flexlb/navi_sched/test/simulator/TrafficReplayStartData.h"
#include "autil/Lock.h"
#include "navi/engine/AsyncPipe.h"
#include "navi/engine/Resource.h"
#include <map>

namespace navi_sched::test {

class TrafficReplayStartR : public navi::Resource {
public:
    void def(navi::ResourceDefBuilder &builder) const override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    bool join(const TrafficReplayStartRequestData &request,
              const navi::AsyncPipePtr &responsePipe);

public:
    static const std::string RESOURCE_ID;

private:
    struct ClientState {
        int64_t firstRequestTimeMs = 0;
        navi::AsyncPipePtr responsePipe;
    };

    struct ReplayGroupState {
        std::map<size_t, ClientState> clients;
        size_t clientCount = 0;
        int64_t startDelayMs = 0;
        bool started = false;
    };

private:
    autil::ThreadMutex _lock;
    std::map<std::string, ReplayGroupState> _groups;
};

} // namespace navi_sched::test
