#pragma once

#include "rtp_llm/flexlb/navi_sched/resource/ReplayClientR.h"
#include "autil/Log.h"
#include "navi/engine/Navi.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace navi_sched {

class ReplayControlGraph {
public:
    ReplayControlGraph(const navi::NaviPtr &navi,
                       const ReplayClientRPtr &clientResource,
                       const std::string &clientBizName,
                       const std::string &masterBizName);
    ~ReplayControlGraph();

    bool start();
    void stop();
    bool running() const;

private:
    static std::unique_ptr<navi::GraphDef> buildGraph(
        const std::string &clientBizName, const std::string &masterBizName);
    void supervise();

private:
    struct State {
        std::atomic<bool> running{false};
        std::atomic<bool> stopping{false};
        std::mutex mutex;
        std::condition_variable condition;
    };

    navi::NaviPtr _navi;
    ReplayClientRPtr _clientResource;
    std::string _clientBizName;
    std::string _masterBizName;
    navi::NaviUserResultPtr _userResult;
    std::shared_ptr<State> _state;
    std::thread _supervisorThread;

private:
    AUTIL_LOG_DECLARE();
};

} // namespace navi_sched
