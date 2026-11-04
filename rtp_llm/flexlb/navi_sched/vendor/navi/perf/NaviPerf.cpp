#include "navi/perf/NaviPerf.h"
#include "navi/perf/NaviPerfThread.h"

namespace navi {

NaviPerf::NaviPerf()
    : _inited(false)
{
}

NaviPerf::~NaviPerf() {
    stop();
    for (const auto &pair : _pidMap) {
        delete pair.second;
    }
}

void NaviPerf::addPid(pid_t pid) {
    auto thread = new NaviPerfThread(pid);
    _pidMap.emplace(pid, thread);
}

bool NaviPerf::init(const NaviLoggerPtr &logger) {
    _logger.object = this;
    _logger.logger = logger;
    _logger.addPrefix("navi perf");
    if (!initEpollThread()) {
        return false;
    }
    if (!initPerfThread()) {
        return false;
    }
    if (!startEpollThread()) {
        return false;
    }
    _inited = true;
    return true;
}

bool NaviPerf::initEpollThread() {
    constexpr int32_t epoll_thread_count = 5;
    _epollThread.reserve(epoll_thread_count);
    for (auto i = 0; i < epoll_thread_count; ++i) {
        NaviPerfEpollThreadPtr epollThread(new NaviPerfEpollThread());
        if (!epollThread->init(_logger.logger)) {
            return false;
        }
        _epollThread.emplace_back(std::move(epollThread));
    }
    return true;
}

bool NaviPerf::initPerfThread() {
    auto epollThreadCount = _epollThread.size();
    size_t i = 0;
    for (auto &pair : _pidMap) {
        auto perfThread = pair.second;
        if (!perfThread->init(_logger.logger)) {
            return false;
        }
        const auto &epollThread = _epollThread[i++ % epollThreadCount];
        if (!epollThread->addPerfThread(perfThread)) {
            return false;
        }
    }
    return true;
}

bool NaviPerf::startEpollThread() {
    for (uint32_t i = 0; i < _epollThread.size(); ++i) {
        if (!_epollThread[i]->start(i)) {
            NAVI_LOG(ERROR, "start navi perf epoll thread failed");
            return false;
        }
    }
    return true;
}

NaviPerfThread *NaviPerf::getNaviPerfThread(pid_t pid) {
    if (!_inited) {
        return nullptr;
    }
    auto it = _pidMap.find(pid);
    if (_pidMap.end() == it){
        NAVI_KERNEL_LOG(DEBUG,
                        "can't find sample thread [%d], not in init list", pid);
        return nullptr;
    }
    return it->second;
}

void NaviPerf::stop() {
    for (const auto &epollThread : _epollThread) {
        epollThread->stop();
    }
    _epollThread.clear();
}

}

