#pragma once

#include "navi/log/NaviLogger.h"
#include "navi/engine/RunGraphParams.h"
#include "navi/perf/NaviPerfEpollThread.h"
#include "navi/perf/NaviPerfResult.h"
#include <sys/types.h>

namespace navi {

class NaviPerfThread;

class NaviPerf
{
public:
    NaviPerf();
    ~NaviPerf();
private:
    NaviPerf(const NaviPerf &);
    NaviPerf &operator=(const NaviPerf &);
public:
    void addPid(pid_t pid);
    bool init(const NaviLoggerPtr &logger);
    void stop();
public:
    NaviPerfThread *getNaviPerfThread(pid_t pid);
private:
    bool initEpollThread();
    bool initPerfThread();
    bool startEpollThread();    
private:
    DECLARE_LOGGER();
    NaviUnorderedMap<pid_t, NaviPerfThread *> _pidMap;
    std::vector<NaviPerfEpollThreadPtr> _epollThread;
    bool _inited;
};

}

