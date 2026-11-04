#pragma once

#include "navi/log/NaviLogger.h"
#include <autil/Thread.h>

namespace navi {

class NaviPerfThread;

class NaviPerfEpollThread
{
public:
    NaviPerfEpollThread();
    ~NaviPerfEpollThread();
private:
    NaviPerfEpollThread(const NaviPerfEpollThread &);
    NaviPerfEpollThread &operator=(const NaviPerfEpollThread &);
public:
    bool init(const NaviLoggerPtr &logger);
    bool addPerfThread(NaviPerfThread *perfThread);
    bool start(uint32_t id);
    void stop();
private:
    bool createPipes();
    bool createThread(uint32_t id);
    void workLoop();
private:
    DECLARE_LOGGER();
    int _epollFd;
    int _pipes[2];
    volatile bool _stop;
    autil::ThreadPtr _thread;
};

NAVI_TYPEDEF_PTR(NaviPerfEpollThread);

}
