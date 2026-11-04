#pragma once

#include <atomic>
#include <iosfwd>
#include <stdint.h>
#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/epollsocketevent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iocomponent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/socket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/thread.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadmutex.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/transport.h"
#include "autil/Thread.h"

namespace anet {

/**
 * Class IoWorker is response for watch on a set of file descriptor and
 * dispatch the events on them.
 * There are two mode to run an IoWorker instance. a) Run it with a standalone
 * thread. In this case, call IoWorker::start(), then a new thread will be
 * created to do the io staff work. b) Call IoWorker::step() directly. Then
 * the calling thread is response for calling the epoll functions and handle
 * the read/write events.
 */
class IoWorker {
    friend class TcpComponentIdleTimeTest_testClientIdleTime_Test;
    friend class TcpComponentIdleTimeTest_testDefaultIdleTime_Test;
    friend class TcpComponentIdleTimeTest_testIdleTimeSetViaListen_Test;
    friend class TcpComponentIdleTimeTest_testIdleTimeZero_Test;
    friend class TcpComponentIdleTimeTest_testIdleTimeNegative_Test;
    friend class TCPConnectionTest_testMemLeak_Test;
    friend class ConnectionTest_testHTTPHandler_Test;
    friend class ConnectionTest_testHTTPTimeoutBeforeSend_Test;
    friend class ConnectionTest_testHTTPTimeoutAfterSend_Test;
    friend class TransportTest;
    friend class TransportTest_testConstructor_Test;
    friend class IoWorkerTest_testAddComponent_Test;
    friend class IoWorkerTest_testInitEpollWaitTimeoutFromEnv_Test;

public:
    IoWorker();

    ~IoWorker();

    /**
     * @param enableTimeoutThread enable separate timeout thread or not
     * @param interval of checkTimeout()
     * @param promotePriority promote the proproty of ANET thread
     * @return Return true if start success, else return false.
     */
    bool start(bool enableTimeoutThread, int64_t interval, bool promotePriority = false);

    /**
     * Stop ANET. It will stop all threads of ANET.
     * In this function, _stopMutex will be locked.
     *
     * @return Return true if stop success, else return false.
     */
    bool stop();

    /**
     * Waiting for read/write thread exit.
     *
     * @return Return true if wait success, else return false.
     */
    bool wait();

    void run();
    void eventLoop();

    /**
     * Run iteration in single thread. In this function,
     * eventIteration and timeoutIteration will be called.
     *
     * @param timeStamp: execute time
     */
    void step(int64_t timeStamp);

    void postCommand(const Transport::CommandType type, IOComponent *ioc);

    EPollSocketEvent *getSocketEvent();
    void closeComponents();
    int dump(std::ostringstream &buf);
    void getTcpConnStats(std::vector<ConnStat> &connStats);

    void setName(const char *name);

private:
    /**
     * process Command in queue _commands;
     */
    void processCommands();
    void addComponent(IOComponent *ioc);
    void removeComponent(IOComponent *ioc);
    /**
     * @return how much longer until the next check is overdue
     */
    int64_t checkTimeout(int64_t now);

private:
    // mutex for _iocList and _commands
    ThreadMutex _mutex;
    ThreadMutex _stopMutex;
    EPollSocketEvent _socketEvent;
    autil::ThreadPtr _ioThread;
    std::atomic<bool> _stop{false}; // stopping flag
    bool _started{false};
    IOComponent *_iocListHead, *_iocListTail; // IOComponent list
    std::vector<Transport::TransportCommand> _commands;
    int _epollWaitTimeoutMs{100};
    int64_t _nextCheckTime{0};
    bool _enableCheckTimeout{false};
    int64_t _checkTimeoutInterval;

public:
    static __thread int64_t _loopTime;
};

} // namespace anet
