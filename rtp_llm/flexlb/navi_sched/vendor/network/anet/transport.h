#pragma once
#include <atomic>
#include <iosfwd>
#include <list>
#include <queue>
#include <stdint.h>
#include <string>
#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/epollsocketevent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iocomponent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ipacketstreamer.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iserveradapter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/itransport.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/socket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/thread.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadcond.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadmutex.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/timeutil.h"
#include "autil/Thread.h"

namespace anet {

class IoWorker;

/**
 * This class controls behavior of ANET. There are two work modes:
 * Multi-Thread and Single-Thread. In Multi-Thread: socket read/write
 * and timeout checking working in different threads. In
 * Single-Thread, all these jobs working in single threads.
 * When using Multi-Thread, its recommend to start by start(bool)
 * and stop by stop(). When using Single-Thread, its recommend
 * to call runIteration() explicit. (More samples in anet/util/)
 */

enum ListenFdThreadModeEnum {
    SHARE_THREAD, // default: share thread with normal fd
    EXCLUSIVE_LISTEN_THREAD,
};

class Transport : public ITransport {
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
    friend class TransportTest_testRun_Test;
    friend class TransportTest_testParseLocalAddr_Test;
    friend class TransportTest_testBindLocalAddress_Test;

public:
    enum CommandType {
        TC_ADD_IOC = 1,
        TC_REMOVE_IOC
    };

    struct TransportCommand {
        CommandType type;
        IOComponent *ioc;
    };

    Transport();

    explicit Transport(int ioThreadNum, ListenFdThreadModeEnum mode = SHARE_THREAD);

    ~Transport();

    /**
     * Run ANET in multi thread.
     * It will start two threads, one for read/write, one for
     * timeout checking
     *
     * @param promotePriority promote the proproty of ANET thread
     * @return Return true if start success, else return false.
     */
    bool start(bool promotePriority = false);

    /**
     * Stop ANET. It will stop all threads of ANET.
     * In this function, _stopMutex will be locked.
     *
     * @return Return true if stop success, else return false.
     */
    bool stop();

    /**
     * Waiting for read/write thread and timeout checking thread
     * exit. It's not used in single thread mode.
     *
     * @return Return true if wait success, else return false.
     */
    bool wait();

    /**
     * Run iteration in single thread. In this function,
     * eventIteration and timeoutIteration will be called.
     *
     * @param now: execute time
     */
    void runIteration(int64_t &now);

    /**
     * Start anet as single thread. In this function, runTteration()
     * will be called. When start ANET using run(), ANET will run
     * in single thread mode. Both eventIteation and timeoutIteration
     * will work in one thread under this mode.
     */
    /* override */ void run();

    /**
     * Stop anet when using single thread. It's very simple in this
     * function. Just modigy _stop to true.
     */
    void stopRun();

    /**
     * Implement ITranport interfaces.
     * @ret always return 0 in ANET case.
     */
    /* override */ void runOnce() {
        int64_t now = TimeUtil::getTime();
        runIteration(now);
    }

    /**
     * Implement ITranport interfaces. Reset() is not valid for ANET.
     */
    /* override */ void reset() { ; }

    /*
     * Implement ITranport interfaces.
     */
    /* override */ void runInThread() { start(); }

    /**
     * listen to an address
     *
     * @param spec: formats:  [upd|tcp]:ip:port
     * @param streamer: streamer to encode/decode packets
     * @param serverAdapter: callback object to server incoming packets
     * @param postPacketTimeout: postpacket timeout (default 5000 ms)
     * @param maxIdleTime: connection will be closed after maxIdleTime
     *                     (default 15min)
     * @param backlog: listen(2)'s backlog
     * @return an iocomponent
     */
    IOComponent *listen(const char *spec,
                        IPacketStreamer *streamer,
                        IServerAdapter *serverAdapter,
                        int postPacketTimeout = DEFAULT_POSTPACKET_TIMEOUT,
                        int maxIdleTime = MAX_IDLE_TIME_IN_MILLSECONDS,
                        int backlog = LISTEN_BACKLOG);
    /**
     * listen to localhost tcp address
     * if env support ipv6 then use ipv6 :: as address for listen
     * or use ipv4 0.0.0.0 for listen
     */
    IOComponent *listen(uint16_t port,
                        IPacketStreamer *streamer,
                        IServerAdapter *serverAdapter,
                        int postPacketTimeout = DEFAULT_POSTPACKET_TIMEOUT,
                        int maxIdleTime = MAX_IDLE_TIME_IN_MILLSECONDS,
                        int backlog = LISTEN_BACKLOG);

    /**
     * connect to server at address spec
     *
     * @param spec: format [upd|tcp]:ip:port
     * @param streamer: streamer to encode/decode packets
     * @return an iocomponent
     */
    Connection *connect(const char *spec,
                        IPacketStreamer *streamer,
                        bool autoReconn = false,
                        CONNPRIORITY prio = ANET_PRIORITY_NORMAL);

    /**
     * min 10ms, max 10000ms, default 100ms
     */
    void setTimeoutLoopInterval(int64_t ms);

    /**
     * Add ioc to timeout thread's checking list.
     *
     * @param ioc IOComponent to be checked periodically by
     * timeout thread
     */
    void addToCheckingList(IOComponent *ioc);
    void collectNewComponets();
    /**
     * one iteration of processing socket related events
     * @param now current time in microsecond
     */
    void eventIteration(int64_t &now);

    /**
     * one iteration of checking timeout
     * @param now current time in microsecond
     */
    void timeoutIteration(int64_t now);

    int64_t getCheckTimeoutDelayed();

    void closeComponents();

    /**
     * dump the content of this transport for debugging purpose.
     */
    void dump(std::ostringstream &buf);

    Connection *connectWithAddr(const char *localSpec,
                                const char *remoteSpec,
                                IPacketStreamer *streamer,
                                bool autoReconn = false,
                                CONNPRIORITY prio = ANET_PRIORITY_NORMAL);

    /**
     * determine which ioworker will own the @param ioc
     */
    IoWorker *getBelongedWorker(const IOComponent *ioc);
    virtual void getTcpConnStats(std::vector<ConnStat> &connStats);

    void setName(const std::string &name);

    const std::string &getName() const { return _name; }

protected:
    /**
     * Time out checking.
     */
    void timeoutLoop();
    Connection *doConnect(
        const char *localIp, const char *remoteSpec, IPacketStreamer *streamer, bool autoReconn, CONNPRIORITY prio);
    void parseLocalAddr(const char *localAddr, char *ipStr, int ipStrBufLen, uint16_t &port);
    bool bindLocalAddress(Socket *socket, const char *localAddr);

private:
    void initialize();

    /**
     * getChunkId determines which IoWorker instance is response for handling
     * the read/write events on a given IOComponent. Currently, only simply
     * hash the file descriptor of the socket. This function could be improved
     * if more complicated strategy needed.
     */
    inline int getChunkId(const IOComponent *ioc) const;

private:
    ThreadMutex _mutex;
    ThreadMutex _stopMutex;
    int _ioThreadNum;
    IoWorker *_ioWorkers;
    const ListenFdThreadModeEnum _listenFdThreadMode;
    int _listenThreadNum;

    autil::ThreadPtr _timeoutThread;
    int64_t _checkTimeoutDelayed{0};
    ThreadCond _timeoutCond;
    bool _enableTimeoutThread{true};
    std::atomic<bool> _stop{false}; // stopping flag
    bool _started;
    int64_t _nextCheckTime;
    int64_t _timeoutLoopInterval;

    std::list<IOComponent *> _checkingList;       // timeout checking list
    std::vector<IOComponent *> _newCheckingList;  // new timeout checking list
    std::atomic_bool _newCheckingListEmpty{true}; // new timeout checking list is empty or not

    std::string _name;
};

} // namespace anet
