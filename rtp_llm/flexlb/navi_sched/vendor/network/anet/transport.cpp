#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/transport.h"

#include <assert.h>
#include <list>
#include <ostream>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connectionpriority.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/debug.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iocomponent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ioworker.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/log.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/socket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/tcpacceptor.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/tcpcomponent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/thread.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadmutex.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/timeutil.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/transportlist.h"
#include "autil/EnvUtil.h"
#include "autil/NetUtil.h"

namespace anet {
class Connection;
class IPacketStreamer;
class IServerAdapter;
struct ConnStat;
} // namespace anet

using namespace std;

namespace anet {

namespace {
const int MAX_IO_THREAD_NUM = 32;
} // anonymous namespace

static int SimpleHashStrategy(int fd, int ioThreadNum) { return (fd == -1) ? 0 : (fd % ioThreadNum); }

static int PrioritySeparateStrategy(int ioType, int priority, int fd, int ioThreadNum) {
    if (ioType == IOC_TCPACCEPTOR || priority == ANET_PRIORITY_HIGH) {
        return ioThreadNum;
    } else
        return SimpleHashStrategy(fd, ioThreadNum);
}

Transport::Transport() : _listenFdThreadMode(SHARE_THREAD), _listenThreadNum(0) {
    _ioThreadNum = 1;

    initialize();
}

Transport::Transport(int ioThreadNum, ListenFdThreadModeEnum mode) : _listenFdThreadMode(mode) {
    if (ioThreadNum < 1) {
        ioThreadNum = 1;
    } else if (ioThreadNum > MAX_IO_THREAD_NUM) {
        ANET_LOG(
            INFO, "io thread num [%d] exceed [%d], set to [%d]", ioThreadNum, MAX_IO_THREAD_NUM, MAX_IO_THREAD_NUM);
        ioThreadNum = MAX_IO_THREAD_NUM;
    }
    _ioThreadNum = ioThreadNum;
    if (_listenFdThreadMode == EXCLUSIVE_LISTEN_THREAD) {
        _listenThreadNum = 1;
    } else {
        _listenThreadNum = 0;
    }

    initialize();
}

void Transport::initialize() {
    _ioWorkers = new IoWorker[_ioThreadNum + _listenThreadNum];
    _stop = false;
    _started = false;
    _nextCheckTime = 0;
    _timeoutLoopInterval = 100000; // default 100ms
    _enableTimeoutThread = autil::EnvUtil::getEnv("ANET_ENABLE_TIMEOUT_THREAD", _enableTimeoutThread);
    /* Register the object into the global list. */
    addTransportToList(this);
}

Transport::~Transport() {
    closeComponents();
    delTransportFromList(this);
    delete[] _ioWorkers;
}

/**
 * start two threads, one for read/write, one for timeout checking
 *
 * @return true - OK, false - Fail
 */
bool Transport::start(bool promotePriority) {
    MutexGuard guard(&_stopMutex);
    if (_started) {
        return false;
    }
    _started = true;
    signal(SIGPIPE, SIG_IGN);
    for (int k = 0; k < _ioThreadNum + _listenThreadNum; ++k) {
        auto ret = _ioWorkers[k].start(_enableTimeoutThread, _timeoutLoopInterval, promotePriority);
        if (!ret) {
            ANET_LOG(ERROR, "io worker[%d] start failed", k);
            return false;
        }
    }
    if (_enableTimeoutThread) {
        _timeoutThread = autil::Thread::createThread(std::bind(&Transport::timeoutLoop, this), "ANetTimeout");
        if (!_timeoutThread) {
            ANET_LOG(ERROR, "create timeout thread failed");
            return false;
        }
    }
    ANET_LOG(INFO,
             "transport start success, name [%s], threadNum [%d], promote priority [%d]",
             _name.c_str(),
             _ioThreadNum + _listenThreadNum,
             promotePriority);
    return true;
}

/**
 * set stop flag for read/write thread and timeout checking thread.
 *
 * @return true - OK, false - Fail
 */
bool Transport::stop() {
    MutexGuard guard(&_stopMutex);
    _stop = true;
    for (int k = 0; k < _ioThreadNum + _listenThreadNum; ++k) {
        _ioWorkers[k].stop();
    }
    return true;
}

/**
 * waiting for all threads exit
 *
 * @return true - OK, false - Fail
 */
bool Transport::wait() {
    for (int k = 0; k < _ioThreadNum + _listenThreadNum; ++k) {
        _ioWorkers[k].wait();
    }

    if (_enableTimeoutThread && _timeoutThread) {
        {
            MutexGuard guard(&_timeoutCond);
            _timeoutCond.signal();
        }
        _timeoutThread->join();
    }
    closeComponents();
    return true;
}

void Transport::eventIteration(int64_t &now) {
    now = TimeUtil::getTime();
    for (int k = 0; k < _ioThreadNum + _listenThreadNum; ++k) {
        _ioWorkers[k].step(now);
    }
}

/**
 * calling timeoutIteration() every 100ms
 */
void Transport::timeoutLoop() {
    while (!_stop) {
        auto waitTime = _timeoutLoopInterval / 1000;
        if (0 == waitTime) {
            waitTime = 100;
        }
        {
            MutexGuard guard(&_timeoutCond);
            _timeoutCond.wait(waitTime);
        }
        int64_t now = TimeUtil::getTime();
        timeoutIteration(now);
    }
}

void Transport::setTimeoutLoopInterval(int64_t ms) {
    // min 5ms, max 10000ms
    if (ms < 5) {
        ANET_LOG(WARN, "timeoutLoopInterval %ldms is too short", ms);
    } else if (ms > 10000) {
        ANET_LOG(WARN, "timeoutLoopInterval %ldms is too long", ms);
    }

    _timeoutLoopInterval = ms * 1000; // ms -> us
}

void Transport::closeComponents() {
    collectNewComponets();
    for (list<IOComponent *>::iterator it = _checkingList.begin(); it != _checkingList.end();) {
        IOComponent *ioc = *it;
        ANET_LOG(DEBUG, "IOC(%p)->subRef(), [%d]", ioc, ioc->getRef());
        ioc->close();
        ioc->subRef();
        it = _checkingList.erase(it);
    }
    _checkingList.clear();

    for (int k = 0; k < _ioThreadNum + _listenThreadNum; ++k) {
        _ioWorkers[k].closeComponents();
    }
}

void Transport::collectNewComponets() {
    // first check if new checking list is empty, avoid pthread mutex lock
    bool expectEmpty = false;
    if (_newCheckingListEmpty.compare_exchange_strong(expectEmpty, true)) {
        MutexGuard guard(&_mutex);
        _checkingList.insert(_checkingList.end(), _newCheckingList.begin(), _newCheckingList.end());
        _newCheckingList.clear();
        return;
    }
}

void Transport::timeoutIteration(int64_t now) {
    if (_nextCheckTime > 0 /* skip first iteration */
        && now > (_nextCheckTime + _timeoutLoopInterval)) {
        _checkTimeoutDelayed++;
        ANET_LOG_INTERVAL(/*logIntervalInSeconds=*/60,
                          WARN,
                          "timeoutIteration delayed! now(%ld), _nextCheckTime(%ld), _timeoutLoopInterval(%ld)",
                          now,
                          _nextCheckTime,
                          _timeoutLoopInterval);
    }
    collectNewComponets();
    list<IOComponent *>::iterator it = _checkingList.begin();
    while (it != _checkingList.end()) {
        IOComponent *ioc = *it;
        if (ioc->checkTimeout(now)) {
            ++it;
        } else {
            ioc->subRef();
            it = _checkingList.erase(it);
            ANET_LOG(DEBUG, "checkTimeout remove ioc: %p from checkingList", ioc);
        }
    }
    if (now > INT64_MAX - _timeoutLoopInterval) {
        _nextCheckTime = INT64_MAX;
    } else {
        _nextCheckTime = now + _timeoutLoopInterval;
    }
}

int64_t Transport::getCheckTimeoutDelayed() { return _checkTimeoutDelayed; }

void Transport::runIteration(int64_t &now) {
    eventIteration(now);
    if (now >= _nextCheckTime) {
        timeoutIteration(now);
    }
}

void Transport::run() {
    int64_t now;
    while (!_stop) {
        runIteration(now);
    }
}

void Transport::stopRun() { _stop = true; }

IOComponent *Transport::listen(const char *spec,
                               IPacketStreamer *streamer,
                               IServerAdapter *serverAdapter,
                               int postPacketTimeout,
                               int maxIdleTime,
                               int backlog) {
    MutexGuard guard(&_stopMutex);
    if (_stop) {
        ANET_LOG(SPAM, "Transport(%p) Stoped!", this);
        return NULL;
    }
    if (NULL == spec || NULL == streamer || NULL == serverAdapter) {
        ANET_LOG(WARN, "Invalid parameters for listen(%p,%p,%p)", spec, streamer, serverAdapter);
        return NULL;
    }

    // Server Socket
    Socket *socket = new Socket();
    DBGASSERT(socket);
    socket->setAddrSpec(spec);
    if (socket->getProtocolType() == (int)SOCK_STREAM) {
        // TCPAcceptor
        TCPAcceptor *acceptor =
            new TCPAcceptor(this, socket, streamer, serverAdapter, postPacketTimeout, maxIdleTime, backlog);
        DBGASSERT(acceptor);
        if (!acceptor->init()) {
            delete acceptor;
            return NULL;
        }
        return acceptor;
    } else {
        ANET_LOG(WARN, "SOCK_DGRAM server does not support yet, spec %s", spec);
        delete socket;
    }

    return NULL;
}

IOComponent *Transport::listen(uint16_t port,
                               IPacketStreamer *streamer,
                               IServerAdapter *serverAdapter,
                               int postPacketTimeout,
                               int maxIdleTime,
                               int backlog) {
    std::string spec("tcp:");
    if (autil::NetUtil::SupportIPv6()) {
        spec += ":::" + std::to_string(port);
    } else {
        spec += "0.0.0.0:" + std::to_string(port);
    }
    return listen(spec.c_str(), streamer, serverAdapter, postPacketTimeout, maxIdleTime, backlog);
}

Connection *Transport::connect(const char *spec, IPacketStreamer *streamer, bool autoReconn, CONNPRIORITY prio) {
    return doConnect(NULL, spec, streamer, autoReconn, prio);
}

IoWorker *Transport::getBelongedWorker(const IOComponent *ioc) {
    int chunkId = getChunkId(ioc);
    return &_ioWorkers[chunkId];
}

void Transport::addToCheckingList(IOComponent *ioc) {
    if (_enableTimeoutThread == false)
        return;
    ioc->addRef();
    {
        MutexGuard guard(&_mutex);
        _newCheckingList.push_back(ioc);
    }

    // should after checking list set to queue
    _newCheckingListEmpty.store(false);
}

int Transport::getChunkId(const IOComponent *ioc) const {
    if (ioc == NULL) {
        return 0;
    }

    if (_listenFdThreadMode == EXCLUSIVE_LISTEN_THREAD) {
        assert(_listenThreadNum > 0);
        IOComponent *pioc = const_cast<IOComponent *>(ioc);
        return PrioritySeparateStrategy(
            ioc->getType(), pioc->getSocket()->getPriority(), pioc->getSocket()->getSocketHandle(), _ioThreadNum);
    } else {
        IOComponent *pioc = const_cast<IOComponent *>(ioc);
        return SimpleHashStrategy(pioc->getSocket()->getSocketHandle(), _ioThreadNum);
    }
}

void Transport::dump(ostringstream &buf) {
    char timestr[32];
    TimeUtil::getTimeStr(_nextCheckTime, timestr);

    int totalIOC = 0;
    buf << "------Transport State------\n";
    buf << "Address: " << this << "\tStarted: " << _started << "\tStopping: " << _stop << endl;
    buf << "\tTimeout interval: " << _timeoutLoopInterval << "us\t" << "Next check: " << _nextCheckTime << "-"
        << timestr << endl;

    for (int k = 0; k < _ioThreadNum + _listenThreadNum; ++k) {
        totalIOC += _ioWorkers[k].dump(buf);
    }
    buf << "\nTotally dumped IOC: " << totalIOC << endl;
}

void Transport::setName(const std::string &name) {
    if (_enableTimeoutThread && _timeoutThread) {
        _timeoutThread->setName(name + "ANetTo");
    }
    for (int k = 0; k < _ioThreadNum + _listenThreadNum; ++k) {
        _ioWorkers[k].setName((name + "ANetIo").c_str());
    }
    _name = name;
}

Connection *Transport::connectWithAddr(
    const char *localAddr, const char *remoteSpec, IPacketStreamer *streamer, bool autoReconn, CONNPRIORITY prio) {
    if (localAddr == NULL) {
        ANET_LOG(WARN, "Invalid localspec for connect");
        return NULL;
    }
    return doConnect(localAddr, remoteSpec, streamer, autoReconn, prio);
}

Connection *Transport::doConnect(
    const char *localAddr, const char *remoteSpec, IPacketStreamer *streamer, bool autoReconn, CONNPRIORITY prio) {
    MutexGuard guard(&_stopMutex);
    if (_stop) {
        return NULL;
    }

    if (NULL == remoteSpec || NULL == streamer) {
        ANET_LOG(WARN, "Invalid parameters for connect(%s,%p,%d)", remoteSpec, streamer, autoReconn);
        return NULL;
    }

    Socket *socket = new Socket();
    assert(socket);

    if (socket->setAddrSpec(remoteSpec) == false) {
        delete socket;
        ANET_LOG(ERROR, "setAddress error: %s", remoteSpec);
        return NULL;
    }

    if (!socket->setReuseAddress(true)) {
        ANET_LOG(WARN, "set reuse address [%s] failed with localAddr [%s]", remoteSpec, localAddr);
    }
    if (socket->setPriority(prio)) {
        ANET_LOG(WARN, "set priority [%d] to addr [%s] failed", prio, remoteSpec);
    }
    if (localAddr && socket->getProtocolFamily() != AF_UNIX) {
        if (!bindLocalAddress(socket, localAddr)) {
            delete socket;
            ANET_LOG(WARN, "Bind localip [%s] to socket failed", localAddr);
            return NULL;
        }
    }

    /* Now we only support steam, instead of dgram type. */
    if (socket->getProtocolType() == (int)SOCK_STREAM) {
        TCPComponent *component = new TCPComponent(this, socket);
        assert(component);
        if (!component->init()) {
            delete component;
            ANET_LOG(ERROR, "Failed to init TCPComponent(%s).", remoteSpec);
            return NULL;
        }
        component->setAutoReconn(autoReconn);
        component->createConnection(streamer, NULL);
        ANET_LOG(INFO,
                 "socket(fd:%d) connected, connection %p, ioc %p, remote %s, local %s",
                 component->getSocket()->getSocketHandle(),
                 component->getConnection(),
                 component,
                 remoteSpec,
                 localAddr);
        addToCheckingList(component);
        return component->getConnection();
    } else {
        ANET_LOG(WARN, "SOCK_DGRAM does not support yet, spec: %s", remoteSpec);
        delete socket;
    }
    return NULL;
}

void Transport::getTcpConnStats(vector<ConnStat> &connStats) {
    for (int k = 0; k < _ioThreadNum + _listenThreadNum; ++k) {
        _ioWorkers[k].getTcpConnStats(connStats);
    }
}

bool Transport::bindLocalAddress(Socket *socket, const char *localAddr) {
    char ipStr[512];
    ipStr[0] = '\0';
    uint16_t port;
    parseLocalAddr(localAddr, ipStr, 512, port);
    return socket->bindLocalAddress(ipStr, port);
}

void Transport::parseLocalAddr(const char *localAddr, char *ipStr, int ipStrBufLen, uint16_t &port) {
    const char *sep = strchr(localAddr, ':');
    size_t ipLen = 0;
    if (sep != NULL) {
        ipLen = sep - localAddr;
    } else {
        ipLen = strlen(localAddr);
    }

    if (ipLen >= ipStrBufLen) {
        if (ipStrBufLen >= 1) {
            ipStr[0] = '\0';
        }
        return;
    }
    strncpy(ipStr, localAddr, ipLen);
    ipStr[ipLen] = '\0';

    port = 0;
    if (sep != NULL && ipLen < (int)strlen(localAddr)) {
        port = (uint16_t)atoi(sep + 1);
    }
}

} // namespace anet
