#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/tcpacceptor.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/controlpacket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ilogger.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iocomponent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/ioworker.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/iserveradapter.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/log.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/socket.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/epollsocketevent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/tcpcomponent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/threadmutex.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/transport.h"

namespace anet {
class IPacketStreamer;

TCPAcceptor::TCPAcceptor(Transport *owner,
                         Socket *socket,
                         IPacketStreamer *streamer,
                         IServerAdapter *serverAdapter,
                         int timeout,
                         int maxIdleTimeInMillseconds,
                         int backlog)
    : IOComponent(owner, socket) {
    _streamer = streamer;
    _serverAdapter = serverAdapter;
    _timeout = timeout;
    _maxIdleTimeInMillseconds = maxIdleTimeInMillseconds;
    _backlog = backlog;
    _lastAcceptedComponent = NULL;
    _type = IOC_TCPACCEPTOR;
}

bool TCPAcceptor::init(bool isServer) {
    _socket->setSoBlocking(false);
    bool rc = ((Socket *)_socket)->listen(_backlog);
    if (rc) {
        initialize();
        setState(ANET_CONNECTED); // may be need new state
        {
            MutexGuard lock(&_socketMutex);
            _isSocketInEpoll = true;
            rc = _socketEvent->addEvent(_socket, true, false, this);
            if (!rc)
                return rc;
        }
        _belongedWorker->postCommand(Transport::TC_ADD_IOC, this);
    }
    return rc;
}

bool TCPAcceptor::handleReadEvent() {
    lock();
    Socket *socket;
    while ((socket = ((Socket *)_socket)->accept()) != NULL) {
        ANET_LOG(DEBUG, "New connection coming. fd=%d", socket->getSocketHandle());
        TCPComponent *component = new TCPComponent(_owner, socket);
        assert(component);
        component->setMaxIdleTime(_maxIdleTimeInMillseconds);
        if (!component->init(true)) {
            delete component; /**@TODO: may coredump?*/
            unlock();
            return true;
        }
        Connection *conn = component->createConnection(_streamer, _serverAdapter);
        conn->setQueueTimeout(_timeout);
        _serverAdapter->handlePacket(conn, &ControlPacket::ReceiveNewConnectionPacket);

        _lastAcceptedComponent = component;
        _owner->addToCheckingList(component);

        // transport's Read Write Thread and Timeout thread will have their
        // own reference of component, so we need to subRef() the initial one
        component->subRef();
    }
    unlock();
    return true;
}

bool TCPAcceptor::handleErrorEvent() {
    close();
    return false;
}

void TCPAcceptor::close() {
    lock();
    if (getState() != ANET_CLOSED) {
        closeSocketNoLock();
        setState(ANET_CLOSED);
    }
    _belongedWorker->postCommand(Transport::TC_REMOVE_IOC, this);
    unlock();
}

/*
 * 超时检查
 *
 * @param    now 当前时间(单位us)
 */
bool TCPAcceptor::checkTimeout(int64_t now) { return true; }
} // namespace anet
