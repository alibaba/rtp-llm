
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/tools/ConnectionPool.h"

#include <assert.h>
#include <iosfwd>
#include <utility>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/connection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/transport.h"
#include "alog/Logger.h"
#include "autil/TimeUtility.h"

namespace anet {
namespace tools {

using namespace std;
using namespace autil;

AUTIL_LOG_SETUP(anet.tools, ConnectionPool);

ConnectionPool::ConnectionPool(anet::IPacketStreamer *packetStreamer) {
    _packetStreamer = packetStreamer;
    _anetTransport = new anet::Transport();
    _connectionArgs = new ConnectionArgs();
    _opened = false;
}

ConnectionPool::~ConnectionPool() {
    close();
    if (nullptr != _anetTransport) {
        delete _anetTransport;
        _anetTransport = nullptr;
    }
    if (nullptr != _connectionArgs) {
        delete _connectionArgs;
        _connectionArgs = nullptr;
    }
}

bool ConnectionPool::open() {
    if (true == _opened) {
        return true;
    }
    if (0 != _connectionArgs->timeoutCheckIntervalMs) {
        _anetTransport->setTimeoutLoopInterval(_connectionArgs->timeoutCheckIntervalMs);
    }
    bool rnt1 = _anetTransport->start();
    if (false == rnt1) {
        AUTIL_LOG(ERROR, "anet transport start failed!");
        return false; // ERROR_TRANSPORT_START_FAILED;
    }
    AUTIL_LOG(INFO, "anet transport start success!");
    _opened = true;
    return true;
}

void ConnectionPool::close() {
    if (false == _opened) {
        return;
    }
    ScopedLock scoped_lock(_connectionMapMutex);
    for (map<string, list<anet::Connection *>>::iterator itr = _connectionMap.begin(); itr != _connectionMap.end();
         itr++) {
        list<anet::Connection *> &connList = itr->second;
        for (list<anet::Connection *>::iterator connItr = connList.begin(); connItr != connList.end(); connItr++) {
            anet::Connection *conn = (*connItr);
            if (false == conn->isClosed()) {
                conn->close();
            }
            conn->subRef();
        }
    }
    _connectionMap.clear();
    assert(_anetTransport);
    bool success = (_anetTransport->stop() && _anetTransport->wait());
    if (false == success) {
        AUTIL_LOG(ERROR, "stop anet transport failed!");
        return; // ERROR_TRANSPORT_STOP_FAILED;
    }
    _opened = false;
    return;
}

Connection *ConnectionPool::getConnection(const std::string &spec) {
    anet::Connection *resultConn = nullptr;
    ScopedLock scoped_lock(_connectionMapMutex);
    map<string, list<anet::Connection *>>::iterator itr = _connectionMap.find(spec);
    if (_connectionMap.end() == itr) {
        resultConn = openConnection(spec);
        if (nullptr == resultConn) {
            AUTIL_LOG(ERROR, "open connection failed : address [%s]", spec.c_str());
        }
        return resultConn;
    } else {
        list<anet::Connection *> &connList = itr->second;
        while (!connList.empty()) {
            anet::Connection *tempConn = connList.back();
            connList.pop_back();
            if (tempConn->isClosed()) {
                tempConn->subRef();
                AUTIL_LOG(DEBUG, "http anet connection [0x%p] to [%s] closed", tempConn, spec.c_str());
                continue;
            } else {
                resultConn = tempConn;
                break;
            }
        }
        if (nullptr == resultConn) {
            resultConn = openConnection(spec);
            if (nullptr == resultConn) {
                AUTIL_LOG(ERROR, "open connection failed : address [%s]", spec.c_str());
            }
        }
        return resultConn;
    }
}

void ConnectionPool::recycleConnection(const std::string &spec, anet::Connection *connection) {
    if (connection->isClosed()) {
        connection->subRef();
        return;
    }
    ScopedLock scoped_lock(_connectionMapMutex);
    map<string, list<anet::Connection *>>::iterator itr = _connectionMap.find(spec);
    if (_connectionMap.end() == itr) {
        list<anet::Connection *> connList;
        connList.push_back(connection);
        _connectionMap.insert(pair<string, list<anet::Connection *>>(spec, connList));
    } else {
        list<anet::Connection *> &connList = itr->second;
        connList.push_back(connection);
    }
}

Connection *ConnectionPool::openConnection(const string &spec) {
    anet::Connection *connection = _anetTransport->connect(spec.c_str(), _packetStreamer, false);
    if (nullptr != connection) {
        if (0 != _connectionArgs->queueLimit) {
            connection->setQueueLimit(_connectionArgs->queueLimit);
        }
        if (0 != _connectionArgs->queueTimeoutMs) {
            connection->setQueueTimeout(_connectionArgs->queueTimeoutMs);
        }
    } else {
        AUTIL_LOG(WARN, "open connection failed : address [%s]", spec.c_str());
    }
    return connection;
}

} // namespace tools
} // namespace anet
