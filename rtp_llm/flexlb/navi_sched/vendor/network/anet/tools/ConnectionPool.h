#pragma once

#include <list>
#include <map>
#include <memory>
#include <stdint.h>
#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/anet.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/atomic.h"
#include "autil/Lock.h"
#include "autil/Log.h"

namespace anet {
namespace tools {

struct ConnectionArgs {
    ConnectionArgs() {
        timeoutCheckIntervalMs = 0;
        queueLimit = 0;
        queueTimeoutMs = 0;
    }
    int32_t timeoutCheckIntervalMs;
    int32_t queueLimit;
    int32_t queueTimeoutMs;
};

class ConnectionPool {
public:
    explicit ConnectionPool(anet::IPacketStreamer *packetStreamer);
    ~ConnectionPool();

private:
    ConnectionPool(const ConnectionPool &);
    ConnectionPool &operator=(const ConnectionPool &);

public:
    // open和close方法需要显式的调用，无锁，只能调用一次，在单线程里面操作
    bool open();
    void close();

    anet::Connection *getConnection(const std::string &spec);
    void recycleConnection(const std::string &spec, anet::Connection *connection);

    inline void setTimeoutCheckInterval(int32_t timeoutCheckIntervalMs) {
        _connectionArgs->timeoutCheckIntervalMs = timeoutCheckIntervalMs;
    }
    inline void setQueueLimit(int32_t queueLimit) { _connectionArgs->queueLimit = queueLimit; }
    inline void setQueueTimeout(int32_t queueTimeoutMs) { _connectionArgs->queueTimeoutMs = queueTimeoutMs; }

private:
    anet::Connection *openConnection(const std::string &spec);

private:
    anet::IPacketStreamer *_packetStreamer;
    anet::Transport *_anetTransport;
    std::map<std::string, std::list<anet::Connection *>> _connectionMap;
    autil::ThreadMutex _connectionMapMutex;
    ConnectionArgs *_connectionArgs;
    bool _opened;

private:
    AUTIL_LOG_DECLARE();
};

typedef std::shared_ptr<ConnectionPool> ConnectionPoolPtr;

} // namespace tools
} // namespace anet
