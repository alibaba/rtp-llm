#pragma once

#include <unordered_map>

#include "rtp_llm/flexlb/navi_sched/vendor/network/anet/transport.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/Connection.h"
#include "autil/Lock.h"
#include "autil/LoopThread.h"

namespace multi_call {

class ConnectionFactory;

class ConnectionPool
{
public:
    ConnectionPool();
    virtual ~ConnectionPool();

private:
    ConnectionPool(const ConnectionPool &);
    ConnectionPool &operator=(const ConnectionPool &);

public:
    virtual bool init(const ProtocolConfig &config);
    virtual void stop();
    ConnectionPtr getConnection(const std::string &spec);
    void syncConnection(const std::set<std::string> &keepSpecs);

private:
    ConnectionPtr addConnection(const std::string &spec);
    ConnectionPtr doGetConnection(const std::string &spec);
    void removeConnection(const std::string &spec);
    std::vector<std::string> getRemoveList(const std::set<std::string> &keepSpecs);
    // virtual for ut
    virtual anet::Transport *createTransport(uint32_t threadNum);

protected:
    anet::Transport *_transport;
    size_t _threadNum;
    size_t _queueSize;
    uint32_t _queueTimeout;
    int64_t _queueConsumeTimeout;
    std::shared_ptr<ConnectionFactory> _connectionFactory;
    autil::ReadWriteLock _mapLock;
    std::unordered_map<std::string, ConnectionPtr> _connectionMap;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ConnectionPool);

} // namespace multi_call

