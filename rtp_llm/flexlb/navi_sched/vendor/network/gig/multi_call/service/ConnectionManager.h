#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ConnectionPool.h"
#include "autil/LockFreeThreadPool.h"

namespace multi_call {

class ConnectionManager
{
public:
    ConnectionManager();
    virtual ~ConnectionManager();

private:
    ConnectionManager(const ConnectionManager &);
    ConnectionManager &operator=(const ConnectionManager &);

public:
    bool init(const ConnectionConfig &config, const MiscConfig &miscConf);
    ConnectionPtr getConnection(const std::string &spec, ProtocolType type);
    void syncConnection(const std::set<std::string> &keepSpecs);
    void stop();
    bool supportProtocol(ProtocolType type) const;
    ConnectionPoolPtr getConnectionPool(ProtocolType type);
    autil::LockFreeThreadPool *getCallBackThreadPool() const {
        return _callBackThreadPool;
    }

private:
    // virtual for ut
    virtual ConnectionPoolPtr createConnectionPool(ProtocolType type);

private:
    ConnectionPoolPtr _poolVector[MC_PROTOCOL_UNKNOWN + 1];
    autil::LockFreeThreadPool *_callBackThreadPool;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ConnectionManager);

} // namespace multi_call

