#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ConnectionPool.h"

namespace multi_call {

class ConnectionPoolFactory
{
public:
    static ConnectionPoolPtr createConnectionPool(ProtocolType type);

private:
    AUTIL_LOG_DECLARE();
};

} // namespace multi_call

