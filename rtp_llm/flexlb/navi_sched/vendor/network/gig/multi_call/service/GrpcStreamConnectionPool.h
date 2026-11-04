#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/client/GrpcClientWorker.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ConnectionPool.h"

namespace multi_call {

class GrpcStreamConnectionPool : public ConnectionPool
{
public:
    GrpcStreamConnectionPool();
    ~GrpcStreamConnectionPool();

private:
    GrpcStreamConnectionPool(const GrpcStreamConnectionPool &);
    GrpcStreamConnectionPool &operator=(const GrpcStreamConnectionPool &);

public:
    bool init(const ProtocolConfig &config) override;
    void stop() override;

private:
    GrpcClientWorkerPtr _worker;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GrpcStreamConnectionPool);

} // namespace multi_call

