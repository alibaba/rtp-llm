#pragma once

#include <grpc++/alarm.h>
#include <grpc++/generic/async_generic_service.h>
#include <grpc++/grpc++.h>
#include <grpc++/impl/codegen/byte_buffer.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/GigAgent.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc/GigRpcWorker.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/RandomGenerator.h"
#include "autil/LockFreeThreadPool.h"
#include "autil/Thread.h"

namespace multi_call {

typedef std::shared_ptr<grpc::ServerCompletionQueue> ServerCompletionQueuePtr;

struct ServerCompletionQueueStatus {
    ServerCompletionQueueStatus(const ServerCompletionQueuePtr &cq_) : stopped(false), cq(cq_) {
    }
    volatile bool stopped;
    autil::ReadWriteLock enqueueLock;
    ServerCompletionQueuePtr cq;
};

MULTI_CALL_TYPEDEF_PTR(ServerCompletionQueueStatus);

class GigRpcServer;

class GrpcServerWorker : public GigRpcWorker
{
public:
    GrpcServerWorker(GigRpcServer &owner);
    ~GrpcServerWorker();

public:
    bool init(const GrpcServerDescription &desc);

private:
    GrpcServerWorker(const GrpcServerWorker &);
    GrpcServerWorker &operator=(const GrpcServerWorker &);

public:
    void stop();
    int32_t getPort() const;
    const GigAgentPtr &getAgent() const;
    std::vector<int32_t> getIoPids() const;

private:
    // virtual for ut
    virtual grpc::SslServerCredentialsOptions getSslOptions(const SecureConfig &secureConfig) const;
    void workLoop(size_t idx, ServerCompletionQueueStatusPtr cqsPtr);

public:
    grpc::AsyncGenericService &getAsyncGenericService() {
        return _genericService;
    }

private:
    GigRpcServer &_owner;
    GrpcServerDescription _desc;
    RandomGenerator _randomGenerator;
    std::vector<autil::ThreadPtr> _workThreads;
    std::vector<::grpc::Alarm *> _shutdownAlarms;
    std::vector<ServerCompletionQueueStatusPtr> _completionQueues;
    grpc::ServerBuilder _builder;
    std::unique_ptr<grpc::Server> _server;
    grpc::AsyncGenericService _genericService;
    autil::LockFreeThreadPool *_callBackThreadPool;
    int32_t _listenPort;
    int32_t *_threadPids = NULL;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GrpcServerWorker);

} // namespace multi_call

