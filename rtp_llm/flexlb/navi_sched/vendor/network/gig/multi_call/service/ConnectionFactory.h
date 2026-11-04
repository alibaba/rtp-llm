#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/client/GrpcClientWorker.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/GrpcConnection.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/GrpcStreamConnection.h"

namespace grpc {
class ChannelCredentials;
class ChannelArguments;
} // namespace grpc

namespace multi_call {

struct GrpcChannelInitParams {
    int64_t keepAliveInterval = 0;
    int64_t keepAliveTimeout = 0;
};

class ConnectionFactory
{
public:
    ConnectionFactory(size_t queueSize, uint32_t queueTimeout, int64_t queueConsumeTimeout)
        : _queueSize(queueSize)
        , _queueTimeout(queueTimeout)
        , _queueConsumeTimeout(queueConsumeTimeout) {
    }
    virtual ~ConnectionFactory() {
    }

private:
    ConnectionFactory(const ConnectionFactory &);
    ConnectionFactory &operator=(const ConnectionFactory &);

public:
    virtual ConnectionPtr createConnection(const std::string &spec) = 0;

protected:
    size_t _queueSize;
    uint32_t _queueTimeout;
    int64_t _queueConsumeTimeout;

protected:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ConnectionFactory);

class GrpcConnectionFactory : public ConnectionFactory
{
public:
    GrpcConnectionFactory(const GrpcClientWorkerPtr &worker, const SecureConfig &secureConfig)
        : ConnectionFactory(0, 0, 0)
        , _worker(worker)
        , _secureConfig(secureConfig) {
    }
    ~GrpcConnectionFactory() {
    }
    void initChannelArgs(GrpcChannelInitParams params = {});
    ConnectionPtr createConnection(const std::string &spec) override = 0;
    // virtual for ut
    virtual grpc::SslCredentialsOptions getSslOptions() const;

protected:
    GrpcClientWorkerPtr _worker;
    SecureConfig _secureConfig;
    std::shared_ptr<grpc::ChannelCredentials> _channelCredentials;
    std::shared_ptr<grpc::ChannelArguments> _channelArgs;
};

class GrpcStreamConnectionFactory : public GrpcConnectionFactory
{
public:
    GrpcStreamConnectionFactory(const GrpcClientWorkerPtr &worker, const SecureConfig &secureConfig)
        : GrpcConnectionFactory(worker, secureConfig)
        , _objectPool(std::make_shared<ObjectPool>()) {
    }
    ~GrpcStreamConnectionFactory() {
    }
    ConnectionPtr createConnection(const std::string &spec) override {
        return std::make_shared<GrpcStreamConnection>(_worker, spec, _channelCredentials,
                                                      _channelArgs, _objectPool);
    }

private:
    ObjectPoolPtr _objectPool;
};

} // namespace multi_call
