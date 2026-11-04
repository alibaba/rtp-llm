#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/client/GrpcClientWorker.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/Connection.h"
#include "autil/Lock.h"

namespace grpc {
class ChannelCredentials;
class ChannelArguments;
} // namespace grpc

namespace multi_call {

typedef std::shared_ptr<grpc::ChannelInterface> GrpcChannelPtr;

class GrpcConnection : public Connection
{
public:
    GrpcConnection(const GrpcClientWorkerPtr &grpcWorker, const std::string &spec,
                   const std::shared_ptr<grpc::ChannelCredentials> &channelCredentials,
                   const std::shared_ptr<grpc::ChannelArguments> &channelArgs, ProtocolType type);
    ~GrpcConnection();

private:
    GrpcConnection(const GrpcConnection &);
    GrpcConnection &operator=(const GrpcConnection &);

public:
    void post(const RequestPtr &request, const CallBackPtr &callBack) override = 0;
    GrpcChannelPtr getChannel();

protected:
    GrpcClientWorkerPtr _grpcWorker;
    autil::ReadWriteLock _channelLock;
    GrpcChannelPtr _channel;
    std::shared_ptr<grpc::ChannelCredentials> _channelCredentials;
    std::shared_ptr<grpc::ChannelArguments> _channelArgs;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GrpcConnection);

} // namespace multi_call

