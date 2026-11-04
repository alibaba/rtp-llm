#pragma once

#include <grpc++/generic/generic_stub.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/client/CompletionQueueStatus.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/GrpcConnection.h"

namespace multi_call {

class GigStreamBase;
class GigStreamRequest;
class GrpcClientStreamHandler;

class GrpcStreamConnection : public GrpcConnection,
                             public std::enable_shared_from_this<GrpcStreamConnection>
{
public:
    GrpcStreamConnection(const GrpcClientWorkerPtr &grpcWorker, const std::string &spec,
                         const std::shared_ptr<grpc::ChannelCredentials> &channelCredentials,
                         const std::shared_ptr<grpc::ChannelArguments> &channelArgs,
                         const ObjectPoolPtr &objectPool);
    ~GrpcStreamConnection();

private:
    GrpcStreamConnection(const GrpcStreamConnection &);
    GrpcStreamConnection &operator=(const GrpcStreamConnection &);

public:
    void post(const RequestPtr &request, const CallBackPtr &callBack) override;
    std::shared_ptr<GrpcClientStreamHandler>
    createGrpcHandler(const std::string &bizName, const GigStreamRequest *request, PartIdTy partId);

private:
    bool createGrpcStream(const std::shared_ptr<GigStreamBase> &clientStream, int64_t handlerId,
                          const GigStreamRequest *request, bool corked,
                          grpc::ClientContext *clientContext, PartIdTy partId,
                          CompletionQueueStatusPtr &cqs,
                          grpc::GenericClientAsyncReaderWriter *&stream);

private:
    ObjectPoolPtr _objectPool;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GrpcStreamConnection);

} // namespace multi_call
