#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/GrpcStreamConnection.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/client/GrpcClientStreamHandler.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceResource.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigClientStream.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamRequest.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, GrpcStreamConnection);

GrpcStreamConnection::GrpcStreamConnection(
    const GrpcClientWorkerPtr &grpcWorker, const std::string &spec,
    const std::shared_ptr<grpc::ChannelCredentials> &channelCredentials,
    const std::shared_ptr<grpc::ChannelArguments> &channelArgs, const ObjectPoolPtr &objectPool)
    : GrpcConnection(grpcWorker, spec, channelCredentials, channelArgs, MC_PROTOCOL_GRPC_STREAM)
    , _objectPool(objectPool) {
    AUTIL_LOG(INFO, "create grpc stream connection spec [%s] allocId [%lu]", _spec.c_str(),
              _allocId);
}

GrpcStreamConnection::~GrpcStreamConnection() {
}

void GrpcStreamConnection::post(const RequestPtr &request, const CallBackPtr &callBack) {
    auto streamRequest = dynamic_pointer_cast<GigStreamRequest>(request);
    assert(streamRequest);
    streamRequest->addHandler(shared_from_this(), callBack);
}

GrpcClientStreamHandlerPtr GrpcStreamConnection::createGrpcHandler(const std::string &bizName,
                                                                   const GigStreamRequest *request,
                                                                   PartIdTy partId) {
    auto clientStream = request->getStream();
    if (!clientStream) {
        AUTIL_LOG(DEBUG, "null client stream");
        return nullptr;
    }
    bool corked = true;
    GrpcClientStreamHandlerPtr streamHandler(
        new GrpcClientStreamHandler(clientStream, partId, bizName, getSpec(), corked));
    streamHandler->setObjectPool(_objectPool);
    // streamHandler->setCallBackThreadPool(_callBackThreadPool);
    grpc::GenericClientAsyncReaderWriter *grpcStream = nullptr;
    CompletionQueueStatusPtr cqs;
    if (!createGrpcStream(clientStream, streamHandler->getHandlerId(), request, corked,
                          streamHandler->getClientContext(), partId, cqs, grpcStream)) {
        return GrpcClientStreamHandlerPtr();
    }
    streamHandler->setGrpcReaderWriter(cqs, grpcStream);
    return streamHandler;
}

bool GrpcStreamConnection::createGrpcStream(const std::shared_ptr<GigStreamBase> &clientStream,
                                            int64_t handlerId, const GigStreamRequest *request,
                                            bool corked, grpc::ClientContext *clientContext,
                                            PartIdTy partId, CompletionQueueStatusPtr &cqs,
                                            grpc::GenericClientAsyncReaderWriter *&stream) {
    auto channel = getChannel();
    if (!channel) {
        AUTIL_LOG(DEBUG, "create grpc stream failed, null channel");
        return false;
    }
    auto timeout = request->getTimeout();
    if (timeout != INFINITE_TIMEOUT) {
        auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout);
        clientContext->set_deadline(deadline);
    }
    if (corked) {
        clientContext->set_initial_metadata_corked(true);
    }
    clientContext->AddMetadata(GIG_GRPC_REQUEST_INFO_KEY, request->getAgentQueryInfo());
    clientContext->AddMetadata(GIG_GRPC_TYPE_KEY, GIG_GRPC_TYPE_STREAM);
    clientContext->AddMetadata(GIG_GRPC_HANDLER_ID_KEY, autil::StringUtil::toString(handlerId));
    const auto &param = clientStream->getParam();
    clientContext->AddMetadata(GIG_GRPC_GLOBAL_START_TIME_US_KEY,
                               autil::StringUtil::toString(param.globalStartTimeUs));
    if (param.collectFlagsFunc) {
        auto flags = param.collectFlagsFunc(partId);
        if (0 != flags) {
            clientContext->AddMetadata(GIG_GRPC_COLLECT_FLAGS_KEY,
                                       autil::StringUtil::toString(flags));
        }
    }
    if (!param.mallocTraceStr.empty()) {
        clientContext->AddMetadata(GIG_GRPC_MALLOC_TRACE_KEY, param.mallocTraceStr);
    }
    grpc::GenericStub stub(channel);
    cqs = _grpcWorker->getCompletionQueue(_allocId);
    assert(cqs);
    assert(cqs->cq);

    stream = stub.PrepareCall(clientContext, request->getMethodName(), cqs->cq.get()).release();
    return cqs != nullptr && stream != nullptr;
}

} // namespace multi_call
