#pragma once

#include <grpc++/generic/async_generic_service.h>
#include <grpc++/generic/generic_stub.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/client/CompletionQueueStatus.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamHandlerBase.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamMessage.h"

// #define SET_DEBUG

namespace multi_call {

class GigStreamBase;
class GigStreamRequest;
struct SingleCallInfo;
class CallBack;
class SearchServiceProvider;
class Response;
struct CompletionQueueStatus;

class GrpcClientStreamHandler : public GigStreamHandlerBase
{
public:
    GrpcClientStreamHandler(const std::shared_ptr<GigStreamBase> &stream, PartIdTy partId,
                            const std::string &bizName, const std::string &spec, bool corked);
    ~GrpcClientStreamHandler();

private:
    GrpcClientStreamHandler(const GrpcClientStreamHandler &);
    GrpcClientStreamHandler &operator=(const GrpcClientStreamHandler &);

public:
    grpc::ClientContext *getClientContext();
    void setGrpcReaderWriter(const CompletionQueueStatusPtr &cqs,
                             grpc::GenericClientAsyncReaderWriter *readerWriter);
    void setRequestInfo(GigStreamRequest *request, const std::shared_ptr<CallBack> &callBack);
    void setCallInfo(SingleCallInfo *callInfo);
    GigStreamRpcInfo snapshotStreamRpcInfo() const;

    void setObjectPool(const ObjectPoolPtr &objectPool) {
#ifdef SET_DEBUG
        _objectPool = objectPool;
        if (_objectPool) {
            autil::ScopedLock lock(_objectPool->lock);
            _objectPool->set.insert(this);
        }
#endif
    }

public:
    bool init() override;
    void abort();

private:
    void initCallback(bool ok) override;
    GigStreamClosureBase *doSend(bool sendEof, grpc::ByteBuffer *message,
                                 GigStreamClosureBase *closure, bool &ok) override;
    void tryCancel() override;
    GigStreamClosureBase *receiveNext() override;
    bool doStreamReceive(const std::shared_ptr<GigStreamBase> &stream,
                         const GigStreamMessage &message) override;
    void doStreamReceiveCancel(const std::shared_ptr<GigStreamBase> &stream,
                               const GigStreamMessage &message, MultiCallErrorCode ec) override;
    bool ignoreReceiveError() override;
    GigStreamClosureBase *finish() override;
    void doFinishCallback(bool ok) override;

    void clean() override;
    void updateClockDrift();
    void rpcBegin();

private:
    // receive
    std::string _bizName;
    std::string _spec;
    grpc::ClientContext _context;
    CompletionQueueStatusPtr _cqs;
    grpc::GenericClientAsyncReaderWriter *_grpcReaderWriter;
    GigStreamRequest *_request;
    SingleCallInfo *_callInfo;
    std::shared_ptr<CallBack> _callBack;
    std::shared_ptr<SearchServiceProvider> _provider;
    std::shared_ptr<Response> _response;
    RequestType _requestType;
    bool _needRpcBegin;
    std::atomic<bool> _cleaned;
    grpc::Status _status;
    bool _corked;
    bool _isRetry;
#ifdef SET_DEBUG
    ObjectPoolPtr _objectPool;
#endif
private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(GrpcClientStreamHandler);

} // namespace multi_call

