#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/client/GrpcClientStreamHandler.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/controller/ControllerFeedBack.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/client/CompletionQueueStatus.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigAgent.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigStreamHeader.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/CallBack.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceResource.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigClientStream.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamClosure.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamRequest.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamResponse.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/ProtobufByteBufferUtil.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, GrpcClientStreamHandler);

GrpcClientStreamHandler::GrpcClientStreamHandler(const std::shared_ptr<GigStreamBase> &stream,
                                                 PartIdTy partId, const std::string &bizName,
                                                 const std::string &spec, bool corked)
    : GigStreamHandlerBase(partId, stream)
    , _bizName(bizName)
    , _spec(spec)
    , _grpcReaderWriter(nullptr)
    , _request(nullptr)
    , _callInfo(nullptr)
    , _requestType(RT_NORMAL)
    , _needRpcBegin(true)
    , _cleaned(true)
    , _corked(corked)
    , _isRetry(false) {
    HANDLER_LOG(TRACE3,
                "client handler [%p] constructed, stream [%p] type: %s, biz: "
                "%s, spec: %s",
                this, _stream.get(), typeid(*this).name(), _bizName.c_str(), _spec.c_str());
    _streamRpc = true;
    if (stream) {
        _asyncIo = stream->isAsyncMode();
    }
}

GrpcClientStreamHandler::~GrpcClientStreamHandler() {
    if (!_cleaned.load()) {
        HANDLER_LOG(ERROR, "handler not cleaned: %s", typeid(*this).name());
        _streamState.logState(true);
    }
    if (_cancelBuffer.valid) {
        HANDLER_LOG(DEBUG, "cancel message not sent, this type: %s", typeid(*this).name());
        _streamState.logState();
    }
    auto count = closureCount();
    if (count > 0) {
        HANDLER_LOG(ERROR, "closure count: %ld, not zero when finish, %s", count,
                    typeid(*this).name());
        _streamState.logState(true);
    }
    notifyFinish();
    HANDLER_LOG(TRACE3, "begin delete client grpc stream");
    delete _grpcReaderWriter;
    HANDLER_LOG(TRACE3, "end delete client handler [%p]", this);
#ifdef SET_DEBUG
    if (_objectPool) {
        autil::ScopedLock lock(_objectPool->lock);
        _objectPool->set.erase(this);
    }
#endif
}

grpc::ClientContext *GrpcClientStreamHandler::getClientContext() {
    return &_context;
}

void GrpcClientStreamHandler::setGrpcReaderWriter(
    const CompletionQueueStatusPtr &cqs, grpc::GenericClientAsyncReaderWriter *readerWriter) {
    _cqs = cqs;
    _grpcReaderWriter = readerWriter;
}

void GrpcClientStreamHandler::setRequestInfo(GigStreamRequest *request,
                                             const CallBackPtr &callBack) {
    _request = request;
    _callBack = callBack;
    _isRetry = callBack->isRetry();
    const auto &resource = _callBack->getResource();
    _requestType = resource->getRequestType();
    _provider = resource->getProvider(_isRetry);
    _response = resource->getResponse(_isRetry);
}

void GrpcClientStreamHandler::setCallInfo(SingleCallInfo *callInfo) {
    _callInfo = callInfo;
}

GigStreamRpcInfo GrpcClientStreamHandler::snapshotStreamRpcInfo() const {
    GigStreamRpcInfo rpcInfo;
    _streamRpcInfoController.snapshot(rpcInfo);
    rpcInfo.spec = _spec;
    rpcInfo.isRetry = _isRetry;
    return rpcInfo;
}

bool GrpcClientStreamHandler::init() {
    HANDLER_LOG(TRACE3, "begin init, host: %s", _spec.c_str());
    if (!_corked) {
        rpcBegin();
    }
    _cleaned.store(false, std::memory_order_relaxed);
    autil::ScopedReadWriteLock lock(_cqs->enqueueLock, 'r');
    if (!_cqs->stopped) {
        if (_corked) {
            _grpcReaderWriter->StartCall(nullptr);
            dynamic_cast<GigStreamClosureBase *>(getInitClosure())->run(true);
        } else {
            _grpcReaderWriter->StartCall(getInitClosure());
        }
        return true;
    } else {
        _streamState.setErrorCode(MULTI_CALL_ERROR_STREAM_STOPPED);
        _streamState.initFailed();
        runNext(false);
        return false;
    }
}

void GrpcClientStreamHandler::initCallback(bool ok) {
    if (ok) {
        setSendStatus(SS_IDLE);
        _streamState.endInit();
        HANDLER_LOG(TRACE3, "init success");
    } else {
        HANDLER_LOG(TRACE3, "init failed");
        _streamState.setErrorCode(MULTI_CALL_ERROR_STREAM_INIT);
        _streamState.initFailed();
    }
}

GigStreamClosureBase *GrpcClientStreamHandler::doSend(bool sendEof, grpc::ByteBuffer *message,
                                                      GigStreamClosureBase *closure, bool &ok) {
    if (_corked) {
        rpcBegin();
    }
    if (_provider) {
        _provider->incStreamQueryCount();
    }

    autil::ScopedReadWriteLock lock(_cqs->enqueueLock, 'r');
    if (!_cqs->stopped) {
        _grpcReaderWriter->Write(*message, closure);
        return nullptr;
    } else {
        ok = false;
        return closure;
    }
}

GigStreamClosureBase *GrpcClientStreamHandler::receiveNext() {
    if (_streamState.receiveEof()) {
        return nullptr;
    }
    _receiveBuffer.Clear();
    auto closure = getReceiveClosure();
    autil::ScopedReadWriteLock lock(_cqs->enqueueLock, 'r');
    if (!_cqs->stopped) {
        _grpcReaderWriter->Read(&_receiveBuffer, closure);
        return nullptr;
    } else {
        return closure;
    }
}

bool GrpcClientStreamHandler::doStreamReceive(const std::shared_ptr<GigStreamBase> &stream,
                                              const GigStreamMessage &message) {
    if (!_request) {
        return false;
    }
    if (_provider) {
        _provider->incStreamResponseCount();
    }
    return _request->receive(_callInfo, stream, message);
}

void GrpcClientStreamHandler::doStreamReceiveCancel(const std::shared_ptr<GigStreamBase> &stream,
                                                    const GigStreamMessage &message,
                                                    MultiCallErrorCode ec) {
    if (!_request) {
        return;
    }
    _request->receiveCancel(_callInfo, stream, message, ec);
}

bool GrpcClientStreamHandler::ignoreReceiveError() {
    return false;
}

GigStreamClosureBase *GrpcClientStreamHandler::finish() {
    auto cancelled = _streamState.isCancelled();
    auto closure = getFinishClosure();
    autil::ScopedReadWriteLock lock(_cqs->enqueueLock, 'r');
    if (cancelled) {
        HANDLER_LOG(TRACE3, "try client cancel");
        _context.TryCancel();
    }
    HANDLER_LOG(TRACE3, "client finishing");
    if (!_cqs->stopped) {
        _grpcReaderWriter->Finish(&_status, closure);
        HANDLER_LOG(TRACE3, "client finish called");
        return nullptr;
    } else {
        return closure;
    }
}

void GrpcClientStreamHandler::doFinishCallback(bool ok) {
    clean();
}

void GrpcClientStreamHandler::clean() {
    bool expect = false;
    if (!_cleaned.compare_exchange_weak(expect, true)) {
        return;
    }
    const auto &serverMetaMap = _context.GetServerTrailingMetadata();
    auto it = serverMetaMap.find(GIG_GRPC_RESPONSE_INFO_KEY);
    string responseInfo;
    if (serverMetaMap.end() != it) {
        const auto &stringRef = it->second;
        responseInfo = string(stringRef.data(), stringRef.length());
    }
    updateClockDrift();
    if (_callBack) {
        _callBack->run(nullptr, _streamState.getErrorCode(), "", responseInfo);
        if (1 == _callBack.use_count()) {
            _callBack->clearResource();
        } else {
            _callBack.reset();
        }
    }
    HANDLER_LOG(TRACE3, "client finished");
}



void GrpcClientStreamHandler::updateClockDrift() {
    if (!_provider || 0 == _netLatency) {
        return;
    }
    // minus means drift from this clock
    auto clockDrift = -(_timeDiff - (_netLatency / 2));
    _provider->updateClockDrift(clockDrift);
}

void GrpcClientStreamHandler::tryCancel() {
    _context.TryCancel();
}

void GrpcClientStreamHandler::abort() {
    _streamState.setErrorCode(MULTI_CALL_ERROR_STREAM_STOPPED);
    _streamState.initFailed();
    stealStream();
    _callBack.reset();
}

void GrpcClientStreamHandler::rpcBegin() {
    if (!_needRpcBegin) {
        return;
    }
    _needRpcBegin = false;
    if (_response) {
        _response->rpcBegin();
    }
    if (_provider) {
        _provider->updateRequestCounter(_isRetry, _requestType);
    }
}

} // namespace multi_call
