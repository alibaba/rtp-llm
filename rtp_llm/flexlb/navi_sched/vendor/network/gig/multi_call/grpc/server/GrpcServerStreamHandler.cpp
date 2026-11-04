#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/server/GrpcServerStreamHandler.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamClosure.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/ProtobufByteBufferUtil.h"

using namespace std;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, GrpcServerStreamHandler);

GrpcServerStreamHandler::GrpcServerStreamHandler(GrpcServerWorker *worker,
                                                 const ServerCompletionQueueStatusPtr &cqs)
    : GigStreamHandlerBase(0, GigStreamBasePtr())
    , _worker(worker)
    , _cqs(cqs)
    , _grpcStream(&_serverContext)
    , _triggered(false)
    , _finished(false)
    , _beginTime(0)
    , _beginTimeNs(0) {
    assert(_worker);
    assert(_cqs);
    HANDLER_LOG(DEBUG, "server handler [%p] constructed, worker [%p], type: %s", this, _worker,
                typeid(*this).name());
}

GrpcServerStreamHandler::~GrpcServerStreamHandler() {
    if (_cancelBuffer.valid) {
        HANDLER_LOG(DEBUG, "cancel message not sent, this type: %s", typeid(*this).name());
    }
    auto count = closureCount();
    if (count > 0) {
        HANDLER_LOG(ERROR, "closure count: %ld, not zero when finish, this type: %s", count,
                    typeid(*this).name());
        _streamState.logState(true);
    }
    HANDLER_LOG(DEBUG, "begin notifyFinish in grpc server stream destruct");
    notifyFinish();
    HANDLER_LOG(DEBUG, "deconstructed [%p], worker [%p], type: %s", this, _worker,
                typeid(*this).name());
}

void GrpcServerStreamHandler::triggerNew() {
    if (!_triggered && !_cqs->stopped) {
        _triggered = true;
        GrpcServerStreamHandlerPtr handler(new GrpcServerStreamHandler(_worker, _cqs));
        handler->setCallBackThreadPool(_callBackThreadPool);
        handler->init();
    }
}

bool GrpcServerStreamHandler::init() {
    autil::ScopedReadWriteLock lock(_cqs->enqueueLock, 'r');
    if (!_cqs->stopped) {
        _worker->getAsyncGenericService().RequestCall(&_serverContext, &_grpcStream, _cqs->cq.get(),
                                                      _cqs->cq.get(), getInitClosure());
        return true;
    } else {
        _streamState.setErrorCode(MULTI_CALL_ERROR_STREAM_STOPPED);
        _streamState.initFailed();
        runNext(false);
        return false;
    }
}

void GrpcServerStreamHandler::initCallback(bool ok) {
    _beginTime = autil::TimeUtility::currentTime();
    _beginTimeNs = autil::TimeUtility::monotonicTimeNs();
    triggerNew();
    if (ok) {
        setSendStatus(SS_IDLE);
        _streamState.endInit();
        if (!receiveInitQuery()) {
            HANDLER_LOG(ERROR, "init query failed");
            return;
        }
        StreamProcessScope streamProcessScope(false, _partId, _stream.get());
        receiveBeginQuery();
    } else {
        HANDLER_LOG(DEBUG, "handler init failed");
        _streamState.setErrorCode(MULTI_CALL_ERROR_STREAM_INIT);
        _streamState.initFailed();
    }
}

bool GrpcServerStreamHandler::receiveBeginQuery() {

    std::string requestInfo;
    getServerContextMetaValue(GIG_GRPC_REQUEST_INFO_KEY, requestInfo);
    std::string rpcType;
    if (getServerContextMetaValue(GIG_GRPC_TYPE_KEY, rpcType)) {
        _streamRpc = (rpcType == GIG_GRPC_TYPE_STREAM);
    }
    std::string handlerId;
    if (getServerContextMetaValue(GIG_GRPC_HANDLER_ID_KEY, handlerId)) {
        _peerId = autil::StringUtil::fromString<int64_t>(handlerId);
        HANDLER_LOG(DEBUG, "peerId [%p]", (void *)_peerId);
    }
    auto queryInfo = _worker->getAgent()->getQueryInfo(requestInfo);
    queryInfo->degradeLevel(1.0f);
    setQueryInfo(queryInfo);
    return true;
}

bool GrpcServerStreamHandler::receiveInitQuery() {
    const auto &methodName = _serverContext.method();
    auto streamCreator = _worker->getStreamService(methodName);
    if (!streamCreator) {
        HANDLER_LOG(ERROR, "method[%s] not found", methodName.c_str());
        _streamState.setErrorCode(MULTI_CALL_ERROR_NO_METHOD);
        return false;
    }
    StreamRpcParam param;
    {
        uint64_t collectFlags = 0;
        std::string str;
        if (getServerContextMetaValue(GIG_GRPC_COLLECT_FLAGS_KEY, str)) {
            if (autil::StringUtil::fromString<uint64_t>(str, collectFlags)) {
                param.collectFlags = collectFlags;
            }
        }
    }
    {
        int64_t globalStartTimeUs = 0;
        std::string str;
        if (getServerContextMetaValue(GIG_GRPC_GLOBAL_START_TIME_US_KEY, str)) {
            if (autil::StringUtil::fromString<int64_t>(str, globalStartTimeUs)) {
                param.globalStartTimeUs = globalStartTimeUs;
            }
        }
    }
    {
        std::string mallocTraceStr;
        if (getServerContextMetaValue(GIG_GRPC_MALLOC_TRACE_KEY, mallocTraceStr)) {
            param.mallocTraceStr = mallocTraceStr;
        }
    }
    auto stream = streamCreator->create(param);
    if (!stream) {
        HANDLER_LOG(ERROR, "create stream failed, method [%s]", methodName.c_str());
        _streamState.setErrorCode(MULTI_CALL_ERROR_STREAM_CREATOR);
        return false;
    }
    stream->setHandler(shared_from_this());
    _stream = stream;
    return true;
}

bool GrpcServerStreamHandler::getServerContextMetaValue(const std::string &key,
                                                        std::string &value) const {
    const auto &clientMetaMap = _serverContext.client_metadata();
    auto it = clientMetaMap.find(key);
    if (clientMetaMap.end() != it) {
        const auto &stringRef = it->second;
        value = string(stringRef.data(), stringRef.length());
        return true;
    } else {
        return false;
    }
}

bool GrpcServerStreamHandler::hasMetaKey(const std::string &key) const {
    const auto &clientMetaMap = _serverContext.client_metadata();
    return clientMetaMap.end() != clientMetaMap.find(key);
}

GigStreamClosureBase *GrpcServerStreamHandler::doSend(bool sendEof, grpc::ByteBuffer *message,
                                                      GigStreamClosureBase *closure, bool &ok) {

    autil::ScopedReadWriteLock lock(_cqs->enqueueLock, 'r');
    if (!_cqs->stopped) {
        if (_finished) {
            ok = true;
            return closure;
        }
        if (sendEof) {
            clean();
            auto statusCode =
                _streamState.hasError() ? grpc::StatusCode::CANCELLED : grpc::StatusCode::OK;
            grpc::Status status(statusCode, "");
            _grpcStream.WriteAndFinish(*message, grpc::WriteOptions(), status, closure);
            _finished = true;
        } else {
            _grpcStream.Write(*message, closure);
        }
        return nullptr;
    } else {
        ok = false;
        return closure;
    }
}

GigStreamClosureBase *GrpcServerStreamHandler::receiveNext() {
    _receiveBuffer.Clear();
    auto closure = getReceiveClosure();
    autil::ScopedReadWriteLock lock(_cqs->enqueueLock, 'r');
    if (!_cqs->stopped) {
        _grpcStream.Read(&_receiveBuffer, closure);
        return nullptr;
    } else {
        return closure;
    }
}

bool GrpcServerStreamHandler::ignoreReceiveError() {
    return _streamState.sendEof();
}

GigStreamClosureBase *GrpcServerStreamHandler::finish() {
    autil::ScopedReadWriteLock lock(_cqs->enqueueLock, 'r');
    auto closure = getFinishClosure();
    if (_finished) {
        dynamic_cast<GigStreamClosureBase *>(closure)->run(true);
        return nullptr;
    }
    clean();
    auto statusCode = _streamState.hasError() ? grpc::StatusCode::CANCELLED : grpc::StatusCode::OK;
    grpc::Status status(statusCode, "");
    _streamState.logState();
    HANDLER_LOG(DEBUG, "server finishing, closureCount: %ld", closureCount());
    if (!_cqs->stopped) {
        _grpcStream.Finish(status, closure);
        HANDLER_LOG(DEBUG, "server finishing called, closureCount: %ld", closureCount());
        return nullptr;
    } else {
        return closure;
    }
}

void GrpcServerStreamHandler::clean() {
    auto queryInfo = stealQueryInfo();
    if (queryInfo) {
        auto latency = getLatency();
        auto responseInfo = queryInfo->finish(latency, _streamState.getErrorCode(), MAX_WEIGHT);
        _serverContext.AddTrailingMetadata(GIG_GRPC_RESPONSE_INFO_KEY, responseInfo);
        queryInfo.reset();
    }
}

void GrpcServerStreamHandler::setQueryInfo(const QueryInfoPtr &queryInfo) {
    autil::ScopedLock lock(_queryInfoLock);
    _queryInfo = queryInfo;
}

QueryInfoPtr GrpcServerStreamHandler::getQueryInfo() const {
    autil::ScopedLock lock(_queryInfoLock);
    return _queryInfo;
}

std::string GrpcServerStreamHandler::getPeer() const {
    // remove ipv4 prefix
    const auto &uri = _serverContext.peer();
    if (uri.size() > 5) {
        return uri.substr(5);
    } else {
        return uri;
    }
}

QueryInfoPtr GrpcServerStreamHandler::stealQueryInfo() {
    autil::ScopedLock lock(_queryInfoLock);
    auto info = _queryInfo;
    _queryInfo.reset();
    return info;
}

int64_t GrpcServerStreamHandler::getLatency() const {
    return (autil::TimeUtility::currentTime() - _beginTime) / FACTOR_US_TO_MS;
}

void GrpcServerStreamHandler::doFinishCallback(bool ok) {
}

} // namespace multi_call
