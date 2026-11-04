#include "navi/distribute/NaviServerStream.h"
#include "navi/distribute/NaviServerStreamCreator.h"
#include "navi/engine/NaviSnapshot.h"
#include "navi/engine/RemoteSubGraphBase.h"

namespace navi {

class NaviServerStreamQueueItem : public TaskQueueScheduleItemBase {
public:
    NaviServerStreamQueueItem(const std::shared_ptr<NaviServerStream> &stream, multi_call::PartIdTy partId)
        : _stream(stream)
        , _partId(partId)
    {
    }

public:
    void process() override {
        if (_stream) {
            auto handle = _stream->beginTaskQueue(_partId, _schedInfo);
            _stream->setTaskQueueHolder(stealTaskQueueHolder());
            if (!_stream->asyncReceive(_partId)) {
                _stream->sendCancel(_partId, nullptr);
            }
            _stream->endQueueHandle(handle);
            _stream.reset();
        }
    }
    void destroy() override {
        if (_stream) {
            _stream->sendCancel(_partId, nullptr);
            _stream.reset();
        }
        delete this;
    }
private:
    std::shared_ptr<NaviServerStream> _stream;
    multi_call::PartIdTy _partId;
};

NaviServerStream::NaviServerStream(const multi_call::StreamRpcParam &param,
                                   NaviServerStreamCreator *creator,
                                   const std::shared_ptr<NaviSnapshot> &snapshot)
    : NaviStreamBase(nullptr)
    , _creator(creator)
    , _snapshot(snapshot)
{
    setParam(param);
    _logger.addPrefix("server");
    initThreadPool();
    bool mallocTrace = !param.mallocTraceStr.empty();
    MallocTraceParam mallocParam;
    if (mallocTrace) {
        mallocParam.init(param.mallocTraceStr);
    }
    initMetric(CollectFlags(param.collectFlags),
               1,
               snapshot ? snapshot->getNaviPerf() : nullptr,
               false,
               mallocTrace ? &mallocParam : nullptr,
               nullptr);
}

NaviServerStream::~NaviServerStream() {
    _creator->decQueryCount();
}

void NaviServerStream::initThreadPool() {
    if (_snapshot) {
        setThreadPool(_snapshot->getDefaultThreadPool());
    }
}

google::protobuf::Message *NaviServerStream::newReceiveMessage(
        google::protobuf::Arena *arena) const
{
    return NaviStreamBase::newReceiveMessageBase(arena);
}

bool NaviServerStream::receive(const multi_call::GigStreamMessage &message) {
    auto naviRequest = dynamic_cast<NaviMessage *>(message.message);
    if (!naviRequest) {
        NAVI_LOG(ERROR, "invalid message type");
        return false;
    }
    if (!naviRequest->domain_graph().sub_graph().empty()) {
        _needGraphDef = false;
        if (!runGraph(naviRequest, message.arena)) {
            return false;
        }
    } else if (_needGraphDef) {
        NAVI_LOG(ERROR, "lack graph def");
        return false;
    }
    NAVI_LOG(SCHEDULE2, "receive msg, handler [%p]", (void *)message.handlerId);
    if (naviRequest->border_datas_size() > 0) {
        return NaviStreamBase::receiveBase(message);
    }
    return true;
}

void NaviServerStream::receiveCancel(const multi_call::GigStreamMessage &message,
                                     multi_call::MultiCallErrorCode ec)
{
    NaviStreamBase::receiveCancelBase(message, ec);
}

bool NaviServerStream::notifyInitReceive(multi_call::PartIdTy partId) {
    multi_call::GigStreamMessage message;
    multi_call::MultiCallErrorCode ec = multi_call::MULTI_CALL_ERROR_NONE;
    if (!peekMessage(partId, message, ec)) {
        return false;
    }
    if (multi_call::MULTI_CALL_ERROR_NONE != ec) {
        return false;
    }
    if (!message.message) {
        return false;
    }
    auto naviRequest = dynamic_cast<NaviMessage *>(message.message);
    if (!naviRequest) {
        NAVI_LOG(ERROR, "invalid message type");
        return false;
    }
    if (!_snapshot) {
        NAVI_LOG(ERROR, "navi snapshot is null");
        return false;
    }
    auto &pbParams = naviRequest->params();
    const auto &taskQueueName = pbParams.task_queue_name();
    SessionId sessionId;
    sessionId.instance = pbParams.id().instance();
    sessionId.queryId = pbParams.id().query_id();
    auto thisPtr = std::dynamic_pointer_cast<NaviServerStream>(shared_from_this());
    auto item = new NaviServerStreamQueueItem(thisPtr, partId);
    {
        NaviLoggerScope _(nullptr);
        _snapshot->runStreamSession(taskQueueName, sessionId, getParam().globalStartTimeUs, item, _logger.logger);
    }
    return true;
}

bool NaviServerStream::notifyReceive(multi_call::PartIdTy partId) {
    if (_initMessage && isAsyncMode()) {
        _initMessage = false;
        return notifyInitReceive(partId);
    } else {
        _initMessage = false;
        return NaviStreamBase::notifyReceiveBase(shared_from_this(), partId);
    }
}

bool NaviServerStream::runGraph(NaviMessage *request, const ArenaPtr &arena) {
    if (initialized()) {
        NAVI_LOG(ERROR, "graph is running");
        return false;
    }
    assert(_snapshot && "snapshot must not be nullptr");
    {
        if (!_logger.logger) {
            const auto &pbParams = request->params();
            SessionId sessionId;
            sessionId.instance = pbParams.id().instance();
            sessionId.queryId = pbParams.id().query_id();
            _logger.logger = _snapshot->getObjectLogger(sessionId).logger;
        }
        auto streamMetric = getStreamMetric();
        NaviLoggerScope _(_logger);
        if (!_snapshot->doRunStreamSession(request,
                                           arena,
                                           request->params(),
                                           shared_from_this(),
                                           streamMetric ? streamMetric->getMemLeakCollector() : nullptr)) {
            NAVI_LOG(WARN, "snapshot run graph failed");
            return false;
        }
    }
    if (!initialized()) {
        NAVI_LOG(ERROR, "run graph failed");
        return false;
    }
    NAVI_LOG(SCHEDULE1, "end run graph");
    return true;
}

void NaviServerStream::setTaskQueueHolder(TaskQueueHolder token) {
    _taskQueueHolder = std::move(token);
}

TaskQueueHolder NaviServerStream::stealTaskQueueHolder() {
    return std::move(_taskQueueHolder);
}

void *NaviServerStream::beginReceive(multi_call::PartIdTy partId) {
    return NaviStreamBase::beginReceive(partId);
}

void NaviServerStream::endReceive(void *handle, bool fake) {
    NaviStreamBase::endHandle(handle, fake);
}

void *NaviServerStream::beginSend(multi_call::PartIdTy partId) {
    return NaviStreamBase::beginSend(partId);
}

void NaviServerStream::endSend(void *handle, bool fake) {
    NaviStreamBase::endHandle(handle, fake);
}

}
