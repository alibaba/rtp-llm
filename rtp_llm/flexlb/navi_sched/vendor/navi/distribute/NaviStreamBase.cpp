#include "navi/distribute/NaviStreamBase.h"

namespace navi {

NaviStreamBase::NaviStreamBase(const NaviLoggerPtr &logger)
    : _logger(this, "stream", logger)
    , _initialized(false)
{
    NAVI_LOG(SCHEDULE1, "stream base constructed");
}

NaviStreamBase::~NaviStreamBase() {
}

void NaviStreamBase::initMetric(CollectFlags collectFlags,
                                NaviPartId partCount,
                                NaviPerf *naviPerf,
                                bool isClient,
                                const MallocTraceParam *mallocParam,
                                std::shared_ptr<MemLeakCollector> memLeakCollector) {
    if (!collectFlags.needCollect()) {
        return;
    }
    if (collectFlags.collectMalloc() && !memLeakCollector) {
        memLeakCollector = std::make_shared<MemLeakCollector>();
    }
    _streamMetric = std::make_shared<StreamMetric>(partCount,
                                                   naviPerf,
                                                   std::move(memLeakCollector),
                                                   isClient,
                                                   collectFlags,
                                                   mallocParam);
}

google::protobuf::Message *NaviStreamBase::newReceiveMessageBase(
        google::protobuf::Arena *arena) const
{
    return google::protobuf::Arena::CreateMessage<NaviMessage>(arena);
}

bool NaviStreamBase::receiveBase(const multi_call::GigStreamMessage &message) {
    NaviLoggerScope _(_logger);
    DomainHolderScope<GraphDomainRemoteBase> scope(_domainHolder);
    auto domain = scope.getDomain();
    if (!domain) {
        std::string msg;
        if (message.message) {
            msg = message.message->DebugString();
        }
        NAVI_LOG(SCHEDULE1,
                 "receive failed, null domain, partId: %d, handler: [%p], msg: %s",
                 message.partId,
                 (void *)message.handlerId,
                 msg.c_str());
        return false;
    }
    if (!domain->receive(message)) {
        NAVI_LOG(ERROR, "receive failed, partId: %d, handler: [%p]", message.partId, (void *)message.handlerId);
        return false;
    }
    return true;
}

void NaviStreamBase::receiveCancelBase(const multi_call::GigStreamMessage &message,
                                       multi_call::MultiCallErrorCode ec)
{
    NaviLoggerScope _(_logger);
    DomainHolderScope<GraphDomainRemoteBase> scope(_domainHolder);
    auto domain = scope.getDomain();
    if (!domain) {
        NAVI_LOG(SCHEDULE1,
                 "receive cancel failed, null domain, partId: %d, handler: [%p]",
                 message.partId,
                 (void *)message.handlerId);
        return;
    }
    domain->receiveCancel(message, ec);
}

bool NaviStreamBase::notifyReceiveBase(const multi_call::GigStreamBasePtr &stream, multi_call::PartIdTy partId) {
    if (!_threadPool) {
        NAVI_LOG(DEBUG, "no receive thread pool, partId [%d], use sync receive", partId);
        return stream->asyncReceive(partId);
    }
    auto item = new NaviStreamReceiveItem(stream, this, partId);
    _threadPool->push(stream->getParam().globalStartTimeUs, 0, item, true);
    return true;
}

bool NaviStreamBase::setDomain(const GraphDomainRemoteBasePtr &domain) {
    NAVI_LOG(SCHEDULE1, "set domain [%p]", domain.get());
    auto ret = _domainHolder.init(domain);
    _initialized = true;
    _domainGdb = domain.get();
    return ret;
}

void NaviStreamBase::setThreadPool(const std::shared_ptr<NaviThreadPool> &threadPool) {
    _threadPool = threadPool;
}

bool NaviStreamBase::initialized() {
    return _initialized;
}

void NaviStreamBase::finish() {
    NAVI_LOG(SCHEDULE1, "finish");
    _domainHolder.terminate();
}

void *NaviStreamBase::beginTaskQueue(multi_call::PartIdTy partId, const ScheduleInfo &schedInfo) {
    auto streamMetric = getStreamMetric();
    if (streamMetric) {
        return streamMetric->beginTaskQueue(partId, schedInfo);
    } else {
        return nullptr;
    }
}

void *NaviStreamBase::beginReceiveQueue(multi_call::PartIdTy partId, const ScheduleInfo &schedInfo) {
    auto streamMetric = getStreamMetric();
    if (streamMetric) {
        return streamMetric->beginReceiveQueue(partId, schedInfo);
    } else {
        return nullptr;
    }
}

void *NaviStreamBase::beginReceive(multi_call::PartIdTy partId) {
    auto streamMetric = getStreamMetric();
    if (streamMetric) {
        return streamMetric->beginReceive(partId);
    } else {
        return nullptr;
    }
}

void *NaviStreamBase::beginSend(multi_call::PartIdTy partId) {
    auto streamMetric = getStreamMetric();
    if (streamMetric) {
        return streamMetric->beginSend(partId);
    } else {
        return nullptr;
    }
}

void NaviStreamBase::endQueueHandle(void *handle) {
    auto streamMetric = getStreamMetric();
    if (streamMetric) {
        streamMetric->endQueueHandle(handle);
    }
}

void NaviStreamBase::endHandle(void *handle, bool fake) {
    auto streamMetric = getStreamMetric();
    if (streamMetric) {
        streamMetric->endHandle(handle, fake);
    }
}

}
