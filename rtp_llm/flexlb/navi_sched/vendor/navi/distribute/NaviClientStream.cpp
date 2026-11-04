#include "navi/distribute/NaviClientStream.h"
#include "navi/engine/RemoteSubGraphBase.h"

namespace navi {

NaviClientStream::NaviClientStream(const multi_call::StreamRpcParam &param,
                                   const std::string &bizName,
                                   const NaviLoggerPtr &logger)
    : GigClientStream(bizName, NAVI_RPC_METHOD_NAME)
    , NaviStreamBase(logger)
{
    setParam(param);
    _logger.addPrefix("client: [%s]", bizName.c_str());
    NAVI_LOG(SCHEDULE1, "constructed");
}

NaviClientStream::~NaviClientStream() {
    NAVI_LOG(SCHEDULE1, "destructed");
}

google::protobuf::Message *NaviClientStream::newReceiveMessage(
        google::protobuf::Arena *arena) const
{
    return NaviStreamBase::newReceiveMessageBase(arena);
}

bool NaviClientStream::receive(const multi_call::GigStreamMessage &message) {
    return NaviStreamBase::receiveBase(message);
}

void NaviClientStream::receiveCancel(const multi_call::GigStreamMessage &message,
                                     multi_call::MultiCallErrorCode ec)
{
    NaviStreamBase::receiveCancelBase(message, ec);
}

bool NaviClientStream::notifyReceive(multi_call::PartIdTy partId) {
    return NaviStreamBase::notifyReceiveBase(shared_from_this(), partId);
}

void NaviClientStream::initMetric(NaviPerf *naviPerf,
                                  CollectFlags flags,
                                  const MallocTraceParam *mallocParam,
                                  const std::shared_ptr<MemLeakCollector> &memLeakCollector) {
    NaviStreamBase::initMetric(flags, getPartCount(), naviPerf, true, mallocParam, memLeakCollector);
}

void *NaviClientStream::beginReceive(multi_call::PartIdTy partId) {
    return NaviStreamBase::beginReceive(partId);
}

void NaviClientStream::endReceive(void *handle, bool fake) {
    NaviStreamBase::endHandle(handle, fake);
}

void *NaviClientStream::beginSend(multi_call::PartIdTy partId) {
    return NaviStreamBase::beginSend(partId);
}

void NaviClientStream::endSend(void *handle, bool fake) {
    NaviStreamBase::endHandle(handle, fake);
}

}
