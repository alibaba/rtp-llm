#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigServerStream.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/grpc/server/GrpcServerStreamHandler.h"

using namespace std;

namespace multi_call {

AUTIL_DECLARE_AND_SETUP_LOGGER(multi_call, GigServerStream);

GigServerStream::GigServerStream() {
}

GigServerStream::~GigServerStream() {
}

void GigServerStream::setHandler(const GigStreamHandlerBasePtr &handler) {
    _handler = handler;
}

std::shared_ptr<QueryInfo> GigServerStream::getQueryInfo() const {
    auto serverHandler = dynamic_cast<GrpcServerStreamHandler *>(_handler.get());
    if (!serverHandler) {
        return nullptr;
    }
    return serverHandler->getQueryInfo();
}

std::string GigServerStream::getPeer() const {
    auto serverHandler = dynamic_cast<GrpcServerStreamHandler *>(_handler.get());
    if (!serverHandler) {
        return std::string();
    }
    return serverHandler->getPeer();
}

int64_t GigServerStream::getBeginTime() const {
    if (_handler) {
        return _handler->getBeginTime();
    } else {
        return autil::TimeUtility::currentTime();
    }
}

int64_t GigServerStream::getBeginTimeNs() const {
    if (_handler) {
        return _handler->getBeginTimeNs();
    } else {
        return autil::TimeUtility::monotonicTimeNs();
    }
}

bool GigServerStream::send(PartIdTy partId, bool eof, google::protobuf::Message *message) {
    if (!_handler) {
        return false;
    }
    auto sendMessage = GigStreamHandlerBase::createSendBufferMessage(eof, false, message);
    if (eof) {
        _handler->endReceive(true);
    }
    return _handler->send(sendMessage);
}

void GigServerStream::sendCancel(PartIdTy partId, google::protobuf::Message *message) {
    if (!_handler) {
        return;
    }
    auto sendMessage = GigStreamHandlerBase::createSendBufferMessage(true, true, message);
    _handler->send(sendMessage);
}

} // namespace multi_call
