#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigClientStream.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigClientStreamImpl.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/stream/GigStreamRequest.h"

using namespace std;

namespace multi_call {

AUTIL_DECLARE_AND_SETUP_LOGGER(multi_call, GigServerStream);

GigClientStream::GigClientStream(const std::string &bizName, const std::string &methodName)
    : RequestGenerator(bizName, INVALID_SOURCE_ID, INVALID_VERSION_ID)
    , _impl(new GigClientStreamImpl(this))
    , _methodName(methodName)
    , _timeout(DEFAULT_TIMEOUT)
    , _hasError(false)
    , _forceStop(false)
    , _allowLack(true)
    , _disableDetection(false) {
}

GigClientStream::~GigClientStream() {
    delete _impl;
}

GigClientStreamImpl *GigClientStream::getImpl() {
    return _impl;
}

void GigClientStream::enablePartId(PartIdTy partId) {
    _partIds.insert(partId);
}

const absl::flat_hash_set<PartIdTy> &GigClientStream::getPartIds() const {
    return _partIds;
}

PartIdTy GigClientStream::getPartCount() const {
    return _impl->getPartCount();
}

GigStreamRpcInfo GigClientStream::getStreamRpcInfo(PartIdTy partId) const {
    return _impl->getStreamRpcInfo(partId);
}

bool GigClientStream::hasError() const {
    return _hasError;
}

void GigClientStream::generate(PartIdTy partCnt, PartRequestMap &requestMap) {
    const auto &partIds = getPartIds();
    if (partIds.empty()) {
        for (int32_t partId = 0; partId < partCnt; partId++) {
            createRequest(partId, requestMap);
        }
    } else {
        for (auto partId : partIds) {
            if (partId >= partCnt) {
                AUTIL_LOG(ERROR, "generate failed, invalid partId: %d, partCount: %d", partId,
                          partCnt);
                _hasError = true;
                requestMap.clear();
                return;
            }
            createRequest(partId, requestMap);
        }
    }
}

void GigClientStream::createRequest(PartIdTy partId, PartRequestMap &requestMap) {
    auto request = new GigStreamRequest();
    request->setStream(this);
    request->setMethodName(_methodName);
    request->setTimeout(_timeout);
    request->disableProbe(getDisableProbe());
    requestMap.emplace(partId, RequestPtr(request));
}

bool GigClientStream::send(PartIdTy partId, bool eof, google::protobuf::Message *message) {
    return _impl->send(partId, eof, message);
}

void GigClientStream::sendCancel(PartIdTy partId, google::protobuf::Message *message) {
    _impl->sendCancel(partId, message);
}

} // namespace multi_call
