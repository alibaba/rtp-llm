#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/rpc/GigClosure.h"

#include <grpc++/impl/codegen/proto_utils.h>

#include "rtp_llm/flexlb/navi_sched/vendor/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/agent/QueryInfo.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/QuerySession.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/ProtobufUtil.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/http_arpc/HTTPRPCController.h"
#include "autil/Log.h"

using namespace std;
using namespace google::protobuf;

namespace multi_call {

AUTIL_DECLARE_AND_SETUP_LOGGER(multi_call, GigClosure);

GigClosure::GigClosure()
    : _compatibleInfo(nullptr)
    , _needFinishQueryInfo(false)
    , _targetWeight(MAX_WEIGHT)
    , _startTime(autil::TimeUtility::currentTime()) {
}

GigClosure::~GigClosure() {
    if (_needFinishQueryInfo && _queryInfo && !_queryInfo->isFinished()) {
        AUTIL_INTERVAL_LOG(600, ERROR,
                           "query info not finished[%s], "
                           "this may lead to serious gig misbehavior",
                           _queryInfo->toString().c_str());
    }

}

const QuerySessionPtr &GigClosure::getQuerySession() const {
    return _querySession;
}

void GigClosure::setQuerySession(const QuerySessionPtr &session) {
    _querySession = session;
}

void GigClosure::setQueryInfo(const QueryInfoPtr &queryInfo) {
    _queryInfo = queryInfo;
}

const QueryInfoPtr &GigClosure::getQueryInfo() const {
    return _queryInfo;
}

void GigClosure::setNeedFinishQueryInfo(bool needFinish) {
    _needFinishQueryInfo = needFinish;
}

int64_t GigClosure::getStartTime() const {
    return _startTime;
}

GigRpcController &GigClosure::getController() {
    return _controller;
}

MultiCallErrorCode GigClosure::getErrorCode() const {
    return _controller.getErrorCode();
}

void GigClosure::setErrorCode(MultiCallErrorCode ec) {
    _controller.setErrorCode(ec);
}

void GigClosure::setTargetWeight(WeightTy targetWeight) {
    _targetWeight = targetWeight;
}

WeightTy GigClosure::getTargetWeight() const {
    return _targetWeight;
}

RequestType GigClosure::getRequestType() const {
    if (_queryInfo) {
        return _queryInfo->requestType();
    } else {
        return RT_NORMAL;
    }
}

float GigClosure::degradeLevel(float level) const {
    if (!_queryInfo) {
        return MIN_PERCENT;
    }
    return _queryInfo->degradeLevel(level);
}

void GigClosure::setCompatibleFieldInfo(const CompatibleFieldInfo *info) {
    _compatibleInfo = info;
}

void GigClosure::fillCompatibleInfo(google::protobuf::Message *message, MultiCallErrorCode ec,
                                    const std::string &responseInfo) const {
    if (!_compatibleInfo) {
        return;
    }
    ProtobufCompatibleUtil::setErrorCodeField(message, _compatibleInfo->ecField, ec);
    ProtobufCompatibleUtil::setGigMetaField(message, _compatibleInfo->responseInfoField,
                                            responseInfo);
}





} // namespace multi_call
