#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/Request.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigAgent.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/MiscUtil.h"
#include "autil/legacy/base64.h"

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, Request);

Request::Request(ProtocolType type, const std::shared_ptr<google::protobuf::Arena> &arena)
    : _arena(arena)
    , _type(type)
    , _timeout(DEFAULT_TIMEOUT)
    , _providerAttributeMap(NULL) {
    if (_arena) {
        _queryInfo.reset(google::protobuf::Arena::CreateMessage<GigQueryInfo>(_arena.get()),
                         freeProtoMessage);
    } else {
        _queryInfo.reset(new GigQueryInfo(), freeProtoMessage);
    }
    setRequestType(RT_NORMAL);
    setUserRequestType(RT_NORMAL);
    setLatencyLimit(INVALID_FILTER_VALUE);
    setLoadBalanceLatencyLimit(INVALID_FILTER_VALUE);
    setProviderWeight(MAX_WEIGHT);

    _queryInfo->set_gig_query_checksum(GIG_QUERY_CHECKSUM);
    setErrorRatioLimit(INVALID_FILTER_VALUE);
    setBeginServerDegradeLatency(INVALID_FILTER_VALUE);
    setBeginDegradeLatency(INVALID_FILTER_VALUE);
    setBeginServerDegradeErrorRatio(MAX_PERCENT);
    setBeginDegradeErrorRatio(MAX_PERCENT);
}

void Request::setBizName(const std::string &bizName) {
    _queryInfo->set_biz_name(bizName);
}

const std::string &Request::getBizName() const {
    return _queryInfo->biz_name();
}

void Request::setPartId(PartIdTy partId) {
    _queryInfo->set_part_id(partId);
}

PartIdTy Request::getPartId() const {
    return _queryInfo->part_id();
}

void Request::setRequestType(RequestType type) {
    _queryInfo->set_request_type(type);
}

RequestType Request::getRequestType() const {
    return RequestType(_queryInfo->request_type());
}

void Request::setUserRequestType(RequestType type) {
    _queryInfo->set_user_request_type(type);
}

RequestType Request::getUserRequestType() const {
    return RequestType(_queryInfo->user_request_type());
}

void Request::setErrorRatioLimit(float limit) {
    _queryInfo->set_error_ratio_limit(limit);
}

float Request::getErrorRatioLimit() const {
    return _queryInfo->error_ratio_limit();
}

void Request::setLatencyLimit(float limit) {
    _queryInfo->set_latency_limit_ms(limit);
}

float Request::getLatencyLimit() const {
    return _queryInfo->latency_limit_ms();
}

void Request::setLoadBalanceLatencyLimit(float limit) {
    _queryInfo->set_load_balance_latency_limit_ms(limit);
}

float Request::getLoadBalanceLatencyLimit() const {
    return _queryInfo->load_balance_latency_limit_ms();
}

void Request::setBeginServerDegradeLatency(float latency) {
    _queryInfo->set_begin_server_degrade_latency(latency);
}

float Request::getBeginServerDegradeLatency() const {
    return _queryInfo->begin_server_degrade_latency();
}

void Request::setBeginDegradeLatency(float latency) {
    _queryInfo->set_begin_degrade_latency(latency);
}

float Request::getBeginDegradeLatency() const {
    return _queryInfo->begin_degrade_latency();
}

void Request::setProviderWeight(WeightTy weight) {
    _queryInfo->set_gig_weight(weight);
}

int32_t Request::getProviderWeight() const {
    return _queryInfo->gig_weight();
}

void Request::setProviderAttributes(const NodeAttributeMap *attrs) {
    _providerAttributeMap = attrs;
}

std::string Request::getProviderAttribute(const std::string &key) const {
    if (!_providerAttributeMap) {
        return std::string();
    }
    auto iter = _providerAttributeMap->find(key);
    if (_providerAttributeMap->end() != iter) {
        return iter->second;
    } else {
        return std::string();
    }
}

void Request::setBeginServerDegradeErrorRatio(float ratio) {
    _queryInfo->set_begin_server_degrade_error_ratio(ratio);
}

float Request::getBeginServerDegradeErrorRatio() const {
    return _queryInfo->begin_server_degrade_error_ratio();
}

void Request::setBeginDegradeErrorRatio(float ratio) {
    _queryInfo->set_begin_degrade_error_ratio(ratio);
}

float Request::getBeginDegradeErrorRatio() const {
    return _queryInfo->begin_degrade_error_ratio();
}

void Request::setReturnMetaEnv(bool b) {
    _queryInfo->set_return_meta_env(b);
}

bool Request::getReturnMetaEnv() const {
    return _queryInfo->return_meta_env();
}

PropagationStats *Request::getPropagationStats() {
    return _queryInfo->mutable_propagation_stats();
}





std::string Request::getAgentQueryInfo() const {
    return _queryInfo->SerializeAsString();
}

void Request::embedGigData(std::string &body, GigMetaEmbedType metaType) const {
    if (metaType == META_BT_NONE || body.find(GIG_DATA_C) != std::string::npos) {
        return;
    }
    std::string gigData = autil::legacy::Base64EncodeFast(getAgentQueryInfo());
    switch (metaType) {
    case META_BT_IGRAPH: {
        std::size_t pos = body.find("?config%7B");
        if (pos != std::string::npos) {
            std::string meta;
            if (pos > 0) {
                meta.append("&");
            }
            meta.append(GIG_DATA_C).append("=").append(gigData);
            body.insert(pos, meta);
            break;
        }
        // fall through
    }
    case META_BT_APPEND:
        if (body.length() > 0) {
            body.append("&");
        }
        body.append(GIG_DATA_C).append("=").append(gigData);
        break;
    default:
        break;
    }
}

} // namespace multi_call
