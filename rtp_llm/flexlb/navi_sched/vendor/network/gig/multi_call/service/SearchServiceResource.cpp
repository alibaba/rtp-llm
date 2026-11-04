#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceResource.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/proto/GigAgent.pb.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceReplica.h"
#include "autil/StringConvertor.h"
#include "autil/StringUtil.h"

using namespace std;
using namespace autil;
namespace multi_call {

AUTIL_LOG_SETUP(multi_call, SearchServiceResource);

SearchServiceResource::SearchServiceResource(const std::string &bizName, const string &requestId,
                                             SourceIdTy sourceId, const RequestPtr &request,
                                             const SearchServiceReplicaPtr &replica,
                                             const SearchServiceProviderPtr &provider,
                                             RequestType requestType)
    : _bizName(bizName)
    , _requestId(requestId)
    , _sourceId(sourceId)
    , _request(request)
    , _replica(replica)
    , _provider(provider)
    , _requestType(requestType)
    , _partIdIndex(INVALID_PART_ID)
    , _callBeginTime(0)
    , _disableRetry(false)
    , _hasError(false) {
    assert(_request);
    assert(_provider);
}

SearchServiceResource::~SearchServiceResource() {
}

void SearchServiceResource::updateHealthStatus(bool isRetry) {
    const auto &providerPtr = getProvider(isRetry);
    assert(providerPtr);
    auto &provider = *providerPtr;
    const auto &response = getResponse(isRetry);
    assert(response);
    assert(_flowControlConfig);
    ControllerFeedBack feedBack(response->getStat());
    initFeedBack(*_flowControlConfig, _replica, feedBack);

    provider.updateWeight(feedBack);

    feedBack.mirrorResponse = true;
    updateMirrorResponse(feedBack);

    _replica->getReplicaController()->update(provider.getControllerChain(), feedBack.metricLimits);
}

void SearchServiceResource::initFeedBack(const FlowControlConfig &config,
                                         const SearchServiceReplicaPtr &replica,
                                         ControllerFeedBack &feedBack) {
    feedBack.minWeight = config.minWeight;
    auto &metricLimits = feedBack.metricLimits;
    metricLimits.errorRatioLimit = config.errorRatioLimit;
    metricLimits.latencyUpperLimitMs = config.latencyUpperLimitMs;
    metricLimits.latencyUpperLimitPercent = 1.0f + config.latencyUpperLimitPercent;
    metricLimits.fullDegradeLatency = config.fullDegradeLatency;
    replica->getReplicaController()->fillReplicaControllerInfo(feedBack);
}

void SearchServiceResource::updateMirrorResponse(ControllerFeedBack &feedBack) {
    if (!feedBack.stat.agentInfo) {
        return;
    }
    const auto &propagationStats = feedBack.stat.agentInfo->propagation_stats();
    for (int32_t i = 0; i < propagationStats.stats_size(); i++) {
        const auto &propagationStat = propagationStats.stats(i);
        auto provider = _replica->getProviderByAgentId(propagationStat.latency().agent_id());
        if (!provider) {
            continue;
        }
        feedBack.stat.statIndex = i;
        provider->updateLoadBalance(feedBack, true);
    }
}

void SearchServiceResource::updateProviderFromHeartbeat(const SearchServiceReplicaPtr &replica,
                                                        const SearchServiceProviderPtr &provider,
                                                        const PropagationStatDef &propagationStat,
                                                        int64_t netLatencyUs) {
    if (!replica || !provider) {
        return;
    }
    QueryResultStatistic stat;
    stat.heartbeatPropagationStat = &propagationStat;
    ControllerFeedBack feedBack(stat);
    auto flowConfigPtr = replica->getLastFlowControlConfig();
    if (flowConfigPtr) {
        initFeedBack(*flowConfigPtr, replica, feedBack);
    } else {
        FlowControlConfig defaultConfig;
        initFeedBack(defaultConfig, replica, feedBack);
    }
    feedBack.mirrorResponse = true;
    provider->updateNetLatency(netLatencyUs);
    provider->updateLoadBalance(feedBack, true);
    AUTIL_LOG(DEBUG, "update from heartbeat success, provider: %s", provider->getNodeId().c_str());
}

bool SearchServiceResource::prepareForRetry(const SearchServiceProviderPtr &retryProvider) {
    auto retryResponse = createResponse(_response->getPartCount(), _response->getPartId(),
                                        _response->getVersion(), retryProvider);
    if (!retryResponse) {
        return false;
    }
    _retryProvider = retryProvider;
    _retryResponse = retryResponse;
    return true;
}

bool SearchServiceResource::init(PartIdTy partCount, PartIdTy partId, VersionTy version,
                                 const FlowControlConfigPtr &flowControlConfig) {
    auto response = createResponse(partCount, partId, version, _provider);
    if (!response) {
        return false;
    }
    _response = response;
    _flowControlConfig = flowControlConfig;
    return true;
}

ResponsePtr SearchServiceResource::createResponse(PartIdTy partCount, PartIdTy partId,
                                                  VersionTy version,
                                                  const SearchServiceProviderPtr &provider) {
    auto response = _request->newResponse();
    if (!response) {
        return ResponsePtr();
    }
    auto protocolType = _request->getProtocolType();
    response->setBizName(_bizName);
    response->setRequestId(_requestId);
    response->setSpecStr(provider->getSpecStr(protocolType));
    response->setNodeId(provider->getNodeId());
    response->setPartCount(partCount);
    response->setPartId(partId);
    response->setVersion(version);
    response->setProtocolType(protocolType);
    return response;
}

const ResponsePtr &SearchServiceResource::getResponse(bool isRetry) {
    return isRetry ? _retryResponse : _response;
}

ResponsePtr SearchServiceResource::getReturnedResponse() const {
    if (_response->isReturned() && !_response->isFailed()) {
        // response
        return _response;
    } else if (_retryResponse && _retryResponse->isReturned() && !_retryResponse->isFailed()) {
        // retryResponse
        return _retryResponse;
    } else if (_response->isReturned()) {
        // response
        return _response;
    } else if (_retryResponse && _retryResponse->isReturned()) {
        // retryResponse
        return _retryResponse;
    } else {
        return nullptr;
    }
}

ResponsePtr SearchServiceResource::stealReturnedResponse() {
    ResponsePtr ret;
    if (_response->isReturned() && !_response->isFailed()) {
        // response
        ret = _response;
        _response.reset();
    } else if (_retryResponse && _retryResponse->isReturned() && !_retryResponse->isFailed()) {
        // retryResponse
        ret = _retryResponse;
        _retryResponse.reset();
    } else if (_response->isReturned()) {
        // response
        ret = _response;
        _response.reset();
    } else if (_retryResponse && _retryResponse->isReturned()) {
        // retryResponse
        ret = _retryResponse;
        _retryResponse.reset();
    }
    return ret;
}

const SearchServiceProviderPtr &SearchServiceResource::getProvider(bool isRetry) const {
    return isRetry ? _retryProvider : _provider;
}





void SearchServiceResource::fillRequestQueryInfo(bool isRetry, RequestType type) {
    if (!isRetry) {
        if (_hasQueryInfoFilled) {
            return;
        }
        _hasQueryInfoFilled = true;
    } else {
        if (_hasQueryInfoRetryFilled) {
            return;
        }
        _hasQueryInfoRetryFilled = true;
    }
    const auto &provider = getProvider(isRetry);
    if (!_flowControlConfig || !_replica) {
        return;
    }
    const auto &serverBizName = provider->getServerBizName();
    if (!serverBizName.empty()) {
        _request->setBizName(serverBizName);
    } else {
        _request->setBizName(_bizName);
    }
    _request->setPartId(getPartId());
    _request->setRequestType(_requestType);
    auto latencyLimit = _flowControlConfig->fullDegradeLatency *
                        (1.0f + _flowControlConfig->latencyUpperLimitPercent);
    auto loadBalanceLatencyLimit = latencyLimit;
    auto errorRatioLimit = INVALID_FILTER_VALUE;
    auto bestChain = _replica->getBestChain();
    if (bestChain) {
        errorRatioLimit =
            bestChain->errorRatioController.getLegalLimit(_flowControlConfig) * SATURATION_FACTOR;
        latencyLimit =
            min(latencyLimit,
                bestChain->latencyController.getLegalLimit(_flowControlConfig) * SATURATION_FACTOR);
        loadBalanceLatencyLimit =
            min(loadBalanceLatencyLimit,
                bestChain->latencyController.getLoadBalanceLegalLimit(_flowControlConfig) *
                    SATURATION_FACTOR);
    }
    _request->setLatencyLimit(latencyLimit);
    _request->setLoadBalanceLatencyLimit(loadBalanceLatencyLimit);
    _request->setErrorRatioLimit(errorRatioLimit);
    auto providerWeight = provider->getWeight();
    auto replicaAvgWeight = _replica->getAvgWeight();
    auto realWeight = providerWeight / replicaAvgWeight * MAX_WEIGHT_FLOAT;
    _request->setProviderWeight(realWeight);
    _request->setBeginServerDegradeLatency(_flowControlConfig->beginServerDegradeLatency);
    _request->setBeginDegradeLatency(_flowControlConfig->beginDegradeLatency);
    _request->setBeginServerDegradeErrorRatio(_flowControlConfig->beginServerDegradeErrorRatio);
    _request->setBeginDegradeErrorRatio(_flowControlConfig->beginDegradeErrorRatio);
    _request->setProviderAttributes(&provider->getNodeAttributeMap());
    if (!provider->getNodeMetaEnv().valid()) {
        _request->setReturnMetaEnv(true);
    }
    _replica->fillPropagationStat(provider->getAgentId(), _request->getProtobufArena().get(),
                                  *_request->getPropagationStats());

}

} // namespace multi_call
