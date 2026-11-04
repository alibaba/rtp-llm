#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ChildNodeCaller.h"

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/interface/QuerySession.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/FlowControlParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ResourceComposer.h"

using namespace std;

namespace multi_call {

AUTIL_LOG_SETUP(multi_call, ChildNodeCaller);

ChildNodeCaller::ChildNodeCaller(const SearchServiceSnapshotPtr &snapshot,
                                 const FlowConfigSnapshotPtr &flowConfigSnapshot,
                                 const CallDelegationThreadPtr &callDelegationThread,
                                 const RequestGeneratorVec &generatorVec,
                                 const RetryLimitCheckerPtr &retryLimitChecker,
                                 const LatencyTimeSnapshotPtr &latencyTimeSnapshot,
                                 const QuerySessionPtr &querySession)
    : _snapshot(snapshot)
    , _flowConfigSnapshot(flowConfigSnapshot)
    , _callDelegationThread(callDelegationThread)
    , _generatorVec(generatorVec)
    , _retryLimitChecker(retryLimitChecker)
    , _latencyTimeSnapshot(latencyTimeSnapshot)
    , _querySession(querySession)
    , _earlyTerminationEnabled(false)
    , _retryEnabled(false)
    , _singleRetryEnable(false)
    , _canRetry(false)
    , _disableDetection(false) {
    _callBeginTime = autil::TimeUtility::currentTime();
}

ChildNodeCaller::~ChildNodeCaller() {
}

ChildNodeReplyPtr ChildNodeCaller::call() {
    SearchServiceResourceVector resourceVec;
    return call(resourceVec);
}

ChildNodeReplyPtr ChildNodeCaller::call(SearchServiceResourceVector &resourceVec) {
    if (initResource(resourceVec)) {
        for (const auto &resource : resourceVec) {
            doCall(resource);
        }
    }
    if (isDetectionOn() && !resourceVec.empty()) {
        _callDelegationThread->pushWorkItem(
            new CallDelegationWorkItem(_caller, _querySession->getLoadBalancerContext()));
    }
    return _reply;
}

uint64_t ChildNodeCaller::getMaxTimeout(const SearchServiceResourceVector &resourceVec) {
    uint64_t timeout = 0;
    for (const auto &resource : resourceVec) {
        timeout = std::max(timeout, resource->getTimeout());
    }
    return timeout;
}

void ChildNodeCaller::afterCall(Closure *closure) {
    _caller->afterCall(closure);
}

bool ChildNodeCaller::initResource(SearchServiceResourceVector &resourceVec) {
    initCaller();
    if (!_snapshot) {
        AUTIL_LOG(ERROR, "snapshot is NULL");
        return false;
    }
    fillSourceId();
    prepareSearchResource(resourceVec);
    return true;
}

void ChildNodeCaller::fillSourceId() {
    for (const auto &generator : _generatorVec) {
        if (!generator) {
            continue;
        }
        if (generator->getSourceId() == INVALID_SOURCE_ID) {
            generator->setSourceId(_snapshot->getRandomSourceId());
        }
    }
}

void ChildNodeCaller::initCaller() {
    vector<string> bizNameVec;
    vector<string> flowControlStrategyVec;
    bizNameVec.reserve(_generatorVec.size());
    for (const auto &generator : _generatorVec) {
        if (!generator) {
            AUTIL_LOG(ERROR, "generator is NULL");
            continue;
        }
        if (!generator->getDisableRetry()) {
            _canRetry = true;
        }
        bizNameVec.push_back(generator->getBizName());
        flowControlStrategyVec.push_back(generator->getFlowControlStrategy());
    }
    _replyInfoCollector.reset(new ReplyInfoCollector(bizNameVec));
    _replyInfoCollector->setSessionBiz(_querySession->getBiz());
    _replyInfoCollector->setSessionSrc(_querySession->getUserData(RPC_DATA_SRC));
    _replyInfoCollector->setSessionSrcAb(_querySession->getUserData(RPC_DATA_SRC_AB));
    _replyInfoCollector->setStressTest(_querySession->getUserData(RPC_DATA_STRESS_TEST));

    _reply.reset(new ChildNodeReply(_flowConfigSnapshot, _replyInfoCollector, _retryLimitChecker,
                                    _latencyTimeSnapshot));
    _flowConfigSnapshot->getFlowControlSwitch(bizNameVec, flowControlStrategyVec,
                                              _earlyTerminationEnabled, _retryEnabled,
                                              _singleRetryEnable);
    _reply->setSingleRetryEnabled(_singleRetryEnable);
    if (isDetectionOn()) {
        _reply->prepareCallDelegationStatistic(bizNameVec, flowControlStrategyVec);
    }

    _caller.reset(new Caller(_reply, _querySession->getSessionContext()));
}

void ChildNodeCaller::prepareSearchResource(SearchServiceResourceVector &resourceVec) {
    for (const auto &generator : _generatorVec) {
        if (!generator) {
            continue;
        }
        size_t providerCount = 0;
        const auto &bizName = generator->getBizName();

        ResourceComposer resourceComposer(_snapshot, _flowConfigSnapshot);
        auto expectProviderCount = resourceComposer.prepareResource(
            _querySession, generator, _replyInfoCollector, resourceVec, providerCount);

        _replyInfoCollector->addRequestCount(bizName, 1);
        _replyInfoCollector->addExpectProviderCount(bizName, expectProviderCount);
        _reply->addExpectProviderCount(expectProviderCount);
        if (isDetectionOn()) {
            _reply->collectStatistic(bizName, providerCount, generator->getDisableRetry());
        }
    }
    _reply->setRpcTimeout(getMaxTimeout(resourceVec));
    for (const auto &resource : resourceVec) {
        resource->setCallBegTime(_callBeginTime);
    }
}

bool ChildNodeCaller::isDetectionOn() const {
    return (!_disableDetection) && (_earlyTerminationEnabled || isRetryOn());
}

bool ChildNodeCaller::isRetryOn() const {
    return _canRetry && (_retryEnabled || _singleRetryEnable);
}

const CallerPtr &ChildNodeCaller::getCaller() const {
    return _caller;
}

void ChildNodeCaller::doCall(const SearchServiceResourcePtr &resource) {
    _caller->call(resource, false);
    if (resource->isNormalRequest()) {
        _reply->appendSearchServiceResource(resource);
    }
}

} // namespace multi_call
