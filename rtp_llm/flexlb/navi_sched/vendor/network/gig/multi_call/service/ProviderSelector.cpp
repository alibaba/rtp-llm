#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ProviderSelector.h"

#include <string>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/FlowControlParam.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/ResourceComposer.h"
#include "autil/Log.h"

using namespace std;
using namespace autil;

namespace multi_call {
AUTIL_LOG_SETUP(multi_call, ProviderSelector);

ProviderSelector::ProviderSelector(const std::string &bizName) : _bizName(bizName) {
}

ProviderSelector::~ProviderSelector() {
}

void ProviderSelector::selectFromNormalVersion(
    const RequestGeneratorPtr &generator, const SearchServiceSnapshotInVersionPtr &versionSnapshot,
    const FlowControlConfigPtr &flowControlConfig, const FlowControlParam &flowControlParam,
    PartIdTy partId, const RequestPtr &request, bool probeOnly,
    SearchServiceResourceVector &searchResourceVec) {
    SearchServiceReplicaPtr replica;
    SearchServiceProviderPtr provider;
    SearchServiceProviderPtr probeProvider;
    RequestType probeType = RT_PROBE;
    SourceIdTy sourceId = generator->getSourceId();
    auto tags = generator->cloneMatchTags();
    versionSnapshot->selectProvider(partId, sourceId, flowControlParam, tags, replica, provider,
                                    probeProvider, probeType);
    auto partCnt = versionSnapshot->getPartCount();
    auto version = versionSnapshot->getVersion();
    if (!probeOnly) {
        ResourceComposer::doCreateSearchServiceResource(
            generator, tags, _bizName, request, partCnt, partId, version, replica, provider,
            RT_NORMAL, flowControlConfig, searchResourceVec);
        if (!provider || !replica) {
            AUTIL_INTERVAL_LOG(200, WARN,
                               "no provider found for biz[%s] partId[%d], maybe "
                               "degrade triggered",
                               _bizName.c_str(), partId);
        }
    }
    AUTIL_LOG(DEBUG, "provider: %s, probe: %s", provider ? provider->getNodeId().c_str() : "",
              probeProvider ? probeProvider->getNodeId().c_str() : "");
    if ((probeOnly || provider != probeProvider) && (!generator->getDisableProbe())) {
        ResourceComposer::doCreateSearchServiceResource(
            generator, tags, _bizName, request, partCnt, partId, version, replica, probeProvider,
            probeType, flowControlConfig, searchResourceVec);
    }
}

void ProviderSelector::selectFromCopyVersion(
    const RequestGeneratorPtr &generator, const SearchServiceSnapshotInVersionPtr &versionSnapshot,
    PartIdTy partId, const RequestPtr &request, SearchServiceResourceVector &searchResourceVec) {
    SearchServiceReplicaPtr replica;
    SearchServiceProviderPtr provider;
    auto tags = generator->cloneMatchTags();
    versionSnapshot->selectProbeProvider(partId, tags, replica, provider);

    auto partCnt = versionSnapshot->getPartCount();
    auto version = versionSnapshot->getVersion();
    ResourceComposer::doCreateSearchServiceResource(generator, tags, _bizName, request, partCnt,
                                                    partId, version, replica, provider, RT_COPY,
                                                    FlowControlConfigPtr(), searchResourceVec);
}

uint32_t ProviderSelector::select(const std::shared_ptr<CachedRequestGenerator> &generator,
                                  const SearchServiceSnapshotInVersionPtr &versionSnapshot,
                                  const FlowConfigSnapshotPtr &flowConfigSnapshot,
                                  const ReplyInfoCollectorPtr &replyInfoCollector, bool probeOnly,
                                  SearchServiceResourceVector &searchResourceVec) {
    if (!versionSnapshot) {
        replyInfoCollector->addErrorCode(_bizName, MULTI_CALL_REPLY_ERROR_VERSION_NOT_EXIST);
        return 1;
    }

    auto realGenerator = generator->getGenerator();
    const auto &strategy = realGenerator->getFlowControlStrategy();
    const FlowControlConfigPtr &flowControlConfig =
        flowConfigSnapshot->getFlowControlConfig(_bizName, strategy);
    AUTIL_LOG(DEBUG,
              "biz [%s] strategy [%s], disable retry: %d, disable probe: %d, disable degrade: %d, "
              "ignore weight label: %d, flow config: [%s]",
              realGenerator->getBizName().c_str(), strategy.c_str(),
              realGenerator->getDisableRetry(), realGenerator->getDisableProbe(),
              realGenerator->getDisableDegrade(),
              realGenerator->getIgnoreWeightLabelInConsistentHash(),
              flowControlConfig ? ToJsonString(*flowControlConfig).c_str() : "null");

    auto partCnt = versionSnapshot->getPartCount();
    PartRequestMap requestMap;
    generator->generate(partCnt, requestMap);
    if (realGenerator->hasError()) {
        replyInfoCollector->addErrorCode(_bizName, MULTI_CALL_ERROR_GENERATE);
        return partCnt;
    }
    ResourceComposer::fillUserRequestType(realGenerator, requestMap);

    FlowControlParam flowControlParam(flowControlConfig);
    flowControlParam.partitionCount = requestMap.size();
    flowControlParam.disableDegrade = realGenerator->getDisableDegrade();
    flowControlParam.ignoreWeightLabelInConsistentHash =
        realGenerator->getIgnoreWeightLabelInConsistentHash();
    auto isCopyVersion = versionSnapshot->isCopyVersion();
    size_t index = 0;
    for (auto it = requestMap.begin(); it != requestMap.end(); it++, index++) {
        PartIdTy pid = it->first;
        const auto &request = it->second;
        if (!request) {
            replyInfoCollector->addErrorCode(_bizName, MULTI_CALL_REPLY_ERROR_REQUEST_NOT_EXIST);
            AUTIL_LOG(ERROR, "generated request is null");
            continue;
        }
        if (!isCopyVersion) {
            flowControlParam.partitionIndex = index;
            selectFromNormalVersion(realGenerator, versionSnapshot, flowControlConfig,
                                    flowControlParam, pid, request, probeOnly, searchResourceVec);
        } else {
            selectFromCopyVersion(realGenerator, versionSnapshot, pid, request, searchResourceVec);
        }
    }
    return requestMap.size();
}

SearchServiceProviderPtr
ProviderSelector::selectBackupProvider(const SearchServiceSnapshotInVersionPtr &versionSnapshot,
                                       const SearchServiceProviderPtr &provider, PartIdTy partId,
                                       SourceIdTy sourceId, const MatchTagMapPtr &matchTagMap,
                                       const FlowControlConfigPtr &flowControlConfig) {
    return versionSnapshot->getBackupProvider(provider, partId, sourceId, matchTagMap,
                                              flowControlConfig);
}

} // namespace multi_call
