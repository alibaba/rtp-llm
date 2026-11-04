#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/CachedRequestGenerator.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceProvider.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceSnapshot.h"

namespace multi_call {

class ProviderSelector
{
public:
    ProviderSelector(const std::string &bizName);
    virtual ~ProviderSelector();

private:
    ProviderSelector(const ProviderSelector &) = delete;
    ProviderSelector &operator=(const ProviderSelector &) = delete;

public:
    uint32_t select(const std::shared_ptr<CachedRequestGenerator> &generator,
                    const SearchServiceSnapshotInVersionPtr &versionSnapshot,
                    const FlowConfigSnapshotPtr &flowConfigSnapshot,
                    const ReplyInfoCollectorPtr &replyInfoCollector, bool probeOnly,
                    SearchServiceResourceVector &searchResourceVec);

    SearchServiceProviderPtr
    selectBackupProvider(const SearchServiceSnapshotInVersionPtr &versionSnapshot,
                         const SearchServiceProviderPtr &provider, PartIdTy partId,
                         SourceIdTy sourceId, const MatchTagMapPtr &matchTagMap,
                         const FlowControlConfigPtr &flowControlConfig);

private:
    void selectFromNormalVersion(const RequestGeneratorPtr &generator,
                                 const SearchServiceSnapshotInVersionPtr &versionSnapshot,
                                 const FlowControlConfigPtr &flowControlConfig,
                                 const FlowControlParam &flowControlParam, PartIdTy partId,
                                 const RequestPtr &request, bool probeOnly,
                                 SearchServiceResourceVector &searchResourceVec);
    void selectFromCopyVersion(const RequestGeneratorPtr &generator,
                               const SearchServiceSnapshotInVersionPtr &versionSnapshot,
                               PartIdTy partId, const RequestPtr &request,
                               SearchServiceResourceVector &searchResourceVec);

protected:
    std::string _bizName;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(ProviderSelector);

} // namespace multi_call

