#pragma once

#include <string>
#include <vector>

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/common/common.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/CachedRequestGenerator.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/SearchServiceSnapshot.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/util/RandomGenerator.h"

namespace multi_call {

class VersionSelector
{
public:
    VersionSelector(const std::string &bizName)
        : _bizName(bizName)
        , _randomGenerator(autil::TimeUtility::currentTimeInMicroSeconds()) {
    }
    virtual ~VersionSelector() {
    }

private:
    VersionSelector(const VersionSelector &) = delete;
    VersionSelector &operator=(const VersionSelector &) = delete;

public:
    virtual bool select(const std::shared_ptr<CachedRequestGenerator> &generator,
                        const SearchServiceSnapshotPtr &snapshot) = 0;

    virtual const SearchServiceSnapshotInVersionPtr &getBizVersionSnapshot() = 0;

    virtual const SearchServiceSnapshotInVersionPtr &getProbeBizVersionSnapshot() = 0;

    virtual const std::vector<SearchServiceSnapshotInVersionPtr> &getCopyBizVersionSnapshots() = 0;

protected:
    bool needCopy(size_t copyProviderCount, size_t normalProviderCount);

protected:
    std::string _bizName;
    RandomGenerator _randomGenerator;
};

MULTI_CALL_TYPEDEF_PTR(VersionSelector);

} // namespace multi_call

