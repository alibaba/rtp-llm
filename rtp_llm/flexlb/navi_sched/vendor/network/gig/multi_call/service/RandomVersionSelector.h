#pragma once

#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/config/MultiCallConfig.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/service/VersionSelector.h"
#include "rtp_llm/flexlb/navi_sched/vendor/network/gig/multi_call/subscribe/SubscribeServiceManager.h"
#include "autil/Lock.h"

namespace multi_call {

class RandomVersionSelector : public VersionSelector
{
public:
    RandomVersionSelector(const std::string &bizName,
                          const SubscribeServiceManagerPtr &subscribeManager);

public:
    bool select(const std::shared_ptr<CachedRequestGenerator> &generator,
                const SearchServiceSnapshotPtr &snapshot) override;

    const SearchServiceSnapshotInVersionPtr &getBizVersionSnapshot() override {
        return _versionSnapshot;
    }

    const SearchServiceSnapshotInVersionPtr &getProbeBizVersionSnapshot() override {
        return _probeVersionSnapshot;
    }

    const std::vector<SearchServiceSnapshotInVersionPtr> &getCopyBizVersionSnapshots() override {
        return _copyVersionSnapshots;
    }

private:
    SearchServiceSnapshotInBizPtr
    getBizSnapshot(const std::shared_ptr<CachedRequestGenerator> &generator,
                   const SearchServiceSnapshotPtr &snapshot);

    void selectVersionSnapshot(const std::shared_ptr<CachedRequestGenerator> &generator,
                               const SearchServiceSnapshotInBizPtr &bizSnapshot, VersionTy version,
                               VersionTy preferVersion);
    void selectCopyVersionSnapshots(const SearchServiceSnapshotInBizPtr &bizSnapshot,
                                    VersionTy version);
    void selectProbeVersionSnapshot(const SearchServiceSnapshotInBizPtr &bizSnapshot,
                                    VersionTy version, VersionTy preferVersion);

    VersionTy selectVersion(const std::shared_ptr<CachedRequestGenerator> &generator,
                            const SearchServiceSnapshotInBizPtr &bizSnapshot, SourceIdTy sourceId);
    void getWeightVec(const std::shared_ptr<CachedRequestGenerator> &generator,
                      const SearchServiceSnapshotInBizPtr &bizSnapshot,
                      std::vector<std::pair<VersionTy, WeightTy>> &completeWeightVec,
                      std::vector<std::pair<VersionTy, WeightTy>> &notCompleteWeightVec,
                      std::vector<std::pair<VersionTy, WeightTy>> &stopWeightVec,
                      std::map<VersionTy, size_t> &verisonNormalProviderCountMap);
    VersionTy selectVersionByWeightOrProviderCount(
        const SearchServiceSnapshotInBizPtr &bizSnapshot,
        const std::vector<std::pair<VersionTy, WeightTy>> &weightVec,
        const std::map<VersionTy, size_t> &versionNormalProviderCountMap, SourceIdTy sourceId);
    VersionTy selectVersionByWeightVec(WeightTy sumWeight,
                                       const std::vector<std::pair<VersionTy, WeightTy>> &weightVec,
                                       SourceIdTy sourceId);

private:
    SubscribeServiceManagerPtr _subscribeServiceManager;
    autil::ThreadMutex _mutex;
    SearchServiceSnapshotInVersionPtr _versionSnapshot;
    SearchServiceSnapshotInVersionPtr _probeVersionSnapshot;
    std::vector<SearchServiceSnapshotInVersionPtr> _copyVersionSnapshots;

private:
    AUTIL_LOG_DECLARE();
};

MULTI_CALL_TYPEDEF_PTR(RandomVersionSelector);

} // namespace multi_call

