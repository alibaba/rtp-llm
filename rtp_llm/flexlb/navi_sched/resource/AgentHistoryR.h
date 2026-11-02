#pragma once

#include "aios/navi_sched/common/NaviSchedCommon.h"
#include "aios/navi_sched/data/NaviSchedLatencySample.h"
#include "aios/navi_sched/resource/LatencyHistorySampler.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "aios/navi_sched/resource/AgentSchedulerConfigR.h"
#include "navi/engine/Resource.h"
#include "autil/Lock.h"
#include <memory>
#include <string>
#include <vector>

namespace navi_sched {

struct NaviSchedLatencyModelSnapshot;

class AgentHistoryR : public navi::Resource {
public:
    using History = LatencyHistorySampler::History;
    using HistoryConstPtr = LatencyHistorySampler::HistoryConstPtr;

    struct AppendResult {
        bool appended = false;
        bool fitReady = false;
        bool persistReady = false;
        bool rampReplay = false;
        bool trainingDeferred = false;
        bool lateRampSample = false;
    };

public:
    AgentHistoryR() = default;

public:
    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    bool load(NonLinearPrefillModelR &model,
              std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot,
              bool &fitOnStart);
    bool load(NonLinearPrefillModelR &model,
              std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot,
              bool &fitOnStart,
              const LatencyResidualModel::RuntimeConfig &residualConfig);
    bool append(const NaviSchedLatencySample &sample,
                NaviSchedLatencyTrainingPhase trainingPhase,
                AppendResult &result);
    bool append(const NaviSchedLatencySample &sample, bool &fitReady, bool &persistReady);
    std::vector<NaviSchedLatencySample> getFitBatch() const;
    std::vector<NaviSchedLatencySample> getOnlineFitBatch() const;
    std::vector<std::vector<NaviSchedLatencySample>> getOnlineFitBatches() const;
    void clearPendingFitSamples();
    std::vector<std::vector<NaviSchedLatencySample>>
    getOnlineFitReplayBatches(bool includePendingTail = false) const;
    HistoryConstPtr getHistory() const;
    size_t getSessionSampleCount() const;
    size_t getRetainedSampleCountAtMost(double latencyMs) const;
    size_t getFitBatchSize() const;
    bool isWarmupPending() const;
    int64_t getWarmupLocalBatchIdCutoff() const;
    bool requiresParameterValidation() const;
    bool makeSavePayload(
        const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot,
        const NonLinearPrefillModelR::OptimizerState &optimizerState,
        std::string &payload) const;
    bool savePayload(const std::string &payload) const;
    bool save(const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot,
              const NonLinearPrefillModelR::OptimizerState &optimizerState) const;
    bool enabled() const;
    void applyRuntimeConfig(const AgentSchedulerConfig &config);
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    bool advanceFitCadence(size_t historySize, size_t &samplesSinceFit) const;

private:
    mutable autil::ThreadMutex _lock;
    LatencyHistorySampler _sampler;
    std::string _stateFile;
    std::string _formulaFingerprint;
    std::string _historyFingerprint;
    std::string _binaryFingerprint;
    size_t _historyLimit = 2000;
    size_t _historyBucketCapacity = 64;
    size_t _fitBatchSize = 4;
    size_t _fitInterval = 0;
    size_t _persistInterval = 256;
    size_t _warmupSampleCount = 0;
    size_t _warmupSamplesSkipped = 0;
    int64_t _warmupLocalBatchIdCutoff = 0;
    size_t _samplesSinceFit = 0;
    size_t _pendingFitSampleCount = 0;
    size_t _samplesSincePersist = 0;
    bool _rampCollectionObserved = false;
    bool _rampCollectionCompleted = false;
    bool _historyRestored = false;
    bool _parametersRestoredWithoutHistory = false;
};

NAVI_TYPEDEF_PTR(AgentHistoryR);

}
