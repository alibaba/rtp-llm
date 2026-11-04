#pragma once

#include "rtp_llm/flexlb/navi_sched/data/AgentLatencyToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/data/LatencyModelGlobalData.h"
#include "rtp_llm/flexlb/navi_sched/data/MasterLatencyModelWarmupToAgentData.h"
#include "rtp_llm/flexlb/navi_sched/data/NaviSchedLatencySample.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentHistoryR.h"
#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyResidualModel.h"
#include "autil/Lock.h"
#include "navi/engine/Resource.h"
#include <atomic>
#include <deque>
#include <memory>
#include <optional>
#include <string>

class NaviSchedResourceTest;

namespace navi_sched {

struct NaviSchedLatencyModelSnapshot {
    int32_t formula = 0;
    int64_t binaryVersion = 0;
    int64_t revision = 0;
    std::vector<double> parameters;
    NonLinearPrefillModelR::OptimizerState optimizerState;
    int32_t residualVersion = LatencyResidualModel::VERSION;
    std::vector<double> residualParameters =
        LatencyResidualModel::makeInitialParameters();
    int64_t residualLastUpdateTimeUs = 0;
    bool overloadPredictionTrusted = false;
    bool warmStarted = false;
    bool onlineFitUpdate = false;
    uint64_t sessionSampleCount = 0;

    bool valid() const;
};

NAVI_TYPEDEF_PTR(NaviSchedLatencyModelSnapshot);

class LatencyModelR : public navi::Resource {
public:
    enum class GlobalZeroPointApplyResult {
        APPLIED,
        STALE,
        INCOMPATIBLE,
    };

    void def(navi::ResourceDefBuilder &builder) const override;
    bool config(navi::ResourceConfigContext &ctx) override;
    navi::ErrorCode init(navi::ResourceInitContext &ctx) override;
    bool recordSample(
        const NaviSchedLatencySample &sample,
        std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot,
        bool &saveNeeded,
        NaviSchedLatencyTrainingPhase trainingPhase =
            NaviSchedLatencyTrainingPhase::NORMAL);
    bool fitHistoryOnStart(
        std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot);
    double predict(const NaviSchedLatencyFeatures &features) const;
    double predict(const NaviSchedLatencyFeatures &features,
                   const std::vector<double> &parameters) const;
    std::optional<NonLinearPrefillModelR::NonlinearOperatingPoint>
    getNonlinearOperatingPoint(const NaviSchedLatencyFeatures &features,
                               const std::vector<double> &parameters) const;
    NonLinearPrefillModelR::OptimizerState getOptimizerState() const;
    bool saveState();
    bool applyColdStartCandidates(
        const MasterLatencyModelWarmupToAgentData &warmup,
        std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot);
    GlobalZeroPointApplyResult applyGlobalZeroPoint(const LatencyModelGlobalData &global);
    LatencyModelGlobalData getGlobalZeroPoint() const;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> getSnapshot() const;
    bool applyRuntimeConfig(const AgentSchedulerConfig &config);
    navi::SummaryErrorCode getSummary(const std::string &query,
                                      navi::SummaryBuilder &builder) const override;

public:
    static const std::string RESOURCE_ID;

private:
    friend class ::NaviSchedResourceTest;

    enum class ColdStartSelectionResult {
        PENDING,
        LOCAL_SELECTED,
        IMPORTED_ACCEPTED,
        IMPORTED_REJECTED,
    };

    static constexpr size_t COLD_START_MIN_HISTORY_SAMPLE_COUNT = 32;
    static constexpr int64_t COLD_START_SOURCE_MIN_REVISION = 1024;

    bool fit(const std::vector<NaviSchedLatencySample> &samples,
             std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot);
    bool fit(
        const std::vector<std::vector<NaviSchedLatencySample>> &batches,
        std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot);
    std::vector<NaviSchedLatencySample> getOnlineFitBatch() const;
    std::vector<std::vector<NaviSchedLatencySample>> getOnlineFitBatches() const;
    bool selectColdStartCandidate(
        std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot,
        ColdStartSelectionResult &result);
    bool addColdStartCandidates(
        const std::vector<NaviSchedLatencyModelCandidate> &candidates);
    bool fitParameters(const std::vector<NaviSchedLatencySample> &samples,
                       const std::vector<double> &initialParameters,
                       std::vector<double> &updatedParameters,
                       double learningRateScale,
                       double currentSampleWeight = 1.0,
                       const std::vector<double> *parameterLearningRateScales = nullptr,
                       bool rejectNonPositiveWaitingCandidate = false);
    bool fitOnlineParameters(
        const std::vector<NaviSchedLatencySample> &samples,
        const std::vector<double> &initialParameters,
        std::vector<double> &updatedParameters, double learningRateScale,
        double currentSampleWeight,
        const std::vector<double> *parameterLearningRateScales = nullptr,
                       bool rejectNonPositiveWaitingCandidate = false);
    bool fitHistoryParameters(
        const std::deque<NaviSchedLatencySample> &history,
        const std::vector<double> &initialParameters,
        std::vector<double> &updatedParameters);
    bool refitColdStartCandidate(
        const std::vector<double> &initialParameters,
        const NaviSchedLatencyOptimizerState &initialOptimizerState,
        const std::vector<double> &initialResidualParameters,
        int64_t initialResidualLastUpdateTimeUs,
        std::vector<double> &updatedParameters,
        std::vector<double> &updatedResidualParameters,
        int64_t &updatedResidualLastUpdateTimeUs,
        int64_t &localUpdateCount);
    bool fitOnlineBatch(
        const std::vector<NaviSchedLatencySample> &samples,
        const std::vector<double> &initialParameters,
        const std::vector<double> &initialResidualParameters,
        int64_t initialResidualLastUpdateTimeUs,
        const AgentSchedulerConfig &runtimeConfig,
        std::vector<double> &updatedParameters,
        std::vector<double> &updatedResidualParameters,
        int64_t &updatedResidualLastUpdateTimeUs,
        bool rejectNonPositiveWaitingCandidate = false);
    bool calculateAbsoluteError(
        const AgentHistoryR::History &history,
        const std::vector<double> &parameters,
        const std::vector<double> &residualParameters, double &error) const;
    int64_t calculateRefittedCandidateRevision(
        int64_t sourceRevision, int64_t localUpdateCount) const;
    bool shouldTrustLocked(int64_t revision) const;
    void refreshTrustLocked(
        std::shared_ptr<const NaviSchedLatencyModelSnapshot> &updatedSnapshot);
    void clearColdStartCandidates();
    void publishSnapshot(
        const std::shared_ptr<const NaviSchedLatencyModelSnapshot> &snapshot);
    LatencyResidualModel::RuntimeConfig getResidualRuntimeConfig() const;

private:
    RESOURCE_DEPEND_DECLARE();

private:
    RESOURCE_DEPEND_ON(NonLinearPrefillModelR, _nonLinearPrefillModelR);
    RESOURCE_DEPEND_ON(AgentHistoryR, _agentHistoryR);
    RESOURCE_DEPEND_ON_FALSE(AgentSchedulerConfigR, _agentSchedulerConfigR);
    bool _fitOnStart = false;
    bool _restoredTrustedModel = false;
    std::string _latencyModelFingerprint;
    size_t _coldStartFitEpochs = 100;
    int64_t _trustMinRevision = 256;
    double _zeroAnchorMaxMs = NonLinearPrefillModelR::DEFAULT_ZERO_ANCHOR_MAX_MS;
    size_t _zeroAnchorMinSamples = 8;
    bool _rampFinished = false;
    mutable autil::ThreadMutex _persistMutex;
    mutable autil::ThreadMutex _fitMutex;
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> _snapshot;
    std::shared_ptr<const LatencyResidualModel::RuntimeConfig>
        _residualRuntimeConfig =
            std::make_shared<const LatencyResidualModel::RuntimeConfig>();
    double _fixedOnlineLearningRateScale = 6.0;
    double _fixedResidualLearningRate = 0.03;
    std::atomic<uint64_t> _onlineFitCount{0};
    std::atomic<uint64_t> _rampDeferredSampleCount{0};
    std::atomic<uint64_t> _rampReplayCount{0};
    std::atomic<uint64_t> _lateRampSampleCount{0};
    std::vector<NaviSchedLatencyModelCandidate> _coldStartCandidates;
    std::vector<bool> _coldStartCandidateImported;
    std::shared_ptr<const LatencyModelGlobalData> _globalZeroPoint;
    std::atomic<uint64_t> _coldStartAppliedCount{0};
    std::atomic<uint64_t> _coldStartSelectedCount{0};
    std::atomic<uint64_t> _coldStartRefitRejectedCount{0};
    std::atomic<uint64_t> _globalZeroPointAppliedCount{0};
    std::atomic<uint64_t> _globalZeroPointIncompatibleCount{0};
};

NAVI_TYPEDEF_PTR(LatencyModelR);

} // namespace navi_sched
