#pragma once

#include "aios/navi_sched/scheduler/CostSchedulerConfig.h"
#include "aios/navi_sched/scheduler/CostSchedulerKernelInternal.h"
#include "aios/navi_sched/resource/SchedulerConfigR.h"
#include "aios/network/gig/multi_call/util/RandomGenerator.h"
#include "navi/engine/Data.h"
#include "navi/engine/Kernel.h"

#include <atomic>
#include <deque>
#include <limits>
#include <memory>
#include <vector>

namespace navi {
class GraphBuilder;
class N;
}

namespace navi_sched {

extern const std::string COST_SCHEDULER_GRAPH_DATA_TYPE_ID;

class CostSchedulerGraphState {
public:
    static std::shared_ptr<CostSchedulerGraphState> create(
        const NaviSchedScheduleContextPtr &context, const CostSchedulerConfig &config,
        size_t workerCount,
        std::shared_ptr<const std::vector<double>> randomWeights = nullptr);
    ~CostSchedulerGraphState();

    bool runRow(size_t workerIndex, size_t iteration);
    bool prepareWorker(size_t workerIndex);
    bool finishPreparation();
    bool finishRow(size_t iteration);
    bool runProjection(size_t workerIndex, size_t iteration);
    bool finishProjection(size_t iteration);
    bool shouldContinue() const;
    bool runFinalRow(size_t workerIndex);
    bool finalize(int64_t replayRefinementEvaluations = -1);
    bool resetWeights(const double *weights, size_t weightCount);

    const NaviSchedScheduleContextPtr &context() const;
    const double *weights() const;
    const double *gradients() const;
    size_t weightCount() const;
    double forwardValue() const;
    bool failed() const;
    size_t actualLoopCount() const;
    const std::string &stopReason() const;

private:
    friend class CostSchedulerGraphKernelUtil;
    struct WorkerState;

    CostSchedulerGraphState(NaviSchedScheduleContextPtr context,
                            CostSchedulerConfig config, size_t workerCount);
    bool init(std::shared_ptr<const std::vector<double>> randomWeights);
    void beginOptimization();
    void fail();
    bool updateBestResult(double loss);
    bool evaluateDiscreteCandidate(size_t iteration, double continuousLoss);
    std::pair<size_t, size_t> nodeRange(size_t workerIndex) const;
    std::pair<size_t, size_t> requestRange(size_t workerIndex) const;

private:
    NaviSchedScheduleContextPtr _context;
    CostSchedulerConfig _config;
    size_t _workerCount = 0;
    size_t _requestCount = 0;
    size_t _nodeCount = 0;
    size_t _weightCount = 0;
    int64_t _totalTokenCount = 0;
    double _tokenNormalizationDenominator = 0.0;
    std::unique_ptr<int64_t[]> _requestTokenCounts;
    std::unique_ptr<double[]> _inverseRequestTokenCounts;
    std::unique_ptr<float[]> _costs;
    std::unique_ptr<double[]> _weights;
    std::unique_ptr<double[]> _gradients;
    std::unique_ptr<uint32_t[]> _bestNodeIndexes;
    std::unique_ptr<uint32_t[]> _candidateNodeIndexes;
    std::vector<std::unique_ptr<WorkerState>> _workers;
    std::vector<CostSchedulerKernelInternal::GradientSpan> _gradientSpans;
    std::vector<double> _weightsBeforeUpdate;
    std::vector<double> _weightsBeforeProjection;
    std::unique_ptr<CostSchedulerKernelInternal::ReplayInfo> _replayInfo;
    std::atomic<bool> _failed{false};
    double _forwardValue = 0.0;
    double _bestLoss = std::numeric_limits<double>::infinity();
    double _stepAlpha = 0.0;
    double _repairAlpha = 0.0;
    CostSchedulerKernelInternal::DiscreteForwardWorkspace _discreteForwardWorkspace;
    int64_t _runBeginUs = 0;
    int64_t _stepBeginUs = 0;
    int64_t _optimizationBeginUs = 0;
    int64_t _preparationBeginUs = 0;
    int64_t _preparationReadyUs = 0;
    int64_t _preparationInitCpuUs = 0;
    int64_t _lastLoopEndUs = 0;
    int64_t _lastLoopReplayCollectionUs = 0;
    int64_t _replayCollectionBaselineUs = 0;
    size_t _actualLoopCount = 0;
    size_t _allowedLoopCount = 0;
    std::string _stopReason = "max_loop_count";
    std::shared_ptr<const std::vector<double>> _randomWeights;
    size_t _randomWeightCount = 0;
    bool _parallelPreparation = false;
    bool _discreteRepairActive = false;
    bool _discreteEvaluated = false;
    double _discreteForwardValue = 0.0;
    double _discreteGap = 0.0;
};

NAVI_TYPEDEF_PTR(CostSchedulerGraphState);

class CostSchedulerGraphData final : public navi::Data {
public:
    explicit CostSchedulerGraphData(CostSchedulerGraphStatePtr state,
                                    NaviSchedScheduleContextPtr context = nullptr);

    CostSchedulerGraphStatePtr state;
    NaviSchedScheduleContextPtr context;
    bool continueLoop = true;
};

NAVI_TYPEDEF_PTR(CostSchedulerGraphData);

class CostSchedulerPrepareKernel final : public navi::Kernel {
public:
    static const std::string KERNEL_ID;

    CostSchedulerPrepareKernel();
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    static constexpr size_t RANDOM_WEIGHT_COUNT = 1UL << 22;

    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON_FALSE(SchedulerConfigR, _schedulerConfigR);

    CostSchedulerConfig _config;
    size_t _workerCount = 0;
    std::shared_ptr<const std::vector<double>> _randomWeights;
    uint64_t _randomWeightSeed = 0;
    multi_call::RandomGenerator _randomGenerator;
};

class CostSchedulerRowKernel final : public navi::Kernel {
public:
    static const std::string KERNEL_ID;

    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    size_t _workerIndex = 0;
};

class CostSchedulerPreparationKernel final : public navi::Kernel {
public:
    static const std::string KERNEL_ID;

    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    size_t _workerIndex = 0;
};

class CostSchedulerPreparationBarrierKernel final : public navi::Kernel {
public:
    static const std::string KERNEL_ID;

    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    size_t _workerCount = 0;
};

class CostSchedulerLoopKernel final : public navi::Kernel {
public:
    static const std::string KERNEL_ID;

    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    size_t _workerCount = 0;
    std::deque<CostSchedulerGraphDataPtr> _pending;
    bool _active = false;
    bool _sourceEof = false;
    bool _outputEof = false;
};

class CostSchedulerGradientBarrierKernel final : public navi::Kernel {
public:
    static const std::string KERNEL_ID;

    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    size_t _workerCount = 0;
};

class CostSchedulerProjectionKernel final : public navi::Kernel {
public:
    static const std::string KERNEL_ID;

    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    size_t _workerIndex = 0;
};

class CostSchedulerProjectionBarrierKernel final : public navi::Kernel {
public:
    static const std::string KERNEL_ID;

    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    size_t _workerCount = 0;
};

class CostSchedulerFinalRowKernel final : public navi::Kernel {
public:
    static const std::string KERNEL_ID;

    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    size_t _workerIndex = 0;
};

class CostSchedulerFinalizeKernel final : public navi::Kernel {
public:
    static const std::string KERNEL_ID;

    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;
};

class CostSchedulerGraphBuilder {
public:
    static size_t getWorkerCount(int32_t configuredParallelism);
    static bool add(navi::GraphBuilder &builder, const navi::N &context,
                    const navi::N &result, const std::string &schedulerKernelConfig);

private:
    static std::string makePrepareConfig(const CostSchedulerConfig &config,
                                         size_t workerCount);
    static std::string makeWorkerConfig(size_t workerIndex);
    static std::string makeBarrierConfig(size_t workerCount);
    static bool add(navi::GraphBuilder &builder, const navi::N &context,
                    const navi::N &result, const CostSchedulerConfig &config,
                    size_t workerCount);
};

}
