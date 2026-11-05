#include "aios/navi_sched/scheduler/CostSchedulerGraph.h"

#include "aios/navi_sched/scheduler/CostSchedulerColdStartStrategy.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "aios/navi_sched/scheduler/MasterScheduleContextData.h"
#include "aios/navi_sched/scheduler/SchedulerBaseKernel.h"
#include "autil/TimeUtility.h"
#include "autil/legacy/json.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/Type.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <sched.h>
#include <unistd.h>

namespace navi_sched {

const std::string COST_SCHEDULER_GRAPH_DATA_TYPE_ID = "NaviSchedCostSchedulerGraphType";

const std::string CostSchedulerPrepareKernel::KERNEL_ID =
    "navi_sched.cost_scheduler_prepare_k";
const std::string CostSchedulerPreparationKernel::KERNEL_ID =
    "navi_sched.cost_scheduler_preparation_k";
const std::string CostSchedulerPreparationBarrierKernel::KERNEL_ID =
    "navi_sched.cost_scheduler_preparation_barrier_k";
const std::string CostSchedulerLoopKernel::KERNEL_ID = "navi_sched.cost_scheduler_loop_k";
const std::string CostSchedulerRowKernel::KERNEL_ID = "navi_sched.cost_scheduler_row_k";
const std::string CostSchedulerGradientBarrierKernel::KERNEL_ID =
    "navi_sched.cost_scheduler_gradient_barrier_k";
const std::string CostSchedulerProjectionKernel::KERNEL_ID =
    "navi_sched.cost_scheduler_projection_k";
const std::string CostSchedulerProjectionBarrierKernel::KERNEL_ID =
    "navi_sched.cost_scheduler_projection_barrier_k";
const std::string CostSchedulerFinalRowKernel::KERNEL_ID =
    "navi_sched.cost_scheduler_final_row_k";
const std::string CostSchedulerFinalizeKernel::KERNEL_ID =
    "navi_sched.cost_scheduler_finalize_k";

size_t CostSchedulerGraphBuilder::getWorkerCount(int32_t configuredParallelism)
{
    if (configuredParallelism > 0) {
        return static_cast<size_t>(configuredParallelism);
    }
    cpu_set_t available;
    CPU_ZERO(&available);
    if (sched_getaffinity(0, sizeof(available), &available) == 0 &&
        CPU_COUNT(&available) > 0) {
        return static_cast<size_t>(CPU_COUNT(&available));
    }
    auto cpuCount = sysconf(_SC_NPROCESSORS_ONLN);
    if (cpuCount <= 0) {
        return 1;
    }
    return static_cast<size_t>(cpuCount);
}

std::string CostSchedulerGraphBuilder::makePrepareConfig(
    const CostSchedulerConfig &config, size_t workerCount)
{
    autil::legacy::json::JsonMap attrs;
    attrs["lambda"] = config.lambda;
    attrs["alpha"] = config.alpha;
    attrs["alpha_decay"] = config.alphaDecay;
    attrs["min_alpha"] = config.minAlpha;
    attrs["max_loop_count"] = config.maxLoopCount;
    attrs["min_loop_count"] = config.minLoopCount;
    attrs["optimization_time_budget_us"] = config.optimizationTimeBudgetUs;
    attrs["parallelism"] = config.parallelism;
    attrs["latency_weight_concavity"] = config.latencyWeightConcavity;
    attrs["latency_token_normalization_power"] = config.latencyTokenNormalizationPower;
    attrs["initial_weight_seed"] = config.initialWeightSeed;
    attrs["discrete_check_interval"] = config.discreteCheckInterval;
    attrs["discrete_gap_threshold"] = config.discreteGapThreshold;
    attrs["discrete_repair_alpha_decay"] = config.discreteRepairAlphaDecay;
    attrs["worker_count"] = workerCount;
    return autil::legacy::ToJsonString(attrs, true);
}

std::string CostSchedulerGraphBuilder::makeWorkerConfig(size_t workerIndex)
{
    autil::legacy::json::JsonMap attrs;
    attrs["worker_index"] = workerIndex;
    return autil::legacy::ToJsonString(attrs, true);
}

std::string CostSchedulerGraphBuilder::makeBarrierConfig(size_t workerCount)
{
    autil::legacy::json::JsonMap attrs;
    attrs["worker_count"] = workerCount;
    return autil::legacy::ToJsonString(attrs, true);
}

bool CostSchedulerGraphBuilder::add(navi::GraphBuilder &builder, const navi::N &context,
                                    const navi::N &result,
                                    const std::string &schedulerKernelConfig)
{
    CostSchedulerConfig config;
    try {
        autil::legacy::FastFromJsonString(config, schedulerKernelConfig);
    } catch (const autil::legacy::ExceptionBase &) {
        return false;
    }
    if (!config.valid()) {
        return false;
    }
    auto workerCount = getWorkerCount(config.parallelism);
    return workerCount > 0 && add(builder, context, result, config, workerCount);
}

bool CostSchedulerGraphBuilder::add(navi::GraphBuilder &builder, const navi::N &context,
                                    const navi::N &result,
                                    const CostSchedulerConfig &config, size_t workerCount)
{
    auto prepare = builder.node("navi_sched_cost_prepare")
                       .kernel(CostSchedulerPrepareKernel::KERNEL_ID)
                       .jsonAttrs(makePrepareConfig(config, workerCount));
    context.out("context").to(prepare.in("context")).require(true);
    auto preparationBarrier =
        builder.node("navi_sched_cost_preparation_barrier")
            .kernel(CostSchedulerPreparationBarrierKernel::KERNEL_ID)
            .jsonAttrs(makeBarrierConfig(workerCount));
    for (size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex) {
        auto preparation =
            builder.node("navi_sched_cost_preparation_" + std::to_string(workerIndex))
                .kernel(CostSchedulerPreparationKernel::KERNEL_ID)
                .jsonAttrs(makeWorkerConfig(workerIndex));
        prepare.out("workers")
            .asGroup()
            .index(static_cast<navi::IndexType>(workerIndex))
            .to(preparation.in("state"))
            .require(true);
        preparation.out("done")
            .to(preparationBarrier.in("parts").asGroup().index(
                static_cast<navi::IndexType>(workerIndex)))
            .require(true);
    }
    auto loop = builder.node("navi_sched_cost_loop")
                    .kernel(CostSchedulerLoopKernel::KERNEL_ID)
                    .jsonAttrs(makeBarrierConfig(workerCount));
    preparationBarrier.out("ready")
        .to(loop.in("states").asGroup().index(0))
        .require(false);
    auto gradientBarrier = builder.node("navi_sched_cost_gradient_barrier")
                               .kernel(CostSchedulerGradientBarrierKernel::KERNEL_ID)
                               .jsonAttrs(makeBarrierConfig(workerCount));
    auto projectionBarrier = builder.node("navi_sched_cost_projection_barrier")
                                 .kernel(CostSchedulerProjectionBarrierKernel::KERNEL_ID)
                                 .jsonAttrs(makeBarrierConfig(workerCount));
    for (size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex) {
        auto row = builder.node("navi_sched_cost_row_" + std::to_string(workerIndex))
                       .kernel(CostSchedulerRowKernel::KERNEL_ID)
                       .jsonAttrs(makeWorkerConfig(workerIndex));
        loop.out("workers")
            .asGroup()
            .index(static_cast<navi::IndexType>(workerIndex))
            .to(row.in("state"))
            .require(true);
        row.out("done")
            .to(gradientBarrier.in("parts").asGroup().index(
                static_cast<navi::IndexType>(workerIndex)))
            .require(true);
        auto projection =
            builder.node("navi_sched_cost_projection_" + std::to_string(workerIndex))
                .kernel(CostSchedulerProjectionKernel::KERNEL_ID)
                .jsonAttrs(makeWorkerConfig(workerIndex));
        gradientBarrier.out("workers")
            .asGroup()
            .index(static_cast<navi::IndexType>(workerIndex))
            .to(projection.in("state"))
            .require(true);
        projection.out("done")
            .to(projectionBarrier.in("parts").asGroup().index(
                static_cast<navi::IndexType>(workerIndex)))
            .require(true);
    }
    projectionBarrier.out("loop").to(loop.in("states").asGroup().index(1)).require(false);
    auto finalize = builder.node("navi_sched_cost_finalize")
                        .kernel(CostSchedulerFinalizeKernel::KERNEL_ID);
    for (size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex) {
        auto row =
            builder.node("navi_sched_cost_final_row_" + std::to_string(workerIndex))
                .kernel(CostSchedulerFinalRowKernel::KERNEL_ID)
                .jsonAttrs(makeWorkerConfig(workerIndex));
        projectionBarrier.out("finish")
            .asGroup()
            .index(static_cast<navi::IndexType>(workerIndex))
            .to(row.in("state"))
            .require(true);
        row.out("done")
            .to(finalize.in("parts").asGroup().index(
                static_cast<navi::IndexType>(workerIndex)))
            .require(true);
    }
    finalize.out("result").to(result.in("result")).require(true);
    return true;
}

struct CostSchedulerGraphState::WorkerState {
    std::unique_ptr<CostSchedulerKernelInternal::NodeGradient[]> nodeGradients;
    std::unique_ptr<double[]> gradients;
    std::unique_ptr<double[]> projectionWorkspace;
    double forwardValue = 0.0;
    double finalForwardValue = 0.0;
    int64_t forwardUs = 0;
    int64_t gradientUs = 0;
    int64_t backwardUs = 0;
    int64_t updateUs = 0;
    int64_t projectionUs = 0;
    CostSchedulerKernelInternal::PreparationTiming preparationTiming;
    int64_t preparationStartUs = 0;
    int64_t preparationEndUs = 0;
    int64_t preparationCpuUs = 0;
    int64_t weightInitializationUs = 0;
    bool prepared = false;
};

class CostSchedulerGraphKernelUtil {
public:
    static bool getInput(navi::KernelComputeContext &ctx, CostSchedulerGraphDataPtr &data,
                         bool &eof)
    {
        navi::DataPtr input;
        if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), input, eof)) {
            return false;
        }
        data = std::dynamic_pointer_cast<CostSchedulerGraphData>(input);
        return true;
    }

    static bool getGroupInput(navi::KernelComputeContext &ctx,
                              CostSchedulerGraphDataPtr &data, bool &eof)
    {
        navi::GroupDatas inputs;
        if (!ctx.getGroupInput(0, inputs)) {
            return false;
        }
        eof = inputs.eof();
        for (const auto &input : inputs) {
            if (input.hasValue && input.data) {
                data = std::dynamic_pointer_cast<CostSchedulerGraphData>(input.data);
                return true;
            }
        }
        return true;
    }

    static navi::ErrorCode setWorkerOutputs(navi::KernelComputeContext &ctx,
                                            const CostSchedulerGraphDataPtr &data,
                                            size_t workerCount, bool eof)
    {
        for (size_t workerIndex = 0; workerIndex < workerCount; ++workerIndex) {
            if (!ctx.setOutput(
                    navi::PortIndex(static_cast<navi::IndexType>(workerIndex), 0), data,
                    eof)) {
                return navi::EC_ABORT;
            }
        }
        return navi::EC_NONE;
    }

    static navi::ErrorCode setOutput(navi::KernelComputeContext &ctx,
                                     const navi::DataPtr &data, bool eof)
    {
        if (!ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
            return navi::EC_ABORT;
        }
        return navi::EC_NONE;
    }

    static int64_t maxWorkerTime(
        const std::vector<std::unique_ptr<CostSchedulerGraphState::WorkerState>> &workers,
        int64_t CostSchedulerGraphState::WorkerState::*member)
    {
        int64_t result = 0;
        for (const auto &worker : workers) {
            result = std::max(result, worker.get()->*member);
        }
        return result;
    }

    static double sumWorkerValue(
        const std::vector<std::unique_ptr<CostSchedulerGraphState::WorkerState>> &workers,
        double CostSchedulerGraphState::WorkerState::*member)
    {
        double result = 0.0;
        for (const auto &worker : workers) {
            result += worker.get()->*member;
        }
        return result;
    }
};

std::shared_ptr<CostSchedulerGraphState> CostSchedulerGraphState::create(
    const NaviSchedScheduleContextPtr &context, const CostSchedulerConfig &config,
    size_t workerCount, std::shared_ptr<const std::vector<double>> randomWeights)
{
    auto state = std::shared_ptr<CostSchedulerGraphState>(
        new CostSchedulerGraphState(context, config, workerCount));
    return state->init(std::move(randomWeights)) ? state : nullptr;
}

CostSchedulerGraphState::CostSchedulerGraphState(NaviSchedScheduleContextPtr context,
                                                 CostSchedulerConfig config,
                                                 size_t workerCount)
    : _context(std::move(context)), _config(config), _workerCount(workerCount)
{
}

CostSchedulerGraphState::~CostSchedulerGraphState()
{
    if (_context) {
        _context->clearScheduleView();
    }
}

bool CostSchedulerGraphState::init(
    std::shared_ptr<const std::vector<double>> randomWeights)
{
    _preparationBeginUs = autil::TimeUtility::monotonicTimeUs();
    const auto initCpuBeginUs = autil::TimeUtility::cpuThreadTimeUs();
    _optimizationBeginUs = _preparationBeginUs;
    const auto initBeginUs = _preparationBeginUs;
    if (!_context || !_config.valid() || 0 == _workerCount) {
        return false;
    }
    if (_context->getScheduleBeginTimeUs() > 0) {
        _optimizationBeginUs = _context->getScheduleBeginTimeUs();
        _config.optimizationTimeBudgetUs =
            _context->getScheduleDeadlineUs() - _optimizationBeginUs;
    } else {
        _replayCollectionBaselineUs = _context->getReplayCollectionTimeUs();
    }
    _requestCount = _context->requestCount();
    _nodeCount = _context->nodeCount();
    if (0 == _requestCount || 0 == _nodeCount ||
        _nodeCount > std::numeric_limits<uint32_t>::max() ||
        _requestCount > std::numeric_limits<size_t>::max() / _nodeCount) {
        return false;
    }
    _weightCount = _requestCount * _nodeCount;
    _context->prepareCacheMatchPrefixes();
    _requestTokenCounts.reset(new int64_t[_requestCount]);
    _inverseRequestTokenCounts.reset(new double[_requestCount]);
    _totalTokenCount = CostSchedulerKernelInternal::fillRequestTokenCounts(
        *_context, _requestTokenCounts.get(), _inverseRequestTokenCounts.get(),
        _config.latencyTokenNormalizationPower);
    _tokenNormalizationDenominator =
        CostSchedulerKernelInternal::calculateTokenNormalizationDenominator(
            _requestTokenCounts.get(), _requestCount,
            _config.latencyTokenNormalizationPower);
    if (_tokenNormalizationDenominator <= 0.0) {
        return false;
    }
    if (_weightCount > std::numeric_limits<size_t>::max() / 2) {
        return false;
    }
    _costs.reset(new float[_weightCount * 2]);
    if (_context->isCollectingReplay()) {
        const auto replayBeginUs = autil::TimeUtility::monotonicTimeUs();
        _replayInfo = std::make_unique<CostSchedulerKernelInternal::ReplayInfo>();
        _replayInfo->init(true, _config.lambda, _config.alpha, _config.alphaDecay,
                          _config.minAlpha, static_cast<size_t>(_config.maxLoopCount),
                          _config.optimizationTimeBudgetUs);
        _replayInfo->minLoopCount = _config.minLoopCount;
        _replayInfo->parallelism = _workerCount;
        _replayInfo->latencyWeightConcavity = _config.latencyWeightConcavity;
        _replayInfo->latencyTokenNormalizationPower =
            _config.latencyTokenNormalizationPower;
        _replayInfo->initialWeightSeed = _config.initialWeightSeed;
        _replayInfo->discreteCheckInterval =
            static_cast<size_t>(_config.discreteCheckInterval);
        _replayInfo->discreteGapThreshold = _config.discreteGapThreshold;
        _replayInfo->discreteRepairAlphaDecay = _config.discreteRepairAlphaDecay;
        _context->addReplayCollectionTimeUs(autil::TimeUtility::monotonicTimeUs() -
                                            replayBeginUs);
    }
    _allowedLoopCount = static_cast<size_t>(_config.maxLoopCount);
    _weights.reset(new double[_weightCount]);
    _bestNodeIndexes.reset(new uint32_t[_requestCount]);
    _candidateNodeIndexes.reset(new uint32_t[_requestCount]);
    const auto randomWeightCount = randomWeights ? randomWeights->size() : 0;
    _parallelPreparation =
        randomWeightCount > 0 && 0 == (randomWeightCount & (randomWeightCount - 1));
    if (_parallelPreparation) {
        // Preparation workers can outlive the kernel that publishes the table.
        _randomWeights = std::move(randomWeights);
        _randomWeightCount = randomWeightCount;
    } else {
        CostSchedulerKernelInternal::initializeWeights(
            _weights.get(), _requestCount, _nodeCount, _config.initialWeightSeed);
    }
    if (_replayInfo) {
        const auto replayBeginUs = autil::TimeUtility::monotonicTimeUs();
        _gradients.reset(new double[_weightCount]);
        _weightsBeforeProjection.resize(_weightCount);
        _context->addReplayCollectionTimeUs(autil::TimeUtility::monotonicTimeUs() -
                                            replayBeginUs);
    }
    _workers.reserve(_workerCount);
    _gradientSpans.resize(_workerCount);
    for (size_t workerIndex = 0; workerIndex < _workerCount; ++workerIndex) {
        auto worker = std::make_unique<WorkerState>();
        auto nodes = nodeRange(workerIndex);
        _gradientSpans[workerIndex].nodeBegin = nodes.first;
        _gradientSpans[workerIndex].nodeEnd = nodes.second;
        auto localNodeCount = nodes.second - nodes.first;
        auto localWeightCount = localNodeCount * _requestCount;
        worker->nodeGradients.reset(
            new CostSchedulerKernelInternal::NodeGradient[localNodeCount]);
        worker->gradients.reset(new double[localWeightCount]);
        auto requests = requestRange(workerIndex);
        auto requestRangeSize = requests.second - requests.first;
        if (requestRangeSize > 0) {
            auto workspaceSize =
                _nodeCount *
                std::min(CostSchedulerKernelInternal::PROJECTION_REQUEST_TILE_SIZE,
                         requestRangeSize);
            worker->projectionWorkspace.reset(new double[workspaceSize]);
        }
        _workers.emplace_back(std::move(worker));
    }
    if (!_parallelPreparation) {
        beginOptimization();
    }
    _preparationInitCpuUs = autil::TimeUtility::cpuThreadTimeUs() - initCpuBeginUs;
    _preparationReadyUs = autil::TimeUtility::monotonicTimeUs();
    if (_context->_scheduleId < 10 || _context->_scheduleId % 100 == 0) {
        NAVI_KERNEL_LOG(INFO,
                        "[navi-cost-boundary-timing] schedule_id=%lu phase=init "
                        "activation_to_init_us=%ld init_us=%ld workers=%lu",
                        _context->_scheduleId, initBeginUs - _optimizationBeginUs,
                        autil::TimeUtility::monotonicTimeUs() - initBeginUs,
                        _workerCount);
    }
    return true;
}

void CostSchedulerGraphState::beginOptimization()
{
    _lastLoopEndUs = _optimizationBeginUs;
    _runBeginUs = bool(_replayInfo) ? _optimizationBeginUs : 0;
    _stepBeginUs = _runBeginUs;
    if (_replayInfo) {
        const auto replayBeginUs = autil::TimeUtility::monotonicTimeUs();
        _replayInfo->beginRun(_weights.get(), _weightCount);
        _weightsBeforeUpdate.assign(_weights.get(), _weights.get() + _weightCount);
        _weightsBeforeProjection = _weightsBeforeUpdate;
        _context->addReplayCollectionTimeUs(autil::TimeUtility::monotonicTimeUs() -
                                            replayBeginUs);
    }
}

bool CostSchedulerGraphState::prepareWorker(size_t workerIndex)
{
    if (failed() || workerIndex >= _workers.size()) {
        fail();
        return false;
    }
    auto &worker = *_workers[workerIndex];
    if (worker.prepared) {
        return true;
    }
    auto nodes = nodeRange(workerIndex);
    auto requests = requestRange(workerIndex);
    worker.preparationStartUs = autil::TimeUtility::monotonicTimeUs();
    const auto cpuBeginUs = autil::TimeUtility::cpuThreadTimeUs();
    if (!CostSchedulerKernelInternal::fillCostsForNodeRange(
            *_context, _requestTokenCounts.get(), nodes.first, nodes.second, _costs.get(),
            &worker.preparationTiming)) {
        fail();
        return false;
    }
    const auto weightsBeginUs = autil::TimeUtility::monotonicTimeUs();
    if (_parallelPreparation &&
        !CostSchedulerKernelInternal::initializeWeightRequestRange(
            _weights.get(), *_context, _randomWeights ? _randomWeights->data() : nullptr, _randomWeightCount, requests.first,
            requests.second)) {
        fail();
        return false;
    }
    worker.preparationCpuUs = autil::TimeUtility::cpuThreadTimeUs() - cpuBeginUs;
    worker.preparationEndUs = autil::TimeUtility::monotonicTimeUs();
    worker.weightInitializationUs = worker.preparationEndUs - weightsBeginUs;
    worker.prepared = true;
    return true;
}

bool CostSchedulerGraphState::finishPreparation()
{
    if (failed()) {
        return false;
    }
    for (const auto &worker : _workers) {
        if (!worker->prepared) {
            fail();
            return false;
        }
    }
    const auto endUs = autil::TimeUtility::monotonicTimeUs();
    _context->_costPrepareUs = endUs - _preparationBeginUs;
    if (_context->_costPrepareUs >= 10'000) {
        static std::atomic<int64_t> lastLogUs{0};
        auto previousLogUs = lastLogUs.load(std::memory_order_relaxed);
        if (endUs - previousLogUs >= 1'000'000 &&
            lastLogUs.compare_exchange_strong(previousLogUs, endUs,
                                              std::memory_order_relaxed)) {
            size_t slowestWorker = 0;
            int64_t workerWallSumUs = 0, workerCpuSumUs = 0;
            int64_t matchSumUs = 0, costSumUs = 0, weightsSumUs = 0;
            int64_t dispatchWaitMaxUs = 0, lastWorkerEndUs = 0;
            for (size_t index = 0; index < _workers.size(); ++index) {
                const auto &worker = *_workers[index];
                const auto wallUs = worker.preparationEndUs - worker.preparationStartUs;
                const auto &slowest = *_workers[slowestWorker];
                if (wallUs > slowest.preparationEndUs - slowest.preparationStartUs) {
                    slowestWorker = index;
                }
                workerWallSumUs += wallUs;
                workerCpuSumUs += worker.preparationCpuUs;
                matchSumUs += worker.preparationTiming.matchUs;
                costSumUs += worker.preparationTiming.costEvaluationUs;
                weightsSumUs += worker.weightInitializationUs;
                dispatchWaitMaxUs = std::max(
                    dispatchWaitMaxUs, worker.preparationStartUs - _preparationReadyUs);
                lastWorkerEndUs = std::max(lastWorkerEndUs, worker.preparationEndUs);
            }
            const auto &slowest = *_workers[slowestWorker];
            const auto nodes = nodeRange(slowestWorker);
            NAVI_KERNEL_LOG(
                INFO,
                "[navi-cost-prepare-timing] schedule_id=%lu requests=%lu nodes=%lu "
                "workers=%lu "
                "total_us=%ld init_us=%ld init_cpu_us=%ld dispatch_wait_max_us=%ld "
                "worker_wall_sum_us=%ld worker_cpu_sum_us=%ld match_sum_us=%ld "
                "cost_eval_sum_us=%ld weights_sum_us=%ld barrier_wait_us=%ld "
                "slowest_worker=%lu slowest_node_begin=%lu slowest_node_end=%lu "
                "slowest_wall_us=%ld slowest_cpu_us=%ld slowest_match_us=%ld "
                "slowest_cost_eval_us=%ld slowest_weights_us=%ld collect_replay=%d",
                _context->_scheduleId, _requestCount, _nodeCount, _workerCount,
                _context->_costPrepareUs, _preparationReadyUs - _preparationBeginUs,
                _preparationInitCpuUs, dispatchWaitMaxUs, workerWallSumUs, workerCpuSumUs,
                matchSumUs, costSumUs, weightsSumUs, endUs - lastWorkerEndUs,
                slowestWorker, nodes.first, nodes.second,
                slowest.preparationEndUs - slowest.preparationStartUs,
                slowest.preparationCpuUs, slowest.preparationTiming.matchUs,
                slowest.preparationTiming.costEvaluationUs,
                slowest.weightInitializationUs, _context->isCollectingReplay() ? 1 : 0);
        }
    }
    if (_parallelPreparation) {
        beginOptimization();
    }
    if (_context->_scheduleId < 10 || _context->_scheduleId % 100 == 0) {
        NAVI_KERNEL_LOG(INFO,
                        "[navi-cost-boundary-timing] schedule_id=%lu phase=prepared "
                        "activation_elapsed_us=%ld",
                        _context->_scheduleId,
                        autil::TimeUtility::monotonicTimeUs() - _optimizationBeginUs);
    }
    return true;
}

std::pair<size_t, size_t> CostSchedulerGraphState::nodeRange(size_t workerIndex) const
{
    return {_nodeCount * workerIndex / _workerCount,
            _nodeCount * (workerIndex + 1) / _workerCount};
}

std::pair<size_t, size_t> CostSchedulerGraphState::requestRange(size_t workerIndex) const
{
    return {_requestCount * workerIndex / _workerCount,
            _requestCount * (workerIndex + 1) / _workerCount};
}

void CostSchedulerGraphState::fail() { _failed.store(true, std::memory_order_relaxed); }

bool CostSchedulerGraphState::updateBestResult(double loss)
{
    if (!std::isfinite(loss)) {
        return false;
    }
    if (loss < _bestLoss) {
        _bestLoss = loss;
        _bestNodeIndexes.swap(_candidateNodeIndexes);
    }
    return true;
}

bool CostSchedulerGraphState::evaluateDiscreteCandidate(size_t iteration,
                                                        double continuousLoss)
{
    auto discreteLoss = CostSchedulerKernelInternal::forwardDiscrete(
        *_context, _costs.get(), _candidateNodeIndexes.get(), _requestTokenCounts.get(),
        _inverseRequestTokenCounts.get(), _tokenNormalizationDenominator, _config.lambda,
        _config.latencyWeightConcavity, _discreteForwardWorkspace);
    _discreteEvaluated = true;
    _discreteForwardValue = discreteLoss;
    if (!updateBestResult(discreteLoss)) {
        return false;
    }
    auto gap = (discreteLoss - continuousLoss) /
               std::max(std::abs(continuousLoss), std::numeric_limits<double>::epsilon());
    _discreteGap = gap;
    if (!_discreteRepairActive &&
        iteration % static_cast<size_t>(_config.discreteCheckInterval) != 0) {
        return true;
    }
    if (gap > _config.discreteGapThreshold) {
        _repairAlpha = std::max(_config.minAlpha,
                                (_discreteRepairActive ? _repairAlpha : _stepAlpha) *
                                    _config.discreteRepairAlphaDecay);
        if (_repairAlpha <= 0.0) {
            _repairAlpha = std::numeric_limits<double>::min();
        }
        _discreteRepairActive = true;
    }
    if (_discreteRepairActive) {
        _stepAlpha = _repairAlpha;
    }
    return true;
}

bool CostSchedulerGraphState::runRow(size_t workerIndex, size_t iteration)
{
    if (failed() || workerIndex >= _workers.size()) {
        fail();
        return false;
    }
    if (iteration >= _allowedLoopCount) {
        return true;
    }
    if (!prepareWorker(workerIndex)) {
        return false;
    }
    if (iteration == 0) {
        const auto requests = requestRange(workerIndex);
        if (!CostSchedulerKernelInternal::selectWeightRequestRange(
                _weights.get(), _requestCount, _nodeCount, requests.first,
                requests.second, _candidateNodeIndexes.get(),
                _config.initialWeightSeed)) {
            fail();
            return false;
        }
    }
    auto range = nodeRange(workerIndex);
    if (range.first == range.second) {
        return true;
    }
    auto &worker = *_workers[workerIndex];
    auto beginUs = autil::TimeUtility::monotonicTimeUs();
    worker.forwardValue = CostSchedulerKernelInternal::forwardNodeRange(
        *_context, _costs.get(), _weights.get(), _requestTokenCounts.get(),
        _inverseRequestTokenCounts.get(), _tokenNormalizationDenominator, _config.lambda,
        _config.latencyWeightConcavity, range.first, range.second,
        worker.nodeGradients.get(), nullptr, nullptr);
    auto forwardEndUs = bool(_replayInfo) ? autil::TimeUtility::monotonicTimeUs() : 0;
    if (!std::isfinite(worker.forwardValue)) {
        fail();
        return false;
    }
    CostSchedulerKernelInternal::backwardNodeRange(
        *_context, _costs.get(), _weights.get(), _requestTokenCounts.get(),
        _inverseRequestTokenCounts.get(), _requestCount, range.first, range.second,
        worker.nodeGradients.get(), worker.gradients.get());
    auto backwardEndUs = autil::TimeUtility::monotonicTimeUs();
    worker.gradientUs = backwardEndUs - beginUs;
    auto gradients = worker.gradients.get();
    _gradientSpans[workerIndex].values = gradients;
    auto updateEndUs = bool(_replayInfo) ? autil::TimeUtility::monotonicTimeUs() : 0;
    worker.forwardUs = forwardEndUs - beginUs;
    worker.backwardUs = backwardEndUs - forwardEndUs;
    worker.updateUs = updateEndUs - backwardEndUs;
    return true;
}

bool CostSchedulerGraphState::finishRow(size_t iteration)
{
    if (failed()) {
        return false;
    }
    if (iteration >= _allowedLoopCount) {
        return true;
    }
    _forwardValue = CostSchedulerGraphKernelUtil::sumWorkerValue(
        _workers, &WorkerState::forwardValue);
    if (!std::isfinite(_forwardValue)) {
        fail();
        return false;
    }
    _discreteEvaluated = false;
    if (iteration == 0) {
        const auto initialLoss = CostSchedulerKernelInternal::forwardDiscrete(
            *_context, _costs.get(), _candidateNodeIndexes.get(),
            _requestTokenCounts.get(), _inverseRequestTokenCounts.get(),
            _tokenNormalizationDenominator, _config.lambda,
            _config.latencyWeightConcavity, _discreteForwardWorkspace);
        if (!updateBestResult(initialLoss)) {
            fail();
            return false;
        }
    }
    _stepAlpha = CostSchedulerKernelInternal::getStepAlpha(
        _config.alpha, _config.alphaDecay, _config.minAlpha, iteration);
    if (iteration > 0 && !evaluateDiscreteCandidate(iteration, _forwardValue)) {
        fail();
        return false;
    }
    if (_replayInfo) {
        auto updateBeginUs = autil::TimeUtility::monotonicTimeUs();
        for (size_t workerIndex = 0; workerIndex < _workerCount; ++workerIndex) {
            const auto nodes = nodeRange(workerIndex);
            std::copy(_workers[workerIndex]->gradients.get(),
                      _workers[workerIndex]->gradients.get() +
                          (nodes.second - nodes.first) * _requestCount,
                      _gradients.get() + nodes.first * _requestCount);
        }
        _weightsBeforeProjection = _weightsBeforeUpdate;
        if (!CostSchedulerKernelInternal::applyGradient(_weightsBeforeProjection.data(),
                                                        0, _weightCount, _gradients.get(),
                                                        _stepAlpha)) {
            fail();
            return false;
        }
        auto updateUs = autil::TimeUtility::monotonicTimeUs() - updateBeginUs;
        _replayInfo->beginStep(iteration, _stepAlpha, _forwardValue, _gradients.get(),
                               _weightsBeforeUpdate.data(), _weightCount,
                               CostSchedulerGraphKernelUtil::maxWorkerTime(
                                   _workers, &WorkerState::forwardUs),
                               CostSchedulerGraphKernelUtil::maxWorkerTime(
                                   _workers, &WorkerState::backwardUs));
        if (_discreteEvaluated) {
            _replayInfo->recordDiscreteEvaluation(_discreteForwardValue, _discreteGap,
                                                  _discreteRepairActive);
        }
        _replayInfo->recordBeforeProjection(_weightsBeforeProjection.data(), _weightCount,
                                            updateUs);
        _context->addReplayCollectionTimeUs(autil::TimeUtility::monotonicTimeUs() -
                                            updateBeginUs);
    }
    return true;
}

bool CostSchedulerGraphState::runProjection(size_t workerIndex, size_t iteration)
{
    if (failed() || workerIndex >= _workers.size()) {
        fail();
        return false;
    }
    if (iteration >= _allowedLoopCount) {
        return true;
    }
    auto range = requestRange(workerIndex);
    if (range.first == range.second) {
        return true;
    }
    auto &worker = *_workers[workerIndex];
    auto beginUs = autil::TimeUtility::monotonicTimeUs();
    if (!CostSchedulerKernelInternal::projectRequestRange(
            _weights.get(), _requestCount, _nodeCount, range.first, range.second,
            worker.projectionWorkspace.get(), _candidateNodeIndexes.get(),
            _gradientSpans.data(), _gradientSpans.size(), _stepAlpha,
            _config.initialWeightSeed)) {
        fail();
        return false;
    }
    worker.projectionUs = autil::TimeUtility::monotonicTimeUs() - beginUs;
    return true;
}

bool CostSchedulerGraphState::finishProjection(size_t iteration)
{
    if (failed()) {
        return false;
    }
    if (iteration >= _allowedLoopCount) {
        return true;
    }
    auto nowUs = autil::TimeUtility::monotonicTimeUs();
    _actualLoopCount = iteration + 1;
    _context->recordCostGradientIteration(
        CostSchedulerGraphKernelUtil::maxWorkerTime(_workers, &WorkerState::gradientUs) +
        CostSchedulerGraphKernelUtil::maxWorkerTime(_workers,
                                                    &WorkerState::projectionUs));
    if (_replayInfo) {
        const auto replayBeginUs = autil::TimeUtility::monotonicTimeUs();
        _replayInfo->recordAfterProjection(_weights.get(), _weightCount,
                                           CostSchedulerGraphKernelUtil::maxWorkerTime(
                                               _workers, &WorkerState::projectionUs),
                                           0, nowUs - _stepBeginUs);
        if (_actualLoopCount < static_cast<size_t>(_config.maxLoopCount)) {
            _weightsBeforeUpdate.assign(_weights.get(), _weights.get() + _weightCount);
            _weightsBeforeProjection = _weightsBeforeUpdate;
            _stepBeginUs = nowUs;
        }
        _context->addReplayCollectionTimeUs(autil::TimeUtility::monotonicTimeUs() -
                                            replayBeginUs);
        nowUs = autil::TimeUtility::monotonicTimeUs();
    }
    const auto replayCollectionUs =
        _context->getReplayCollectionTimeUs() - _replayCollectionBaselineUs;
    auto elapsedUs =
        std::max<int64_t>(0, nowUs - _optimizationBeginUs);
    auto lastLoopUs = std::max<int64_t>(
        0, nowUs - _lastLoopEndUs);
    _lastLoopEndUs = nowUs;
    _lastLoopReplayCollectionUs = replayCollectionUs;
    if (iteration == 0 &&
        (_context->_scheduleId < 10 || _context->_scheduleId % 100 == 0)) {
        NAVI_KERNEL_LOG(
            INFO,
            "[navi-cost-boundary-timing] schedule_id=%lu phase=first_projection "
            "activation_elapsed_us=%ld forward_worker_max_us=%ld "
            "backward_worker_max_us=%ld projection_worker_max_us=%ld",
            _context->_scheduleId, elapsedUs,
            CostSchedulerGraphKernelUtil::maxWorkerTime(_workers,
                                                        &WorkerState::forwardUs),
            CostSchedulerGraphKernelUtil::maxWorkerTime(_workers,
                                                        &WorkerState::backwardUs),
            CostSchedulerGraphKernelUtil::maxWorkerTime(_workers,
                                                        &WorkerState::projectionUs));
    }
    if (!CostSchedulerKernelInternal::shouldRunNextLoop(
            _actualLoopCount, static_cast<size_t>(_config.maxLoopCount),
            _config.optimizationTimeBudgetUs, elapsedUs, lastLoopUs,
            _config.minLoopCount)) {
        _allowedLoopCount = _actualLoopCount;
        _stopReason = _actualLoopCount >= static_cast<size_t>(_config.maxLoopCount)
                          ? "max_loop_count"
                          : "time_budget";
    }
    _context->recordCostOptimizerProgress(_actualLoopCount,
                                          static_cast<size_t>(_config.maxLoopCount),
                                          _stopReason == "time_budget");
    return true;
}

bool CostSchedulerGraphState::shouldContinue() const
{
    return !failed() && _actualLoopCount < _allowedLoopCount;
}

bool CostSchedulerGraphState::runFinalRow(size_t workerIndex)
{
    if (failed() || workerIndex >= _workers.size()) {
        fail();
        return false;
    }
    auto range = nodeRange(workerIndex);
    if (range.first == range.second) {
        return true;
    }
    auto &worker = *_workers[workerIndex];
    worker.finalForwardValue = CostSchedulerKernelInternal::forwardNodeRange(
        *_context, _costs.get(), _weights.get(), _requestTokenCounts.get(),
        _inverseRequestTokenCounts.get(), _tokenNormalizationDenominator, _config.lambda,
        _config.latencyWeightConcavity, range.first, range.second, nullptr, nullptr,
        nullptr);
    if (!std::isfinite(worker.finalForwardValue)) {
        fail();
        return false;
    }
    return true;
}

bool CostSchedulerGraphState::finalize(int64_t replayRefinementEvaluations)
{
    const auto finalizeBeginUs = autil::TimeUtility::monotonicTimeUs();
    const auto optimizationEndUs = autil::TimeUtility::monotonicTimeUs();
    if (failed()) {
        _context->fail("scheduler_error");
        return false;
    }
    _forwardValue = CostSchedulerGraphKernelUtil::sumWorkerValue(
        _workers, &WorkerState::finalForwardValue);
    auto finalForwardValue = _forwardValue;
    auto finalDiscreteLoss = CostSchedulerKernelInternal::forwardDiscrete(
        *_context, _costs.get(), _candidateNodeIndexes.get(), _requestTokenCounts.get(),
        _inverseRequestTokenCounts.get(), _tokenNormalizationDenominator, _config.lambda,
        _config.latencyWeightConcavity, _discreteForwardWorkspace);
    if (!updateBestResult(finalDiscreteLoss)) {
        fail();
        _context->fail("scheduler_error");
        return false;
    }
    const auto scoringEndUs = autil::TimeUtility::monotonicTimeUs();
    const auto refinementBudget =
        replayRefinementEvaluations >= 0
            ? -1
            : CostSchedulerKernelInternal::discreteRefinementBudgetUs(
                  _config.optimizationTimeBudgetUs,
                  autil::TimeUtility::monotonicTimeUs() - _optimizationBeginUs);
    const auto maxEvaluations = replayRefinementEvaluations >= 0
                                    ? static_cast<size_t>(replayRefinementEvaluations)
                                    : std::numeric_limits<size_t>::max();
    CostSchedulerKernelInternal::DiscreteRefinementStats refinementStats;
    const auto auditAssignment = _context->_scheduleId % 100 == 0;
    std::vector<size_t> initialNodeRequests;
    const auto initialLoss = _bestLoss;
    if (auditAssignment) {
        initialNodeRequests.assign(_nodeCount, 0);
        for (size_t i = 0; i < _requestCount; ++i) {
            ++initialNodeRequests[_bestNodeIndexes[i]];
        }
    }
    if (refinementBudget != 0 &&
        !CostSchedulerKernelInternal::refineDiscreteAssignment(
            *_context, _costs.get(), _bestNodeIndexes.get(), _requestTokenCounts.get(),
            _inverseRequestTokenCounts.get(), _tokenNormalizationDenominator,
            _config.lambda, _config.latencyWeightConcavity, _bestLoss, refinementBudget,
            maxEvaluations, refinementStats)) {
        fail();
        _context->fail("scheduler_error");
        return false;
    }
    const auto refinementEndUs = autil::TimeUtility::monotonicTimeUs();
    if (!CostSchedulerKernelInternal::selectNodes(*_context, _bestNodeIndexes.get()) ||
        !_context->setScheduleLoss(_bestLoss)) {
        fail();
        _context->fail("scheduler_error");
        return false;
    }
    _context->recordDiscreteSearchLosses(initialLoss, _bestLoss);
    _forwardValue = _bestLoss;
    if (auditAssignment) {
        std::vector<size_t> finalNodeRequests(_nodeCount, 0);
        for (size_t i = 0; i < _requestCount; ++i) {
            ++finalNodeRequests[_bestNodeIndexes[i]];
        }
        NAVI_KERNEL_LOG(
            INFO,
            "[navi-geometry-search-audit] path=parallel_graph schedule_id=%lu "
            "schedule_time_us=%ld requests=%lu candidates=%lu budget_us=%ld "
            "evaluations=%lu moves=%lu budget_exhausted=%d "
            "initial_loss=%.9f final_loss=%.9f gradient_loops=%lu stop_reason=%s",
            _context->_scheduleId, _context->getScheduleTimeUs(), _requestCount,
            _nodeCount, refinementBudget, refinementStats.evaluations,
            refinementStats.moves, refinementStats.budgetExhausted, initialLoss,
            _bestLoss, _actualLoopCount, _stopReason.c_str());
        for (size_t i = 0; i < _nodeCount; ++i) {
            const auto &node = _context->getNode(i);
            NAVI_KERNEL_LOG(
                INFO,
                "[navi-geometry-candidate-audit] path=parallel_graph schedule_id=%lu "
                "schedule_time_us=%ld node_index=%lu agent=%s inflight=%d "
                "queue_ms=%.9f model_revision=%ld initial_requests=%lu "
                "final_requests=%lu",
                _context->_scheduleId, _context->getScheduleTimeUs(), i,
                node.getAgentId().c_str(), node.getInflightRequests(),
                node.getEstimatedQueueWaitMs(), node.getLatencyModelRevision(),
                initialNodeRequests[i], finalNodeRequests[i]);
        }
    }
    const auto replayBeginUs = autil::TimeUtility::monotonicTimeUs();
    if (_replayInfo) {
        _replayInfo->finishRun(_weights.get(), _weightCount, finalForwardValue, _bestLoss,
                               _bestNodeIndexes.get(), _requestCount,
                               autil::TimeUtility::monotonicTimeUs() - _runBeginUs,
                               _actualLoopCount, _stopReason, refinementStats.evaluations,
                               refinementStats.moves);
        _replayInfo->finish(*_context, _costs.get(), _weights.get(), _requestCount,
                            _nodeCount);
    }
    const auto replayEndUs = autil::TimeUtility::monotonicTimeUs();
    if (_context->_scheduleId < 10 || _context->_scheduleId % 100 == 0) {
        NAVI_KERNEL_LOG(
            INFO,
            "[navi-cost-phase-timing] schedule_id=%lu requests=%lu candidates=%lu "
            "prepare_gradient_graph_us=%ld exact_score_us=%ld refinement_us=%ld "
            "select_audit_us=%ld replay_json_us=%ld finalize_total_us=%ld "
            "gradient_loops=%lu replay=%d",
            _context->_scheduleId, _requestCount, _nodeCount,
            optimizationEndUs - _optimizationBeginUs, scoringEndUs - finalizeBeginUs,
            refinementEndUs - scoringEndUs, replayBeginUs - refinementEndUs,
            replayEndUs - replayBeginUs, replayEndUs - finalizeBeginUs, _actualLoopCount,
            static_cast<int>(bool(_replayInfo)));
    }
    _context->clearScheduleView();
    return true;
}

bool CostSchedulerGraphState::resetWeights(const double *weights, size_t weightCount)
{
    if (!weights || weightCount != _weightCount || failed()) {
        return false;
    }
    std::copy(weights, weights + weightCount, _weights.get());
    if (_replayInfo) {
        _weightsBeforeUpdate.assign(weights, weights + weightCount);
        _replayInfo->optimizationRuns.back().initialWeights.assign(weights,
                                                                   weights + weightCount);
    }
    return true;
}

const NaviSchedScheduleContextPtr &CostSchedulerGraphState::context() const
{
    return _context;
}

const double *CostSchedulerGraphState::weights() const { return _weights.get(); }

const double *CostSchedulerGraphState::gradients() const { return _gradients.get(); }

size_t CostSchedulerGraphState::weightCount() const { return _weightCount; }

double CostSchedulerGraphState::forwardValue() const { return _forwardValue; }

bool CostSchedulerGraphState::failed() const
{
    return _failed.load(std::memory_order_relaxed);
}

size_t CostSchedulerGraphState::actualLoopCount() const { return _actualLoopCount; }

const std::string &CostSchedulerGraphState::stopReason() const { return _stopReason; }

CostSchedulerGraphData::CostSchedulerGraphData(CostSchedulerGraphStatePtr stateValue,
                                               NaviSchedScheduleContextPtr contextValue)
    : navi::Data(COST_SCHEDULER_GRAPH_DATA_TYPE_ID)
    , state(std::move(stateValue))
    , context(state ? state->context() : std::move(contextValue))
{
}

class CostSchedulerGraphType final : public navi::Type {
public:
    CostSchedulerGraphType()
        : navi::Type(NAVI_TYPE_PARAM, COST_SCHEDULER_GRAPH_DATA_TYPE_ID)
    {
    }
};

void CostSchedulerPrepareKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .input("context", MASTER_SCHEDULE_CONTEXT_TYPE_ID)
        .outputGroup("workers", COST_SCHEDULER_GRAPH_DATA_TYPE_ID);
}

CostSchedulerPrepareKernel::CostSchedulerPrepareKernel()
    : _randomGenerator(autil::TimeUtility::currentTimeInMicroSeconds())
{
    (void)_randomGenerator.get();
}

bool CostSchedulerPrepareKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "lambda", _config.lambda, _config.lambda);
    NAVI_JSONIZE(ctx, "alpha", _config.alpha, _config.alpha);
    NAVI_JSONIZE(ctx, "alpha_decay", _config.alphaDecay, _config.alphaDecay);
    NAVI_JSONIZE(ctx, "min_alpha", _config.minAlpha, _config.minAlpha);
    NAVI_JSONIZE(ctx, "min_loop_count", _config.minLoopCount, _config.minLoopCount);
    NAVI_JSONIZE(ctx, "max_loop_count", _config.maxLoopCount, _config.maxLoopCount);
    NAVI_JSONIZE(ctx, "optimization_time_budget_us", _config.optimizationTimeBudgetUs,
                 _config.optimizationTimeBudgetUs);
    NAVI_JSONIZE(ctx, "parallelism", _config.parallelism, _config.parallelism);
    NAVI_JSONIZE(ctx, "latency_weight_concavity", _config.latencyWeightConcavity,
                 _config.latencyWeightConcavity);
    NAVI_JSONIZE(ctx, "latency_token_normalization_power",
                 _config.latencyTokenNormalizationPower,
                 _config.latencyTokenNormalizationPower);
    NAVI_JSONIZE(ctx, "initial_weight_seed", _config.initialWeightSeed,
                 _config.initialWeightSeed);
    NAVI_JSONIZE(ctx, "discrete_check_interval", _config.discreteCheckInterval,
                 _config.discreteCheckInterval);
    NAVI_JSONIZE(ctx, "discrete_gap_threshold", _config.discreteGapThreshold,
                 _config.discreteGapThreshold);
    NAVI_JSONIZE(ctx, "discrete_repair_alpha_decay", _config.discreteRepairAlphaDecay,
                 _config.discreteRepairAlphaDecay);
    NAVI_JSONIZE(ctx, "worker_count", _workerCount, _workerCount);
    return _config.valid() && _workerCount > 0;
}

navi::ErrorCode CostSchedulerPrepareKernel::init(navi::KernelInitContext &)
{
    NAVI_KERNEL_LOG(INFO,
                    "cost_scheduler_effective_config {\"lambda\":%.17g,"
                    "\"alpha\":%.17g,\"alpha_decay\":%.17g,"
                    "\"min_alpha\":%.17g,\"max_loop_count\":%d,"
                    "\"optimization_time_budget_us\":%ld,\"parallelism\":%d,"
                    "\"worker_count\":%lu,\"latency_weight_concavity\":%.17g,"
                    "\"latency_token_normalization_power\":%.17g,"
                    "\"initial_weight_seed\":%lu,"
                    "\"discrete_check_interval\":%d,"
                    "\"discrete_gap_threshold\":%.17g,"
                    "\"discrete_repair_alpha_decay\":%.17g}",
                    _config.lambda, _config.alpha, _config.alphaDecay, _config.minAlpha,
                    _config.maxLoopCount, _config.optimizationTimeBudgetUs,
                    _config.parallelism, _workerCount, _config.latencyWeightConcavity,
                    _config.latencyTokenNormalizationPower, _config.initialWeightSeed,
                    _config.discreteCheckInterval, _config.discreteGapThreshold,
                    _config.discreteRepairAlphaDecay);
    _randomWeightSeed = _config.initialWeightSeed > 0
                            ? _config.initialWeightSeed
                            : static_cast<uint64_t>(autil::TimeUtility::currentTime());
    std::mt19937_64 random(_randomWeightSeed);
    std::uniform_real_distribution<double> distribution(1.0, 2.0);
    auto randomWeights = std::make_shared<std::vector<double>>(RANDOM_WEIGHT_COUNT);
    for (auto &weight : *randomWeights) {
        weight = distribution(random);
    }
    _randomWeights = std::move(randomWeights);
    return navi::EC_NONE;
}

navi::ErrorCode CostSchedulerPrepareKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr input;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), input, eof)) {
        return navi::EC_NONE;
    }
    CostSchedulerGraphDataPtr output;
    if (input) {
        auto config = _schedulerConfigR ? _schedulerConfigR->get().config.cost : _config;
        auto randomWeightSeed =
            config.initialWeightSeed > 0 ? config.initialWeightSeed : _randomWeightSeed;
        if (randomWeightSeed != _randomWeightSeed) {
            std::mt19937_64 random(randomWeightSeed);
            std::uniform_real_distribution<double> distribution(1.0, 2.0);
            auto randomWeights =
                std::make_shared<std::vector<double>>(RANDOM_WEIGHT_COUNT);
            for (auto &weight : *randomWeights) {
                weight = distribution(random);
            }
            _randomWeights = std::move(randomWeights);
            _randomWeightSeed = randomWeightSeed;
        }
        auto context = MasterScheduleContextData::getContext(input);
        if (!SchedulerBaseKernel::validateScheduleInput(context)) {
            output = std::make_shared<CostSchedulerGraphData>(nullptr, context);
        } else {
            CostSchedulerColdStartStrategy::Result coldStartResult;
            const auto coldStartApplied =
                context->_coldStartRoutingStats.evaluated ||
                CostSchedulerColdStartStrategy::apply(*context, _randomGenerator,
                                                      coldStartResult);
            if (coldStartApplied) {
                NAVI_KERNEL_LOG(DEBUG,
                                "cold-start routing evaluated, targetInflight [%.3f], "
                                "trustedNodes [%lu], untrustedNodes [%lu], "
                                "trustedRequests [%lu], untrustedRequests [%lu], "
                                "rejectedRequests [%lu], untrustedCapacity [%lu]",
                                context->_coldStartRoutingStats.targetInflight,
                                context->_coldStartRoutingStats.trustedNodeCount,
                                context->_coldStartRoutingStats.untrustedNodeCount,
                                context->_coldStartRoutingStats.trustedRequestCount,
                                context->_coldStartRoutingStats.untrustedRequestCount,
                                context->_coldStartRoutingStats.rejectedRequestCount,
                                context->_coldStartRoutingStats.untrustedCapacity);
            }
            if (!coldStartApplied) {
                context->fail("scheduler_error");
                output = std::make_shared<CostSchedulerGraphData>(nullptr, context);
            } else if (!context->_scheduleViewActive) {
                context->setScheduleLoss(0.0);
                output = std::make_shared<CostSchedulerGraphData>(nullptr, context);
            } else {
                auto state = CostSchedulerGraphState::create(
                    context, config, _workerCount, _randomWeights);
                if (!state) {
                    context->fail("scheduler_error");
                    output = std::make_shared<CostSchedulerGraphData>(nullptr, context);
                } else {
                    output = std::make_shared<CostSchedulerGraphData>(std::move(state));
                }
            }
        }
    }
    return CostSchedulerGraphKernelUtil::setWorkerOutputs(ctx, output, _workerCount, eof);
}

void CostSchedulerPreparationKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .input("state", COST_SCHEDULER_GRAPH_DATA_TYPE_ID)
        .output("done", COST_SCHEDULER_GRAPH_DATA_TYPE_ID);
}

bool CostSchedulerPreparationKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "worker_index", _workerIndex, _workerIndex);
    return true;
}

navi::ErrorCode CostSchedulerPreparationKernel::compute(navi::KernelComputeContext &ctx)
{
    CostSchedulerGraphDataPtr data;
    bool eof = false;
    if (!CostSchedulerGraphKernelUtil::getInput(ctx, data, eof)) {
        return navi::EC_NONE;
    }
    if (data && data->state) {
        data->state->prepareWorker(_workerIndex);
    }
    return CostSchedulerGraphKernelUtil::setOutput(ctx, data, eof);
}

void CostSchedulerPreparationBarrierKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .inputGroup("parts", COST_SCHEDULER_GRAPH_DATA_TYPE_ID, navi::IT_REQUIRE,
                    navi::IT_REQUIRE)
        .output("ready", COST_SCHEDULER_GRAPH_DATA_TYPE_ID);
}

bool CostSchedulerPreparationBarrierKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "worker_count", _workerCount, _workerCount);
    return _workerCount > 0;
}

navi::ErrorCode CostSchedulerPreparationBarrierKernel::compute(
    navi::KernelComputeContext &ctx)
{
    CostSchedulerGraphDataPtr data;
    bool eof = false;
    if (!CostSchedulerGraphKernelUtil::getGroupInput(ctx, data, eof)) {
        return navi::EC_NONE;
    }
    if (data && data->state) {
        data->state->finishPreparation();
    }
    return CostSchedulerGraphKernelUtil::setOutput(ctx, data, eof);
}

void CostSchedulerLoopKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .inputGroup("states", COST_SCHEDULER_GRAPH_DATA_TYPE_ID, navi::IT_OPTIONAL,
                    navi::IT_REQUIRE)
        .outputGroup("workers", COST_SCHEDULER_GRAPH_DATA_TYPE_ID);
}

bool CostSchedulerLoopKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "worker_count", _workerCount, _workerCount);
    return _workerCount > 0;
}

navi::ErrorCode CostSchedulerLoopKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::GroupDatas inputs;
    if (!ctx.getGroupInput(0, inputs) || inputs.size() != 2) {
        return navi::EC_ABORT;
    }
    CostSchedulerGraphDataPtr feedback;
    for (size_t index = 0; index < inputs.size(); ++index) {
        const auto &input = inputs[index];
        if (!input.hasValue) {
            continue;
        }
        auto data = std::dynamic_pointer_cast<CostSchedulerGraphData>(input.data);
        if (0 == index) {
            if (data) {
                _pending.emplace_back(std::move(data));
            }
            _sourceEof = _sourceEof || input.eof;
        } else {
            feedback = std::move(data);
        }
    }
    CostSchedulerGraphDataPtr output;
    if (feedback) {
        if (!_active) {
            return navi::EC_ABORT;
        }
        if (feedback->continueLoop) {
            output = std::move(feedback);
        } else {
            _active = false;
        }
    }
    if (!output && !_active && !_pending.empty()) {
        output = std::move(_pending.front());
        _pending.pop_front();
        _active = true;
    }
    if (output) {
        return CostSchedulerGraphKernelUtil::setWorkerOutputs(ctx, output, _workerCount,
                                                              false);
    }
    if (_sourceEof && !_active && _pending.empty() && !_outputEof) {
        _outputEof = true;
        return CostSchedulerGraphKernelUtil::setWorkerOutputs(ctx, nullptr, _workerCount,
                                                              true);
    }
    if (_outputEof) {
        ctx.setIgnoreDeadlock();
    }
    return navi::EC_NONE;
}

void CostSchedulerRowKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .input("state", COST_SCHEDULER_GRAPH_DATA_TYPE_ID)
        .output("done", COST_SCHEDULER_GRAPH_DATA_TYPE_ID);
}

bool CostSchedulerRowKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "worker_index", _workerIndex, _workerIndex);
    return true;
}

navi::ErrorCode CostSchedulerRowKernel::compute(navi::KernelComputeContext &ctx)
{
    CostSchedulerGraphDataPtr data;
    bool eof = false;
    if (!CostSchedulerGraphKernelUtil::getInput(ctx, data, eof)) {
        return navi::EC_NONE;
    }
    if (data && data->state) {
        data->state->runRow(_workerIndex, data->state->actualLoopCount());
    }
    return CostSchedulerGraphKernelUtil::setOutput(ctx, data, eof);
}

void CostSchedulerGradientBarrierKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .inputGroup("parts", COST_SCHEDULER_GRAPH_DATA_TYPE_ID, navi::IT_REQUIRE,
                    navi::IT_REQUIRE)
        .outputGroup("workers", COST_SCHEDULER_GRAPH_DATA_TYPE_ID);
}

bool CostSchedulerGradientBarrierKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "worker_count", _workerCount, _workerCount);
    return _workerCount > 0;
}

navi::ErrorCode CostSchedulerGradientBarrierKernel::compute(
    navi::KernelComputeContext &ctx)
{
    CostSchedulerGraphDataPtr data;
    bool eof = false;
    if (!CostSchedulerGraphKernelUtil::getGroupInput(ctx, data, eof)) {
        return navi::EC_NONE;
    }
    if (data && data->state) {
        data->state->finishRow(data->state->actualLoopCount());
    }
    return CostSchedulerGraphKernelUtil::setWorkerOutputs(ctx, data, _workerCount, eof);
}

void CostSchedulerProjectionKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .input("state", COST_SCHEDULER_GRAPH_DATA_TYPE_ID)
        .output("done", COST_SCHEDULER_GRAPH_DATA_TYPE_ID);
}

bool CostSchedulerProjectionKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "worker_index", _workerIndex, _workerIndex);
    return true;
}

navi::ErrorCode CostSchedulerProjectionKernel::compute(navi::KernelComputeContext &ctx)
{
    CostSchedulerGraphDataPtr data;
    bool eof = false;
    if (!CostSchedulerGraphKernelUtil::getInput(ctx, data, eof)) {
        return navi::EC_NONE;
    }
    if (data && data->state) {
        data->state->runProjection(_workerIndex, data->state->actualLoopCount());
    }
    return CostSchedulerGraphKernelUtil::setOutput(ctx, data, eof);
}

void CostSchedulerProjectionBarrierKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .inputGroup("parts", COST_SCHEDULER_GRAPH_DATA_TYPE_ID, navi::IT_REQUIRE,
                    navi::IT_REQUIRE)
        .output("loop", COST_SCHEDULER_GRAPH_DATA_TYPE_ID)
        .outputGroup("finish", COST_SCHEDULER_GRAPH_DATA_TYPE_ID);
}

bool CostSchedulerProjectionBarrierKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "worker_count", _workerCount, _workerCount);
    return _workerCount > 0;
}

navi::ErrorCode CostSchedulerProjectionBarrierKernel::compute(
    navi::KernelComputeContext &ctx)
{
    CostSchedulerGraphDataPtr data;
    bool eof = false;
    if (!CostSchedulerGraphKernelUtil::getGroupInput(ctx, data, eof)) {
        return navi::EC_NONE;
    }
    if (eof) {
        auto ec = CostSchedulerGraphKernelUtil::setOutput(ctx, nullptr, true);
        if (navi::EC_NONE != ec) {
            return ec;
        }
        return CostSchedulerGraphKernelUtil::setWorkerOutputs(ctx, nullptr, _workerCount,
                                                              true);
    }
    if (data && data->state) {
        auto state = data->state;
        state->finishProjection(state->actualLoopCount());
        data->continueLoop = state->shouldContinue();
    } else if (data) {
        data->continueLoop = false;
    }
    auto ec = CostSchedulerGraphKernelUtil::setOutput(ctx, data, false);
    if (navi::EC_NONE != ec || !data || data->continueLoop) {
        return ec;
    }
    return CostSchedulerGraphKernelUtil::setWorkerOutputs(ctx, data, _workerCount, false);
}

void CostSchedulerFinalRowKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .input("state", COST_SCHEDULER_GRAPH_DATA_TYPE_ID)
        .output("done", COST_SCHEDULER_GRAPH_DATA_TYPE_ID);
}

bool CostSchedulerFinalRowKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "worker_index", _workerIndex, _workerIndex);
    return true;
}

navi::ErrorCode CostSchedulerFinalRowKernel::compute(navi::KernelComputeContext &ctx)
{
    CostSchedulerGraphDataPtr data;
    bool eof = false;
    if (!CostSchedulerGraphKernelUtil::getInput(ctx, data, eof)) {
        return navi::EC_NONE;
    }
    if (data && data->state) {
        data->state->runFinalRow(_workerIndex);
    }
    return CostSchedulerGraphKernelUtil::setOutput(ctx, data, eof);
}

void CostSchedulerFinalizeKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .inputGroup("parts", COST_SCHEDULER_GRAPH_DATA_TYPE_ID, navi::IT_REQUIRE,
                    navi::IT_REQUIRE)
        .output("result", MASTER_SCHEDULE_CONTEXT_TYPE_ID);
}

navi::ErrorCode CostSchedulerFinalizeKernel::compute(navi::KernelComputeContext &ctx)
{
    CostSchedulerGraphDataPtr data;
    bool eof = false;
    if (!CostSchedulerGraphKernelUtil::getGroupInput(ctx, data, eof)) {
        return navi::EC_NONE;
    }
    navi::DataPtr output;
    if (data) {
        if (data->state) {
            data->state->finalize();
        }
        output = std::make_shared<MasterScheduleContextData>(data->context);
    }
    return CostSchedulerGraphKernelUtil::setOutput(ctx, output, eof);
}

REGISTER_TYPE(CostSchedulerGraphType);
REGISTER_KERNEL(CostSchedulerPrepareKernel);
REGISTER_KERNEL(CostSchedulerPreparationKernel);
REGISTER_KERNEL(CostSchedulerPreparationBarrierKernel);
REGISTER_KERNEL(CostSchedulerLoopKernel);
REGISTER_KERNEL(CostSchedulerRowKernel);
REGISTER_KERNEL(CostSchedulerGradientBarrierKernel);
REGISTER_KERNEL(CostSchedulerProjectionKernel);
REGISTER_KERNEL(CostSchedulerProjectionBarrierKernel);
REGISTER_KERNEL(CostSchedulerFinalRowKernel);
REGISTER_KERNEL(CostSchedulerFinalizeKernel);

}
