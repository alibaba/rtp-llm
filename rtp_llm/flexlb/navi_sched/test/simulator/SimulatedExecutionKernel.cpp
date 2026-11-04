#include "rtp_llm/flexlb/navi_sched/test/simulator/SimulatedExecutionR.h"
#include "autil/TimeUtility.h"
#include "navi/engine/Kernel.h"
#include <algorithm>
#include <deque>

namespace navi_sched::test {

class SimulatedExecutionKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();

private:
    struct ScheduledBatch {
        int64_t beginTimeUs = 0;
        int64_t deadlineTimeUs = 0;
        double modelPredictedLatencyMs = 0.0;
        double predictedLatencyMs = 0.0;
        double executionLatencyMs = 0.0;
        std::vector<SimulatedRequest> requests;
    };

private:
    bool scheduleNext(navi::KernelComputeContext &ctx);
    void startSubmitted();
    void startNextBatch();
    void finishReady();
    void logExecutionSummary() const;

private:
    KERNEL_DEPEND_ON(SimulatedExecutionR, _simulatedExecutionR);
    // Request notifications and execution deadlines serialize through one pipe.
    navi::AsyncPipePtr _activationPipe;
    std::deque<ScheduledBatch> _pendingBatches;
    ScheduledBatch _runningBatch;
    int64_t _firstBatchBeginTimeUs = 0;
    int64_t _lastBatchFinishTimeUs = 0;
    double _totalExecutionLatencyMs = 0.0;
    double _totalPredictionErrorMs = 0.0;
    double _totalPredictionAbsoluteErrorMs = 0.0;
    double _totalPredictionSquaredErrorMs = 0.0;
    double _totalPredictionRelativeAbsoluteError = 0.0;
    size_t _executedBatchCount = 0;
    size_t _executedRequestCount = 0;
    bool _running = false;
    bool _finishRequested = false;
};

void SimulatedExecutionKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.agent_simulate_execution_k").output("output", "");
}

navi::ErrorCode SimulatedExecutionKernel::init(navi::KernelInitContext &ctx)
{
    _activationPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    if (!_activationPipe || !_simulatedExecutionR->setRequestPipe(_activationPipe)) {
        return navi::EC_INIT_SUB_GRAPH;
    }
    return navi::EC_NONE;
}

navi::ErrorCode SimulatedExecutionKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    _activationPipe->getData(data, eof);
    startSubmitted();
    _finishRequested = _finishRequested || eof;
    finishReady();
    if (_finishRequested && !_running && _pendingBatches.empty() &&
        0 == _simulatedExecutionR->getInflightRequestCount()) {
        logExecutionSummary();
        if (!_simulatedExecutionR->finishResidentGraph()) {
            NAVI_KERNEL_LOG(WARN, "finish simulated Agent resident graph failed");
        }
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true);
        return navi::EC_NONE;
    }
    if (!scheduleNext(ctx)) {
        NAVI_KERNEL_LOG(ERROR, "schedule simulated execution timer failed");
        return navi::EC_ABORT;
    }
    return navi::EC_NONE;
}

bool SimulatedExecutionKernel::scheduleNext(navi::KernelComputeContext &ctx)
{
    if (!_running) {
        return true;
    }
    auto remainingUs = _runningBatch.deadlineTimeUs - autil::TimeUtility::currentTime();
    auto timeoutMs =
        std::max<int64_t>(1, (std::max<int64_t>(0, remainingUs) + 999) / 1000);
    return ctx.registerPeriodicPipe(_activationPipe, timeoutMs, 1);
}

void SimulatedExecutionKernel::startSubmitted()
{
    std::vector<SimulatedRequest> requests;
    _finishRequested = _simulatedExecutionR->takeSubmitted(requests) || _finishRequested;
    for (size_t beginIndex = 0; beginIndex < requests.size();) {
        auto endIndex = beginIndex + 1;
        if (0 != requests[beginIndex].batchId) {
            while (endIndex < requests.size() &&
                   requests[beginIndex].batchId == requests[endIndex].batchId) {
                ++endIndex;
            }
        }
        std::vector<SimulatedRequest> batch;
        batch.reserve(endIndex - beginIndex);
        for (auto index = beginIndex; index < endIndex; ++index) {
            batch.emplace_back(requests[index]);
        }
        ScheduledBatch scheduledBatch{0, 0, 0.0, 0.0, 0.0, std::move(batch)};
        _pendingBatches.emplace_back(std::move(scheduledBatch));
        beginIndex = endIndex;
    }
    startNextBatch();
}

void SimulatedExecutionKernel::startNextBatch()
{
    if (_running) {
        return;
    }
    while (!_pendingBatches.empty()) {
        _runningBatch = std::move(_pendingBatches.front());
        _pendingBatches.pop_front();
        if (_simulatedExecutionR->startBatch(_runningBatch.requests)) {
            break;
        }
        _runningBatch.requests.clear();
    }
    if (_runningBatch.requests.empty()) {
        return;
    }
    _runningBatch.modelPredictedLatencyMs =
        _simulatedExecutionR->predictModelLatencyMs(_runningBatch.requests);
    auto latency = _simulatedExecutionR->predictBatch(_runningBatch.requests);
    _runningBatch.predictedLatencyMs = latency.predictedLatencyMs;
    _runningBatch.executionLatencyMs = latency.executionLatencyMs;
    auto waitLatencyMs =
        _simulatedExecutionR->getWaitLatencyMs(_runningBatch.executionLatencyMs);
    _runningBatch.beginTimeUs = autil::TimeUtility::currentTime();
    if (0 == _firstBatchBeginTimeUs) {
        _firstBatchBeginTimeUs = _runningBatch.beginTimeUs;
    }
    _runningBatch.deadlineTimeUs =
        _runningBatch.beginTimeUs + static_cast<int64_t>(waitLatencyMs * 1000.0);
    _running = true;
}

void SimulatedExecutionKernel::finishReady()
{
    auto nowUs = autil::TimeUtility::currentTime();
    if (!_running || _runningBatch.deadlineTimeUs > nowUs) {
        return;
    }
    _simulatedExecutionR->finishBatch(_runningBatch.requests,
                                      _runningBatch.executionLatencyMs);
    ++_executedBatchCount;
    _executedRequestCount += _runningBatch.requests.size();
    _totalExecutionLatencyMs += _runningBatch.executionLatencyMs;
    auto predictionErrorMs =
        _runningBatch.modelPredictedLatencyMs - _runningBatch.executionLatencyMs;
    _totalPredictionErrorMs += predictionErrorMs;
    _totalPredictionAbsoluteErrorMs += std::abs(predictionErrorMs);
    _totalPredictionSquaredErrorMs += predictionErrorMs * predictionErrorMs;
    _totalPredictionRelativeAbsoluteError +=
        std::abs(predictionErrorMs) / std::max(1e-12, _runningBatch.executionLatencyMs);
    _lastBatchFinishTimeUs = nowUs;
    _runningBatch.requests.clear();
    _running = false;
    startNextBatch();
}

void SimulatedExecutionKernel::logExecutionSummary() const
{
    const auto elapsedMs =
        _firstBatchBeginTimeUs > 0 && _lastBatchFinishTimeUs >= _firstBatchBeginTimeUs
            ? _simulatedExecutionR->getLogicalElapsedMs(
                  static_cast<double>(_lastBatchFinishTimeUs - _firstBatchBeginTimeUs) /
                  1000.0)
            : 0.0;
    NAVI_KERNEL_LOG(INFO,
                    "simulated execution summary, batchCount [%lu], requestCount [%lu], "
                    "logicalBusyMs [%.3f], logicalElapsedMs [%.3f]",
                    _executedBatchCount, _executedRequestCount, _totalExecutionLatencyMs,
                    elapsedMs);
    NAVI_KERNEL_LOG(
        INFO, "simulated cache summary, cachedBlockCount [%lu], evictedBlockCount [%lu]",
        _simulatedExecutionR->getCachedBlockCount(),
        _simulatedExecutionR->getEvictedBlockCount());
    if (_executedBatchCount > 0) {
        NAVI_KERNEL_LOG(INFO,
                        "simulated model fit summary, batchCount [%lu], "
                        "predictionBiasMs [%.12f], "
                        "predictionMaeMs [%.12f], predictionRmseMs [%.12f], "
                        "predictionMeanRelativeAbsoluteError [%.12f]",
                        _executedBatchCount,
                        _totalPredictionErrorMs / _executedBatchCount,
                        _totalPredictionAbsoluteErrorMs / _executedBatchCount,
                        std::sqrt(_totalPredictionSquaredErrorMs / _executedBatchCount),
                        _totalPredictionRelativeAbsoluteError / _executedBatchCount);
    }
}

REGISTER_KERNEL(SimulatedExecutionKernel);

} // namespace navi_sched::test
