#include "aios/navi_sched/scheduler/MasterScheduleResultGraph.h"

#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "autil/TimeUtility.h"
#include "navi/engine/Type.h"
#include <algorithm>

namespace navi_sched {

const std::string MASTER_SCHEDULE_RESULT_TYPE_ID = "NaviSchedMasterScheduleResultType";
const std::string MasterScheduleResultPrepareKernel::KERNEL_ID =
    "navi_sched.master_schedule_result_prepare_k";
const std::string MasterScheduleResultWorkerKernel::KERNEL_ID =
    "navi_sched.master_schedule_result_worker_k";
const std::string MasterScheduleResultBarrierKernel::KERNEL_ID =
    "navi_sched.master_schedule_result_barrier_k";

MasterScheduleResultState::MasterScheduleResultState(NaviSchedScheduleContextPtr value)
    : context(std::move(value))
    , completion(context ? std::move(context->_scheduleRoundCompletion) : nullptr)
    , beginUs(autil::TimeUtility::monotonicTimeUs())
{
}

void MasterScheduleResultState::partition(size_t workerCount)
{
    const auto begin = autil::TimeUtility::monotonicTimeUs();
    auto &c = *context;
    c.clearScheduleView();
    c._dispatchHandles.assign(c.requestCount(), MasterDispatchHandle());
    c._responseDeliveryFailedIndexes.clear();
    _nodeRequests.resize(c.nodeCount());
    for (size_t index = 0; index < c.requestCount(); ++index) {
        const auto node = c._selectedNodeIndexes[index];
        if (node < _nodeRequests.size()) {
            _nodeRequests[node].push_back(index);
        }
    }
    const auto activeNodes = std::count_if(
        _nodeRequests.begin(), _nodeRequests.end(), [](const auto &v) { return !v.empty(); });
    workers.resize(std::max<size_t>(1, std::min<size_t>(workerCount, activeNodes)));
    auto lightestWorker = [&]() -> Worker & {
        return *std::min_element(workers.begin(), workers.end(),
            [](const auto &a, const auto &b) { return a.requests.size() < b.requests.size(); });
    };
    for (size_t node = 0; node < _nodeRequests.size(); ++node) {
        const auto &indexes = _nodeRequests[node];
        if (!indexes.empty()) {
            auto &worker = lightestWorker();
            worker.nodes.push_back(node);
            worker.requests.insert(worker.requests.end(), indexes.begin(), indexes.end());
        }
    }
    for (size_t index = 0; index < c.requestCount(); ++index) {
        if (c._selectedNodeIndexes[index] >= _nodeRequests.size()) {
            lightestWorker().requests.push_back(index);
        }
    }
    _requestWorkers.resize(c.requestCount());
    for (size_t worker = 0; worker < workers.size(); ++worker) {
        for (auto index : workers[worker].requests) {
            _requestWorkers[index] = worker;
        }
    }
    for (auto &response : c._deferredResponses) {
        response.decision.scheduleLoss = c._scheduleLoss;
        response.decision.scheduleLatencyUs = c._scheduleLatencyUs;
    }
    reserveDispatchUs = autil::TimeUtility::monotonicTimeUs();
    c._resultPartitionUs = reserveDispatchUs - begin;
}

void MasterScheduleResultState::reserveWorker(size_t workerIndex, MasterStateR &masterState)
{
    auto &c = *context;
    auto &worker = workers[workerIndex];
    const auto begin = autil::TimeUtility::monotonicTimeUs();
    worker.reserveBeginUs = begin;
    for (auto node : worker.nodes) {
        masterState.reserveScheduleNodeBatch(c, node, _nodeRequests[node]);
    }
    worker.reserveUs = autil::TimeUtility::monotonicTimeUs() - begin;
}

void MasterScheduleResultState::prepareDelivery(size_t workerCount)
{
    reserveBarrierUs = autil::TimeUtility::monotonicTimeUs();
    const auto count = context->requestCount();
    const auto active = std::max<size_t>(1, std::min(workerCount, (count + 31) / 32));
    deliveryWorkers.resize(active);
    for (size_t index = 0; index < active; ++index) {
        deliveryWorkers[index].begin = count * index / active;
        deliveryWorkers[index].end = count * (index + 1) / active;
    }
    deliveryDispatchUs = autil::TimeUtility::monotonicTimeUs();
}

void MasterScheduleResultState::deliverWorker(size_t workerIndex)
{
    auto &c = *context;
    auto &worker = deliveryWorkers[workerIndex];
    const auto begin = autil::TimeUtility::monotonicTimeUs();
    worker.beginUs = begin;
    const bool validResponses = c._responseRequests.size() == c.requestCount();
    for (auto index = worker.begin; index < worker.end; ++index) {
        if (!validResponses || !c._responseRequests[index] ||
            !c._responseRequests[index]->valid() ||
            !c._responseRequests[index]->finish(c.makeResponse(c._decisions[index], index))) {
            worker.failures.push_back(index);
            worker.success = false;
        }
    }
    if (workerIndex == 0 && validResponses) {
        for (const auto &deferred : c._deferredResponses) {
            const bool delivered = deferred.request && deferred.request->valid() &&
                deferred.request->finish(c.makeResponse(deferred.decision));
            worker.success = delivered && worker.success;
        }
    }
    worker.elapsedUs = autil::TimeUtility::monotonicTimeUs() - begin;
}

void MasterScheduleResultState::collectDeliveryFailures()
{
    deliveryBarrierUs = autil::TimeUtility::monotonicTimeUs();
    for (const auto &delivery : deliveryWorkers) {
        workers[0].success = workers[0].success && delivery.success;
        for (auto index : delivery.failures) {
            workers[_requestWorkers[index]].failures.push_back(index);
        }
    }
    cleanupDispatchUs = autil::TimeUtility::monotonicTimeUs();
    context->_resultFailureCollectUs = cleanupDispatchUs - deliveryBarrierUs;
}

void MasterScheduleResultState::cleanupWorker(size_t workerIndex)
{
    auto &c = *context;
    auto &worker = workers[workerIndex];
    const auto begin = autil::TimeUtility::monotonicTimeUs();
    worker.cleanupBeginUs = begin;
    std::vector<MasterDispatchHandle> failed;
    failed.reserve(worker.failures.size());
    for (auto index : worker.failures) {
        failed.push_back(c._dispatchHandles[index]);
    }
    MasterDispatchHandle::fail(failed);
    worker.cleanupUs = autil::TimeUtility::monotonicTimeUs() - begin;
}

void MasterScheduleResultState::merge()
{
    cleanupBarrierUs = autil::TimeUtility::monotonicTimeUs();
    auto &c = *context;
    c._resultWorkerCount = workers.size();
    c._resultDeliveryWorkerCount = deliveryWorkers.size();
    c._resultReserveUs = 0;
    c._resultDeliveryUs = 0;
    c._resultFailureCleanupUs = 0;
    c._resultReportReplayUs = 0;
    c._resultReplayLogUs = 0;
    c._resultReserveStageUs = reserveBarrierUs - reserveDispatchUs;
    c._resultDeliveryStageUs = deliveryBarrierUs - deliveryDispatchUs;
    c._resultCleanupStageUs = cleanupBarrierUs - cleanupDispatchUs;
    c._resultReserveDispatchDelayUs = 0;
    c._resultDeliveryDispatchDelayUs = 0;
    c._resultCleanupDispatchDelayUs = 0;
    for (const auto &worker : workers) {
        c._responseDeliveryFailedIndexes.insert(c._responseDeliveryFailedIndexes.end(),
            worker.failures.begin(), worker.failures.end());
        c._resultReserveUs = std::max(c._resultReserveUs, worker.reserveUs);
        c._resultFailureCleanupUs = std::max(c._resultFailureCleanupUs, worker.cleanupUs);
        c._resultReserveDispatchDelayUs = std::max(
            c._resultReserveDispatchDelayUs, worker.reserveBeginUs - reserveDispatchUs);
        c._resultCleanupDispatchDelayUs = std::max(
            c._resultCleanupDispatchDelayUs, worker.cleanupBeginUs - cleanupDispatchUs);
    }
    for (const auto &worker : deliveryWorkers) {
        c._resultDeliveryUs = std::max(c._resultDeliveryUs, worker.elapsedUs);
        c._resultDeliveryDispatchDelayUs = std::max(
            c._resultDeliveryDispatchDelayUs, worker.beginUs - deliveryDispatchUs);
    }
    c._resultMergeUs = autil::TimeUtility::monotonicTimeUs() - cleanupBarrierUs;
}

void MasterScheduleResultPrepareKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .input("result", MASTER_SCHEDULE_CONTEXT_TYPE_ID)
        .outputGroup("workers", MASTER_SCHEDULE_RESULT_TYPE_ID);
}

bool MasterScheduleResultPrepareKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "worker_count", _workerCount, _workerCount);
    return _workerCount > 0;
}

navi::ErrorCode MasterScheduleResultPrepareKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr input;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), input, eof)) {
        return navi::EC_NONE;
    }
    std::shared_ptr<MasterScheduleResultData> output;
    size_t activeWorkers = 0;
    if (input) {
        auto context = MasterScheduleContextData::getContext(input);
        if (!context) {
            return navi::EC_ABORT;
        }
        auto state = std::make_shared<MasterScheduleResultState>(context);
        context->_scheduleLatencyUs = std::max<int64_t>(
            0, state->beginUs - context->_scheduleBeginTimeUs);
        if (!context->valid()) {
            context->fail("scheduler_error");
            context->finish();
            state->serialFinished = true;
        } else if (context->_replay || context->requestCount() < MIN_PARALLEL_REQUESTS ||
                   _workerCount == 1) {
            if (!finishContext(*context)) {
                NAVI_KERNEL_LOG(WARN, "finish serial master schedule result failed");
            }
            state->serialFinished = true;
        } else if (!prepareContext(*context, state->beginUs)) {
            reportResultTiming(*context, state->beginUs);
            state->serialFinished = true;
        } else {
            state->partition(_workerCount);
            activeWorkers = state->workers.size();
        }
        output = std::make_shared<MasterScheduleResultData>(std::move(state));
    }
    for (size_t index = 0; index < _workerCount; ++index) {
        if (!ctx.setOutput(navi::PortIndex(static_cast<navi::IndexType>(index), 0),
                           index == 0 || index < activeWorkers ? output : nullptr, eof)) {
            return navi::EC_ABORT;
        }
    }
    return navi::EC_NONE;
}

void MasterScheduleResultWorkerKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .input("state", MASTER_SCHEDULE_RESULT_TYPE_ID)
        .output("done", MASTER_SCHEDULE_RESULT_TYPE_ID);
}

bool MasterScheduleResultWorkerKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "worker_index", _workerIndex, _workerIndex);
    NAVI_JSONIZE(ctx, "phase", _phase, _phase);
    return _phase >= 0 && _phase <= 2;
}

navi::ErrorCode MasterScheduleResultWorkerKernel::compute(navi::KernelComputeContext &ctx)
{
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    navi::DataPtr input;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), input, eof)) {
        return navi::EC_NONE;
    }
    const auto data = std::dynamic_pointer_cast<MasterScheduleResultData>(input);
    if (input && !data) {
        return navi::EC_ABORT;
    }
    if (data && !data->state->serialFinished) {
        auto &state = *data->state;
        if (_phase == 0) {
            state.reserveWorker(_workerIndex, *_masterStateR);
        } else if (_phase == 1) {
            state.deliverWorker(_workerIndex);
        } else {
            state.cleanupWorker(_workerIndex);
        }
    }
    const auto outputBeginUs = autil::TimeUtility::monotonicTimeUs();
    const auto success = ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), input, eof);
    const auto endUs = autil::TimeUtility::monotonicTimeUs();
    if (endUs - beginUs >= 50'000) {
        NAVI_KERNEL_LOG(INFO,
            "[navi-master-result-worker-call-timing] schedule_id=%lu phase=%d worker=%lu "
            "begin_us=%ld output_begin_us=%ld end_us=%ld body_us=%ld output_us=%ld",
            data ? data->state->getScheduleId() : 0, _phase, _workerIndex,
            beginUs, outputBeginUs, endUs, outputBeginUs - beginUs, endUs - outputBeginUs);
    }
    return success ? navi::EC_NONE : navi::EC_ABORT;
}

void MasterScheduleResultBarrierKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .inputGroup("parts", MASTER_SCHEDULE_RESULT_TYPE_ID)
        .outputGroup("workers", MASTER_SCHEDULE_RESULT_TYPE_ID);
}

bool MasterScheduleResultBarrierKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "worker_count", _workerCount, _workerCount);
    NAVI_JSONIZE(ctx, "phase", _phase, _phase);
    return _workerCount > 0 && (_phase == 1 || _phase == 2);
}

navi::ErrorCode MasterScheduleResultBarrierKernel::compute(navi::KernelComputeContext &ctx)
{
    const auto beginUs = autil::TimeUtility::monotonicTimeUs();
    navi::GroupDatas inputs;
    if (!ctx.getGroupInput(0, inputs)) {
        return navi::EC_NONE;
    }
    const auto inputReadyUs = autil::TimeUtility::monotonicTimeUs();
    std::shared_ptr<MasterScheduleResultData> output;
    for (const auto &input : inputs) {
        if (!input.hasValue || !input.data) {
            continue;
        }
        auto data = std::dynamic_pointer_cast<MasterScheduleResultData>(input.data);
        if (!data || !data->state || (output && output->state != data->state)) {
            return navi::EC_ABORT;
        }
        output = std::move(data);
    }
    size_t activeWorkers = 0;
    if (output && !output->state->serialFinished) {
        auto &state = *output->state;
        if (_phase == 1) {
            state.prepareDelivery(_workerCount);
            activeWorkers = state.deliveryWorkers.size();
        } else {
            state.collectDeliveryFailures();
            int64_t latestWorkerDoneUs = state.deliveryDispatchUs;
            for (const auto &worker : state.deliveryWorkers) {
                latestWorkerDoneUs = std::max(
                    latestWorkerDoneUs, worker.beginUs + worker.elapsedUs);
            }
            if (state.deliveryBarrierUs - latestWorkerDoneUs >= 50'000 ||
                state.getScheduleId() % 100 == 0) {
                NAVI_KERNEL_LOG(INFO,
                    "[navi-master-delivery-barrier-timing] schedule_id=%lu requests=%lu "
                    "workers=%lu latest_worker_done_us=%ld entry_us=%ld input_ready_us=%ld "
                    "ready_us=%ld worker_done_to_entry_us=%ld input_us=%ld collect_us=%ld",
                    state.getScheduleId(), state.context->requestCount(),
                    state.deliveryWorkers.size(), latestWorkerDoneUs, beginUs, inputReadyUs,
                    state.deliveryBarrierUs, beginUs - latestWorkerDoneUs,
                    inputReadyUs - beginUs, state.cleanupDispatchUs - inputReadyUs);
            }
            activeWorkers = state.workers.size();
        }
    }
    for (size_t index = 0; index < _workerCount; ++index) {
        if (!ctx.setOutput(navi::PortIndex(static_cast<navi::IndexType>(index), 0),
                          index == 0 || index < activeWorkers ? output : nullptr, inputs.eof())) {
            return navi::EC_ABORT;
        }
    }
    return navi::EC_NONE;
}

class MasterScheduleResultType : public navi::Type {
public:
    MasterScheduleResultType() : navi::Type(NAVI_TYPE_PARAM, MASTER_SCHEDULE_RESULT_TYPE_ID) {}
};

REGISTER_TYPE(MasterScheduleResultType);
REGISTER_KERNEL(MasterScheduleResultPrepareKernel);
REGISTER_KERNEL(MasterScheduleResultWorkerKernel);
REGISTER_KERNEL(MasterScheduleResultBarrierKernel);

}
