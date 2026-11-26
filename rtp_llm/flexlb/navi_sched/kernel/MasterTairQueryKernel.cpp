#include "aios/navi_sched/kernel/MasterTairQueryKernel.h"
#include "autil/TimeUtility.h"
#include <algorithm>

namespace navi_sched {
const std::string MasterTairQueryKernel::KERNEL_ID = "navi_sched.master_tair_query_k";


void MasterTairQueryKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .input("context", MASTER_SCHEDULE_CONTEXT_TYPE_ID, navi::IT_OPTIONAL, true)
        .output("queried_context", MASTER_SCHEDULE_CONTEXT_TYPE_ID);
}


navi::ErrorCode MasterTairQueryKernel::init(navi::KernelInitContext &ctx)
{
    _completionPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    _deadlinePipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    return _completionPipe && _deadlinePipe ? navi::EC_NONE : navi::EC_INIT_SUB_GRAPH;
}


void MasterTairQueryKernel::applyResults()
{
    _pending->context->_tairKvcmQueryStatuses.assign(_pending->context->nodeCount(), "NOT_CONFIGURED");
    if (_pending->tairKvcmNodeIndexes.empty()) return;
    if (_pending->tairKvcmNodeIndexes.size() != _pending->tairKvcmResults.size()) {
        NAVI_KERNEL_LOG(ERROR, "Tair query node mapping does not match results");
        return;
    }
    for (size_t i = 0; i < _pending->tairKvcmResults.size(); ++i) {
        const auto &result = _pending->tairKvcmResults[i];
        for (const auto n : _pending->tairKvcmNodeIndexes[i])
            _pending->context->_tairKvcmQueryStatuses.at(n) = tairKvcmQueryStatusName(result.status);
        NAVI_KERNEL_LOG(INFO,
            "[navi-master-tair-query] schedule_id=%lu query_index=%lu status=%s "
            "rpc_sent=%d rpc_count=%lu retries=%lu node_count=%lu request_count=%lu physical_objects=%lu "
            "remote_hit_objects=%lu query_us=%ld request_bytes=%lu response_bytes=%lu",
            _pending->context->_scheduleId, i, tairKvcmQueryStatusName(result.status),
            int(_batch->wasSent(i)), _batch->rpcCount(i), _batch->retryCount(i), _pending->tairKvcmNodeIndexes[i].size(),
            _pending->context->requestCount(), _pending->tairKvcmQueries[i].objects.size(),
            static_cast<size_t>(std::count(result.remoteHits.begin(), result.remoteHits.end(), uint8_t(1))),
            result.elapsedUs, result.requestBytes, result.responseBytes);
        if (result.status != TairKvcmQueryStatus::OK) continue;
        auto blocks = std::make_shared<const NaviSchedExternalCacheReadyBlocks>(
            getTairKvcmReadyBlocks(_pending->tairKvcmQueries[i], _pending->tairKvcmResults[i]));
        if (blocks->full.empty() && blocks->tail.empty()) continue;
        for (const auto n : _pending->tairKvcmNodeIndexes[i]) {
            if (!_pending->context->setRemoteCacheReadyBlocks(n, blocks))
                NAVI_KERNEL_LOG(ERROR, "remote cache result arrived after input preparation, node [%lu]", n);
        }
    }
}


navi::ErrorCode MasterTairQueryKernel::emitBatch(navi::KernelComputeContext &ctx)
{
    if (_deadlineActive) {
        ctx.stopPeriodicAsyncPipe(_deadlinePipe);
        _deadlineActive = false;
    }
    applyResults();
    _queryR->reportBatch(*_batch);
    if (!ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), _pending, _pendingEof))
        return navi::EC_ABORT;
    _batch.reset();
    _pending.reset();
    // Emit one context per compute. A queued cache-disabled round still needs
    // its own activation when the preceding round finishes asynchronously.
    if (!_queued.empty()) _completionPipe->setDataOverride(nullptr, false, true);
    return navi::EC_NONE;
}


void MasterTairQueryKernel::startBatch(MasterScheduleContextDataPtr data, bool eof,
                                      navi::KernelComputeContext &ctx)
{
    _pending = std::move(data);
    _pendingEof = eof;
    _batch = _queryR->startBatch(_pending, _pending->context->_requests,
                                _pending->context->_scheduleId, _completionPipe, ++_round);
    if (!_batch->complete()) {
        const auto remainingUs = _pending->context->getScheduleDeadlineUs() -
                                 autil::TimeUtility::monotonicTimeUs();
        if (remainingUs < 1000 ||
            !ctx.registerPeriodicPipe(_deadlinePipe, remainingUs / 1000, 1)) {
            _queryR->expirePending(_batch);
        } else {
            _deadlineActive = true;
        }
    }
}


navi::ErrorCode MasterTairQueryKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr raw;
    bool eof = false;
    while (_completionPipe->getData(raw, eof)) {
        auto completion = std::dynamic_pointer_cast<TairKvcmGrpcCompletionData>(raw);
        if (completion) _queryR->completed(*completion, _batch);
    }
    _deadlinePipe->getData(raw, eof);
    if (_pending && _deadlineActive) {
        int64_t activated = 0, remaining = 0, periodMs = 0, nextTickMs = 0;
        const bool timerKnown = ctx.getPeriodicPipeActivationCount(
            _deadlinePipe, activated, remaining, periodMs, nextTickMs);
        if (!timerKnown || activated > 0 ||
            autil::TimeUtility::monotonicTimeUs() >= _pending->context->getScheduleDeadlineUs())
            _queryR->expirePending(_batch);
    }
    raw.reset();
    if (ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), raw, eof)) {
        auto data = std::dynamic_pointer_cast<MasterScheduleContextData>(raw);
        if (raw && (!data || !data->context)) return navi::EC_ABORT;
        if (data || eof) _queued.emplace_back(std::move(data), eof);
    }
    if (_pending && _batch->complete()) {
        return emitBatch(ctx);
    }
    if (!_pending && !_queued.empty()) {
        auto next = std::move(_queued.front());
        _queued.pop_front();
        if (!next.first) {
            if (!ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, next.second))
                return navi::EC_ABORT;
        } else {
            startBatch(std::move(next.first), next.second, ctx);
            if (_batch->complete()) {
                return emitBatch(ctx);
            }
        }
    }
    return navi::EC_NONE;
}


REGISTER_KERNEL(MasterTairQueryKernel);

} // namespace navi_sched
