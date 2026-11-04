#include "aios/navi_sched/kernel/ReplayControlSyncKernel.h"

#include "aios/navi_sched/proto/ReplayControl.pb.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/rpc_server/NaviArpcResponseData.h"

namespace navi_sched {
namespace {

constexpr int64_t HISTORY_QUERY_TIMEOUT_MS = 10 * 1000;

}

const std::string ReplayControlSyncKernel::KERNEL_ID =
    "navi_sched.replay_control_sync_k";
const std::string ReplayControlSyncKernel::OUTPUT_PORT =
    "navi_sched.replay_control_sync_k.out";

ReplayControlSyncKernel::~ReplayControlSyncKernel() = default;

void ReplayControlSyncKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID).output(OUTPUT_PORT, "");
}

navi::ErrorCode ReplayControlSyncKernel::init(navi::KernelInitContext &ctx)
{
    if (!_requestData || _requestData->isStopping()) {
        return navi::EC_ABORT;
    }
    auto request = dynamic_cast<const ReplayControlRequest *>(
        _requestData->getRequest());
    if (!request) {
        NAVI_KERNEL_LOG(ERROR, "invalid replay control request type");
        return navi::EC_INIT_SUB_GRAPH;
    }
    _responsePipe = ctx.createAsyncPipe();
    _timeoutPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    if (!_responsePipe || !_timeoutPipe) {
        NAVI_KERNEL_LOG(ERROR, "create replay control response pipe failed");
        return navi::EC_INIT_SUB_GRAPH;
    }
    ReplayControlR::Request queued;
    queued.action = request->action();
    queued.mode = request->mode();
    queued.historyMode = request->history_mode();
    queued.startTimeUs = request->start_time_us();
    queued.endTimeUs = request->end_time_us();
    queued.durationMs = request->duration_ms();
    queued.totalQps = request->total_qps();
    queued.loopCount = request->loop_count();
    queued.recentQueryCount = request->recent_query_count();
    ReplayControlResponseDataPtr immediateResponse;
    if (!_replayControlR->processRequestAsync(
            queued, _responsePipe, _historyQueryId, immediateResponse)) {
        NAVI_KERNEL_LOG(ERROR, "submit replay control request failed");
        return navi::EC_INIT_SUB_GRAPH;
    }
    if (immediateResponse) {
        if (navi::EC_NONE != _responsePipe->setData(immediateResponse)) {
            return navi::EC_INIT_SUB_GRAPH;
        }
    } else {
        if (_historyQueryId == 0 ||
            !ctx.registerPeriodicPipe(_timeoutPipe, HISTORY_QUERY_TIMEOUT_MS, 1)) {
            NAVI_KERNEL_LOG(ERROR, "activate replay history query timeout failed");
            return navi::EC_INIT_SUB_GRAPH;
        }
        _timeoutActive = true;
    }
    return navi::EC_NONE;
}

navi::ErrorCode ReplayControlSyncKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    _responsePipe->getData(data, eof);
    if (!data && _timeoutActive) {
        navi::DataPtr timeoutData;
        bool timeoutEof = false;
        _timeoutPipe->getData(timeoutData, timeoutEof);
        int64_t activatedCount = 0;
        int64_t remainingCount = 0;
        int64_t periodMs = 0;
        int64_t nextTickTimeMs = 0;
        if (!ctx.getPeriodicPipeActivationCount(
                _timeoutPipe, activatedCount, remainingCount, periodMs,
                nextTickTimeMs)) {
            return navi::EC_ABORT;
        }
        if (activatedCount > _timeoutActivationCount) {
            _timeoutActivationCount = activatedCount;
            _replayControlR->timeoutHistoryQuery(_historyQueryId);
            _responsePipe->getData(data, eof);
        }
    }
    if (!data) {
        return navi::EC_NONE;
    }
    if (_timeoutActive) {
        _timeoutActive = false;
        ctx.stopPeriodicAsyncPipe(_timeoutPipe);
    }
    auto result = std::dynamic_pointer_cast<ReplayControlResponseData>(data);
    if (!result) {
        NAVI_KERNEL_LOG(ERROR, "invalid replay control response data");
        return navi::EC_ABORT;
    }
    auto response = new ReplayControlResponse();
    response->set_accepted(result->accepted);
    response->set_error(result->error);
    response->set_generation(result->generation);
    response->set_action(result->action == ReplayAction::START ? "start" : "stop");
    response->set_mode(result->mode == ReplayMode::REALTIME ? "realtime" : "fixed_qps");
    response->set_history_mode(result->historyMode == ReplayHistoryMode::RECENT_COUNT
                                   ? "recent_count"
                                   : "time_range");
    response->set_recent_query_count(result->recentQueryCount);
    response->set_history_query_id(result->historyQueryId);
    response->set_expected_client_count(result->expectedClientCount);
    response->set_responded_client_count(result->respondedClientCount);
    response->set_eligible_client_count(result->eligibleClientCount);
    response->set_per_client_qps(result->perClientQps);
    for (const auto &summary : result->clients) {
        auto *client = response->add_clients();
        client->set_client_id(summary.clientId);
        client->set_earliest_event_time_us(summary.earliestEventTimeUs);
        client->set_latest_event_time_us(summary.latestEventTimeUs);
        client->set_sample_count(summary.sampleCount);
        client->set_replayable(summary.replayable);
        client->set_status(summary.status);
        client->set_error(summary.error);
        client->set_selection_generation(summary.selectionGeneration);
        client->set_selected_sample_count(summary.selectedSampleCount);
        client->set_completed_selection_generation(
            summary.completedSelectionGeneration);
    }
    return ctx.setOutput(0, navi::DataPtr(new navi::NaviArpcResponseData(response)), true)
               ? navi::EC_NONE
               : navi::EC_ABORT;
}

REGISTER_KERNEL(ReplayControlSyncKernel);

}
