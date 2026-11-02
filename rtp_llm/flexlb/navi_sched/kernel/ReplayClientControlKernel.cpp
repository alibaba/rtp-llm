#include "aios/navi_sched/kernel/ReplayClientControlKernel.h"

#include "navi/engine/NaviRegisterMacro.h"

namespace navi_sched {

const std::string ReplayClientControlKernel::KERNEL_ID =
    "navi_sched.replay_client_control_k";

ReplayClientControlKernel::~ReplayClientControlKernel()
{
    if (_replayClientR && _historyPipe) {
        _replayClientR->unsubscribeHistory(_historyPipe);
    }
    if (_replayClientR && _historyQueryResponsePipe) {
        _replayClientR->unsubscribeHistoryQueryResponse(
            _historyQueryResponsePipe);
    }
}

void ReplayClientControlKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .output("capability", ReplayClientCapabilityData::TYPE_ID)
        .output("history_response", ReplayHistoryQueryResponseData::TYPE_ID);
}

navi::ErrorCode ReplayClientControlKernel::init(navi::KernelInitContext &ctx)
{
    _historyPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    _historyQueryResponsePipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    if (!_historyPipe || !_historyQueryResponsePipe ||
        !_replayClientR->subscribeHistory(_historyPipe) ||
        !_replayClientR->subscribeHistoryQueryResponse(
            _historyQueryResponsePipe)) {
        NAVI_KERNEL_LOG(ERROR, "subscribe replay client history failed");
        return navi::EC_INIT_SUB_GRAPH;
    }
    return navi::EC_NONE;
}

navi::ErrorCode ReplayClientControlKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr historyData;
    bool historyEof = false;
    if (_historyPipe->getData(historyData, historyEof)) {
        if (historyEof) {
            if (!ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr,
                               true)) {
                return navi::EC_ABORT;
            }
        } else {
            auto capability =
                std::dynamic_pointer_cast<ReplayClientCapabilityData>(historyData);
            if (!capability ||
                !ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), capability,
                               false)) {
                return navi::EC_ABORT;
            }
        }
    }
    navi::DataPtr responseData;
    bool responseEof = false;
    if (_historyQueryResponsePipe->getData(responseData, responseEof)) {
        if (responseEof) {
            if (!ctx.setOutput(navi::PortIndex(1, navi::INVALID_INDEX), nullptr,
                               true)) {
                return navi::EC_ABORT;
            }
        } else {
            auto response = std::dynamic_pointer_cast<
                ReplayHistoryQueryResponseData>(responseData);
            if (!response ||
                !ctx.setOutput(navi::PortIndex(1, navi::INVALID_INDEX), response,
                               false)) {
                return navi::EC_ABORT;
            }
            NAVI_KERNEL_LOG(INFO,
                            "sent replay history response, query [%lu], client [%s]",
                            response->queryId,
                            response->summary.clientId.c_str());
        }
    }
    return navi::EC_NONE;
}

REGISTER_KERNEL(ReplayClientControlKernel);

}
