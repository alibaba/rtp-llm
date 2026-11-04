#include "rtp_llm/flexlb/navi_sched/kernel/ReplayMasterHistoryKernel.h"

#include "rtp_llm/flexlb/navi_sched/data/ReplayControlData.h"
#include "navi/engine/NaviRegisterMacro.h"

namespace navi_sched {

const std::string ReplayMasterHistoryKernel::KERNEL_ID =
    "navi_sched.replay_master_history_k";

void ReplayMasterHistoryKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID)
        .input("history_response", ReplayHistoryQueryResponseData::TYPE_ID);
}

navi::ErrorCode ReplayMasterHistoryKernel::compute(
    navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
        return navi::EC_NONE;
    }
    if (!data) {
        return eof ? navi::EC_NONE : navi::EC_ABORT;
    }
    auto response =
        std::dynamic_pointer_cast<ReplayHistoryQueryResponseData>(data);
    if (!response || response->summary.clientId.empty()) {
        NAVI_KERNEL_LOG(ERROR, "invalid replay history query response");
        return navi::EC_ABORT;
    }
    NAVI_KERNEL_LOG(INFO,
                    "received replay history response, query [%lu], client [%s]",
                    response->queryId, response->summary.clientId.c_str());
    _replayControlR->completeHistoryQuery(*response);
    return navi::EC_NONE;
}

REGISTER_KERNEL(ReplayMasterHistoryKernel);

} // namespace navi_sched
