#pragma once

#include "rtp_llm/flexlb/navi_sched/resource/ReplayControlR.h"
#include "navi/engine/Kernel.h"
#include "navi/rpc_server/NaviArpcRequestData.h"

namespace navi_sched {

class ReplayControlSyncKernel : public navi::Kernel {
public:
    ~ReplayControlSyncKernel() override;
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

public:
    static const std::string KERNEL_ID;
    static const std::string OUTPUT_PORT;

private:
private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(ReplayControlR, _replayControlR);
    KERNEL_NAMED_DATA(navi::NaviArpcRequestData, _requestData,
                      ReplayControlR::RPC_REQUEST_NAME);
    navi::AsyncPipePtr _responsePipe;
    navi::AsyncPipePtr _timeoutPipe;
    uint64_t _historyQueryId = 0;
    int64_t _timeoutActivationCount = 0;
    bool _timeoutActive = false;
};

} // namespace navi_sched
