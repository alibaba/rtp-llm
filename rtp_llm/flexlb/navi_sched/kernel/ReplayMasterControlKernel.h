#pragma once

#include "rtp_llm/flexlb/navi_sched/data/ReplayControlData.h"
#include "rtp_llm/flexlb/navi_sched/resource/ReplayControlR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class ReplayMasterControlKernel : public navi::Kernel {
public:
    ~ReplayMasterControlKernel() override;
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

    static const std::string KERNEL_ID;

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(ReplayControlR, _replayControlR);
    navi::AsyncPipePtr _commandPipe;
    navi::AsyncPipePtr _historyQueryPipe;
    std::string _clientId;
};

} // namespace navi_sched
