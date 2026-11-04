#pragma once

#include "rtp_llm/flexlb/navi_sched/resource/ReplayControlR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class ReplayMasterHistoryKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

    static const std::string KERNEL_ID;

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(ReplayControlR, _replayControlR);
};

} // namespace navi_sched
