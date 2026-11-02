#pragma once

#include "aios/navi_sched/data/ReplayControlData.h"
#include "aios/navi_sched/resource/ReplayClientR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class ReplayClientControlKernel : public navi::Kernel {
public:
    ~ReplayClientControlKernel() override;
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

    static const std::string KERNEL_ID;

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(ReplayClientR, _replayClientR);
    navi::AsyncPipePtr _historyPipe;
    navi::AsyncPipePtr _historyQueryResponsePipe;
};

}
