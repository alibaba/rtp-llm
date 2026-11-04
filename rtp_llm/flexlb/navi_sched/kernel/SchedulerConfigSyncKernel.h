#pragma once

#include "rtp_llm/flexlb/navi_sched/resource/SchedulerConfigR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class SchedulerConfigSyncKernel : public navi::Kernel {
public:
    ~SchedulerConfigSyncKernel() override;
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

public:
    static const std::string KERNEL_ID;

private:
    bool startTimeout(navi::KernelComputeContext &ctx);
    void stopTimeout(navi::KernelComputeContext &ctx);
    bool publishResponse(const navi::AsyncPipePtr &responsePipe,
                         const SchedulerConfigResponseDataPtr &response);

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(SchedulerConfigR, _schedulerConfigR);
    navi::AsyncPipePtr _activationPipe;
    navi::AsyncPipePtr _timeoutPipe;
    int64_t _timeoutActivationCount = 0;
    bool _timeoutActive = false;
};

} // namespace navi_sched
