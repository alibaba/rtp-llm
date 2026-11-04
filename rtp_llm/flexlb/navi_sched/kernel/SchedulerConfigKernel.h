#pragma once

#include "aios/navi_sched/resource/SchedulerConfigR.h"
#include "navi/engine/Kernel.h"
#include "navi/rpc_server/NaviArpcRequestData.h"

namespace navi_sched {

class SchedulerConfigKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

public:
    static const std::string KERNEL_ID;
    static const std::string OUTPUT_PORT;

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(SchedulerConfigR, _schedulerConfigR);
    KERNEL_NAMED_DATA(navi::NaviArpcRequestData,
                      _requestData,
                      SchedulerConfigR::RPC_REQUEST_NAME);
    navi::AsyncPipePtr _responsePipe;
};

}
