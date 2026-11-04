#pragma once

#include "rtp_llm/flexlb/navi_sched/resource/RouteModeR.h"
#include "navi/engine/Kernel.h"
#include "navi/rpc_server/NaviArpcRequestData.h"

namespace navi_sched {

class RouteModeKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

public:
    static const std::string KERNEL_ID;
    static const std::string OUTPUT_PORT;

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(RouteModeR, _routeModeR);
    KERNEL_NAMED_DATA(navi::NaviArpcRequestData,
                      _requestData,
                      RouteModeR::RPC_REQUEST_NAME);
};

} // namespace navi_sched
