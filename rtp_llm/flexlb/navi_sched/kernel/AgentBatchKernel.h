#pragma once

#include "rtp_llm/flexlb/navi_sched/resource/AgentBatchR.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class AgentBatchKernel : public navi::Kernel {
public:
    static void buildGraph(navi::GraphBuilder &builder,
                           uint64_t graphPipeGeneration);
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    void stopBatchTimer(navi::KernelComputeContext &ctx) const;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_DEPEND_ON(AgentBatchR, _agentBatchR);
    // Requests and timer ticks share one activation pipe.  Each wakeup is
    // consumed before the next deadline is registered.
    navi::AsyncPipePtr _activationPipe;
    int64_t _graphPipeGeneration = 0;
};

} // namespace navi_sched
