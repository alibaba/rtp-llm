#include "aios/navi_sched/data/MasterStopAckToAgentData.h"
#include "aios/navi_sched/resource/AgentReportR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class AgentStopAckKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_DEPEND_ON(AgentReportR, _agentReportR);
};

void AgentStopAckKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.agent_stop_ack_k")
        .input("ack", MASTER_STOP_ACK_TO_AGENT_TYPE_ID);
}

navi::ErrorCode AgentStopAckKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
        NAVI_KERNEL_LOG(ERROR, "read Agent stop ACK input failed");
        return navi::EC_ABORT;
    }
    auto ack = std::dynamic_pointer_cast<MasterStopAckToAgentData>(data);
    if (ack) {
        if (_agentReportR->confirmStop(ack->collectorId)) {
            NAVI_KERNEL_LOG(INFO, "agent stopping acknowledged, collectorId [%s]",
                            ack->collectorId.c_str());
        }
    }
    return navi::EC_NONE;
}

REGISTER_KERNEL(AgentStopAckKernel);

}
