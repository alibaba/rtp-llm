#include "aios/navi_sched/data/MasterLatencyModelWarmupToAgentData.h"
#include "aios/navi_sched/resource/AgentReportR.h"
#include "aios/navi_sched/resource/LatencyModelR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class AgentLatencyModelWarmupKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_DEPEND_ON(LatencyModelR, _latencyModelR);
    KERNEL_DEPEND_ON(AgentReportR, _agentReportR);
};

void AgentLatencyModelWarmupKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.agent_latency_model_warmup_k")
        .input("warmup", MASTER_LATENCY_MODEL_WARMUP_TO_AGENT_TYPE_ID);
}

navi::ErrorCode AgentLatencyModelWarmupKernel::compute(
    navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
        NAVI_KERNEL_LOG(ERROR, "read Agent latency warmup input failed");
        return navi::EC_ABORT;
    }
    auto warmup =
        std::dynamic_pointer_cast<MasterLatencyModelWarmupToAgentData>(data);
    if (!warmup) {
        return navi::EC_NONE;
    }
    std::shared_ptr<const NaviSchedLatencyModelSnapshot> updated;
    if (!_latencyModelR->applyColdStartCandidates(*warmup, updated)) {
        NAVI_KERNEL_LOG(ERROR,
                        "reject latency model warmup, fingerprint [%s], "
                        "candidateCount [%lu]",
                        warmup->fingerprint.c_str(), warmup->candidates.size());
        return navi::EC_NONE;
    }
    if (updated) {
        if (!_latencyModelR->saveState()) {
            NAVI_KERNEL_LOG(ERROR,
                            "persist applied latency model warmup failed, "
                            "fingerprint [%s]",
                            warmup->fingerprint.c_str());
        }
        _agentReportR->reportLatencyModelUpdated(updated);
        NAVI_KERNEL_LOG(INFO,
                        "applied latency model warmup, fingerprint [%s], "
                        "candidateCount [%lu], revision [%ld]",
                        warmup->fingerprint.c_str(), warmup->candidates.size(),
                        updated->revision);
    }
    return navi::EC_NONE;
}

REGISTER_KERNEL(AgentLatencyModelWarmupKernel);

}
