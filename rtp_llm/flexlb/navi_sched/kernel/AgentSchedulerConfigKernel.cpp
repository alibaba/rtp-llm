#include "rtp_llm/flexlb/navi_sched/data/MasterSchedulerConfigToAgentData.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentBatchR.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentReportR.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentSchedulerConfigR.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentHistoryR.h"
#include "rtp_llm/flexlb/navi_sched/resource/LatencyModelR.h"
#include "rtp_llm/flexlb/navi_sched/resource/NonLinearPrefillModelR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class AgentSchedulerConfigKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();
    KERNEL_DEPEND_ON(AgentSchedulerConfigR, _agentSchedulerConfigR);
    KERNEL_DEPEND_ON(AgentBatchR, _agentBatchR);
    KERNEL_DEPEND_ON(AgentReportR, _agentReportR);
    KERNEL_DEPEND_ON(AgentHistoryR, _agentHistoryR);
    KERNEL_DEPEND_ON(LatencyModelR, _latencyModelR);
    KERNEL_DEPEND_ON(NonLinearPrefillModelR, _nonLinearPrefillModelR);
};

void AgentSchedulerConfigKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.agent_scheduler_config_k")
        .input("config", MASTER_SCHEDULER_CONFIG_TO_AGENT_TYPE_ID);
}

navi::ErrorCode AgentSchedulerConfigKernel::compute(
    navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
        return navi::EC_NONE;
    }
    auto config = std::dynamic_pointer_cast<MasterSchedulerConfigToAgentData>(data);
    if (data && (!config || !config->valid())) {
        NAVI_KERNEL_LOG(ERROR, "invalid scheduler config payload from Master");
        if (config && config->generation > 0) {
            const auto snapshot = _agentSchedulerConfigR->get();
            _agentReportR->reportSchedulerConfigUpdated(
                config->generation, snapshot.generation, snapshot.configHash,
                snapshot.configJson, "invalid scheduler config payload");
        }
        return navi::EC_NONE;
    }
    if (config) {
        AgentSchedulerConfigR::Snapshot snapshot;
        std::string errorMessage;
        const auto result = _agentSchedulerConfigR->apply(
            config->generation, config->configHash, config->configJson,
            snapshot, errorMessage);
        if (AgentSchedulerConfigR::ApplyResult::APPLIED != result) {
            if (errorMessage.empty()) {
                errorMessage = "scheduler config update is stale";
            }
            NAVI_KERNEL_LOG(ERROR,
                            "reject scheduler config generation [%lu], hash [%s]: %s",
                            config->generation, config->configHash.c_str(),
                            errorMessage.c_str());
            _agentReportR->reportSchedulerConfigUpdated(
                config->generation, snapshot.generation,
                snapshot.configHash, snapshot.configJson, errorMessage);
            return navi::EC_NONE;
        }
        _agentBatchR->setRuntimeConfig(snapshot.config.batchTimeoutMs,
                                       snapshot.config.closedBatchKeepMs);
        _agentHistoryR->applyRuntimeConfig(snapshot.config);
        _nonLinearPrefillModelR->applyRuntimeConfig(snapshot.config);
        const auto latencyModelChanged =
            _latencyModelR->applyRuntimeConfig(snapshot.config);
        _agentReportR->setLatencyValidationEnabled(
            snapshot.config.latencyValidationEnabled);
        if (latencyModelChanged) {
            _agentReportR->reportLatencyModelUpdated(
                _latencyModelR->getSnapshot(), true);
        }
        if (_agentReportR->reportSchedulerConfigUpdated(
                config->generation, snapshot.generation,
                snapshot.configHash, snapshot.configJson, "") == 0) {
            NAVI_KERNEL_LOG(WARN,
                            "scheduler config applied without active Master "
                            "subscriber, generation [%lu]",
                            snapshot.generation);
        }
    }
    if (eof) {
        NAVI_KERNEL_LOG(INFO, "Master scheduler config stream closed");
    }
    return navi::EC_NONE;
}

REGISTER_KERNEL(AgentSchedulerConfigKernel);

} // namespace navi_sched
