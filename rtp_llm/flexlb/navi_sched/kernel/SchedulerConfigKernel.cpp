#include "aios/navi_sched/kernel/SchedulerConfigKernel.h"

#include "aios/navi_sched/proto/SchedulerConfig.pb.h"
#include "navi/engine/NaviRegisterMacro.h"
#include "navi/rpc_server/NaviArpcResponseData.h"

namespace navi_sched {

const std::string SchedulerConfigKernel::KERNEL_ID =
    "navi_sched.scheduler_config_k";
const std::string SchedulerConfigKernel::OUTPUT_PORT =
    "navi_sched.scheduler_config_k.out";

void SchedulerConfigKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name(KERNEL_ID).output(OUTPUT_PORT, "");
}

navi::ErrorCode SchedulerConfigKernel::init(navi::KernelInitContext &ctx)
{
    if (!_requestData || _requestData->isStopping()) {
        return navi::EC_ABORT;
    }
    auto request = dynamic_cast<const SchedulerConfigRequest *>(
        _requestData->getRequest());
    if (!request) {
        NAVI_KERNEL_LOG(ERROR, "invalid scheduler config request type");
        return navi::EC_INIT_SUB_GRAPH;
    }
    _responsePipe = ctx.createAsyncPipe();
    if (!_responsePipe) {
        NAVI_KERNEL_LOG(ERROR, "create scheduler config response pipe failed");
        return navi::EC_INIT_SUB_GRAPH;
    }
    SchedulerConfigR::Request queuedRequest;
    queuedRequest.update = request->update();
    queuedRequest.configJson = request->config_json();
    queuedRequest.hardware = request->hardware();
    queuedRequest.responsePipe = _responsePipe;
    if (!_schedulerConfigR->submit(std::move(queuedRequest))) {
        NAVI_KERNEL_LOG(
            WARN,
            "[navi-scheduler-config-update] role [master], status [rejected], "
            "operation [%s], patch [%s], error [%s]",
            request->update() ? "update" : "get", request->config_json().c_str(),
            "scheduler config sync graph unavailable");
        auto failed = std::make_shared<SchedulerConfigResponseData>();
        const auto snapshot = _schedulerConfigR->get();
        failed->configJson = SchedulerConfigR::toJson(snapshot.config);
        failed->generation = snapshot.generation;
        failed->accepted = false;
        failed->error = "scheduler config sync graph unavailable";
        if (navi::EC_NONE != _responsePipe->setData(failed)) {
            return navi::EC_INIT_SUB_GRAPH;
        }
    }
    return navi::EC_NONE;
}

navi::ErrorCode SchedulerConfigKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (!_responsePipe->getData(data, eof)) {
        return navi::EC_NONE;
    }
    auto result = std::dynamic_pointer_cast<SchedulerConfigResponseData>(data);
    if (!result) {
        NAVI_KERNEL_LOG(ERROR,
                        "scheduler config response pipe returned invalid data");
        return navi::EC_ABORT;
    }
    auto response = new SchedulerConfigResponse();
    response->set_config_json(result->configJson);
    response->set_generation(result->generation);
    response->set_accepted(result->accepted);
    response->set_error(result->error);
    response->set_hardware(result->hardware);
    response->set_target_agent_count(result->targetAgentCount);
    response->set_updated_agent_count(result->updatedAgentCount);
    response->set_failed_agent_count(result->failedAgentCount);
    for (const auto &agentResult : result->agentResults) {
        auto *output = response->add_agent_results();
        output->set_ip(agentResult.ip);
        output->set_hardware(agentResult.hardware);
        output->set_updated(agentResult.updated);
        output->set_error(agentResult.error);
        output->set_effective_hash(agentResult.effectiveHash);
        output->set_effective_config_json(agentResult.effectiveConfigJson);
    }
    ctx.setOutput(
        0, navi::DataPtr(new navi::NaviArpcResponseData(response)), true);
    return navi::EC_NONE;
}

REGISTER_KERNEL(SchedulerConfigKernel);

}
