#include "rtp_llm/flexlb/navi_sched/data/AgentUpdateToMasterData.h"
#include "rtp_llm/flexlb/navi_sched/resource/AgentReportR.h"
#include "navi/engine/Kernel.h"

namespace navi_sched {

class AgentReportSourceKernel : public navi::Kernel {
public:
    AgentReportSourceKernel();
    ~AgentReportSourceKernel();

public:
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode init(navi::KernelInitContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_DEPEND_ON(AgentReportR, _agentReportR);
    std::string _collectorId;
    navi::AsyncPipePtr _asyncPipe;
};

AgentReportSourceKernel::AgentReportSourceKernel() {}

AgentReportSourceKernel::~AgentReportSourceKernel()
{
    if (_asyncPipe) {
        _agentReportR->unsubscribe(_collectorId, _asyncPipe);
    }
}

void AgentReportSourceKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.agent_report_source_k")
        .output("update", AGENT_UPDATE_TO_MASTER_TYPE_ID);
}

bool AgentReportSourceKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "collector_id", _collectorId);
    return !_collectorId.empty();
}

navi::ErrorCode AgentReportSourceKernel::init(navi::KernelInitContext &ctx)
{
    _asyncPipe = ctx.createAsyncPipe();
    if (!_asyncPipe) {
        NAVI_KERNEL_LOG(ERROR,
                        "create Agent report source pipe failed, collectorId [%s]",
                        _collectorId.c_str());
        return navi::EC_INIT_SUB_GRAPH;
    }
    if (!_agentReportR->subscribe(_collectorId, _asyncPipe)) {
        NAVI_KERNEL_LOG(ERROR,
                        "subscribe Agent report source failed, collectorId [%s], "
                        "subscriberCount [%lu]",
                        _collectorId.c_str(),
                        _agentReportR->getSubscriberCount());
        return navi::EC_ABORT;
    }
    return navi::EC_NONE;
}

navi::ErrorCode AgentReportSourceKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (!_asyncPipe->getData(data, eof)) {
        return navi::EC_NONE;
    }
    auto update = std::dynamic_pointer_cast<AgentUpdateToMasterData>(data);
    if (data && !update) {
        NAVI_KERNEL_LOG(ERROR,
                        "invalid Agent report source payload, collectorId [%s], "
                        "typeId [%s]",
                        _collectorId.c_str(), data->getTypeId().c_str());
        return navi::EC_ABORT;
    }
    if (!ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), update, eof)) {
        NAVI_KERNEL_LOG(ERROR,
                        "publish Agent report source output failed, collectorId [%s], "
                        "eof [%d]",
                        _collectorId.c_str(), static_cast<int>(eof));
        return navi::EC_ABORT;
    }
    return navi::EC_NONE;
}

REGISTER_KERNEL(AgentReportSourceKernel);

} // namespace navi_sched
