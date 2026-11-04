#include "rtp_llm/flexlb/navi_sched/kernel/AgentBatchKernel.h"

namespace navi_sched {

void AgentBatchKernel::buildGraph(navi::GraphBuilder &builder,
                                  uint64_t graphPipeGeneration)
{
    builder.node("navi_sched_agent_batch")
        .kernel("navi_sched.agent_batch_k")
        .integerAttr("graph_pipe_generation",
                     static_cast<int64_t>(graphPipeGeneration))
        .out("finished")
        .asGraphOutput("batch_finished");
}

void AgentBatchKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.agent_batch_k").output("finished", "");
}

bool AgentBatchKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "graph_pipe_generation", _graphPipeGeneration,
                 _graphPipeGeneration);
    return _graphPipeGeneration > 0;
}

navi::ErrorCode AgentBatchKernel::init(navi::KernelInitContext &ctx)
{
    _activationPipe = ctx.createAsyncPipe(navi::AS_INDEPENDENT);
    if (!_activationPipe) {
        NAVI_KERNEL_LOG(ERROR,
                        "create Agent batch activation pipe failed, generation [%ld]",
                        _graphPipeGeneration);
        return navi::EC_INIT_SUB_GRAPH;
    }
    if (!_agentBatchR->setGraphPipe(
            static_cast<uint64_t>(_graphPipeGeneration), _activationPipe)) {
        NAVI_KERNEL_LOG(ERROR,
                        "install Agent batch activation pipe failed, generation [%ld], "
                        "resourceGeneration [%lu]",
                        _graphPipeGeneration,
                        _agentBatchR->getGraphPipeGeneration());
        return navi::EC_INIT_SUB_GRAPH;
    }
    auto timeoutMs = _agentBatchR->getNextBatchTimeoutMs();
    if (timeoutMs > 0 && !ctx.registerPeriodicPipe(_activationPipe, timeoutMs, 1)) {
        NAVI_KERNEL_LOG(ERROR,
                        "register initial Agent batch timer failed, timeout [%ldms], "
                        "generation [%ld]",
                        timeoutMs, _graphPipeGeneration);
        return navi::EC_INIT_SUB_GRAPH;
    }
    return navi::EC_NONE;
}

navi::ErrorCode AgentBatchKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    _activationPipe->getData(data, eof);
    if (eof) {
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true);
        return navi::EC_NONE;
    }
    _agentBatchR->expireBatches();
    auto timeoutMs = _agentBatchR->getNextBatchTimeoutMs();
    if (timeoutMs > 0 && !ctx.registerPeriodicPipe(_activationPipe, timeoutMs, 1)) {
        NAVI_KERNEL_LOG(ERROR, "register Agent batch timer failed, timeout [%ldms]",
                        timeoutMs);
        return navi::EC_ABORT;
    }
    if (timeoutMs <= 0) {
        stopBatchTimer(ctx);
    }
    return navi::EC_NONE;
}

void AgentBatchKernel::stopBatchTimer(navi::KernelComputeContext &ctx) const
{
    int64_t activatedCount = 0;
    int64_t remainingCount = 0;
    int64_t periodMs = 0;
    int64_t nextTickTimeMs = 0;
    if (!ctx.getPeriodicPipeActivationCount(_activationPipe, activatedCount,
                                            remainingCount, periodMs, nextTickTimeMs)) {
        return;
    }
    if (nextTickTimeMs < 0) {
        return;
    }
    if (!ctx.stopPeriodicAsyncPipe(_activationPipe)) {
        NAVI_KERNEL_LOG(WARN, "stop Agent batch timer failed");
    }
}

REGISTER_KERNEL(AgentBatchKernel);

} // namespace navi_sched
