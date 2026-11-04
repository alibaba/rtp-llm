#include "aios/navi_sched/data/ClientRequestToMasterData.h"
#include "aios/navi_sched/data/AgentResponseToClientData.h"
#include "aios/navi_sched/data/MasterResponseToClientData.h"
#include "aios/navi_sched/data/NaviSchedRequestTrace.h"
#include "autil/StringUtil.h"
#include "autil/TimeUtility.h"
#include "navi/builder/GraphBuilder.h"
#include "navi/engine/Kernel.h"
#include "navi/log/NaviLogger.h"

namespace navi_sched {
namespace {

const std::string AGENT_REQUEST_NODE = "navi_sched_agent_request";

}

class ClientForwardKernel : public navi::Kernel {
public:
    void def(navi::KernelDefBuilder &builder) const override;
    bool config(navi::KernelConfigContext &ctx) override;
    navi::ErrorCode compute(navi::KernelComputeContext &ctx) override;

private:
    static bool parseAgentAddress(const std::string &address, std::string &host,
                                  int32_t &port);
    static std::unique_ptr<navi::GraphDef> buildAgentGraph(const NaviSchedTarget &target,
                                                           navi::GraphId &graphId);
    navi::ErrorCode outputDecision(navi::KernelComputeContext &ctx,
                                   const MasterResponseToClientDataPtr &decision) const;
    navi::ErrorCode closeOutputs(navi::KernelComputeContext &ctx) const;

private:
    KERNEL_DEPEND_DECLARE();

private:
    KERNEL_NAMED_DATA(ClientRequestToMasterData, _request,
                      CLIENT_REQUEST_TO_MASTER_DATA_NAME);
    bool _collectMetric = false;
};

bool ClientForwardKernel::parseAgentAddress(const std::string &address, std::string &host,
                                            int32_t &port)
{
    if (address.empty()) {
        return false;
    }
    auto separator = address.rfind(':');
    if (std::string::npos == separator || 0 == separator ||
        separator + 1 == address.size() || address.find(':') != separator) {
        return false;
    }
    host = address.substr(0, separator);
    return autil::StringUtil::fromString(address.substr(separator + 1), port) &&
           port > 0 && port <= 65535;
}

std::unique_ptr<navi::GraphDef> ClientForwardKernel::buildAgentGraph(
    const NaviSchedTarget &target, navi::GraphId &graphId)
{
    std::string host;
    int32_t port = 0;
    if (target.bizName.empty() || !parseAgentAddress(target.agentAddress, host, port)) {
        return nullptr;
    }
    auto graphDef = std::make_unique<navi::GraphDef>();
    navi::GraphBuilder builder(graphDef.get());
    graphId = builder.newSubGraph(target.bizName);
    builder.subGraph(graphId).directLocation(target.bizName, host, port);
    auto request = builder.node(AGENT_REQUEST_NODE).kernel("navi_sched.agent_request_k");
    request.out("decision").toForkNodeOutput("decision").require(true);
    request.out("response").toForkNodeOutput("response").require(true);
    if (!builder.ok()) {
        return nullptr;
    }
    navi::GraphBuilder::fillHashKey(graphDef.get());
    return graphDef;
}

void ClientForwardKernel::def(navi::KernelDefBuilder &builder) const
{
    builder.name("navi_sched.client_forward_k")
        .input("decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID)
        .output("decision", MASTER_RESPONSE_TO_CLIENT_TYPE_ID)
        .output("response", AGENT_RESPONSE_TO_CLIENT_TYPE_ID);
}

bool ClientForwardKernel::config(navi::KernelConfigContext &ctx)
{
    NAVI_JSONIZE(ctx, "collect_metric", _collectMetric, _collectMetric);
    return true;
}

navi::ErrorCode ClientForwardKernel::compute(navi::KernelComputeContext &ctx)
{
    navi::DataPtr data;
    bool eof = false;
    if (!ctx.getInput(navi::PortIndex(0, navi::INVALID_INDEX), data, eof)) {
        return navi::EC_NONE;
    }
    auto decision = std::dynamic_pointer_cast<MasterResponseToClientData>(data);
    auto request = std::static_pointer_cast<const ClientRequestToMasterData>(
        _request->shared_from_this());
    if (!decision || decision->version != NAVI_SCHED_DATA_VERSION || !request ||
        !request->valid()) {
        NAVI_KERNEL_LOG(
            ERROR, "invalid client forward input, decision [%p], request [%p], eof [%d]",
            decision.get(), request.get(), eof);
        return navi::EC_ABORT;
    }
    if (decision->target.address.empty()) {
        return outputDecision(ctx, decision);
    }
    const bool tracing = traceRequestBatch(decision->batchId);
    const auto forwardBeginTimeUs = tracing ? autil::TimeUtility::currentTime() : 0;
    if (tracing) {
        NAVI_KERNEL_LOG(INFO,
                        "[navi-request-stage] stage=client_forward_begin request_id=%lu "
                        "schedule_batch_id=%ld expected_batch_size=%ld event_time_us=%ld "
                        "payload_bytes=%zu agent=%s",
                        request->requestId, decision->batchId, decision->expectedBatchSize,
                        forwardBeginTimeUs, request->forwardPayload.size(),
                        decision->target.agentAddress.c_str());
    }
    navi::GraphId graphId = navi::INVALID_GRAPH_ID;
    auto graphDef = buildAgentGraph(decision->target, graphId);
    if (!graphDef) {
        NAVI_KERNEL_LOG(
            ERROR, "invalid agent endpoint, requestId [%lu], bizName [%s], address [%s]",
            request->requestId, decision->target.bizName.c_str(),
            decision->target.agentAddress.c_str());
        return closeOutputs(ctx);
    }
    navi::ForkGraphParam param;
    param.collectMetric = _collectMetric;
    param.errorAsEof = true;
    navi::NamedData namedData;
    namedData.graphId = graphId;
    namedData.name = CLIENT_REQUEST_TO_MASTER_DATA_NAME;
    namedData.data = std::const_pointer_cast<ClientRequestToMasterData>(request);
    param.namedDataVec.emplace_back(std::move(namedData));
    navi::NamedData decisionData;
    decisionData.graphId = graphId;
    decisionData.name = "navi_sched.master_response";
    decisionData.data = decision;
    param.namedDataVec.emplace_back(std::move(decisionData));
    const auto forkBeginTimeUs = tracing ? autil::TimeUtility::currentTime() : 0;
    auto ec = ctx.fork(graphDef.release(), param);
    if (tracing) {
        const auto endTimeUs = autil::TimeUtility::currentTime();
        NAVI_KERNEL_LOG(INFO,
                        "[navi-request-stage] stage=client_forward_end request_id=%lu "
                        "schedule_batch_id=%ld event_time_us=%ld graph_prepare_us=%ld "
                        "fork_us=%ld error_code=%d agent=%s",
                        request->requestId, decision->batchId, endTimeUs,
                        forkBeginTimeUs - forwardBeginTimeUs, endTimeUs - forkBeginTimeUs,
                        ec, decision->target.agentAddress.c_str());
    }
    if (navi::EC_NONE != ec) {
        NAVI_KERNEL_LOG(ERROR,
                        "failed to fork Agent request graph, requestId [%lu], "
                        "agentAddress [%s], errorCode [%d]",
                        request->requestId, decision->target.agentAddress.c_str(), ec);
        return closeOutputs(ctx);
    }
    return navi::EC_NONE;
}

navi::ErrorCode ClientForwardKernel::outputDecision(
    navi::KernelComputeContext &ctx, const MasterResponseToClientDataPtr &decision) const
{
    auto output = decision->makeForwardResponse();
    ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), output, true);
    ctx.setOutput(navi::PortIndex(1, navi::INVALID_INDEX), nullptr, true);
    return navi::EC_NONE;
}

navi::ErrorCode
ClientForwardKernel::closeOutputs(navi::KernelComputeContext &ctx) const
{
    const auto decisionClosed =
        ctx.setOutput(navi::PortIndex(0, navi::INVALID_INDEX), nullptr, true);
    const auto responseClosed =
        ctx.setOutput(navi::PortIndex(1, navi::INVALID_INDEX), nullptr, true);
    return decisionClosed && responseClosed ? navi::EC_NONE : navi::EC_ABORT;
}

REGISTER_KERNEL(ClientForwardKernel);

}
