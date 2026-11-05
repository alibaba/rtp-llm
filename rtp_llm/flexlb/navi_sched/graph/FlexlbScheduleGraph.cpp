#include "aios/navi_sched/graph/FlexlbScheduleGraph.h"
#include "aios/navi_sched/data/FlexlbScheduleData.h"
#include "navi/builder/GraphBuilder.h"
#include <grpc++/create_channel.h>
#include <grpc++/security/credentials.h>
#include <grpc++/client_context.h>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace navi_sched {

FlexlbScheduleGraph::FlexlbScheduleGraph(navi::NaviPtr navi, std::string clientBizName)
    : _navi(std::move(navi)), _clientBizName(std::move(clientBizName))
{
}

std::tuple<int32_t, std::string, std::string> FlexlbScheduleGraph::schedule(
    uint64_t requestId, const std::string &address, const std::string &payload,
    int64_t timeoutMs, const std::vector<std::pair<std::string, std::string>> &metadata)
{
    if (!requestId || address.empty() || payload.empty() || timeoutMs <= 0) {
        throw std::invalid_argument("invalid FlexLB scheduling request");
    }
    auto request = std::make_shared<FlexlbScheduleRequestData>();
    request->payload = payload;
    request->context = std::make_shared<grpc::ClientContext>();
    for (const auto &header : metadata) {
        request->context->AddMetadata(header.first, header.second);
    }
    request->context->set_deadline(
        std::chrono::system_clock::now() + std::chrono::milliseconds(timeoutMs));
    {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_stopped || !_navi || _clientBizName.empty()) {
            throw std::runtime_error("NaviSched FlexLB client is stopped");
        }
        if (!_activeRequests.emplace(requestId, request->context).second) {
            throw std::invalid_argument("duplicate active FlexLB request");
        }
        auto &channel = _channels[address];
        if (!channel) {
            channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
        }
        request->channel = channel;
    }
    try {
        auto graph = std::make_unique<navi::GraphDef>();
        navi::GraphBuilder builder(graph.get());
        auto graphId = builder.newSubGraph(_clientBizName);
        builder.subGraph(graphId).location(_clientBizName, 1, 0).inlineMode(true);
        builder.node("flexlb_schedule").kernel("navi_sched.client_flexlb_k")
            .out("response").asGraphOutput("response");
        if (!builder.ok()) {
            throw std::runtime_error("build FlexLB client graph failed");
        }
        navi::GraphBuilder::fillHashKey(graph.get());
        navi::RunGraphParams params;
        params.setTimeoutMs(timeoutMs);
        navi::NamedData named;
        named.graphId = graphId;
        named.name = FLEXLB_REQUEST_NAME;
        named.data = request;
        if (!params.addNamedData(named)) {
            throw std::runtime_error("attach FlexLB request failed");
        }
        auto result = _navi->runGraph(graph.release(), params);
        if (!result) {
            throw std::runtime_error("start FlexLB client graph failed");
        }
        std::shared_ptr<FlexlbScheduleResponseData> response;
        bool eof = false;
        while (!eof) {
            navi::NaviUserData output;
            if (result->nextData(output, eof) && output.data) {
                response = std::dynamic_pointer_cast<FlexlbScheduleResponseData>(output.data);
            }
        }
        auto status = result->getNaviResult();
        if (status && status->hasError()) {
            if (status->getErrorCode() != navi::EC_TIMEOUT) {
                throw std::runtime_error("FlexLB client graph: " + status->getErrorMessage());
            }
            request->context->TryCancel();
            response = std::make_shared<FlexlbScheduleResponseData>();
            response->statusCode = static_cast<int32_t>(grpc::StatusCode::DEADLINE_EXCEEDED);
            response->errorMessage = "FlexLB client graph deadline exceeded";
        }
        if (!response) {
            throw std::runtime_error("FlexLB client graph returned no response");
        }
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _activeRequests.erase(requestId);
        }
        return {response->statusCode, std::move(response->payload),
                std::move(response->errorMessage)};
    } catch (...) {
        request->context->TryCancel();
        std::lock_guard<std::mutex> lock(_mutex);
        _activeRequests.erase(requestId);
        throw;
    }
}

bool FlexlbScheduleGraph::abort(uint64_t requestId)
{
    std::lock_guard<std::mutex> lock(_mutex);
    auto found = _activeRequests.find(requestId);
    if (found == _activeRequests.end()) {
        return false;
    }
    found->second->TryCancel();
    return true;
}

void FlexlbScheduleGraph::stop()
{
    std::lock_guard<std::mutex> lock(_mutex);
    _stopped = true;
    for (const auto &request : _activeRequests) {
        request.second->TryCancel();
    }
    _channels.clear();
}

}
