#include "rtp_llm/cpp/api_server/WorkerStatusService.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace autil::legacy;
using namespace autil::legacy::json;

namespace rtp_llm {

// refactor(wantuan.tp): EmbeddingEngine should inherit from EngineBase, we need to remove the strange codes here for
WorkerStatusService::WorkerStatusService(const std::shared_ptr<EngineBase>&            engine,
                                         const std::shared_ptr<ConcurrencyController>& controller):
    engine_(engine), controller_(controller) {}

void WorkerStatusService::workerStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                       const http_server::HttpRequest&                         request) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");

    if (is_stopped_.load()) {
        RTP_LLM_LOG_WARNING("called worker status route, but server has been shutdown");
        writer->SetStatus(503, "Service Unavailable");
        writer->Write(R"({"detail":"this server has been shutdown"})");
        return;
    }

    KVCacheInfo cache_status;
    if (engine_) {
        cache_status = engine_->getCacheStatusInfo(-1, true);
    } else {
        RTP_LLM_LOG_WARNING("worker status service call worker status error, engine is null");
    }
    WorkerStatusResponse worker_status_response;
    worker_status_response.cache_status = std::move(cache_status);
    worker_status_response.alive        = true;
    auto response_json_str              = ToJsonString(worker_status_response, /*isCompact=*/true);
    writer->Write(response_json_str);
}

void WorkerStatusService::stop() {
    is_stopped_.store(true);
}

}  // namespace rtp_llm