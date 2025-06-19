#include "rtp_llm/cpp/api_server/WorkerStatusService.h"
#include "rtp_llm/cpp/th_op/ConfigModules.h"

using namespace autil::legacy;
using namespace autil::legacy::json;

namespace rtp_llm {

WorkerStatusService::WorkerStatusService(const std::shared_ptr<EngineBase>&            engine,
                                         const std::shared_ptr<ConcurrencyController>& controller):
    engine_(engine), controller_(controller), load_balance_env_(engine->getDevice()->initParams().misc_config.load_balance) {}

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

    LoadBalanceInfo load_balance_info;
    if (engine_) {
        load_balance_info = engine_->getLoadBalanceInfo();
    } else {
        RTP_LLM_LOG_WARNING("worker status service call worker status error, engine is null");
    }

    int available_concurrency = 0;
    int load_balance_version  = 0;
    if (load_balance_env_ && load_balance_info.step_per_minute > 0 && load_balance_info.step_latency_us > 0) {
        available_concurrency = load_balance_info.step_per_minute;
        load_balance_version  = 1;
    } else {
        if (controller_) {  // controller should not be null
            available_concurrency = controller_->get_available_concurrency();
        } else {
            RTP_LLM_LOG_WARNING("called register worker status route, concurrency controller is null");
        }
    }
    WorkerStatusResponse worker_status_response;
    worker_status_response.available_concurrency = available_concurrency;
    worker_status_response.load_balance_info     = load_balance_info;
    worker_status_response.load_balance_version  = load_balance_version;
    worker_status_response.alive                 = true;
    auto response_json_str                       = ToJsonString(worker_status_response, /*isCompact=*/true);
    writer->Write(response_json_str);
}

void WorkerStatusService::stop() {
    is_stopped_.store(true);
}

}  // namespace rtp_llm