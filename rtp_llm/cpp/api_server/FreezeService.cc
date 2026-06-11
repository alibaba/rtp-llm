#include "rtp_llm/cpp/api_server/FreezeService.h"

#include "autil/legacy/json.h"
#include "autil/legacy/jsonizable.h"

#include "rtp_llm/cpp/engine_base/freeze/FreezeLifecycleController.h"
#include "rtp_llm/cpp/utils/Logger.h"

using namespace autil::legacy;
using namespace autil::legacy::json;

namespace rtp_llm {

namespace {

void prepareJsonResponse(const std::unique_ptr<http_server::HttpResponseWriter>& writer) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
}

std::string errorBody(const std::string& message) {
    JsonMap error_map;
    error_map["error"] = Any(message);
    return ToJsonString(error_map, /*isCompact=*/true);
}

}  // namespace

FreezeService::FreezeService(const std::shared_ptr<EngineBase>& engine): engine_(engine) {}

bool FreezeService::checkEngine(const std::unique_ptr<http_server::HttpResponseWriter>& writer) {
    if (engine_) {
        return true;
    }
    RTP_LLM_LOG_WARNING("freeze service called but engine is null");
    writer->SetStatus(503, "Service Unavailable");
    writer->Write(errorBody("freeze service unavailable, engine is null"));
    return false;
}

void FreezeService::freeze(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                           const http_server::HttpRequest&                         request) {
    prepareJsonResponse(writer);
    if (!checkEngine(writer)) {
        return;
    }

    FreezeAdminRequest admin_request;
    const auto         body = request.GetBody();
    if (!body.empty()) {
        try {
            FromJsonString(admin_request, body);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("freeze failed, invalid request body: %s, exception: [%s]", body.c_str(), e.what());
            writer->SetStatus(400, "Bad Request");
            writer->Write(errorBody("freeze failed, invalid json body"));
            return;
        }
    }
    if (admin_request.mode != "graceful" && admin_request.mode != "force") {
        RTP_LLM_LOG_WARNING("freeze failed, invalid mode: %s", admin_request.mode.c_str());
        writer->SetStatus(400, "Bad Request");
        writer->Write(errorBody("freeze failed, mode must be \"graceful\" or \"force\""));
        return;
    }

    FreezeOptions options;
    options.mode             = admin_request.mode;
    options.drain_timeout_ms = admin_request.drain_timeout_ms;
    options.force            = admin_request.mode == "force";
    options.reason           = admin_request.reason;

    auto&      controller = engine_->freezeController();
    const auto result     = controller.freeze(options);
    if (!result.ok) {
        RTP_LLM_LOG_WARNING("freeze failed: %s", result.message.c_str());
        writer->SetStatus(409, "Conflict");
        writer->Write(errorBody(result.message));
        return;
    }

    FreezeActionResponse response;
    writer->Write(ToJsonString(response, /*isCompact=*/true));
}

void FreezeService::resume(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                           const http_server::HttpRequest&                         request) {
    prepareJsonResponse(writer);
    if (!checkEngine(writer)) {
        return;
    }

    auto&      controller = engine_->freezeController();
    const auto result     = controller.resume();
    if (!result.ok) {
        RTP_LLM_LOG_WARNING("resume failed: %s", result.message.c_str());
        writer->SetStatus(409, "Conflict");
        writer->Write(errorBody(result.message));
        return;
    }

    FreezeActionResponse response;
    writer->Write(ToJsonString(response, /*isCompact=*/true));
}

void FreezeService::freezeStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                 const http_server::HttpRequest&                         request) {
    prepareJsonResponse(writer);
    if (!checkEngine(writer)) {
        return;
    }

    const auto status = engine_->freezeController().status();

    FreezeStatusHttpResponse response;
    response.state                       = freezeStateToString(status.state);
    response.freeze_epoch                = status.freeze_epoch;
    response.kv_memory_state             = status.kv_memory_state;
    response.device_kv_cache_valid       = status.device_kv_cache_valid;
    response.active_request_count        = status.active_request_count;
    response.active_cache_transfer_count = status.active_cache_transfer_count;
    response.gpu_resource_state          = status.gpu_resource_state;
    response.last_error                  = status.last_error;
    writer->Write(ToJsonString(response, /*isCompact=*/true));
}

}  // namespace rtp_llm
