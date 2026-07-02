#include "rtp_llm/cpp/api_server/SleepService.h"

#include "autil/legacy/json.h"
#include "autil/legacy/jsonizable.h"

#include "rtp_llm/cpp/engine_base/sleep/SleepLifecycleController.h"
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

bool findUnsupportedLifecycleControlField(const JsonMap& body_map, std::string& field) {
    for (const auto& candidate : {"phase", "prepare_only", "commit_only"}) {
        if (body_map.find(candidate) != body_map.end()) {
            field = candidate;
            return true;
        }
    }
    return false;
}

}  // namespace

SleepService::SleepService(const std::shared_ptr<EngineBase>& engine): engine_(engine) {}

bool SleepService::checkEngine(const std::unique_ptr<http_server::HttpResponseWriter>& writer) {
    if (engine_) {
        return true;
    }
    RTP_LLM_LOG_WARNING("sleep service called but engine is null");
    writer->SetStatus(503, "Service Unavailable");
    writer->Write(errorBody("sleep service unavailable, engine is null"));
    return false;
}

namespace {

void setErrorStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer, const SleepResult& result) {
    switch (result.code) {
        case SleepResult::Code::DISABLED:
        case SleepResult::Code::UNIMPLEMENTED:
            writer->SetStatus(501, "Not Implemented");
            break;
        case SleepResult::Code::INVALID_ARGUMENT:
            writer->SetStatus(400, "Bad Request");
            break;
        case SleepResult::Code::FAILED_PRECONDITION:
            writer->SetStatus(409, "Conflict");
            break;
        case SleepResult::Code::OK:
            writer->SetStatus(500, "Internal Server Error");
            break;
    }
}

void fillSleepStatusResponse(const SleepStatus& status, SleepStatusHttpResponse& response) {
    response.sleep_mode_enabled          = status.sleep_mode_enabled;
    response.effective                   = status.effective;
    response.supported_levels            = status.supported_levels;
    response.supported_modes             = status.supported_modes;
    response.disabled_reason             = status.disabled_reason;
    response.state                       = sleepStateToString(status.state);
    response.sleep_epoch                 = status.sleep_epoch;
    response.kv_memory_state             = status.kv_memory_state;
    response.device_kv_cache_valid       = status.device_kv_cache_valid;
    response.active_request_count        = status.active_request_count;
    response.active_cache_transfer_count = status.active_cache_transfer_count;
    response.gpu_resource_state          = status.gpu_resource_state;
    response.last_error                  = status.last_error;
}

}  // namespace

void SleepService::sleep(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                         const http_server::HttpRequest&                         request) {
    prepareJsonResponse(writer);
    if (!checkEngine(writer)) {
        return;
    }

    SleepHttpRequest admin_request;
    const auto       body = request.GetBody();
    if (!body.empty()) {
        try {
            auto        parsed     = ParseJson(body);
            auto&       parsed_map = AnyCast<JsonMap>(parsed);
            std::string unsupported_field;
            if (findUnsupportedLifecycleControlField(parsed_map, unsupported_field)) {
                RTP_LLM_LOG_WARNING("sleep failed, unsupported external lifecycle control field: %s",
                                    unsupported_field.c_str());
                writer->SetStatus(400, "Bad Request");
                writer->Write(errorBody("sleep failed, " + unsupported_field + " is unsupported"));
                return;
            }
            FromJson(admin_request, parsed);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("sleep failed, invalid request body: %s, exception: [%s]", body.c_str(), e.what());
            writer->SetStatus(400, "Bad Request");
            writer->Write(errorBody("sleep failed, invalid json body"));
            return;
        }
    }
    if (admin_request.mode != "wait" && admin_request.mode != "abort") {
        RTP_LLM_LOG_WARNING("sleep failed, invalid mode: %s", admin_request.mode.c_str());
        writer->SetStatus(400, "Bad Request");
        writer->Write(errorBody("sleep failed, mode must be \"wait\" or \"abort\""));
        return;
    }
    for (const auto& tag : admin_request.tags) {
        if (tag.empty()) {
            RTP_LLM_LOG_WARNING("sleep failed, empty tag is not supported");
            writer->SetStatus(400, "Bad Request");
            writer->Write(errorBody("sleep failed, tags must be non-empty strings"));
            return;
        }
    }

    SleepOptions options;
    options.level      = admin_request.level;
    options.mode       = admin_request.mode;
    options.timeout_ms = admin_request.timeout_ms;
    options.reason     = admin_request.reason;
    options.tags       = admin_request.tags;

    auto&      controller = engine_->sleepController();
    const auto result     = controller.sleep(options);
    if (!result.ok) {
        RTP_LLM_LOG_WARNING("sleep failed: %s", result.message.c_str());
        setErrorStatus(writer, result);
        writer->Write(errorBody(result.message));
        return;
    }

    SleepActionResponse response;
    writer->Write(ToJsonString(response, /*isCompact=*/true));
}

void SleepService::wakeUp(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                          const http_server::HttpRequest&                         request) {
    prepareJsonResponse(writer);
    if (!checkEngine(writer)) {
        return;
    }
    const auto body = request.GetBody();
    if (!body.empty()) {
        try {
            auto        parsed     = ParseJson(body);
            auto&       parsed_map = AnyCast<JsonMap>(parsed);
            std::string unsupported_field;
            if (findUnsupportedLifecycleControlField(parsed_map, unsupported_field)) {
                RTP_LLM_LOG_WARNING("wake_up failed, unsupported external lifecycle control field: %s",
                                    unsupported_field.c_str());
                writer->SetStatus(400, "Bad Request");
                writer->Write(errorBody("wake_up failed, " + unsupported_field + " is unsupported"));
                return;
            }
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("wake_up failed, invalid request body: %s, exception: [%s]", body.c_str(), e.what());
            writer->SetStatus(400, "Bad Request");
            writer->Write(errorBody("wake_up failed, invalid json body"));
            return;
        }
    }

    auto&      controller = engine_->sleepController();
    const auto result     = controller.wakeUp();
    if (!result.ok) {
        RTP_LLM_LOG_WARNING("wake_up failed: %s", result.message.c_str());
        setErrorStatus(writer, result);
        writer->Write(errorBody(result.message));
        return;
    }

    SleepActionResponse response;
    writer->Write(ToJsonString(response, /*isCompact=*/true));
}

void SleepService::isSleeping(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                              const http_server::HttpRequest&                         request) {
    prepareJsonResponse(writer);
    if (!checkEngine(writer)) {
        return;
    }

    const auto status = engine_->sleepController().status();

    IsSleepingHttpResponse response;
    response.is_sleeping        = status.state == SleepState::SLEEPING;
    response.sleep_mode_enabled = status.sleep_mode_enabled;
    response.effective          = status.effective;
    response.supported_levels   = status.supported_levels;
    response.supported_modes    = status.supported_modes;
    response.state              = sleepStateToString(status.state);
    response.disabled_reason    = status.disabled_reason;
    writer->Write(ToJsonString(response, /*isCompact=*/true));
}

void SleepService::sleepStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                               const http_server::HttpRequest&                         request) {
    prepareJsonResponse(writer);
    if (!checkEngine(writer)) {
        return;
    }

    const auto status = engine_->sleepController().status();

    SleepStatusHttpResponse response;
    fillSleepStatusResponse(status, response);
    writer->Write(ToJsonString(response, /*isCompact=*/true));
}

}  // namespace rtp_llm
