#include "rtp_llm/cpp/api_server/SysCmdService.h"

#include "autil/legacy/json.h"
#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/api_server/LogLevelOps.h"

using namespace autil::legacy;
using namespace autil::legacy::json;

namespace rtp_llm {

void SysCmdService::startProfile(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                 const http_server::HttpRequest&                         request,
                                 const StartProfile&                                     start_profile) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
    try {
        auto body  = request.GetBody();
        auto value = ParseJson(body.empty() ? "{}" : body);
        if (value.GetType() == typeid(std::string)) {
            value = ParseJson(AnyCast<std::string>(value));
        }
        Jsonizable::JsonWrapper json(value);
        std::string             trace_name;
        int                     start_step      = 0;
        int                     num_steps       = 0;
        bool                    enable_all_rank = false;
        json.Jsonize("trace_name", trace_name, trace_name);
        json.Jsonize("start_step", start_step, start_step);
        json.Jsonize("num_steps", num_steps, num_steps);
        json.Jsonize("all_tp", enable_all_rank, enable_all_rank);
        json.Jsonize("enable_all_rank", enable_all_rank, enable_all_rank);
        start_profile(trace_name, start_step, num_steps, enable_all_rank);
        writer->Write(R"({"status":"ok"})");
    } catch (const std::exception& e) {
        writer->SetStatus(400, "Bad Request");
        JsonMap error;
        error["error"] = std::string("Failed to start profile: ") + e.what();
        writer->Write(ToJsonString(error, true));
    }
}

void SysCmdService::setLogLevel(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                const http_server::HttpRequest&                         request) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
    const auto body = request.GetBody();
    try {
        auto body_map = AnyCast<JsonMap>(ParseJson(body));
        auto it       = body_map.find("log_level");
        if (it == body_map.end()) {
            RTP_LLM_LOG_WARNING("set log level failed, request has no log level info, request body: %s", body.c_str());
            writer->Write(R"({"error":"set log level failed, request has no log level info"})");
            return;
        }
        auto value = AnyCast<std::string>(it->second);
        if (torch_ext::setLogLevel(value)) {
            writer->Write(R"({"status":"ok"})");
        } else {
            RTP_LLM_LOG_WARNING("set log level failed, invalid log level: %s", value.c_str());
            writer->Write(R"({"error":"set debug log level failed, invalid log level"})");
        }
        return;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING(
            "set debug log level failed, found exception. request body: %s, exception: [%s]", body.c_str(), e.what());
        writer->Write(R"({"error":"set debug log level failed, exception occurred when parse request"})");
        return;
    }
}

}  // namespace rtp_llm