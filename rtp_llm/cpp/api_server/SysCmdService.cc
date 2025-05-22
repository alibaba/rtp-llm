#include "rtp_llm/cpp/api_server/SysCmdService.h"

#include "autil/legacy/json.h"
#include "autil/legacy/jsonizable.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/api_server/LogLevelOps.h"

using namespace autil::legacy;
using namespace autil::legacy::json;

namespace rtp_llm {

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