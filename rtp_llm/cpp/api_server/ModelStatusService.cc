#include "rtp_llm/cpp/api_server/ModelStatusService.h"
#include "autil/StringUtil.h"
#include "autil/TimeUtility.h"

namespace rtp_llm {

void ModelStatusService::modelStatus(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                     const http_server::HttpRequest&                         request) {

    std::string model_content_format = R"del({
    "object": "list",
    "data": [
        {
            "id": "AsyncModel",
            "object": "model",
            "created": %s,
            "owned_by": "owner",
            "root": null,
            "parent": null,
            "permission": null
        }
    ]
})del";
    const auto  timestamp_s          = autil::TimeUtility::currentTimeInSeconds();
    auto model_content = autil::StringUtil::formatString(model_content_format, std::to_string(timestamp_s).c_str());

    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
    writer->Write(model_content);
}

}  // namespace rtp_llm