#include "rtp_llm/cpp/http_server/http_server/HttpServer.h"
#include "rtp_llm/cpp/disaggregate/rtpllm_master/common/UserRequest.h"
#include "autil/StringUtil.h"

namespace rtp_llm {
namespace rtp_llm_master {

class FakeServer {
public:
    static void appendTokenizeService(http_server::HttpServer& http_server, const std::vector<int>& token_ids) {
        auto route = [token_ids](std::unique_ptr<http_server::HttpResponseWriter> writer,
                        const http_server::HttpRequest&                  request) {
            TokenizeResponse response;
            response.token_ids = token_ids;
            writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
            writer->AddHeader("Content-Type", "application/json");
            writer->Write(autil::legacy::ToJsonString(response));
        };
        http_server.RegisterRoute("POST", "/tokenize", route);
    }

    static std::shared_ptr<http_server::HttpServer>
    initServer(const std::string& spec, const std::string& running_task_list, const std::string& finished_task_list, const std::vector<int>& token_ids = {}) {
        auto server = std::make_shared<http_server::HttpServer>();
        auto route  = [running_task_list, finished_task_list](std::unique_ptr<http_server::HttpResponseWriter> writer,
                                                             const http_server::HttpRequest&                  request) {
            // set a very big last_schedule_time
            std::string response_format = R"del(
        {
            "available_concurrency": 32,
            "available_kv_cache": 0,
            "total_kv_cache": 18416,
            "step_latency_ms": 29.23,
            "step_per_minute": 2052,
            "onflight_requests": 0,
            "iterate_count": 1,
            "version": 0,
            "alive": true,
            "running_task_list": %s,
            "finished_task_list": %s
        })del";
            auto        response =
                autil::StringUtil::formatString(response_format, running_task_list.c_str(), finished_task_list.c_str());
            writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
            writer->AddHeader("Content-Type", "application/json");
            writer->Write(response);
        };
        server->RegisterRoute("GET", "/worker_status", route);
        appendTokenizeService(*server, token_ids);
        if (!server->Start(spec)) {
            throw std::runtime_error("failed to start server");
        }
        return server;
    }
};

}  // namespace rtp_llm_master
}  // namespace rtp_llm