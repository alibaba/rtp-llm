#pragma once

#include "rtp_llm/cpp/api_server/http_server/http_server/HttpResponseWriter.h"
#include "rtp_llm/cpp/api_server/http_server/http_server/HttpRequest.h"
#include "rtp_llm/cpp/api_server/WeightsLoader.h"

namespace autil {
class AtomicCounter;
}

namespace rtp_llm {

class EngineBase;
class GangServer;
class ApiServerMetricReporter;

class LoraService {
public:
    LoraService(const std::shared_ptr<EngineBase>&              engine,
                const std::shared_ptr<GangServer>&              gang_server,
                const std::shared_ptr<WeightsLoader>&           weights_loader,
                const std::map<std::string, std::string>&       lora_infos,
                const std::shared_ptr<ApiServerMetricReporter>& metric_reporter):
        engine_(engine), gang_server_(gang_server), weights_loader_(weights_loader), metric_reporter_(metric_reporter) {
        autil::ScopedTime2 timer;
        if (lora_infos.size() > 1) {
            for (const auto& [adapter_name, lora_path] : lora_infos) {
                addLora(adapter_name, lora_path);
            }
        }
        RTP_LLM_LOG_INFO("cpp update lora weights time: %f s", timer.done_sec());
    }
    ~LoraService() = default;

public:
    void update(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                const http_server::HttpRequest&                         request);
    void addLoraInternal(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                         const http_server::HttpRequest&                         request);
    void removeLoraInternal(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                            const http_server::HttpRequest&                         request);

private:
    bool addLora(const std::string& adapter_name, const std::string& lora_path);
    bool removeLora(const std::string& adapter_name);
    bool hasLora(const std::string& adapter_name, const std::string& lora_path) const;
    std::unordered_map<std::string, std::string> getLoraInfoMap() const;

private:
    std::shared_ptr<EngineBase>                  engine_;
    std::shared_ptr<GangServer>                  gang_server_;
    std::shared_ptr<WeightsLoader>               weights_loader_;
    std::shared_ptr<ApiServerMetricReporter>     metric_reporter_;
    std::unordered_map<std::string, std::string> lora_info_map_;  // {adapter_name: lora_path}
    mutable std::shared_mutex                    lora_info_map_mutex_;
};

}  // namespace rtp_llm
