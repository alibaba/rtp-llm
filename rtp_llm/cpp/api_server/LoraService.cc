#include "rtp_llm/cpp/api_server/LoraService.h"
#include "rtp_llm/cpp/api_server/Exception.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/api_server/AccessLogWrapper.h"
#include "rtp_llm/cpp/api_server/GangServer.h"
#include "rtp_llm/cpp/api_server/ApiServerMetrics.h"
#include "rtp_llm/cpp/engine_base/EngineBase.h"
#include "rtp_llm/cpp/models/lora/LoraManager.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include "autil/AtomicCounter.h"
#include "autil/legacy/jsonizable.h"
#include "autil/TimeUtility.h"

using namespace autil::legacy;
using namespace autil::legacy::json;

namespace rtp_llm {

class VersionInfo: public autil::legacy::Jsonizable {
public:
    VersionInfo()           = default;
    ~VersionInfo() override = default;

public:
    void Jsonize(autil::legacy::Jsonizable::JsonWrapper& json) override {
        json.Jsonize("models_info", models_info, models_info);
        json.Jsonize("peft_info", peft_info, {});
        json.Jsonize("sampler_info", sampler_info, {});
    }

public:
    std::map<std::string, std::string>                        models_info;
    std::map<std::string, std::map<std::string, std::string>> peft_info;
    std::map<std::string, std::string>                        sampler_info;
};

bool LoraService::hasLora(const std::string& adapter_name, const std::string& lora_path) const {
    std::shared_lock<std::shared_mutex> lock(lora_info_map_mutex_);
    return lora_info_map_.count(adapter_name) != 0 && lora_info_map_.at(adapter_name) == lora_path;
}

std::unordered_map<std::string, std::string> LoraService::getLoraInfoMap() const {
    std::shared_lock<std::shared_mutex> lock(lora_info_map_mutex_);
    return lora_info_map_;
}

void LoraService::update(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                         const http_server::HttpRequest&                         request) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");

    if (!ParallelInfo::globalParallelInfo().isMaster()) {
        RTP_LLM_LOG_WARNING("gang worker should not access /update api directly");
        throw HttpApiServerException(HttpApiServerException::UNSUPPORTED_OPERATION,
                                     "gang worker should not access /update api directly");
    }
    if (!engine_) {
        RTP_LLM_LOG_WARNING("update failed, engine is null");
        throw HttpApiServerException(HttpApiServerException::UNKNOWN_ERROR, "update failed, engine is null");
    }
    auto lora_manager = engine_->getLoraManager();
    if (!lora_manager) {
        RTP_LLM_LOG_WARNING("update failed, lora manager is null");
        throw HttpApiServerException(HttpApiServerException::UNKNOWN_ERROR, "update failed, lora manager is null");
    }

    const auto body          = request.GetBody();
    auto       start_time_ms = autil::TimeUtility::currentTimeInMilliSeconds();

    VersionInfo version_info;
    FromJsonString(version_info, body);

    std::map<std::string, std::string> lora_infos;
    if (version_info.peft_info.count("lora_info") == 0) {
        RTP_LLM_LOG_WARNING("called update route but request has no lora_info, request body: %s", body.c_str());
    } else {
        lora_infos = version_info.peft_info.at("lora_info");
    }
    auto err = lora_manager->checkLoraInfoSize(lora_infos);
    if (err) {
        RTP_LLM_LOG_WARNING("lora_infos size is invalid.");
        throw HttpApiServerException(HttpApiServerException::UPDATE_ERROR, err.value());
    }
    // remove lora
    auto lora_info_map = getLoraInfoMap();
    for (const auto& [adapter_name, lora_path] : lora_info_map) {
        if (lora_infos.count(adapter_name) == 0 || lora_infos.at(adapter_name) != lora_path) {
            if (!removeLora(adapter_name)) {
                RTP_LLM_LOG_WARNING("called update route but remove lora failed, adapter name: %s",
                                    adapter_name.c_str());
                throw HttpApiServerException(HttpApiServerException::UNKNOWN_ERROR, "update remove lora failed");
            }
        }
    }
    // add lora
    for (const auto& [adapter_name, lora_path] : lora_infos) {
        if (!hasLora(adapter_name, lora_path)) {
            if (!addLora(adapter_name, lora_path)) {
                RTP_LLM_LOG_WARNING("called update route but add lora failed, adapter name: %s, lora path: %s",
                                    adapter_name.c_str(),
                                    lora_path.c_str());
                throw HttpApiServerException(HttpApiServerException::UNKNOWN_ERROR, "update add lora failed");
            }
        }
    }
    writer->Write(R"("null")");
    if (metric_reporter_) {
        metric_reporter_->reportUpdateQpsMetric();
        metric_reporter_->reportUpdateLatencyMs(autil::TimeUtility::currentTimeInMilliSeconds() - start_time_ms);
    }
    return;
}

void LoraService::addLoraInternal(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                  const http_server::HttpRequest&                         request) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");

    if (!ParallelInfo::globalParallelInfo().isWorker()) {
        RTP_LLM_LOG_WARNING("gang master should not access /add_lora_internal api directly");
        throw HttpApiServerException(HttpApiServerException::UNSUPPORTED_OPERATION,
                                     "gang master should not access /add_lora_internal api directly");
    }

    if (!engine_) {
        RTP_LLM_LOG_WARNING("add lora internal failed, engine is null");
        throw HttpApiServerException(HttpApiServerException::UNKNOWN_ERROR, "add lora internal failed, engine is null");
    }

    const auto body            = request.GetBody();
    auto       body_map        = AnyCast<JsonMap>(ParseJson(body));
    auto       adapter_name_it = body_map.find("adapter_name");
    auto       lora_path_it    = body_map.find("lora_path");
    if (adapter_name_it == body_map.end() || lora_path_it == body_map.end()) {
        RTP_LLM_LOG_WARNING("add lora internal failed, request has no adapter_name or lora_path, request body: %s",
                            body.c_str());
        throw HttpApiServerException(HttpApiServerException::UNKNOWN_ERROR,
                                     "add lora internal failed, request has no adapter_name or lora_path");
    }
    auto adapter_name = AnyCast<std::string>(adapter_name_it->second);
    auto lora_path    = AnyCast<std::string>(lora_path_it->second);
    if (!addLora(adapter_name, lora_path)) {
        RTP_LLM_LOG_WARNING("add lora internal failed, add lora failed, request body: %s", body.c_str());
        throw HttpApiServerException(HttpApiServerException::UNKNOWN_ERROR, "add lora internal failed");
    }
    writer->Write(R"("null")");
    return;
}

void LoraService::removeLoraInternal(const std::unique_ptr<http_server::HttpResponseWriter>& writer,
                                     const http_server::HttpRequest&                         request) {
    writer->SetWriteType(http_server::HttpResponseWriter::WriteType::Normal);
    writer->AddHeader("Content-Type", "application/json");
    if (!ParallelInfo::globalParallelInfo().isWorker()) {
        RTP_LLM_LOG_WARNING("gang master should not access /remove_lora_internal api directly");
        throw HttpApiServerException(HttpApiServerException::UNSUPPORTED_OPERATION,
                                     "gang master should not access /remove_lora_internal api directly");
    }
    if (!engine_) {
        RTP_LLM_LOG_WARNING("remove lora internal failed, engine is null");
        throw HttpApiServerException(HttpApiServerException::UNKNOWN_ERROR,
                                     "remove lora internal failed, engine is null");
    }

    const auto body            = request.GetBody();
    auto       body_map        = AnyCast<JsonMap>(ParseJson(body));
    auto       adapter_name_it = body_map.find("adapter_name");
    if (adapter_name_it == body_map.end()) {
        RTP_LLM_LOG_WARNING("remove lora internal failed, request has no adapter_name, request body: %s", body.c_str());
        throw HttpApiServerException(HttpApiServerException::UNKNOWN_ERROR,
                                     "remove lora internal failed, request has no adapter_name or lora_path");
    }
    auto adapter_name = AnyCast<std::string>(adapter_name_it->second);
    if (!removeLora(adapter_name)) {
        RTP_LLM_LOG_WARNING("remove lora internal failed, add lora failed, request body: %s", body.c_str());
        throw HttpApiServerException(HttpApiServerException::UNKNOWN_ERROR, "remove lora internal failed");
    }
    writer->Write(R"("null")");
    return;
}

bool LoraService::addLora(const std::string& adapter_name, const std::string& lora_path) {
    // forward to worker
    const auto& parallel_info = ParallelInfo::globalParallelInfo();
    if (ParallelInfo::globalParallelInfo().isMaster() && parallel_info.getWorldSize() > 1) {
        if (!gang_server_) {
            RTP_LLM_LOG_WARNING("add lora failed, gang server is null");
            return false;
        }
        std::map<std::string, std::string> lora_info;
        lora_info["adapter_name"] = adapter_name;
        lora_info["lora_path"]    = lora_path;
        gang_server_->requestWorkers(lora_info, "add_lora_internal", true);
    }
    // self add lora
    if (!engine_) {
        RTP_LLM_LOG_WARNING("add lora failed, engine is null");
        return false;
    }
    auto lora_manager = engine_->getLoraManager();
    if (!lora_manager) {
        RTP_LLM_LOG_WARNING("add lora failed, lora manager is null");
        return false;
    }
    std::unique_ptr<rtp_llm::lora::loraLayerWeightsMap> lora_a_weights, lora_b_weights;
    std::tie(lora_a_weights, lora_b_weights) = weights_loader_->loadLoraWeights(adapter_name, lora_path);
    lora_manager->addLora(adapter_name, *lora_a_weights, *lora_b_weights);
    {
        std::unique_lock<std::shared_mutex> lock(lora_info_map_mutex_);
        lora_info_map_[adapter_name] = lora_path;
    }
    RTP_LLM_LOG_INFO("add lora %s", adapter_name.c_str());
    return true;
}

bool LoraService::removeLora(const std::string& adapter_name) {
    // self remove lora
    if (!engine_) {
        RTP_LLM_LOG_WARNING("remove lora failed, engine is null");
        return false;
    }
    auto lora_manager = engine_->getLoraManager();
    if (!lora_manager) {
        RTP_LLM_LOG_WARNING("remove lora failed, lora manager is null");
        return false;
    }
    lora_manager->removeLora(adapter_name);
    {
        std::unique_lock<std::shared_mutex> lock(lora_info_map_mutex_);
        lora_info_map_.erase(adapter_name);
    }
    RTP_LLM_LOG_INFO("remove lora %s", adapter_name.c_str());
    // forward to worker
    const auto& parallel_info = ParallelInfo::globalParallelInfo();
    if (parallel_info.isMaster() && parallel_info.getWorldSize() > 1) {
        if (!gang_server_) {
            RTP_LLM_LOG_WARNING("remove lora failed, gang server is null");
            return false;
        }
        std::map<std::string, std::string> lora_info;
        lora_info["adapter_name"] = adapter_name;
        gang_server_->requestWorkers(lora_info, "remove_lora_internal", true);
    }
    return true;
}

}  // namespace rtp_llm
