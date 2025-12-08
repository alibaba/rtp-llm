#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorStreamStore.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <thread>
#include <chrono>

namespace rtp_llm {

PrefillConnectorStreamStore::PrefillConnectorStreamStore() {}

PrefillConnectorStreamStore::~PrefillConnectorStreamStore() {
    stop();
}

bool PrefillConnectorStreamStore::init() {
    stop_flag_            = false;
    timeout_check_thread_ = std::thread([this]() { this->checkTimeout(); });
    RTP_LLM_LOG_INFO("PrefillConnectorStreamStore init success");
    return true;
}

void PrefillConnectorStreamStore::stop() {
    if (stop_flag_) {
        return;
    }
    stop_flag_ = true;
    if (timeout_check_thread_.joinable()) {
        timeout_check_thread_.join();
    }
}

void PrefillConnectorStreamStore::addStream(const std::string& unique_key, GenerateStreamPtr stream) {
    std::lock_guard<std::mutex> lock(stream_map_mutex_);
    stream_map_[unique_key] = stream;
}

std::shared_ptr<GenerateStream> PrefillConnectorStreamStore::stealStream(const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(stream_map_mutex_);
    auto                        it = stream_map_.find(unique_key);
    if (it == stream_map_.end()) {
        return nullptr;
    }
    auto stream = it->second;
    stream_map_.erase(it);
    return stream;
}

void PrefillConnectorStreamStore::checkTimeout() {
    while (!stop_flag_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));  // 每100ms检查一次
        std::lock_guard<std::mutex> lock(stream_map_mutex_);
        int64_t                     current_time_ms = currentTimeMs();
        for (auto it = stream_map_.begin(); it != stream_map_.end();) {
            auto& [_, stream] = *it;
            if (stream && current_time_ms >= stream->getDeadlineMs()) {
                RTP_LLM_LOG_WARNING(
                    "PrefillConnectorStreamStore: stream timeout, unique_key: %s, deadline_ms: %ld, current_time_ms: %ld",
                    it->first.c_str(),
                    stream->getDeadlineMs(),
                    current_time_ms);
                it = stream_map_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

}  // namespace rtp_llm
