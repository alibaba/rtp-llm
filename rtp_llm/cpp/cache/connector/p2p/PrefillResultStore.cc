#include "rtp_llm/cpp/cache/connector/p2p/PrefillResultStore.h"
#include "rtp_llm/cpp/model_rpc/TensorPbConvert.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace rtp_llm {

void PrefillResultStore::updateDeadline(const std::string& unique_key, int64_t deadline_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto&                       entry = entries_[unique_key];
    if (entry.deadline_ms != deadline_ms) {
        entry.deadline_ms = deadline_ms;
        cv_.notify_all();
    }
}

std::optional<PrefillResultStore::SideChannelData>
PrefillResultStore::publishPrefillPayload(const std::string& unique_key, SideChannelData&& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = entries_.find(unique_key);
    if (stopping_ || it == entries_.end() || it->second.terminal || it->second.consumed
        || currentTimeMs() >= it->second.deadline_ms) {
        return std::nullopt;
    }
    std::optional<SideChannelData> retired;
    retired.swap(it->second.side_channel_data);
    it->second.side_channel_data.emplace(std::move(data));
    cv_.notify_all();
    return retired;
}

bool PrefillResultStore::waitPrefillPayloadReady(const std::string&    unique_key,
                                                 int64_t               deadline_ms,
                                                 std::function<bool()> is_cancelled) {
    if (deadline_ms <= 0 || deadline_ms == std::numeric_limits<int64_t>::max()) {
        return false;
    }
    std::unique_lock<std::mutex> lock(mutex_);
    int                          sleep_ms = 1;
    while (true) {
        auto it = entries_.find(unique_key);
        if (stopping_ || (is_cancelled && is_cancelled()) || it == entries_.end() || it->second.terminal
            || it->second.consumed) {
            return false;
        }
        const auto effective_deadline_ms = std::min(deadline_ms, it->second.deadline_ms);
        if (currentTimeMs() >= effective_deadline_ms) {
            return false;
        }
        if (it->second.side_channel_data) {
            return true;
        }
        const auto next_wake =
            std::min(std::chrono::system_clock::now() + std::chrono::milliseconds(sleep_ms),
                     std::chrono::system_clock::time_point(std::chrono::milliseconds(effective_deadline_ms)));
        cv_.wait_until(lock, next_wake);
        sleep_ms = std::min(sleep_ms * 2, 8);
    }
}

std::optional<PrefillResultStore::SideChannelData>
PrefillResultStore::takePrefillPayload(const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = entries_.find(unique_key);
    if (stopping_ || it == entries_.end() || it->second.terminal || it->second.consumed
        || currentTimeMs() >= it->second.deadline_ms || !it->second.side_channel_data) {
        return std::nullopt;
    }
    std::optional<SideChannelData> consumed;
    consumed.swap(it->second.side_channel_data);
    it->second.consumed = true;
    cv_.notify_all();
    return consumed;
}

std::optional<PrefillResultStore::SideChannelData>
PrefillResultStore::clearPrefillPayload(const std::string& unique_key) {
    std::lock_guard<std::mutex>    lock(mutex_);
    std::optional<SideChannelData> retired;
    auto                           it = entries_.find(unique_key);
    if (it != entries_.end()) {
        retired.swap(it->second.side_channel_data);
    }
    return retired;
}

std::optional<PrefillResultStore::SideChannelData> PrefillResultStore::markTerminal(const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto&                       entry = entries_[unique_key];
    entry.terminal                    = true;
    std::optional<SideChannelData> retired;
    retired.swap(entry.side_channel_data);
    cv_.notify_all();
    return retired;
}

void PrefillResultStore::eraseRequest(const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.erase(unique_key);
    cv_.notify_all();
}

void PrefillResultStore::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    cv_.notify_all();
}

grpc::Status PrefillResultStore::fillStartLoadResponsePayload(const SideChannelData&           data,
                                                              P2PConnectorStartLoadResponsePB& response) {
    try {
        // Clear only the payload so each TensorPB is filled exactly once, including on reuse.
        response.clear_payload();
        // Fill response proto from side-channel data
        auto* payload = response.mutable_payload();
        payload->set_has_first_generate_token(data.has_first_token);
        if (data.has_first_token) {
            payload->set_first_generate_token_id(data.first_token_id);
        }
        payload->set_total_reuse_len(data.total_reuse_len);
        payload->set_local_reuse_len(data.local_reuse_len);
        payload->set_remote_reuse_len(data.remote_reuse_len);
        payload->set_memory_reuse_len(data.memory_reuse_len);
        payload->set_disk_reuse_len(data.disk_reuse_len);

        if (!data.propose_tokens.empty()) {
            auto& propose_tensor = (*payload->mutable_tensors())["propose_tokens"];
            auto* tokens_pb      = propose_tensor.mutable_tensor();
            tokens_pb->set_data_type(TensorPB::INT32);
            tokens_pb->add_shape(data.propose_tokens.size());
            std::vector<int32_t> int32_tokens(data.propose_tokens.begin(), data.propose_tokens.end());
            tokens_pb->set_int32_data(int32_tokens.data(), int32_tokens.size() * sizeof(int32_t));
        }
        const auto fill_tensor = [&](const char* name, const torch::Tensor& tensor) {
            if (!tensor.defined()) {
                // An absent SPOutputBuffer previously supplied a default FP32 PB with no shape.
                (*payload->mutable_tensors())[name].mutable_tensor();
                return;
            }
            if (!tensor.device().is_cpu()) {
                throw std::runtime_error("side-channel tensor must be on CPU");
            }
            switch (tensor.scalar_type()) {
                case torch::kFloat32:
                    break;
                case torch::kFloat16:
                case torch::kBFloat16:
                    if (tensor.numel() == 0) {
                        return;
                    }
                    break;
                case torch::kInt32:
                    // Preserve the old filter, which omitted PBs with only int32_data.
                    return;
                default:
                    throw std::runtime_error("unsupported side-channel tensor dtype");
            }
            TensorPbConvert::torchToPb((*payload->mutable_tensors())[name].mutable_tensor(), tensor);
        };
        fill_tensor("propose_probs", data.propose_probs);
        fill_tensor("propose_hidden", data.propose_hidden);
        for (const auto& [name, tensor] : data.first_token_tensors) {
            if (!tensor.defined() || !tensor.device().is_cpu()) {
                throw std::runtime_error("first token output tensor must be defined and on CPU");
            }
            TensorPbConvert::torchToPb((*payload->mutable_tensors())[name].mutable_tensor(), tensor);
        }
        if (!data.position_ids.empty()) {
            auto& pos_tensor = (*payload->mutable_tensors())["position_ids"];
            auto* pos_pb     = pos_tensor.mutable_tensor();
            pos_pb->set_data_type(TensorPB::INT32);
            pos_pb->add_shape(data.position_ids.size());
            pos_pb->set_int32_data(data.position_ids.data(), data.position_ids.size() * sizeof(int32_t));
        }

        RTP_LLM_LOG_DEBUG("fill response from entry: first_token: %ld, total_reuse: %d, local: %d, remote: %d, "
                          "memory: %d, disk: %d",
                          data.first_token_id,
                          data.total_reuse_len,
                          data.local_reuse_len,
                          data.remote_reuse_len,
                          data.memory_reuse_len,
                          data.disk_reuse_len);

        return grpc::Status::OK;
    } catch (const std::exception& error) {
        response.clear_payload();
        return grpc::Status(grpc::StatusCode::INTERNAL, error.what());
    }
}

}  // namespace rtp_llm
