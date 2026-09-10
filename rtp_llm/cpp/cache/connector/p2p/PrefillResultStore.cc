#include "rtp_llm/cpp/cache/connector/p2p/PrefillResultStore.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PRequestDeadline.h"

#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <limits>

namespace rtp_llm {

PrefillResultStore::PrefillResultStore(int timeout_check_interval_ms, int64_t hold_ms, int64_t terminal_ttl_ms):
    timeout_check_interval_ms_(timeout_check_interval_ms),
    hold_ms_(hold_ms),
    terminal_ttl_ms_(terminal_ttl_ms) {}

PrefillResultStore::~PrefillResultStore() {
    if (cleanup_thread_) {
        cleanup_thread_->stop();
    }
}

bool PrefillResultStore::init() {
    cleanup_thread_ = autil::LoopThread::createLoopThread(
        [this]() { checkTimeout(); }, int64_t(timeout_check_interval_ms_) * 1000, "PrefillResultStoreCleanup");
    return cleanup_thread_ != nullptr;
}

bool PrefillResultStore::registerRequest(const std::string& key, int64_t deadline_ms, int64_t request_deadline_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto&                       entry = entries_[key];
    if (entry.terminal) {
        return false;
    }
    entry.deadline_ms          = deadline_ms;
    entry.terminal_deadline_ms = request_deadline_ms;
    entry.resource_owned       = true;
    cv_.notify_all();
    return true;
}

bool PrefillResultStore::beginTransfer(const std::string& key, int64_t deadline_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                       it = entries_.find(key);
    if (it == entries_.end() || it->second.terminal || !it->second.resource_owned) {
        return false;
    }
    it->second.deadline_ms    = deadline_ms;
    it->second.resource_owned = false;
    cv_.notify_all();
    return true;
}

void PrefillResultStore::notify(const std::string& key, int64_t request_deadline_ms, const Data& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto                  now_ms = currentTimeMs();
    auto [it, inserted]                = entries_.try_emplace(key);
    auto& entry                        = it->second;
    if (inserted) {
        entry.terminal_deadline_ms = normalizeP2PRequestDeadline(request_deadline_ms, now_ms, terminal_ttl_ms_);
        entry.deadline_ms = p2pResourceHoldDeadline(entry.terminal_deadline_ms, now_ms, hold_ms_);
    }
    if (entry.terminal || (!entry.resource_owned && now_ms >= entry.deadline_ms)) {
        entry.terminal = true;
        entry.data.reset();
        cv_.notify_all();
        return;
    }
    entry.data = data;
    cv_.notify_all();
}

bool PrefillResultStore::waitAndFill(const std::string&    key,
                                     int64_t               deadline_ms,
                                     Data&                 data,
                                     std::function<bool()> is_cancelled) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        const auto now_ms = currentTimeMs();
        const auto it     = entries_.find(key);
        if ((is_cancelled && is_cancelled()) || now_ms >= deadline_ms) {
            return false;
        }
        if (it != entries_.end()) {
            auto& entry = it->second;
            if (entry.terminal || (!entry.resource_owned && now_ms >= entry.deadline_ms)) {
                return false;
            }
            if (entry.data) {
                data = std::move(*entry.data);
                entry.data.reset();
                entry.terminal = true;
                cv_.notify_all();
                return true;
            }
        }
        cv_.wait_for(lock, std::chrono::milliseconds(std::min<int64_t>(10, deadline_ms - now_ms)));
    }
}

void PrefillResultStore::waitAndFill(const std::string&               key,
                                     int64_t                          deadline_ms,
                                     P2PConnectorStartLoadResponsePB& response,
                                     std::function<bool()>            is_cancelled) {
    Data data;
    if (!waitAndFill(key, deadline_ms, data, std::move(is_cancelled))) {
        response.set_error_code(transErrorCodeToRPC(ErrorCode::P2P_CONNECTOR_SCHEDULER_FILL_RESPONSE_FAILED));
        response.set_error_message("prefill result unavailable, cancelled, or expired");
        return;
    }
    fillPayload(data, *response.mutable_payload());
    response.set_error_code(ErrorCodePB::NONE_ERROR);
    response.clear_error_message();
}

void PrefillResultStore::seal(const std::string& key, int64_t request_deadline_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto&                       entry = entries_[key];
    entry.terminal_deadline_ms =
        std::max(entry.terminal_deadline_ms,
                 normalizeP2PRequestDeadline(request_deadline_ms, currentTimeMs(), terminal_ttl_ms_));
    entry.data.reset();
    entry.terminal = true;
    cv_.notify_all();
}

void PrefillResultStore::checkTimeout() {
    checkTimeout(currentTimeMs());
}

void PrefillResultStore::checkTimeout(int64_t now_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    bool                        changed = false;
    for (auto it = entries_.begin(); it != entries_.end();) {
        auto& entry = it->second;
        if (entry.resource_owned && !entry.terminal) {
            ++it;
        } else if (now_ms >= entry.terminal_deadline_ms) {
            changed = changed || !entry.terminal;
            it = entries_.erase(it);
        } else {
            if (!entry.terminal && now_ms >= entry.deadline_ms) {
                entry.data.reset();
                entry.terminal = true;
                changed = true;
            }
            ++it;
        }
    }
    lock.unlock();
    if (changed) {
        cv_.notify_all();
    }
}

void PrefillResultStore::fillPayload(const Data& data, SideChannelPayloadPB& payload) {
    payload.set_has_first_generate_token(data.has_first_token);
    if (data.has_first_token) {
        payload.set_first_generate_token_id(data.first_token_id);
    }
    payload.set_total_reuse_len(data.total_reuse_len);
    payload.set_local_reuse_len(data.local_reuse_len);
    payload.set_remote_reuse_len(data.remote_reuse_len);
    payload.set_memory_reuse_len(data.memory_reuse_len);
    payload.set_disk_reuse_len(data.disk_reuse_len);
    if (!data.propose_tokens.empty()) {
        auto* tensor = (*payload.mutable_tensors())["propose_tokens"].mutable_tensor();
        tensor->set_data_type(TensorPB::INT32);
        tensor->add_shape(data.propose_tokens.size());
        std::vector<int32_t> tokens(data.propose_tokens.begin(), data.propose_tokens.end());
        tensor->set_int32_data(tokens.data(), tokens.size() * sizeof(int32_t));
    }
    if (data.propose_probs.data_type() == TensorPB::FP32 || !data.propose_probs.fp16_data().empty()
        || !data.propose_probs.bf16_data().empty() || !data.propose_probs.fp32_data().empty()) {
        (*payload.mutable_tensors())["propose_probs"].mutable_tensor()->CopyFrom(data.propose_probs);
    }
    if (data.propose_hidden.data_type() == TensorPB::FP32 || !data.propose_hidden.fp16_data().empty()
        || !data.propose_hidden.bf16_data().empty() || !data.propose_hidden.fp32_data().empty()) {
        (*payload.mutable_tensors())["propose_hidden"].mutable_tensor()->CopyFrom(data.propose_hidden);
    }
    if (!data.position_ids.empty()) {
        auto* tensor = (*payload.mutable_tensors())["position_ids"].mutable_tensor();
        tensor->set_data_type(TensorPB::INT32);
        tensor->add_shape(data.position_ids.size());
        tensor->set_int32_data(data.position_ids.data(), data.position_ids.size() * sizeof(int32_t));
    }
}

}  // namespace rtp_llm
