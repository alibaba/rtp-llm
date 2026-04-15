#pragma once

#include <string>
#include <vector>
#include <optional>
#include <tuple>
#include <mutex>
#include <unordered_map>

#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class GenerateStream;

/// @brief Mock GenerateStream for testing P2P connector paths.
/// Does NOT implement IGenerateStream (which has been removed).
/// Provides the minimal methods needed for P2P routing and side-channel apply.
class MockGenerateStream {
public:
    MockGenerateStream() = default;

    // Test helpers for P2P routing
    void setUniqueKey(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        unique_key_ = key;
    }
    std::string uniqueKey() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return unique_key_;
    }

    void setDeadlineMs(int64_t ms) {
        deadline_ms_ = ms;
    }
    int64_t deadlineMs() const {
        return deadline_ms_;
    }

    void setRequestId(int64_t id) {
        request_id_ = id;
    }
    int64_t requestId() const {
        return request_id_;
    }

    void setPrefillAddr(const std::string& ip, uint32_t port) {
        prefill_ip_   = ip;
        prefill_port_ = port;
    }
    std::pair<std::string, uint32_t> getPrefillAddr() const {
        return {prefill_ip_, prefill_port_};
    }

    void setPrefillTpSize(int tp_size) {
        prefill_tp_size_ = tp_size;
    }
    int getPrefillTpSize() const {
        return prefill_tp_size_;
    }

    // Token/reuse methods (matching GenerateStream interface subset)
    void appendTokenId(int batch_id, int token_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        token_ids_[batch_id].push_back(token_id);
    }

    std::vector<int> currentExecuteTokens(int batch_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = token_ids_.find(batch_id);
        if (it != token_ids_.end()) {
            return it->second;
        }
        return {};
    }

    void setReuseBlockNum(int reuse_block_num) {
        std::lock_guard<std::mutex> lock(mutex_);
        reuse_block_num_ = reuse_block_num;
    }
    int reuseBlockNum() {
        std::lock_guard<std::mutex> lock(mutex_);
        return reuse_block_num_;
    }

    void setReuseLength(int total, int local, int remote, int memory = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        total_reuse_len_  = total;
        local_reuse_len_  = local;
        remote_reuse_len_ = remote;
        memory_reuse_len_ = memory;
    }
    std::tuple<int, int, int, int> getReuseLength() {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::make_tuple(total_reuse_len_, local_reuse_len_, remote_reuse_len_, memory_reuse_len_);
    }
    void
    setPrefillReuseLength(int reuse_length, int local_reuse_length, int remote_reuse_length, int memory_reuse_length) {
        std::lock_guard<std::mutex> lock(mutex_);
        total_reuse_len_  = reuse_length;
        local_reuse_len_  = local_reuse_length;
        remote_reuse_len_ = remote_reuse_length;
        memory_reuse_len_ = memory_reuse_length;
    }

    void setPositionIds(const std::vector<int32_t>& ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        position_ids_ = ids;
    }
    std::vector<int32_t> getContextPositionIdsPB() {
        std::lock_guard<std::mutex> lock(mutex_);
        return position_ids_;
    }
    void setContextPositionIds(const std::vector<int32_t>& ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        position_ids_ = ids;
    }

    void setStop(ErrorCode error_code, const std::string& error_msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        error_code_ = error_code;
        error_msg_  = error_msg;
    }

    // Setters for testing
    void setTokenIds(int batch_id, const std::vector<int>& tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        token_ids_[batch_id] = tokens;
    }

    void setProposeInfo(const std::vector<int>& tokens, const TensorPB& probs, const TensorPB& hidden) {
        std::lock_guard<std::mutex> lock(mutex_);
        propose_tokens_ = tokens;
        propose_probs_  = probs;
        propose_hidden_ = hidden;
    }

    std::vector<int> getTokenIds(int batch_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = token_ids_.find(batch_id);
        if (it != token_ids_.end()) {
            return it->second;
        }
        return {};
    }

    size_t tokenCount(int batch_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = token_ids_.find(batch_id);
        if (it != token_ids_.end()) {
            return it->second.size();
        }
        return 0;
    }

    // Getters for test validation
    ErrorCode getErrorCode() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_code_;
    }
    std::string getErrorMsg() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_msg_;
    }
    std::vector<int> getProposeTokens() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return propose_tokens_;
    }
    std::vector<int32_t> getPositionIds() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return position_ids_;
    }
    std::tuple<int, int, int, int> getReuseLengthConst() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::make_tuple(total_reuse_len_, local_reuse_len_, remote_reuse_len_, memory_reuse_len_);
    }

private:
    mutable std::mutex                        mutex_;
    std::unordered_map<int, std::vector<int>> token_ids_;
    std::vector<int32_t>                      position_ids_;
    int                                       reuse_block_num_  = 0;
    int                                       total_reuse_len_  = 0;
    int                                       local_reuse_len_  = 0;
    int                                       remote_reuse_len_ = 0;
    int                                       memory_reuse_len_ = 0;
    std::vector<int>                          propose_tokens_;
    TensorPB                                  propose_probs_;
    TensorPB                                  propose_hidden_;
    std::string                               prefill_ip_;
    uint32_t                                  prefill_port_    = 0;
    int                                       prefill_tp_size_ = 1;
    ErrorCode                                 error_code_      = ErrorCode::NONE_ERROR;
    std::string                               error_msg_;
    std::string                               unique_key_;
    int64_t                                   deadline_ms_{0};
    int64_t                                   request_id_{0};
};

/// @brief Mock Meta implementation for testing P2P routing
class MockMeta: public rtp_llm::Meta {
public:
    MockMeta() = default;

    bool enableMemoryCache() const override {
        return false;
    }
    bool enableRemoteCache() const override {
        return false;
    }
    const std::string& trace_id() const override {
        static const std::string s;
        return s;
    }
    const std::string& unique_id() const override {
        static const std::string s;
        return s;
    }
    const std::vector<int64_t>& tokens() const override {
        static const std::vector<int64_t> v;
        return v;
    }

    void setUniqueKey(const std::string& key) {
        routing_ctx_.unique_key = key;
    }
    void setRequestId(int64_t id) {
        routing_ctx_.request_id = id;
    }
    void setDeadlineMs(int64_t ms) {
        routing_ctx_.deadline_ms = ms;
    }
    void setPrefillAddr(const std::string& ip, uint32_t port) {
        routing_ctx_.prefill_addr = {ip, port};
    }
    void setPrefillTpSize(int tp_size) {
        routing_ctx_.prefill_tp_size = tp_size;
    }
    void setMockStream(MockGenerateStream* stream) {
        mock_stream_ = stream;
    }

    std::optional<P2PRoutingContext> p2pRouting() const override {
        if (routing_ctx_.unique_key.empty()) {
            return std::nullopt;
        }
        return routing_ctx_;
    }

    GenerateStream* generateStream() const override {
        return mock_stream_;
    }
    void setStop(ErrorCode, const std::string&) override {
        // No-op for mock
    }

    // Access to mock stream for test convenience
    MockGenerateStream* mockStream() const {
        return mock_stream_;
    }

private:
    P2PRoutingContext   routing_ctx_;
    MockGenerateStream* mock_stream_ = nullptr;
};

}  // namespace rtp_llm
