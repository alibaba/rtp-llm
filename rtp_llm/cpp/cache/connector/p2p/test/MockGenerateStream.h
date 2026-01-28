#pragma once

#include <string>
#include <vector>
#include <optional>
#include <tuple>
#include <mutex>
#include <unordered_map>

#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

/// @brief Mock implementation of IGenerateStream for testing
class MockGenerateStream: public IGenerateStream {
public:
    MockGenerateStream(const std::string& prefill_ip = "127.0.0.1", uint32_t prefill_port = 12345):
        prefill_ip_(prefill_ip), prefill_port_(prefill_port), need_call_prefill_(false) {}

    void appendTokenId(int batch_id, int token_id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        token_ids_[batch_id].push_back(token_id);
    }

    std::vector<int> currentExecuteTokens(int batch_id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = token_ids_.find(batch_id);
        if (it != token_ids_.end()) {
            return it->second;
        }
        return {};
    }

    void appendSPInfo(const std::vector<int>& propose_tokens,
                      const TensorPB&         propose_probs,
                      const TensorPB&         propose_hidden) override {
        std::lock_guard<std::mutex> lock(mutex_);
        propose_tokens_ = propose_tokens;
        propose_probs_  = propose_probs;
        propose_hidden_ = propose_hidden;
    }

    std::optional<std::tuple<std::vector<int>, TensorPB, TensorPB>> getSPInfoPB() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (propose_tokens_.empty()) {
            return std::nullopt;
        }
        return std::make_tuple(propose_tokens_, propose_probs_, propose_hidden_);
    }

    int reuseBlockNum() override {
        std::lock_guard<std::mutex> lock(mutex_);
        return reuse_block_num_;
    }

    std::tuple<int, int, int> getReuseLength() override {
        std::lock_guard<std::mutex> lock(mutex_);
        return std::make_tuple(total_reuse_len_, local_reuse_len_, remote_reuse_len_);
    }

    void setPrefillReuseLength(int reuse_length, int local_reuse_length, int remote_reuse_length) override {
        std::lock_guard<std::mutex> lock(mutex_);
        total_reuse_len_  = reuse_length;
        local_reuse_len_  = local_reuse_length;
        remote_reuse_len_ = remote_reuse_length;
    }

    std::pair<std::string, uint32_t> getPrefillAddr() override {
        return {prefill_ip_, prefill_port_};
    }

    std::vector<int32_t> getContextPositionIdsPB() override {
        std::lock_guard<std::mutex> lock(mutex_);
        return position_ids_;
    }

    void setContextPositionIds(const std::vector<int32_t>& ids) override {
        std::lock_guard<std::mutex> lock(mutex_);
        position_ids_ = ids;
    }

    bool waitForRemoteGenerate() override {
        return true;
    }

    // Get original request (GenerateInputPB) for calling prefill server
    const GenerateInputPB* getOriginalRequest() const override {
        return original_request_ ? &original_request_.value() : nullptr;
    }

    // Check if need to call prefill server
    bool needCallPrefill() const override {
        return need_call_prefill_;
    }

    // Set stream to stop with error code and message
    void setStop(ErrorCode error_code, const std::string& error_msg) override {
        // Mock implementation: just store the error code and message
        std::lock_guard<std::mutex> lock(mutex_);
        error_code_ = error_code;
        error_msg_  = error_msg;
    }

    // Set whether to call prefill server
    void setNeedCallPrefill(bool need_call_prefill) {
        need_call_prefill_ = need_call_prefill;
    }

    // Set original request for prefill call
    void setOriginalRequest(const GenerateInputPB& request) {
        original_request_ = request;
    }

    // Setters for testing
    void setTokenIds(int batch_id, const std::vector<int>& tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        token_ids_[batch_id] = tokens;
    }

    void setPositionIds(const std::vector<int32_t>& ids) {
        std::lock_guard<std::mutex> lock(mutex_);
        position_ids_ = ids;
    }

    void setReuseLength(int total, int local, int remote) {
        std::lock_guard<std::mutex> lock(mutex_);
        total_reuse_len_  = total;
        local_reuse_len_  = local;
        remote_reuse_len_ = remote;
    }

    void setProposeInfo(const std::vector<int>& tokens, const TensorPB& probs, const TensorPB& hidden) {
        std::lock_guard<std::mutex> lock(mutex_);
        propose_tokens_ = tokens;
        propose_probs_  = probs;
        propose_hidden_ = hidden;
    }

    // 获取指定 batch_id 的所有 token ids（用于测试验证）
    std::vector<int> getTokenIds(int batch_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = token_ids_.find(batch_id);
        if (it != token_ids_.end()) {
            return it->second;
        }
        return {};
    }

    // 获取指定 batch_id 的 token 数量
    size_t tokenCount(int batch_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = token_ids_.find(batch_id);
        if (it != token_ids_.end()) {
            return it->second.size();
        }
        return 0;
    }

    // 设置 reuse block num
    void setReuseBlockNum(int reuse_block_num) {
        std::lock_guard<std::mutex> lock(mutex_);
        reuse_block_num_ = reuse_block_num;
    }

private:
    mutable std::mutex                        mutex_;
    std::unordered_map<int, std::vector<int>> token_ids_;
    std::vector<int32_t>                      position_ids_;
    int                                       reuse_block_num_  = 0;
    int                                       total_reuse_len_  = 0;
    int                                       local_reuse_len_  = 0;
    int                                       remote_reuse_len_ = 0;
    std::vector<int>                          propose_tokens_;
    TensorPB                                  propose_probs_;
    TensorPB                                  propose_hidden_;
    std::string                               prefill_ip_;
    uint32_t                                  prefill_port_;
    bool                                      need_call_prefill_;
    std::optional<GenerateInputPB>            original_request_;
    ErrorCode                                 error_code_ = ErrorCode::NONE_ERROR;
    std::string                               error_msg_;
};

}  // namespace rtp_llm
