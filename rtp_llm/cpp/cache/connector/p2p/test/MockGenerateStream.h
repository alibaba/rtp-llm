#pragma once

#include <string>
#include <vector>
#include <optional>
#include <mutex>

#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

/// @brief GenerateStream 的最小具体实现，用于测试 P2P 路由路径，不实现真正的推理逻辑。
class MockGenerateStream: public GenerateStream {
public:
    MockGenerateStream(const std::shared_ptr<GenerateInput>& input):
        GenerateStream(input, createMockModelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr) {}

    ErrorResult<GenerateOutputs> nextOutput() override {
        return ErrorResult<GenerateOutputs>(GenerateOutputs{});
    }
    void updateOutput(const StreamUpdateInfo&) override {}

private:
    static ModelConfig createMockModelConfig() {
        ModelConfig config;
        config.max_seq_len = 4096;  // Set a reasonable max_seq_len for testing
        return config;
    }
};

/// @brief Mock Meta implementation for testing P2P routing.
/// Holds routing context directly; GenerateStream* is optional and only
/// needed for decode-side tests that exercise side-channel apply.
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
    void setGenerateStream(GenerateStream* stream) {
        stream_ = stream;
    }

    std::optional<P2PRoutingContext> p2pRouting() const override {
        if (routing_ctx_.unique_key.empty()) {
            return std::nullopt;
        }
        return routing_ctx_;
    }

    GenerateStream* generateStream() const override {
        return stream_;
    }

private:
    P2PRoutingContext routing_ctx_;
    GenerateStream*   stream_ = nullptr;
};

}  // namespace rtp_llm
