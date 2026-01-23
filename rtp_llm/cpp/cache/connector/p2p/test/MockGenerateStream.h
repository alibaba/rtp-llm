#pragma once

#include <string>
#include <vector>
#include <optional>
#include <tuple>

#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"

namespace rtp_llm {

/// @brief Mock implementation of IGenerateStream for testing
class MockGenerateStream: public IGenerateStream {
public:
    MockGenerateStream(const std::string& prefill_ip = "127.0.0.1", uint32_t prefill_port = 12345):
        prefill_ip_(prefill_ip), prefill_port_(prefill_port) {}

    void appendTokenId(int batch_id, int token_id) override {}

    std::vector<int> currentExecuteTokens(int batch_id) override {
        return {};
    }

    void appendSPInfo(const std::vector<int>& propose_tokens,
                      const TensorPB&         propose_probs,
                      const TensorPB&         propose_hidden) override {}

    std::optional<std::tuple<std::vector<int>, TensorPB, TensorPB>> getSPInfoPB() override {
        return std::nullopt;
    }

    int reuseBlockNum() override {
        return 0;
    }

    std::tuple<int, int, int> getReuseLength() override {
        return {0, 0, 0};
    }

    void setPrefillReuseLength(int reuse_length, int local_reuse_length, int remote_reuse_length) override {}

    std::pair<std::string, uint32_t> getPrefillAddr() override {
        return {prefill_ip_, prefill_port_};
    }

    std::vector<int32_t> getContextPositionIdsPB() override {
        return {};
    }

    void setContextPositionIds(const std::vector<int32_t>& ids) override {}

    bool waitForRemoteGenerate() override {
        return true;
    }

private:
    std::string prefill_ip_;
    uint32_t    prefill_port_;
};

}  // namespace rtp_llm
