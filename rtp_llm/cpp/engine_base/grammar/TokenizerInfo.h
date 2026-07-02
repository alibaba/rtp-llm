#pragma once

#include <string>

namespace rtp_llm {

class ModelConfig;

// Opaque cooked tokenizer-info blob; backend-private representation, only
// reachable through BackendAccess.
class TokenizerInfo {
public:
    TokenizerInfo() = default;

    // Returns empty() on failure (grammar silently disabled). Does not throw.
    // model_vocab_size (model_config.vocab_size) widens the grammar vocab over a padded model vocab.
    static TokenizerInfo fromHuggingFaceTokenizer(const ModelConfig& model_config) noexcept;

    // For tests and future disk-cache rehydration.
    static TokenizerInfo fromOpaque(std::string opaque);

    bool empty() const noexcept {
        return opaque_.empty();
    }

    // Backend-only accessor for the opaque blob (review-enforced, no friend isolation).
    class BackendAccess;

private:
    explicit TokenizerInfo(std::string opaque): opaque_(std::move(opaque)) {}
    std::string opaque_;
    friend class BackendAccess;
};

class TokenizerInfo::BackendAccess {
public:
    static const std::string& opaque(const TokenizerInfo& info) {
        return info.opaque_;
    }
};

}  // namespace rtp_llm
