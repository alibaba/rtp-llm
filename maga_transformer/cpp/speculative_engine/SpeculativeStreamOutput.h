#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include <cstdint>
#include <memory>

namespace ft = fastertransformer;

namespace rtp_llm {

struct SpeculativeExecutorStreamOutput {
public:
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "SpeculativeExecutorStreamOutput { "
                     << "tokens: " << tokens->debugStringWithData<int32_t>();
        if (logits) {
            debug_string << ", logits: " << logits->debugStringWithData<int32_t>();
        }
        if (hidden_states) {
            debug_string << ", hidden_states: " << hidden_states->debugStringWithData<int32_t>();
        }
        if (all_probs) {
            debug_string << ", all_probs" << all_probs->debugStringWithData<int32_t>();
        }
        debug_string << "}";
        return debug_string.str();
    }

public:
    ft::BufferPtr tokens        = nullptr;  // selected tokens
    ft::BufferPtr logits        = nullptr;
    ft::BufferPtr hidden_states = nullptr;
    ft::BufferPtr all_probs     = nullptr;
};

struct SpeculativeSamplerStreamOutput {
public:
    SpeculativeSamplerStreamOutput(size_t        propose_step,
                                   size_t        accepted_token_nums,
                                   ft::BufferPtr accepted_tokens,
                                   ft::BufferPtr logits,
                                   ft::BufferPtr hidden_states):
        propose_step(propose_step),
        accepted_token_nums(accepted_token_nums),
        accepted_tokens(accepted_tokens),
        logits(logits),
        hidden_states(hidden_states) {}

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "SpeculativeSamplerStreamOutput { "
                     << "propose_step: " << propose_step << ", accepted_token_nums: " << accepted_token_nums
                     << ", accepted_tokens: " << accepted_tokens->debugStringWithData<int32_t>();
        if (logits) {
            debug_string << ", logits: " << logits->debugStringWithData<int32_t>();
        }
        if (hidden_states) {
            debug_string << ", hidden_states: " << hidden_states->debugStringWithData<int32_t>();
        }
        debug_string << "}";
        return debug_string.str();
    }

public:
    size_t        propose_step;
    size_t        accepted_token_nums;
    ft::BufferPtr accepted_tokens = nullptr;
    ft::BufferPtr logits = nullptr;
    ft::BufferPtr hidden_states = nullptr;
};

using SpeculativeExecutorStreamOutputPtr = std::shared_ptr<SpeculativeExecutorStreamOutput>;
using SpeculativeSamplerStreamOutputtPtr = std::shared_ptr<SpeculativeSamplerStreamOutput>;
}  // namespace rtp_llm