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

        if (cum_log_probs) {
            debug_string << ", cum_log_probs" << cum_log_probs->debugStringWithData<int32_t>();
        }
        debug_string << "}";
        return debug_string.str();
    }

public:
    ft::BufferPtr tokens        = nullptr;  // selected tokens
    ft::BufferPtr logits        = nullptr;
    ft::BufferPtr hidden_states = nullptr;
    ft::BufferPtr cum_log_probs = nullptr;
};
struct SpeculativeSamplerStreamOutput {
public:
    SpeculativeSamplerStreamOutput(size_t propose_step, size_t accepted_token_nums, ft::BufferPtr accepted_tokens):
        propose_step(propose_step), accepted_token_nums(accepted_token_nums), accepted_tokens(accepted_tokens) {}

    SpeculativeSamplerStreamOutput(size_t        propose_step,
                                   size_t        accepted_token_nums,
                                   ft::BufferPtr accepted_tokens,
                                   ft::BufferPtr logits,
                                   ft::BufferPtr hidden_states,
                                   ft::BufferPtr cum_log_probs):
        propose_step(propose_step),
        accepted_token_nums(accepted_token_nums),
        accepted_tokens(accepted_tokens),
        logits(logits),
        hidden_states(hidden_states),
        cum_log_probs(cum_log_probs) {}

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
        if (cum_log_probs) {
            debug_string << ", cum_log_probs" << cum_log_probs->debugStringWithData<int32_t>();
        }
        debug_string << "}";
        return debug_string.str();
    }

public:
    size_t        propose_step;
    size_t        accepted_token_nums;
    ft::BufferPtr accepted_tokens;
    ft::BufferPtr logits;
    ft::BufferPtr hidden_states;
    ft::BufferPtr cum_log_probs;
};

using SpeculativeExecutorStreamOutputPtr = std::shared_ptr<SpeculativeExecutorStreamOutput>;
using SpeculativeSamplerStreamOutputtPtr = std::shared_ptr<SpeculativeSamplerStreamOutput>;
}  // namespace rtp_llm