#pragma once

#include "maga_transformer/cpp/core/Buffer.h"
#include <cstdint>
#include <memory>



namespace rtp_llm {

struct SpeculativeExecutorStreamOutput {
public:
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "SpeculativeExecutorStreamOutput { propose_step : " << propose_step;
        if (tokens) {
            debug_string << ", tokens: " << tokens->debugStringWithData<int32_t>();
        }
        if (logits) {
            debug_string << ", logits: " << logits->debugStringWithData<int32_t>();
        }
        if (hidden_states) {
            debug_string << ", hidden_states: " << hidden_states->debugStringWithData<int32_t>();
        }
        if (all_probs) {
            debug_string << ", all_probs" << all_probs->debugStringWithData<int32_t>();
        }
        if (softmax_probs) {
            debug_string << ", softmax_probs" << softmax_probs->debugStringWithData<float>();
        }
        debug_string << "}";
        return debug_string.str();
    }

public:
    size_t propose_step = 0;
    rtp_llm::BufferPtr tokens        = nullptr;  // selected tokens
    rtp_llm::BufferPtr logits        = nullptr;
    rtp_llm::BufferPtr hidden_states = nullptr;
    rtp_llm::BufferPtr loss          = nullptr;
    rtp_llm::BufferPtr all_probs     = nullptr;
    rtp_llm::BufferPtr softmax_probs = nullptr;
};

struct SpeculativeSamplerStreamOutput {
public:
    SpeculativeSamplerStreamOutput(size_t        propose_step,
                                   size_t        accepted_token_nums,
                                   rtp_llm::BufferPtr accepted_tokens,
                                   rtp_llm::BufferPtr logits,
                                   rtp_llm::BufferPtr hidden_states,
                                   rtp_llm::BufferPtr loss,
                                   rtp_llm::BufferPtr softmax_probs,
                                   bool          accepted_bouns_token = false):
        propose_step(propose_step),
        accepted_token_nums(accepted_token_nums),
        accepted_tokens(accepted_tokens),
        logits(logits),
        hidden_states(hidden_states),
        loss(loss),
        softmax_probs(softmax_probs),
        acceped_bouns_token(accepted_bouns_token) {}

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "SpeculativeSamplerStreamOutput { "
                     << "propose_step: " << propose_step << ", accepted_token_nums: " << accepted_token_nums
                     << ", accepted_tokens: " << accepted_tokens->debugStringWithData<int32_t>();
        if (logits) {
            debug_string << ", logits: " << logits->debugStringWithData<float>();
        }
        if (hidden_states) {
            debug_string << ", hidden_states: " << hidden_states->debugStringWithData<float>();
        }
        if (softmax_probs) {
            debug_string << ", softmax_probs: " << softmax_probs->debugStringWithData<float>();
        }
        debug_string << "}";
        return debug_string.str();
    }

public:
    size_t        propose_step;
    size_t        accepted_token_nums;
    rtp_llm::BufferPtr accepted_tokens = nullptr;
    rtp_llm::BufferPtr logits = nullptr;
    rtp_llm::BufferPtr hidden_states = nullptr;
    rtp_llm::BufferPtr loss = nullptr;
    rtp_llm::BufferPtr softmax_probs = nullptr;
    bool          acceped_bouns_token = false;
};

using SpeculativeExecutorStreamOutputPtr = std::shared_ptr<SpeculativeExecutorStreamOutput>;
using SpeculativeSamplerStreamOutputtPtr = std::shared_ptr<SpeculativeSamplerStreamOutput>;
}  // namespace rtp_llm
