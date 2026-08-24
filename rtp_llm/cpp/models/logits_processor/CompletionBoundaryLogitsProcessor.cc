#include "rtp_llm/cpp/models/logits_processor/CompletionBoundaryLogitsProcessor.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

void clearTokenFromBitmask(int32_t* row, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    row[token_id / 32] &= ~(1u << (token_id % 32));
}

bool bitmaskAllowsToken(const int32_t* row, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return false;
    }
    return (static_cast<uint32_t>(row[token_id / 32]) & (1u << (token_id % 32))) != 0u;
}

}  // namespace

CompletionBoundaryState::CompletionBoundaryState(std::vector<int32_t> boundary,
                                                 int32_t              input_length,
                                                 bool                 is_beam_search):
    boundary_token_ids(std::move(boundary)), input_length(input_length), is_beam_search(is_beam_search) {
    prefix_table.resize(boundary_token_ids.size(), 0);
    for (size_t i = 1, prefix = 0; i < boundary_token_ids.size(); ++i) {
        while (prefix > 0 && boundary_token_ids[i] != boundary_token_ids[prefix]) {
            prefix = prefix_table[prefix - 1];
        }
        if (boundary_token_ids[i] == boundary_token_ids[prefix]) {
            ++prefix;
        }
        prefix_table[i] = prefix;
    }
}

bool CompletionBoundaryState::finished() const {
    return !boundary_token_ids.empty() && boundary_status == boundary_token_ids.size();
}

void CompletionBoundaryState::advance(int32_t token_id) {
    ++current_output_length;
    if (finished() || boundary_token_ids.empty()) {
        return;
    }
    while (boundary_status > 0 && boundary_token_ids[boundary_status] != token_id) {
        boundary_status = prefix_table[boundary_status - 1];
    }
    if (boundary_token_ids[boundary_status] == token_id) {
        ++boundary_status;
    }
}

CompletionBoundaryLogitsProcessor::CompletionBoundaryLogitsProcessor(
    std::vector<CompletionBoundaryState> states,
    std::vector<int32_t>                 guarded_stop_token_ids,
    int64_t                              request_id,
    std::string                          trace_id):
    states_(std::move(states)),
    guarded_stop_token_ids_(std::move(guarded_stop_token_ids)),
    request_id_(request_id),
    trace_id_(std::move(trace_id)) {
    std::sort(guarded_stop_token_ids_.begin(), guarded_stop_token_ids_.end());
    guarded_stop_token_ids_.erase(
        std::unique(guarded_stop_token_ids_.begin(), guarded_stop_token_ids_.end()), guarded_stop_token_ids_.end());
}

void CompletionBoundaryLogitsProcessor::maskGuardedStops(const torch::Tensor& logits, size_t vocab_size) const {
    for (int32_t token_id : guarded_stop_token_ids_) {
        if (token_id >= 0 && static_cast<size_t>(token_id) < vocab_size) {
            logits[token_id] = BaseLogitsProcessor::neg_inf;
        }
    }
}

void CompletionBoundaryLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK(states_.size() == finish_idx - start_idx);
    for (size_t i = 0; i < states_.size(); ++i) {
        if (!states_[i].finished()) {
            maskGuardedStops(inputs.logits[start_idx + i], inputs.vocab_size);
        }
    }
}

void CompletionBoundaryLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<CompletionBoundaryState> next_states;
    next_states.reserve(src_batch_indices.size());
    for (int src_batch_idx : src_batch_indices) {
        RTP_LLM_CHECK(src_batch_idx >= 0 && static_cast<size_t>(src_batch_idx) < states_.size());
        next_states.push_back(states_[src_batch_idx]);
    }
    states_ = std::move(next_states);
}

void CompletionBoundaryLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(new_tokens.dim() == 2);
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK(states_.size() == static_cast<size_t>(new_tokens.size(0)));
    RTP_LLM_CHECK(num_new_tokens >= 0);
    const auto* data = new_tokens.data_ptr<int32_t>();
    for (size_t i = 0; i < states_.size(); ++i) {
        auto& state = states_[i];
        const int64_t offset = state.is_beam_search ? state.input_length + state.current_output_length : 0;
        RTP_LLM_CHECK_WITH_INFO(offset + num_new_tokens <= new_tokens.size(1),
                                "completion boundary commit exceeds tensor width, offset=%ld num_new_tokens=%d width=%ld",
                                offset,
                                num_new_tokens,
                                new_tokens.size(1));
        for (int32_t j = 0; j < num_new_tokens; ++j) {
            const bool finished_before = state.finished();
            state.advance(data[i * new_tokens.size(1) + offset + j]);
            if (!finished_before && state.finished()) {
                RTP_LLM_LOG_INFO(
                    "completion boundary observed: request_id=%ld trace_id=%s output_tokens=%d batch_idx=%zu",
                    request_id_,
                    trace_id_.c_str(),
                    state.current_output_length,
                    i);
            }
        }
    }
}

bool CompletionBoundaryLogitsProcessor::isSpecVerifyEligible() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return states_.size() == 1 && !states_[0].is_beam_search;
}

bool CompletionBoundaryLogitsProcessor::isStateful() const {
    return isSpecVerifyEligible();
}

int64_t CompletionBoundaryLogitsProcessor::acceptedTokenLen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return states_.size() == 1 && !states_[0].is_beam_search ? states_[0].current_output_length : 0;
}

int CompletionBoundaryLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (states_.size() != 1 || states_[0].is_beam_search || request.propose_step <= 0
        || request.bitmask_cpu_out == nullptr) {
        return request.propose_step;
    }

    CompletionBoundaryState state = states_[0];
    const size_t            words = request.bitmask_size_int32;
    for (int offset = 0; offset <= request.propose_step; ++offset) {
        int32_t* row = request.bitmask_cpu_out + offset * words;
        std::fill_n(row, words, SpecLogitsProcessor::kBitmaskAllowAll);
        if (!state.finished()) {
            for (int32_t token_id : guarded_stop_token_ids_) {
                clearTokenFromBitmask(row, words, token_id);
            }
        }
        if (offset == request.propose_step) {
            return request.propose_step;
        }
        const int32_t draft_token = request.draft_tokens[offset];
        if (!bitmaskAllowsToken(row, words, draft_token)) {
            return offset;
        }
        state.advance(draft_token);
    }
    return request.propose_step;
}

std::vector<size_t> CompletionBoundaryLogitsProcessor::boundaryStatus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<size_t> result;
    result.reserve(states_.size());
    for (const auto& state : states_) {
        result.push_back(state.boundary_status);
    }
    return result;
}

}  // namespace rtp_llm
