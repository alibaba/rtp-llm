#include "rtp_llm/cpp/models/logits_processor/CompletionBoundaryLogitsProcessor.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

void clearTokenFromBitmask(int32_t* row, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    row[token_id / 32] &= ~(1u << (token_id % 32));
}

void setTokenInBitmask(int32_t* row, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    row[token_id / 32] |= 1u << (token_id % 32);
}

bool bitmaskAllowsToken(const int32_t* row, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return false;
    }
    return (static_cast<uint32_t>(row[token_id / 32]) & (1u << (token_id % 32))) != 0u;
}

}  // namespace

bool CompletionBoundarySpec::hasStatefulCompletionFields() const {
    return !think_close_token_ids.empty() || !response_open_token_ids.empty() || !response_close_token_ids.empty()
           || !tools_open_token_ids.empty() || !tools_close_token_ids.empty() || !whitespace_token_ids.empty();
}

bool CompletionBoundarySpec::isStatefulCompletionGuard() const {
    if (response_close_token_ids.empty() || message_close_token_ids.empty()) {
        return false;
    }
    if (tools_open_token_ids.empty() != tools_close_token_ids.empty()) {
        return false;
    }
    return !starts_in_think || (!think_close_token_ids.empty() && !response_open_token_ids.empty());
}

CompletionTokenSequenceState::CompletionTokenSequenceState(std::vector<int32_t> tokens): token_ids(std::move(tokens)) {
    prefix_table.resize(token_ids.size(), 0);
    for (size_t i = 1, prefix = 0; i < token_ids.size(); ++i) {
        while (prefix > 0 && token_ids[i] != token_ids[prefix]) {
            prefix = prefix_table[prefix - 1];
        }
        if (token_ids[i] == token_ids[prefix]) {
            ++prefix;
        }
        prefix_table[i] = prefix;
    }
}

bool CompletionTokenSequenceState::advance(int32_t token_id) {
    if (token_ids.empty()) {
        return false;
    }
    if (status == token_ids.size()) {
        status = prefix_table.back();
    }
    while (status > 0 && token_ids[status] != token_id) {
        status = prefix_table[status - 1];
    }
    if (token_ids[status] == token_id) {
        ++status;
    }
    return status == token_ids.size();
}

void CompletionTokenSequenceState::reset() {
    status = 0;
}

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

CompletionBoundaryState::CompletionBoundaryState(CompletionBoundarySpec spec,
                                                 int32_t                input_length,
                                                 bool                   is_beam_search):
    input_length(input_length),
    is_beam_search(is_beam_search),
    phase(spec.starts_in_think ? Phase::THINK_BODY : Phase::RESPONSE_BODY),
    think_close(spec.think_close_token_ids),
    response_open(spec.response_open_token_ids),
    response_close(spec.response_close_token_ids),
    tools_open(spec.tools_open_token_ids),
    tools_close(spec.tools_close_token_ids),
    message_close(spec.message_close_token_ids),
    whitespace_token_ids(spec.whitespace_token_ids.begin(), spec.whitespace_token_ids.end()) {
    RTP_LLM_CHECK(spec.isStatefulCompletionGuard());
}

bool CompletionBoundaryState::finished() const {
    if (phase != Phase::LEGACY_BOUNDARY) {
        return phase == Phase::COMPLETE;
    }
    return !boundary_token_ids.empty() && boundary_status == boundary_token_ids.size();
}

bool CompletionBoundaryState::shouldForceStop() const {
    return phase == Phase::COMPLETE || phase == Phase::INVALID;
}

void CompletionBoundaryState::advance(int32_t token_id) {
    ++current_output_length;
    if (finished()) {
        return;
    }
    if (phase != Phase::LEGACY_BOUNDARY) {
        advanceStatefulGuard(token_id);
        return;
    }
    if (boundary_token_ids.empty()) {
        return;
    }
    while (boundary_status > 0 && boundary_token_ids[boundary_status] != token_id) {
        boundary_status = prefix_table[boundary_status - 1];
    }
    if (boundary_token_ids[boundary_status] == token_id) {
        ++boundary_status;
    }
}

bool CompletionBoundaryState::isWhitespace(int32_t token_id) const {
    return whitespace_token_ids.find(token_id) != whitespace_token_ids.end();
}

bool CompletionBoundaryState::advancesBodyContent(const CompletionTokenSequenceState& sequence,
                                                  size_t                              previous_status,
                                                  bool                                completed) {
    // Count a body token only when no structural prefix was pending and the
    // current token did not start or complete the close boundary.
    return !completed && previous_status == 0 && sequence.status == 0;
}

bool CompletionBoundaryState::advanceUnexpectedBoundary(
    int32_t token_id, std::initializer_list<CompletionTokenSequenceState*> sequences) {
    for (auto* sequence : sequences) {
        if (!sequence->token_ids.empty() && sequence->advance(token_id)) {
            markInvalid();
            return true;
        }
    }
    return false;
}

void CompletionBoundaryState::resetStructuralSequences() {
    think_close.reset();
    response_open.reset();
    response_close.reset();
    tools_open.reset();
    tools_close.reset();
    message_close.reset();
}

void CompletionBoundaryState::markInvalid() {
    resetStructuralSequences();
    phase = Phase::INVALID;
}

void CompletionBoundaryState::advanceStatefulGuard(int32_t token_id) {
    switch (phase) {
        case Phase::THINK_BODY:
            if (think_close.advance(token_id)) {
                resetStructuralSequences();
                phase = Phase::EXPECT_RESPONSE_OPEN;
                return;
            }
            advanceUnexpectedBoundary(
                token_id, {&response_open, &response_close, &tools_open, &tools_close, &message_close});
            return;
        case Phase::EXPECT_RESPONSE_OPEN:
            if (response_open.advance(token_id)) {
                resetStructuralSequences();
                phase = Phase::RESPONSE_BODY;
                return;
            }
            advanceUnexpectedBoundary(
                token_id, {&think_close, &response_close, &tools_open, &tools_close, &message_close});
            return;
        case Phase::RESPONSE_BODY: {
            const size_t previous_status = response_close.status;
            const bool   completed       = response_close.advance(token_id);
            if (completed) {
                resetStructuralSequences();
                phase = Phase::AFTER_RESPONSE;
            } else {
                if (!isWhitespace(token_id) && advancesBodyContent(response_close, previous_status, completed)) {
                    response_has_content = true;
                }
                advanceUnexpectedBoundary(
                    token_id, {&think_close, &response_open, &tools_open, &tools_close, &message_close});
            }
            return;
        }
        case Phase::AFTER_RESPONSE:
            if (!tools_open.token_ids.empty() && tools_open.advance(token_id)) {
                resetStructuralSequences();
                tools_has_content    = false;
                tools_channel_opened = true;
                phase                = Phase::TOOLS_BODY;
                return;
            }
            if (message_close.advance(token_id)) {
                if (response_has_content && !tools_channel_opened) {
                    resetStructuralSequences();
                    phase = Phase::COMPLETE;
                } else {
                    markInvalid();
                }
                return;
            }
            advanceUnexpectedBoundary(token_id, {&think_close, &response_open, &response_close, &tools_close});
            return;
        case Phase::TOOLS_BODY: {
            const size_t previous_status = tools_close.status;
            const bool   completed       = tools_close.advance(token_id);
            if (completed) {
                if (tools_has_content) {
                    resetStructuralSequences();
                    phase = Phase::EXPECT_MESSAGE_CLOSE;
                } else {
                    markInvalid();
                }
            } else {
                if (!isWhitespace(token_id) && advancesBodyContent(tools_close, previous_status, completed)) {
                    tools_has_content = true;
                }
                advanceUnexpectedBoundary(
                    token_id, {&think_close, &response_open, &response_close, &tools_open, &message_close});
            }
            return;
        }
        case Phase::EXPECT_MESSAGE_CLOSE:
            if (message_close.advance(token_id)) {
                resetStructuralSequences();
                phase = Phase::COMPLETE;
                return;
            }
            advanceUnexpectedBoundary(
                token_id, {&think_close, &response_open, &response_close, &tools_open, &tools_close});
            return;
        case Phase::COMPLETE:
        case Phase::INVALID:
        case Phase::LEGACY_BOUNDARY:
            return;
    }
}

CompletionBoundaryLogitsProcessor::CompletionBoundaryLogitsProcessor(std::vector<CompletionBoundaryState> states,
                                                                     std::vector<int32_t> guarded_stop_token_ids):
    states_(std::move(states)), guarded_stop_token_ids_(std::move(guarded_stop_token_ids)) {
    std::sort(guarded_stop_token_ids_.begin(), guarded_stop_token_ids_.end());
    guarded_stop_token_ids_.erase(std::unique(guarded_stop_token_ids_.begin(), guarded_stop_token_ids_.end()),
                                  guarded_stop_token_ids_.end());
}

void CompletionBoundaryLogitsProcessor::maskGuardedStops(const torch::Tensor& logits, size_t vocab_size) const {
    for (int32_t token_id : guarded_stop_token_ids_) {
        if (token_id >= 0 && static_cast<size_t>(token_id) < vocab_size) {
            logits[token_id] = BaseLogitsProcessor::neg_inf;
        }
    }
}

void CompletionBoundaryLogitsProcessor::forceGuardedStop(const torch::Tensor& logits, size_t vocab_size) const {
    const bool has_valid_stop = std::any_of(guarded_stop_token_ids_.begin(),
                                            guarded_stop_token_ids_.end(),
                                            [vocab_size](int32_t token_id) {
                                                return token_id >= 0 && static_cast<size_t>(token_id) < vocab_size;
                                            });
    if (!has_valid_stop) {
        return;
    }
    logits.fill_(BaseLogitsProcessor::neg_inf);
    for (int32_t token_id : guarded_stop_token_ids_) {
        if (token_id >= 0 && static_cast<size_t>(token_id) < vocab_size) {
            logits[token_id] = 0.0f;
        }
    }
}

void CompletionBoundaryLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK(states_.size() == finish_idx - start_idx);
    for (size_t i = 0; i < states_.size(); ++i) {
        if (states_[i].shouldForceStop()) {
            forceGuardedStop(inputs.logits[start_idx + i], inputs.vocab_size);
        } else if (!states_[i].finished()) {
            maskGuardedStops(inputs.logits[start_idx + i], inputs.vocab_size);
        }
    }
}

void CompletionBoundaryLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::lock_guard<std::mutex>          lock(mutex_);
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
        auto&         state  = states_[i];
        const int64_t offset = state.is_beam_search ? state.input_length + state.current_output_length : 0;
        RTP_LLM_CHECK_WITH_INFO(
            offset + num_new_tokens <= new_tokens.size(1),
            "completion boundary commit exceeds tensor width, offset=%ld num_new_tokens=%d width=%ld",
            offset,
            num_new_tokens,
            new_tokens.size(1));
        for (int32_t j = 0; j < num_new_tokens; ++j) {
            state.advance(data[i * new_tokens.size(1) + offset + j]);
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
    const bool              has_valid_stop = std::any_of(guarded_stop_token_ids_.begin(),
                                            guarded_stop_token_ids_.end(),
                                            [&request](int32_t token_id) {
                                                return token_id >= 0
                                                       && static_cast<size_t>(token_id) < request.vocab_size;
                                            });
    for (int offset = 0; offset <= request.propose_step; ++offset) {
        int32_t* row = request.bitmask_cpu_out + offset * words;
        std::fill_n(row, words, SpecLogitsProcessor::kBitmaskAllowAll);
        if (state.shouldForceStop() && has_valid_stop) {
            std::fill_n(row, words, 0);
            for (int32_t token_id : guarded_stop_token_ids_) {
                setTokenInBitmask(row, words, token_id);
            }
        } else if (!state.finished()) {
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
    std::vector<size_t>         result;
    result.reserve(states_.size());
    for (const auto& state : states_) {
        result.push_back(state.phase == CompletionBoundaryState::Phase::LEGACY_BOUNDARY ?
                             state.boundary_status :
                             static_cast<size_t>(state.phase));
    }
    return result;
}

}  // namespace rtp_llm
