#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include <algorithm>

using namespace std;

namespace rtp_llm {

namespace {

constexpr int32_t kInvalidTokenId = -1;

int32_t firstTokenOrInvalid(const std::vector<int>& token_ids) {
    if (token_ids.empty()) {
        return kInvalidTokenId;
    }
    return token_ids.front();
}

void maskToken(const torch::Tensor& new_tokens_logits, size_t vocab_size, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
        return;
    }
    new_tokens_logits[token_id] = BaseLogitsProcessor::neg_inf;
}

bool isActiveThinkState(const StreamThinkInfo& info) {
    return info.process_state == ThinkProcessState::IN_THINK || info.process_state == ThinkProcessState::CLOSING_THINK;
}

bool advanceAdaptiveStart(StreamThinkInfo& info, int32_t token_id) {
    if (info.process_state == ThinkProcessState::UNDECIDED) {
        if (info.begin_think_token_ids.empty() || token_id != info.begin_think_token_ids.front()) {
            info.process_state = ThinkProcessState::NO_THINK;
            return true;
        }
        info.begin_think_token_index = 1;
        if (info.begin_think_token_index == info.begin_think_token_ids.size()) {
            info.process_state             = ThinkProcessState::IN_THINK;
            info.think_start_output_length = info.current_output_length;
        } else {
            info.process_state = ThinkProcessState::OPENING_THINK;
        }
        return true;
    }
    if (info.process_state != ThinkProcessState::OPENING_THINK) {
        return false;
    }

    const auto expected = info.begin_think_token_ids[info.begin_think_token_index];
    if (token_id != expected) {
        RTP_LLM_LOG_WARNING("adaptive think start token mismatch, expected=%d actual=%d", expected, token_id);
        info.process_state = ThinkProcessState::NO_THINK;
        return true;
    }
    ++info.begin_think_token_index;
    if (info.begin_think_token_index == info.begin_think_token_ids.size()) {
        info.process_state             = ThinkProcessState::IN_THINK;
        info.think_start_output_length = info.current_output_length;
    }
    return true;
}

bool transitionToAfterThinkIfClosed(StreamThinkInfo& info) {
    if (!info.dfa_ptr || !info.dfa_ptr->isFinished()) {
        return false;
    }
    info.process_state = ThinkProcessState::AFTER_THINK;
    return true;
}

int generatedTokens(const SamplerInputs& inputs, size_t batch_idx) {
    int* input_lengths    = inputs.input_lengths.data_ptr<int32_t>();
    int* sequence_lengths = inputs.sequence_lengths.data_ptr<int32_t>();
    return sequence_lengths[batch_idx] - input_lengths[batch_idx];
}

bool thinkBudgetExhausted(const SamplerInputs& inputs, size_t batch_idx, const StreamThinkInfo& info) {
    if (!info.dfa_ptr || info.end_think_token_ids.empty() || info.max_thinking_tokens <= 0) {
        return false;
    }

    const int observed_output_tokens = std::max(generatedTokens(inputs, batch_idx), info.current_output_length);
    return observed_output_tokens - info.think_start_output_length >= info.max_thinking_tokens;
}

bool thinkEndCloseInProgress(const StreamThinkInfo& info) {
    return info.dfa_ptr && info.dfa_ptr->status() > 0;
}

bool consumePendingForcedThinkEndToken(StreamThinkInfo& info, int32_t current_token_id) {
    if (info.pending_forced_think_end_token_ids.empty()) {
        return false;
    }
    const int32_t expected_token_id = info.pending_forced_think_end_token_ids.front();
    info.pending_forced_think_end_token_ids.erase(info.pending_forced_think_end_token_ids.begin());
    if (current_token_id != expected_token_id) {
        RTP_LLM_LOG_WARNING("forced think end token mismatch, expected=%d actual=%d, trust precommitted state",
                            expected_token_id,
                            current_token_id);
    }
    return true;
}

void maskThinkBoundaryTokens(const torch::Tensor& new_tokens_logits, size_t vocab_size, const StreamThinkInfo& info) {
    maskToken(new_tokens_logits, vocab_size, firstTokenOrInvalid(info.begin_think_token_ids));
    maskToken(new_tokens_logits, vocab_size, firstTokenOrInvalid(info.end_think_token_ids));
}

void clearTokenFromBitmask(int32_t* row, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    row[token_id / 32] &= ~(1u << (token_id % 32));
}

void forceTokenInBitmask(int32_t* row, size_t words, int32_t token_id) {
    std::fill_n(row, words, 0);
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    row[token_id / 32] |= (1u << (token_id % 32));
}

bool bitmaskAllowsToken(const int32_t* row, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return false;
    }
    const uint32_t word = static_cast<uint32_t>(row[token_id / 32]);
    return (word & (1u << (token_id % 32))) != 0u;
}

bool specThinkBudgetExhausted(const StreamThinkInfo& info) {
    return info.dfa_ptr && !info.end_think_token_ids.empty() && info.max_thinking_tokens > 0
           && info.current_output_length - info.think_start_output_length >= info.max_thinking_tokens;
}

bool forceThinkEndTokenInBitmask(int32_t* row, size_t words, const StreamThinkInfo& info) {
    if (!info.dfa_ptr || info.dfa_ptr->isFinished() || info.end_think_token_ids.empty()) {
        return false;
    }
    auto next_token_idx = info.dfa_ptr->status();
    if (next_token_idx >= info.end_think_token_ids.size()) {
        return false;
    }
    forceTokenInBitmask(row, words, info.end_think_token_ids[next_token_idx]);
    return true;
}

void applyThinkSpecRowMask(int32_t* row, size_t words, StreamThinkInfo& info) {
    std::fill_n(row, words, SpecLogitsProcessor::kBitmaskAllowAll);
    switch (info.process_state) {
        case ThinkProcessState::UNDECIDED:
            break;
        case ThinkProcessState::OPENING_THINK:
            if (info.begin_think_token_index < info.begin_think_token_ids.size()) {
                forceTokenInBitmask(row, words, info.begin_think_token_ids[info.begin_think_token_index]);
            }
            break;
        case ThinkProcessState::NO_THINK:
        case ThinkProcessState::AFTER_THINK: {
            clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.begin_think_token_ids));
            clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.end_think_token_ids));
            break;
        }
        case ThinkProcessState::IN_THINK: {
            if (transitionToAfterThinkIfClosed(info)) {
                clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.begin_think_token_ids));
                clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.end_think_token_ids));
                break;
            }
            if (thinkEndCloseInProgress(info) || specThinkBudgetExhausted(info)) {
                info.process_state = ThinkProcessState::CLOSING_THINK;
                if (!forceThinkEndTokenInBitmask(row, words, info)) {
                    clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.begin_think_token_ids));
                    clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.end_think_token_ids));
                }
                break;
            }
            clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.begin_think_token_ids));
            break;
        }
        case ThinkProcessState::CLOSING_THINK: {
            if (transitionToAfterThinkIfClosed(info)) {
                clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.begin_think_token_ids));
                clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.end_think_token_ids));
                break;
            }
            if (!forceThinkEndTokenInBitmask(row, words, info)) {
                clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.begin_think_token_ids));
                clearTokenFromBitmask(row, words, firstTokenOrInvalid(info.end_think_token_ids));
            }
            break;
        }
    }
}

void advanceThinkStateForSpec(StreamThinkInfo& info, int32_t token_id) {
    if (consumePendingForcedThinkEndToken(info, token_id)) {
        return;
    }

    info.current_output_length += 1;
    if (advanceAdaptiveStart(info, token_id)) {
        return;
    }
    if (!isActiveThinkState(info) || info.max_thinking_tokens <= 0 || !info.dfa_ptr) {
        return;
    }

    info.dfa_ptr->next(token_id);
    if (info.dfa_ptr->isFinished()) {
        info.process_state = ThinkProcessState::AFTER_THINK;
    } else if (thinkEndCloseInProgress(info)) {
        info.process_state = ThinkProcessState::CLOSING_THINK;
    } else if (info.process_state == ThinkProcessState::CLOSING_THINK) {
        info.process_state = ThinkProcessState::IN_THINK;
    }
}

}  // namespace

ThinkModeLogitsProcessor::ThinkModeLogitsProcessor(std::vector<StreamThinkInfo> think_infos):
    think_infos_(think_infos) {
    std::lock_guard<std::mutex> lock(mutex_);
    publishSpecSnapshotLocked();
};

void ThinkModeLogitsProcessor::publishSpecSnapshotLocked() {
    auto snapshot      = std::make_shared<ThinkModeSpecSnapshot>();
    snapshot->version  = ++spec_snapshot_version_;
    snapshot->eligible = think_infos_.size() == 1 && !think_infos_[0].is_beam_search;
    if (snapshot->eligible) {
        snapshot->info = think_infos_[0].copy();
    }
    std::atomic_store_explicit(
        &spec_snapshot_, std::shared_ptr<const ThinkModeSpecSnapshot>(snapshot), std::memory_order_release);
}

void ThinkModeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK(think_infos_.size() == finish_idx - start_idx);

    for (size_t i = 0; i < think_infos_.size(); ++i) {
        auto&  info      = think_infos_[i];
        size_t batch_idx = i + start_idx;

        switch (info.process_state) {
            case ThinkProcessState::UNDECIDED:
                break;
            case ThinkProcessState::OPENING_THINK:
                if (info.begin_think_token_index < info.begin_think_token_ids.size()) {
                    memFill(inputs.logits[batch_idx],
                            inputs.vocab_size,
                            static_cast<size_t>(info.begin_think_token_ids[info.begin_think_token_index]));
                }
                break;
            case ThinkProcessState::NO_THINK:
            case ThinkProcessState::AFTER_THINK: {
                maskThinkBoundaryTokens(inputs.logits[batch_idx], inputs.vocab_size, info);
                break;
            }
            case ThinkProcessState::IN_THINK: {
                if (transitionToAfterThinkIfClosed(info)) {
                    maskThinkBoundaryTokens(inputs.logits[batch_idx], inputs.vocab_size, info);
                    break;
                }

                if (thinkEndCloseInProgress(info) || thinkBudgetExhausted(inputs, batch_idx, info)) {
                    info.process_state = ThinkProcessState::CLOSING_THINK;
                    forceThinkEndToken(inputs.logits[batch_idx], info, inputs.vocab_size);
                    break;
                }

                maskToken(inputs.logits[batch_idx], inputs.vocab_size, firstTokenOrInvalid(info.begin_think_token_ids));
                break;
            }
            case ThinkProcessState::CLOSING_THINK: {
                if (transitionToAfterThinkIfClosed(info)) {
                    maskThinkBoundaryTokens(inputs.logits[batch_idx], inputs.vocab_size, info);
                    break;
                }

                if (!forceThinkEndToken(inputs.logits[batch_idx], info, inputs.vocab_size)) {
                    maskThinkBoundaryTokens(inputs.logits[batch_idx], inputs.vocab_size, info);
                }
                break;
            }
        }
    }
    publishSpecSnapshotLocked();
}

bool ThinkModeLogitsProcessor::forceThinkEndToken(const torch::Tensor& new_tokens_logits,
                                                  StreamThinkInfo&     info,
                                                  size_t               vocab_size) {
    if (!info.dfa_ptr || info.dfa_ptr->isFinished() || info.end_think_token_ids.empty()) {
        return false;
    }
    auto next_token_idx = info.dfa_ptr->status();
    if (next_token_idx >= info.end_think_token_ids.size()) {
        return false;
    }

    RTP_LLM_LOG_INFO("sampler enforce think end token");
    auto token_id = info.end_think_token_ids[next_token_idx];
    memFill(new_tokens_logits, vocab_size, (size_t)token_id);

    // Beam/multi-sequence updates need src-batch remapping from updateStatus(),
    // and they do not use the normal async device-state fast path. Keep their
    // historical behavior: force logits now, advance DFA when the sampled token
    // is committed by updateStatus().
    if (info.is_beam_search) {
        return true;
    }

    info.dfa_ptr->next(token_id);
    info.pending_forced_think_end_token_ids.push_back(token_id);
    info.current_output_length += 1;
    if (info.dfa_ptr->isFinished()) {
        info.process_state = ThinkProcessState::AFTER_THINK;
    } else {
        info.process_state = ThinkProcessState::CLOSING_THINK;
    }
    return true;
}

void ThinkModeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::lock_guard<std::mutex>  lock(mutex_);
    std::vector<StreamThinkInfo> new_think_infos;
    for (auto src_batch_idx : src_batch_indices) {
        new_think_infos.push_back(think_infos_[src_batch_idx].copy());
    }
    think_infos_ = new_think_infos;
    publishSpecSnapshotLocked();
}

void ThinkModeLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK(think_infos_.size() == (size_t)new_tokens.size(0));

    for (size_t i = 0; i < think_infos_.size(); i++) {
        auto& info = think_infos_[i];
        if ((info.process_state == ThinkProcessState::NO_THINK || info.process_state == ThinkProcessState::AFTER_THINK)
            && info.pending_forced_think_end_token_ids.empty()) {
            info.current_output_length += num_new_tokens;
            continue;
        }
        if (info.pending_forced_think_end_token_ids.empty() && transitionToAfterThinkIfClosed(info)) {
            info.current_output_length += num_new_tokens;
            continue;
        }

        auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;

        if (!info.is_beam_search) {
            RTP_LLM_CHECK_WITH_INFO(num_new_tokens <= new_tokens.size(1),
                                    "think mode commit token count exceeds tensor width, num_new_tokens=%d, "
                                    "new_tokens.size(1)=%ld",
                                    num_new_tokens,
                                    new_tokens.size(1));
        }

        for (size_t j = 0; j < num_new_tokens; ++j) {
            auto current_token_id = new_tokens.data_ptr<int>()[i * new_tokens.size(1) + j + offset];
            if (consumePendingForcedThinkEndToken(info, current_token_id)) {
                continue;
            }

            info.current_output_length += 1;
            if (advanceAdaptiveStart(info, current_token_id)) {
                continue;
            }
            if (!isActiveThinkState(info)) {
                continue;
            }
            if (!info.dfa_ptr) {
                continue;
            }

            info.dfa_ptr->next(current_token_id);
            if (info.dfa_ptr->isFinished()) {
                info.process_state = ThinkProcessState::AFTER_THINK;
            } else if (thinkEndCloseInProgress(info)) {
                info.process_state = ThinkProcessState::CLOSING_THINK;
            } else if (info.process_state == ThinkProcessState::CLOSING_THINK) {
                info.process_state = ThinkProcessState::IN_THINK;
            }
        }
    }
    publishSpecSnapshotLocked();
}

bool ThinkModeLogitsProcessor::isSpecVerifyEligible() const {
    auto snapshot = std::atomic_load_explicit(&spec_snapshot_, std::memory_order_acquire);
    return snapshot && snapshot->eligible;
}

bool ThinkModeLogitsProcessor::isStateful() const {
    return isSpecVerifyEligible();
}

int64_t ThinkModeLogitsProcessor::acceptedTokenLen() const {
    auto snapshot = std::atomic_load_explicit(&spec_snapshot_, std::memory_order_acquire);
    if (!snapshot || !snapshot->eligible) {
        return 0;
    }
    return snapshot->info.current_output_length;
}

int ThinkModeLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    auto snapshot = std::atomic_load_explicit(&spec_snapshot_, std::memory_order_acquire);
    if (!snapshot || !snapshot->eligible || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return request.propose_step;
    }

    StreamThinkInfo state = snapshot->info.copy();
    int             cap   = request.propose_step;
    const size_t    W     = request.bitmask_size_int32;

    for (int offset = 0; offset <= request.propose_step; ++offset) {
        int32_t* row = request.bitmask_cpu_out + offset * W;
        applyThinkSpecRowMask(row, W, state);
        if (offset == request.propose_step) {
            break;
        }

        const int32_t draft_token = request.draft_tokens[offset];
        if (!bitmaskAllowsToken(row, W, draft_token)) {
            cap = offset;
            break;
        }
        advanceThinkStateForSpec(state, draft_token);
    }
    return cap;
}

ThinkModeLogitsProcessorPtr ThinkModeLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                        int32_t                        num) {
    auto generate_config         = generate_input->generate_config;
    auto end_think_token_ids     = generate_config->end_think_token_ids;
    bool has_think_boundary_mask = !generate_config->begin_think_token_ids.empty() || !end_think_token_ids.empty();
    if (!generate_config->enable_think_logits_processor || !has_think_boundary_mask) {
        return nullptr;
    }

    auto thinking_mode = generate_config->thinking_mode;
    if (thinking_mode == ThinkingMode::UNSPECIFIED) {
        thinking_mode = generate_config->in_think_mode ? ThinkingMode::ENABLED : ThinkingMode::DISABLED;
    }
    const bool has_think_budget = (thinking_mode == ThinkingMode::ENABLED || thinking_mode == ThinkingMode::ADAPTIVE)
                                  && generate_config->max_thinking_tokens > 0 && !end_think_token_ids.empty();
    if (thinking_mode == ThinkingMode::ADAPTIVE && generate_config->begin_think_token_ids.empty()) {
        RTP_LLM_LOG_WARNING("adaptive thinking requires non-empty begin_think_token_ids; disable think processor");
        return nullptr;
    }

    auto processor_ptr = std::make_shared<ThinkModeLogitsProcessor>();
    for (size_t i = 0; i < num; i++) {
        std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr;
        if (has_think_budget) {
            dfa_ptr = std::make_shared<StringContainDFA<size_t, int>>(end_think_token_ids);
        }
        StreamThinkInfo              think_info(thinking_mode,
                                   generate_config->max_thinking_tokens,
                                   generate_config->begin_think_token_ids,
                                   end_think_token_ids,
                                   generate_input->inputLength(),
                                   0,
                                   generate_config->hasNumBeams() || generate_config->num_return_sequences > 1,
                                   dfa_ptr);
        std::vector<StreamThinkInfo> think_infos = {think_info};
        auto                         ptr         = std::make_shared<ThinkModeLogitsProcessor>(think_infos);

        processor_ptr->insert(ptr, 1);
    }
    return processor_ptr;
}

std::vector<size_t> ThinkModeLogitsProcessor::thinkEndTokensStatus() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<size_t>         status;
    for (auto think_info : think_infos_) {
        auto dfa = think_info.dfa_ptr;
        status.push_back(dfa ? dfa->status() : 0);
    }
    return status;
}

}  // namespace rtp_llm
