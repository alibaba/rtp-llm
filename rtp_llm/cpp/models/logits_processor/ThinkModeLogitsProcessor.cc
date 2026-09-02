#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"
#include <algorithm>
#include <limits>

using namespace std;

namespace rtp_llm {

namespace {

constexpr int32_t kInvalidTokenId = -1;

int32_t k3BoundaryToMask(const std::vector<int>& boundary, int32_t last_token_id) {
    if (boundary.empty()) {
        return kInvalidTokenId;
    }
    if (boundary.size() < 2) {
        return boundary.front();
    }
    return last_token_id == boundary.front() ? boundary[1] : kInvalidTokenId;
}

void maskToken(const torch::Tensor& new_tokens_logits, size_t vocab_size, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size) {
        return;
    }
    new_tokens_logits[token_id] = BaseLogitsProcessor::neg_inf;
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
    return observed_output_tokens >= info.max_thinking_tokens;
}

void maskStopTokens(const torch::Tensor& new_tokens_logits,
                    size_t                vocab_size,
                    const StreamThinkInfo& info) {
    for (const int token_id : info.masked_stop_token_ids) {
        maskToken(new_tokens_logits, vocab_size, token_id);
    }
}

void clearTokenFromBitmask(int32_t* row, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    row[token_id / 32] &= ~(1u << (token_id % 32));
}

void clearTokensFromBitmask(int32_t* row, size_t words, const std::vector<int>& token_ids) {
    for (const int token_id : token_ids) {
        clearTokenFromBitmask(row, words, token_id);
    }
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
           && info.current_output_length >= info.max_thinking_tokens;
}

bool forceThinkEndTokenInBitmask(int32_t* row, size_t words, const StreamThinkInfo& info) {
    const int32_t token_id = info.nextThinkEndToken();
    if (token_id == kInvalidTokenId) {
        return false;
    }
    forceTokenInBitmask(row, words, token_id);
    return true;
}

void applyThinkSpecRowMask(int32_t* row, size_t words, StreamThinkInfo& info) {
    std::fill_n(row, words, SpecLogitsProcessor::kBitmaskAllowAll);
    switch (info.process_state) {
        case ThinkProcessState::NO_THINK:
        case ThinkProcessState::AFTER_THINK: {
            clearTokenFromBitmask(row, words, info.thinkBoundaryToMask(info.begin_think_token_ids));
            clearTokenFromBitmask(row, words, info.thinkBoundaryToMask(info.end_think_token_ids));
            break;
        }
        case ThinkProcessState::IN_THINK: {
            if (info.thinkEndCloseInProgress() || specThinkBudgetExhausted(info)) {
                info.process_state = ThinkProcessState::CLOSING_THINK;
                if (!forceThinkEndTokenInBitmask(row, words, info)) {
                    clearTokenFromBitmask(row, words, info.thinkBoundaryToMask(info.begin_think_token_ids));
                    clearTokensFromBitmask(row, words, info.masked_stop_token_ids);
                }
                break;
            }
            clearTokenFromBitmask(row, words, info.thinkBoundaryToMask(info.begin_think_token_ids));
            clearTokensFromBitmask(row, words, info.masked_stop_token_ids);
            break;
        }
        case ThinkProcessState::CLOSING_THINK: {
            if (!info.pending_forced_think_end_token_ids.empty()) {
                forceTokenInBitmask(row, words, info.pending_forced_think_end_token_ids.front());
                break;
            }
            if (!forceThinkEndTokenInBitmask(row, words, info)) {
                clearTokenFromBitmask(row, words, info.thinkBoundaryToMask(info.begin_think_token_ids));
                clearTokensFromBitmask(row, words, info.masked_stop_token_ids);
            }
            break;
        }
    }
}

}  // namespace

std::vector<int> makeMaskedStopTokenIds(const std::vector<std::vector<int>>& stop_words_list,
                                        int64_t                              eos_token_id,
                                        const std::vector<int>&              end_think_token_ids) {
    std::vector<int> token_ids;
    for (const auto& stop_word : stop_words_list) {
        if (stop_word.size() == 1) {
            token_ids.push_back(stop_word.front());
        }
    }
    if (eos_token_id >= 0 && eos_token_id <= std::numeric_limits<int>::max()) {
        token_ids.push_back(static_cast<int>(eos_token_id));
    }
    if (!end_think_token_ids.empty()) {
        token_ids.erase(
            std::remove(token_ids.begin(), token_ids.end(), end_think_token_ids.front()), token_ids.end());
    }
    std::sort(token_ids.begin(), token_ids.end());
    token_ids.erase(std::unique(token_ids.begin(), token_ids.end()), token_ids.end());
    return token_ids;
}

bool StreamThinkInfo::isActiveThinkState() const {
    return process_state == ThinkProcessState::IN_THINK || process_state == ThinkProcessState::CLOSING_THINK;
}

bool StreamThinkInfo::thinkEndCloseInProgress() const {
    return dfa_ptr && dfa_ptr->status() > 0;
}

int32_t StreamThinkInfo::nextThinkEndToken() const {
    if (!dfa_ptr || dfa_ptr->isFinished()) {
        return kInvalidTokenId;
    }
    const size_t next_token_idx = dfa_ptr->status();
    return next_token_idx < end_think_token_ids.size() ? end_think_token_ids[next_token_idx] : kInvalidTokenId;
}

int32_t StreamThinkInfo::thinkBoundaryToMask(const std::vector<int>& boundary) const {
    if (model_type == "kimi_k3" && !isActiveThinkState()) {
        return k3BoundaryToMask(boundary, last_token_id);
    }
    return boundary.empty() ? kInvalidTokenId : boundary.front();
}

bool StreamThinkInfo::updateState(int32_t token_id) {
    const bool in_think = isActiveThinkState();
    if (!pending_forced_think_end_token_ids.empty()) {
        pending_forced_think_end_token_ids.erase(pending_forced_think_end_token_ids.begin());
        if (pending_forced_think_end_token_ids.empty() && dfa_ptr && dfa_ptr->isFinished()) {
            process_state = ThinkProcessState::AFTER_THINK;
        }
        return in_think;
    }
    current_output_length += 1;
    last_token_id = token_id;
    if (!in_think || !dfa_ptr) {
        return in_think;
    }
    dfa_ptr->next(token_id);
    if (dfa_ptr->isFinished()) {
        process_state = ThinkProcessState::AFTER_THINK;
    } else if (thinkEndCloseInProgress()) {
        process_state = ThinkProcessState::CLOSING_THINK;
    } else if (process_state == ThinkProcessState::CLOSING_THINK) {
        process_state = ThinkProcessState::IN_THINK;
    }
    return in_think;
}

void StreamThinkInfo::precommitThinkEndToken(int32_t token_id) {
    dfa_ptr->next(token_id);
    pending_forced_think_end_token_ids.push_back(token_id);
    current_output_length += 1;
    last_token_id = token_id;
    process_state = ThinkProcessState::CLOSING_THINK;
}

ThinkModeLogitsProcessor::ThinkModeLogitsProcessor(std::vector<StreamThinkInfo> think_infos):
    think_infos_(std::move(think_infos)) {
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
        auto   logits    = inputs.logits[batch_idx];

        switch (info.process_state) {
            case ThinkProcessState::NO_THINK:
            case ThinkProcessState::AFTER_THINK: {
                maskToken(logits, inputs.vocab_size, info.thinkBoundaryToMask(info.begin_think_token_ids));
                maskToken(logits, inputs.vocab_size, info.thinkBoundaryToMask(info.end_think_token_ids));
                break;
            }
            case ThinkProcessState::IN_THINK: {
                if (info.thinkEndCloseInProgress() || thinkBudgetExhausted(inputs, batch_idx, info)) {
                    info.process_state = ThinkProcessState::CLOSING_THINK;
                    forceThinkEndToken(logits, info, inputs.vocab_size);
                    break;
                }

                maskToken(logits, inputs.vocab_size, info.thinkBoundaryToMask(info.begin_think_token_ids));
                maskStopTokens(logits, inputs.vocab_size, info);
                break;
            }
            case ThinkProcessState::CLOSING_THINK: {
                if (!forceThinkEndToken(logits, info, inputs.vocab_size)) {
                    maskToken(logits, inputs.vocab_size, info.thinkBoundaryToMask(info.begin_think_token_ids));
                    maskStopTokens(logits, inputs.vocab_size, info);
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
    const int32_t token_id = info.nextThinkEndToken();
    if (token_id == kInvalidTokenId) {
        return false;
    }

    RTP_LLM_LOG_INFO("sampler enforce think end token: token_id=%d", token_id);
    memFill(new_tokens_logits, vocab_size, (size_t)token_id);

    // Beam/multi-sequence updates need src-batch remapping from updateStatus(),
    // and they do not use the normal async device-state fast path. Keep their
    // historical behavior: force logits now, advance DFA when the sampled token
    // is committed by updateStatus().
    if (info.is_beam_search) {
        return true;
    }

    info.precommitThinkEndToken(token_id);
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
            info.updateState(current_token_id);
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
        state.updateState(draft_token);
    }
    return cap;
}

ThinkModeLogitsProcessorPtr ThinkModeLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                        int32_t                        num,
                                                                        int64_t                        eos_token_id,
                                                                        const std::string&             model_type) {
    auto generate_config        = generate_input->generate_config;
    auto end_think_token_ids    = generate_config->end_think_token_ids;
    bool has_think_boundary_mask = !generate_config->begin_think_token_ids.empty() || !end_think_token_ids.empty();
    if (!has_think_boundary_mask) {
        return nullptr;
    }

    auto masked_stop_token_ids =
        generate_config->in_think_mode
            ? makeMaskedStopTokenIds(generate_config->stop_words_list, eos_token_id, end_think_token_ids)
            : std::vector<int>{};

    std::vector<StreamThinkInfo> think_infos;
    think_infos.reserve(num);
    for (size_t i = 0; i < num; i++) {
        std::shared_ptr<StringContainDFA<size_t, int>> dfa_ptr;
        if (generate_config->in_think_mode && !end_think_token_ids.empty()) {
            dfa_ptr = std::make_shared<StringContainDFA<size_t, int>>(end_think_token_ids);
        }
        think_infos.emplace_back(generate_config->in_think_mode,
                                 generate_config->max_thinking_tokens,
                                 generate_config->begin_think_token_ids,
                                 end_think_token_ids,
                                 generate_input->inputLength(),
                                 0,
                                 generate_config->hasNumBeams() || generate_config->num_return_sequences > 1,
                                 std::move(dfa_ptr),
                                 masked_stop_token_ids,
                                 model_type);
    }
    return std::make_shared<ThinkModeLogitsProcessor>(std::move(think_infos));
}

std::vector<size_t> ThinkModeLogitsProcessor::thinkEndTokensStatus() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<size_t>         status;
    for (auto think_info : think_infos_) {
        status.push_back(think_info.dfa_ptr ? think_info.dfa_ptr->status() : 0);
    }
    return status;
}

}  // namespace rtp_llm
