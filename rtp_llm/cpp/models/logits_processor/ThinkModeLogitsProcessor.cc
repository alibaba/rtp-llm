#include "rtp_llm/cpp/models/logits_processor/ThinkModeLogitsProcessor.h"

#include <algorithm>

#include "rtp_llm/cpp/utils/Logger.h"

using namespace std;

namespace rtp_llm {

namespace {

constexpr int32_t kInvalidTokenId = -1;

// Pull the boundary token ids that the masking and spec paths both need.
struct BoundaryTokens {
    int32_t begin = kInvalidTokenId;
    int32_t end   = kInvalidTokenId;

    static BoundaryTokens of(const StreamThinkInfo& info) {
        BoundaryTokens b;
        if (!info.begin_think_token_ids.empty()) b.begin = info.begin_think_token_ids.front();
        if (!info.end_think_token_ids.empty())   b.end   = info.end_think_token_ids.front();
        return b;
    }
};

// What to apply to a row of either logits (process()) or a packed bitmask
// (tryAcceptAndFillBitmask()). The two backends share the same decision logic;
// only the way we materialise it differs.
//
//   FORCE_END_TOKEN    sampler must emit info.end_think_token_ids[next_idx]
//   MASK_BOUNDARIES    forbid begin & end-of-think tokens
//   MASK_BEGIN_ONLY    forbid begin-of-think token (we are still thinking)
//   ALLOW_ALL          no-op
enum class MaskAction { ALLOW_ALL, MASK_BOUNDARIES, MASK_BEGIN_ONLY, FORCE_END_TOKEN };

struct MaskDecision {
    MaskAction action       = MaskAction::ALLOW_ALL;
    int32_t    forced_token = kInvalidTokenId;  // valid iff action==FORCE_END_TOKEN
};

bool dfaClosing(const StreamThinkInfo& info) {
    return info.dfa_ptr && info.dfa_ptr->status() > 0;
}

// Choose what to do with this stream's logits/bitmask row given its current
// think state. `tokens_emitted` is how many decode steps we have observed so
// far; only used to detect think-budget exhaustion. The function may transition
// info.process_state (IN_THINK -> CLOSING_THINK / AFTER_THINK), which is shared
// between the sampler and spec paths.
MaskDecision decideMask(StreamThinkInfo& info, int tokens_emitted) {
    // Active thinking ended: drop into AFTER_THINK and forbid further boundary tokens.
    if (info.dfa_ptr && info.dfa_ptr->isFinished()) {
        info.process_state = ThinkProcessState::AFTER_THINK;
    }

    switch (info.process_state) {
        case ThinkProcessState::NO_THINK:
        case ThinkProcessState::AFTER_THINK:
            return {MaskAction::MASK_BOUNDARIES, kInvalidTokenId};

        case ThinkProcessState::IN_THINK: {
            const bool budget_exhausted = info.dfa_ptr && !info.end_think_token_ids.empty()
                                          && info.max_thinking_tokens > 0
                                          && tokens_emitted >= info.max_thinking_tokens;
            if (dfaClosing(info) || budget_exhausted) {
                info.process_state = ThinkProcessState::CLOSING_THINK;
                // fallthrough into CLOSING_THINK handling below.
            } else {
                return {MaskAction::MASK_BEGIN_ONLY, kInvalidTokenId};
            }
        }
            [[fallthrough]];
        case ThinkProcessState::CLOSING_THINK: {
            if (!info.dfa_ptr || info.dfa_ptr->isFinished() || info.end_think_token_ids.empty()) {
                return {MaskAction::MASK_BOUNDARIES, kInvalidTokenId};
            }
            const auto next_idx = info.dfa_ptr->status();
            if (next_idx >= info.end_think_token_ids.size()) {
                return {MaskAction::MASK_BOUNDARIES, kInvalidTokenId};
            }
            return {MaskAction::FORCE_END_TOKEN, info.end_think_token_ids[next_idx]};
        }
    }
    return {MaskAction::ALLOW_ALL, kInvalidTokenId};
}

// Commit one observed token into the think state machine. Used by:
//   - updateStatus()  for tokens the sampler actually emitted
//   - spec walk       for draft tokens the spec verifier is hypothesising on
// Returns true if the token was consumed as a deferred forced-end token (caller
// must skip its own bookkeeping in that case).
bool commitToken(StreamThinkInfo& info, int32_t token_id) {
    if (!info.pending_forced_think_end_token_ids.empty()) {
        const int32_t expected = info.pending_forced_think_end_token_ids.front();
        info.pending_forced_think_end_token_ids.erase(info.pending_forced_think_end_token_ids.begin());
        if (token_id != expected) {
            RTP_LLM_LOG_WARNING("forced think end token mismatch, expected=%d actual=%d, trust precommitted state",
                                expected, token_id);
        }
        return true;
    }

    info.current_output_length += 1;

    const bool active = info.process_state == ThinkProcessState::IN_THINK
                        || info.process_state == ThinkProcessState::CLOSING_THINK;
    if (!active || info.max_thinking_tokens <= 0 || !info.dfa_ptr) {
        return false;
    }

    info.dfa_ptr->next(token_id);
    if (info.dfa_ptr->isFinished()) {
        info.process_state = ThinkProcessState::AFTER_THINK;
    } else if (dfaClosing(info)) {
        info.process_state = ThinkProcessState::CLOSING_THINK;
    } else if (info.process_state == ThinkProcessState::CLOSING_THINK) {
        info.process_state = ThinkProcessState::IN_THINK;
    }
    return false;
}

// --- bitmask helpers (packed 32-bit rows, bit=1 means allowed) ---

void bitmaskClear(int32_t* row, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) return;
    row[token_id / 32] &= ~(1u << (token_id % 32));
}

void bitmaskForceOnly(int32_t* row, size_t words, int32_t token_id) {
    std::fill_n(row, words, 0);
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) return;
    row[token_id / 32] |= (1u << (token_id % 32));
}

bool bitmaskAllows(const int32_t* row, size_t words, int32_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) return false;
    return (static_cast<uint32_t>(row[token_id / 32]) & (1u << (token_id % 32))) != 0u;
}

}  // namespace

ThinkModeLogitsProcessor::ThinkModeLogitsProcessor(std::vector<StreamThinkInfo> think_infos):
    think_infos_(std::move(think_infos)) {
    spec_eligible_ = think_infos_.size() == 1 && !think_infos_[0].is_beam_search;
    std::lock_guard<std::mutex> lock(mutex_);
    publishSpecSnapshotLocked();
}

// Republishes a snapshot on every process()/updateStatus() tick rather than only
// on actual state transitions. The snapshot is a small shared_ptr<StreamThinkInfo>
// and the contained DFA copy is cheap (a small KMP table). Kept simple on purpose.
void ThinkModeLogitsProcessor::publishSpecSnapshotLocked() {
    ++spec_snapshot_version_;
    if (!spec_eligible_) {
        std::atomic_store_explicit(&spec_snapshot_,
                                   std::shared_ptr<const StreamThinkInfo>(),
                                   std::memory_order_release);
        return;
    }
    auto snapshot = std::make_shared<StreamThinkInfo>(think_infos_[0].copy());
    std::atomic_store_explicit(&spec_snapshot_,
                               std::shared_ptr<const StreamThinkInfo>(snapshot),
                               std::memory_order_release);
}

void ThinkModeLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK(think_infos_.size() == finish_idx - start_idx);

    const int32_t* input_lengths    = inputs.input_lengths.data_ptr<int32_t>();
    const int32_t* sequence_lengths = inputs.sequence_lengths.data_ptr<int32_t>();

    for (size_t i = 0; i < think_infos_.size(); ++i) {
        auto&        info      = think_infos_[i];
        const size_t batch_idx = i + start_idx;
        const auto&  row       = inputs.logits[batch_idx];
        const auto   bounds    = BoundaryTokens::of(info);

        const int tokens_emitted = std::max(sequence_lengths[batch_idx] - input_lengths[batch_idx],
                                            info.current_output_length);
        MaskDecision decision = decideMask(info, tokens_emitted);

        // FORCE_END_TOKEN may downgrade to MASK_BOUNDARIES if the chosen token id
        // would be out-of-vocab (defensive: misconfigured per-request override).
        if (decision.action == MaskAction::FORCE_END_TOKEN) {
            if (decision.forced_token < 0 || static_cast<size_t>(decision.forced_token) >= inputs.vocab_size) {
                RTP_LLM_LOG_WARNING("forceThinkEndToken: end_think_token_id=%d out of vocab_size=%zu; skip force",
                                    decision.forced_token, inputs.vocab_size);
                decision.action = MaskAction::MASK_BOUNDARIES;
            }
        }

        switch (decision.action) {
            case MaskAction::ALLOW_ALL:
                break;
            case MaskAction::MASK_BEGIN_ONLY:
                if (bounds.begin >= 0 && static_cast<size_t>(bounds.begin) < inputs.vocab_size) {
                    row[bounds.begin] = BaseLogitsProcessor::neg_inf;
                }
                break;
            case MaskAction::MASK_BOUNDARIES:
                if (bounds.begin >= 0 && static_cast<size_t>(bounds.begin) < inputs.vocab_size) {
                    row[bounds.begin] = BaseLogitsProcessor::neg_inf;
                }
                if (bounds.end >= 0 && static_cast<size_t>(bounds.end) < inputs.vocab_size) {
                    row[bounds.end] = BaseLogitsProcessor::neg_inf;
                }
                break;
            case MaskAction::FORCE_END_TOKEN:
                RTP_LLM_LOG_INFO("sampler enforce think end token");
                memFill(row, inputs.vocab_size, static_cast<size_t>(decision.forced_token));
                if (!info.is_beam_search) {
                    info.dfa_ptr->next(decision.forced_token);
                    info.pending_forced_think_end_token_ids.push_back(decision.forced_token);
                    info.current_output_length += 1;
                    info.process_state = info.dfa_ptr->isFinished() ? ThinkProcessState::AFTER_THINK
                                                                    : ThinkProcessState::CLOSING_THINK;
                }
                break;
        }
    }
    publishSpecSnapshotLocked();
}

void ThinkModeLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    std::lock_guard<std::mutex>  lock(mutex_);
    std::vector<StreamThinkInfo> new_think_infos;
    new_think_infos.reserve(src_batch_indices.size());
    for (auto src_batch_idx : src_batch_indices) {
        new_think_infos.push_back(think_infos_[src_batch_idx].copy());
    }
    think_infos_ = std::move(new_think_infos);
    publishSpecSnapshotLocked();
}

void ThinkModeLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    RTP_LLM_CHECK(2 == new_tokens.dim());
    std::lock_guard<std::mutex> lock(mutex_);
    RTP_LLM_CHECK(think_infos_.size() == (size_t)new_tokens.size(0));

    for (size_t i = 0; i < think_infos_.size(); ++i) {
        auto&      info   = think_infos_[i];
        const bool active = info.process_state == ThinkProcessState::IN_THINK
                            || info.process_state == ThinkProcessState::CLOSING_THINK;
        if (!active && info.pending_forced_think_end_token_ids.empty()) {
            info.current_output_length += num_new_tokens;
            continue;
        }
        if (info.max_thinking_tokens <= 0 || !info.dfa_ptr) {
            info.current_output_length += num_new_tokens;
            continue;
        }
        if (info.pending_forced_think_end_token_ids.empty() && info.dfa_ptr->isFinished()) {
            info.process_state = ThinkProcessState::AFTER_THINK;
            info.current_output_length += num_new_tokens;
            continue;
        }

        const auto offset = info.is_beam_search ? (info.current_output_length + info.input_length) : 0;
        if (!info.is_beam_search) {
            RTP_LLM_CHECK_WITH_INFO(num_new_tokens <= new_tokens.size(1),
                                    "think mode commit token count exceeds tensor width, num_new_tokens=%d, "
                                    "new_tokens.size(1)=%ld",
                                    num_new_tokens, new_tokens.size(1));
        }

        for (int32_t j = 0; j < num_new_tokens; ++j) {
            const int32_t token_id = new_tokens.data_ptr<int>()[i * new_tokens.size(1) + j + offset];
            commitToken(info, token_id);
        }
    }
    publishSpecSnapshotLocked();
}

bool ThinkModeLogitsProcessor::isSpecVerifyEligible() const {
    return spec_eligible_;
}

bool ThinkModeLogitsProcessor::isStateful() const {
    return spec_eligible_;
}

int ThinkModeLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    if (!spec_eligible_ || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return request.propose_step;
    }
    auto snapshot = std::atomic_load_explicit(&spec_snapshot_, std::memory_order_acquire);
    if (!snapshot) return request.propose_step;

    StreamThinkInfo state = snapshot->copy();
    int             cap   = request.propose_step;
    const size_t    W     = request.bitmask_size_int32;

    for (int offset = 0; offset <= request.propose_step; ++offset) {
        int32_t* row = request.bitmask_cpu_out + offset * W;
        std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);

        const auto    bounds   = BoundaryTokens::of(state);
        MaskDecision  decision = decideMask(state, state.current_output_length);
        switch (decision.action) {
            case MaskAction::ALLOW_ALL:
                break;
            case MaskAction::MASK_BEGIN_ONLY:
                bitmaskClear(row, W, bounds.begin);
                break;
            case MaskAction::MASK_BOUNDARIES:
                bitmaskClear(row, W, bounds.begin);
                bitmaskClear(row, W, bounds.end);
                break;
            case MaskAction::FORCE_END_TOKEN:
                bitmaskForceOnly(row, W, decision.forced_token);
                break;
        }

        if (offset == request.propose_step) break;

        const int32_t draft_token = request.draft_tokens[offset];
        if (!bitmaskAllows(row, W, draft_token)) {
            cap = offset;
            break;
        }
        commitToken(state, draft_token);
    }
    return cap;
}

ThinkModeLogitsProcessorPtr ThinkModeLogitsProcessor::fromGenerateInput(std::shared_ptr<GenerateInput> generate_input,
                                                                        int32_t                        num) {
    auto       generate_config         = generate_input->generate_config;
    const auto end_think_token_ids     = generate_config->end_think_token_ids;
    const bool has_think_boundary_mask = !generate_config->begin_think_token_ids.empty() || !end_think_token_ids.empty();
    if (!has_think_boundary_mask) {
        return nullptr;
    }
    const bool has_think_budget =
        generate_config->in_think_mode && generate_config->max_thinking_tokens > 0 && !end_think_token_ids.empty();
    const bool is_beam = generate_config->hasNumBeams() || generate_config->num_return_sequences > 1;

    std::vector<StreamThinkInfo> infos;
    infos.reserve(num);
    for (int32_t i = 0; i < num; ++i) {
        std::shared_ptr<StringContainDFA<size_t, int>> dfa;
        if (has_think_budget) {
            dfa = std::make_shared<StringContainDFA<size_t, int>>(end_think_token_ids);
        }
        infos.emplace_back(generate_config->in_think_mode,
                           generate_config->max_thinking_tokens,
                           generate_config->begin_think_token_ids,
                           end_think_token_ids,
                           generate_input->inputLength(),
                           0,
                           is_beam,
                           dfa);
    }
    return std::make_shared<ThinkModeLogitsProcessor>(std::move(infos));
}

std::vector<size_t> ThinkModeLogitsProcessorTestPeer::thinkEndTokensStatus(ThinkModeLogitsProcessor& proc) {
    std::lock_guard<std::mutex> lock(proc.mutex_);
    std::vector<size_t>         status;
    status.reserve(proc.think_infos_.size());
    for (const auto& info : proc.think_infos_) {
        status.push_back(info.dfa_ptr ? info.dfa_ptr->status() : 0);
    }
    return status;
}

}  // namespace rtp_llm
