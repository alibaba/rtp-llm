#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include <c10/util/Exception.h>
#include <dlpack/dlpack.h>

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

namespace {

ErrorResult<int32_t> validateVocabSize(RtpGrammarMatcher& matcher, size_t model_vocab_size, const char* path) {
    auto grammar_vocab_size_or = matcher.vocabSize();
    if (!grammar_vocab_size_or.ok()) {
        return grammar_vocab_size_or.status();
    }
    int32_t grammar_vocab_size = grammar_vocab_size_or.value();
    if (grammar_vocab_size <= 0) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS,
                         std::string("grammar ") + path + ": invalid grammar vocab size "
                             + std::to_string(grammar_vocab_size));
    }
    if (static_cast<size_t>(grammar_vocab_size) > model_vocab_size) {
        return ErrorInfo(ErrorCode::GRAMMAR_VOCAB_EXCEEDS_MODEL_VOCAB,
                         std::string("grammar vocab exceeds model vocab in ") + path
                             + " (grammar=" + std::to_string(grammar_vocab_size)
                             + ", model=" + std::to_string(model_vocab_size) + ")");
    }
    return ErrorResult<int32_t>(std::move(grammar_vocab_size));
}

ErrorResult<int>
prepareSpecMask(RtpGrammarMatcher& matcher, int64_t eos_token_id, const SpecLogitsProcessorRequest& request) {
    if (eos_token_id < 0 || static_cast<size_t>(eos_token_id) >= request.vocab_size) {
        matcher.markFinished();
        return ErrorInfo(ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB,
                         "grammar MTP verify: eos_token_id (" + std::to_string(eos_token_id)
                             + ") out of model vocab (vocab=" + std::to_string(request.vocab_size) + ")");
    }

    const size_t bitmask_words = request.bitmask_size_int32;
    auto         fail_closed   = [&](const ErrorInfo& error) -> ErrorResult<int> {
        matcher.markFinished();
        forceTokenInBitmask(request.bitmask_cpu_out, bitmask_words, eos_token_id);
        return error;
    };

    auto grammar_vocab_size_or = validateVocabSize(matcher, request.vocab_size, "MTP verify");
    if (!grammar_vocab_size_or.ok()) {
        return fail_closed(grammar_vocab_size_or.status());
    }
    const int32_t grammar_vocab_size = grammar_vocab_size_or.value();

    int                      cap                 = request.propose_step;
    int                      provisional_accepts = 0;
    std::optional<ErrorInfo> verify_error;
    for (int offset = 0; offset <= request.propose_step; ++offset) {
        int32_t* row = request.bitmask_cpu_out + static_cast<size_t>(offset) * bitmask_words;
        std::fill_n(row, bitmask_words, SpecLogitsProcessorRequest::kBitmaskAllowAll);

        bool row_active = !matcher.finished();
        if (row_active) {
            auto terminated = matcher.isTerminated();
            if (!terminated.ok()) {
                verify_error = terminated.status();
                cap          = offset;
                break;
            }
            row_active = !terminated.value();
        }
        if (row_active) {
            int64_t  dl_shape[2];
            DLTensor dl = makeSingleRowBitmaskView(
                row, static_cast<int32_t>(SpecLogitsProcessorRequest::bitmaskWordCount(grammar_vocab_size)), dl_shape);
            auto filled = matcher.fillBitmask(&dl, 0);
            if (!filled.ok()) {
                verify_error = filled.status();
                cap          = offset;
                break;
            }
            // xgrammar returns false for an all-true mask; the row is still active.
            clearBitmaskTokenRange(row, bitmask_words, grammar_vocab_size, static_cast<int64_t>(request.vocab_size));
        } else {
            forceTokenInBitmask(row, bitmask_words, eos_token_id);
        }

        if (offset == request.propose_step) {
            break;
        }
        if (!row_active) {
            cap = offset;
            break;
        }

        const int32_t draft_token = request.draft_tokens[offset];
        if (draft_token < 0 || static_cast<size_t>(draft_token) >= request.vocab_size
            || !bitmaskAllowsToken(row, bitmask_words, draft_token)) {
            cap = offset;
            break;
        }

        auto accepted = matcher.acceptToken(draft_token);
        if (!accepted.ok()) {
            verify_error = accepted.status();
            cap          = offset;
            break;
        }
        if (!accepted.value()) {
            cap = offset;
            break;
        }
        ++provisional_accepts;
    }

    // Verification is speculative: restore the committed matcher state before
    // returning. Actual accepted tokens are committed later through updateStatus().
    auto rollback_error = provisional_accepts > 0 ? matcher.rollback(provisional_accepts) : ErrorInfo::OkStatus();
    if (rollback_error.hasError()) {
        if (verify_error.has_value()) {
            return fail_closed(ErrorInfo(rollback_error.code(),
                                         "grammar MTP verify rollback failed after error: " + verify_error->ToString()
                                             + "; rollback_error=" + rollback_error.ToString()));
        }
        return fail_closed(rollback_error);
    }
    if (verify_error.has_value()) {
        return fail_closed(*verify_error);
    }
    return ErrorResult<int>(std::move(cap));
}

}  // namespace

class GrammarLogitsProcessor::DecodeMaskBuilder final {
public:
    ErrorInfo apply(const torch::Tensor& logits,
                    RtpGrammarMatcher&   matcher,
                    int64_t              accepted_token_len,
                    int64_t              eos_token_id,
                    int32_t              grammar_vocab_size) {
        try {
            if (device_mask_state_.token_len == accepted_token_len) {
                return applyState(logits, device_mask_state_, eos_token_id);
            }

            auto state_or = build(matcher, accepted_token_len, grammar_vocab_size);
            if (!state_or.ok()) {
                device_mask_state_ = finished(accepted_token_len);
                return state_or.status();
            }

            device_mask_state_ = std::move(state_or.value());
            return applyState(logits, device_mask_state_, eos_token_id);
        } catch (const std::bad_alloc& e) {
            device_mask_state_ = finished(accepted_token_len);
            return detail::grammarMaskBuildError("decode", e);
        } catch (const c10::Error& e) {
            device_mask_state_ = finished(accepted_token_len);
            return detail::grammarMaskBuildError("decode", e);
        }
    }

private:
    enum class DeviceMaskMode {
        MASK,
        TERMINATED,
        FINISHED,
    };

    struct DeviceMaskState {
        DeviceMaskMode mode          = DeviceMaskMode::FINISHED;
        int64_t        token_len     = -1;
        bool           mask_required = false;
        torch::Tensor  packed_allow_mask_cpu;
        int32_t        grammar_vocab_size = 0;
    };

    static DeviceMaskState finished(int64_t accepted_token_len) {
        DeviceMaskState state;
        state.token_len = accepted_token_len;
        state.mode      = DeviceMaskMode::FINISHED;
        return state;
    }

    ErrorResult<DeviceMaskState>
    build(RtpGrammarMatcher& matcher, int64_t accepted_token_len, int32_t grammar_vocab_size) {
        DeviceMaskState state;
        state.token_len = accepted_token_len;

        if (matcher.finished()) {
            state.mode = DeviceMaskMode::FINISHED;
            return ErrorResult<DeviceMaskState>(std::move(state));
        }
        auto terminated = matcher.isTerminated();
        if (!terminated.ok()) {
            return terminated.status();
        }
        if (terminated.value()) {
            state.mode = DeviceMaskMode::TERMINATED;
            return ErrorResult<DeviceMaskState>(std::move(state));
        }

        auto    bitmask = prepareMask(grammar_vocab_size);
        int64_t dl_shape[2];
        auto    dl =
            makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), static_cast<int32_t>(bitmask.size(1)), dl_shape);
        auto filled = matcher.fillBitmask(&dl, 0);
        if (!filled.ok()) {
            return filled.status();
        }

        state.mode                  = DeviceMaskMode::MASK;
        state.mask_required         = filled.value();
        state.packed_allow_mask_cpu = filled.value() ? std::move(bitmask) : torch::Tensor{};
        state.grammar_vocab_size    = grammar_vocab_size;
        return ErrorResult<DeviceMaskState>(std::move(state));
    }

    torch::Tensor prepareMask(int32_t grammar_vocab_size) {
        const int32_t words = (grammar_vocab_size + 31) / 32;
        if (!reusable_bitmask_cpu_.defined() || reusable_mask_words_ < words) {
            reusable_bitmask_cpu_ = at::full({1, words}, -1, at::dtype(at::kInt)).pin_memory();
            reusable_mask_words_  = words;
        } else {
            reusable_bitmask_cpu_.fill_(-1);
        }
        return reusable_bitmask_cpu_.narrow(1, 0, words);
    }

    ErrorInfo applyState(const torch::Tensor& logits, const DeviceMaskState& state, int64_t eos_token_id) {
        const size_t logits_vocab_size = static_cast<size_t>(logits.size(0));

        switch (state.mode) {
            case DeviceMaskMode::FINISHED:
                return ErrorInfo::OkStatus();
            case DeviceMaskMode::TERMINATED:
                return forceEos(logits, eos_token_id);
            case DeviceMaskMode::MASK:
                break;
        }

        const size_t mask_vocab_size = static_cast<size_t>(state.grammar_vocab_size);
        if (state.mask_required) {
            auto mask = state.packed_allow_mask_cpu;
#if USING_CUDA
            if (logits.is_cuda()) {
                const int64_t words = state.packed_allow_mask_cpu.size(1);
                if (!reusable_bitmask_gpu_.defined() || reusable_bitmask_gpu_.device() != logits.device()
                    || reusable_bitmask_gpu_.size(1) < words) {
                    reusable_bitmask_gpu_ =
                        torch::empty({1, words}, torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));
                }
                mask = reusable_bitmask_gpu_.narrow(1, 0, words);
                mask.copy_(state.packed_allow_mask_cpu);
            }
#endif
            runtimeApplyPackedMaskLogits(logits, mask, mask_vocab_size);
        }
        if (mask_vocab_size < logits_vocab_size) {
            logits
                .narrow(
                    0, static_cast<int64_t>(mask_vocab_size), static_cast<int64_t>(logits_vocab_size - mask_vocab_size))
                .fill_(BaseLogitsProcessor::neg_inf);
        }
        return ErrorInfo::OkStatus();
    }

    static ErrorInfo forceEos(const torch::Tensor& logits, int64_t token_id) {
        if (token_id < 0 || token_id >= logits.size(0)) {
            return ErrorInfo(ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB,
                             "grammar decode: eos_token_id (" + std::to_string(token_id)
                                 + ") out of logits vocab (vocab=" + std::to_string(logits.size(0)) + ")");
        }
        logits.fill_(BaseLogitsProcessor::neg_inf);
        logits[token_id] = 0.0f;
        return ErrorInfo::OkStatus();
    }

    DeviceMaskState device_mask_state_{};
    torch::Tensor   reusable_bitmask_cpu_;
    torch::Tensor   reusable_bitmask_gpu_;
    int32_t         reusable_mask_words_ = 0;
};

GrammarLogitsProcessor::GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher, int64_t eos_token_id):
    matcher_(std::move(matcher)),
    eos_token_id_(eos_token_id),
    decode_mask_builder_(std::make_unique<DecodeMaskBuilder>()) {}

GrammarLogitsProcessor::~GrammarLogitsProcessor() = default;

std::optional<ErrorInfo>
GrammarLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (!matcher_) {
        return std::nullopt;
    }
    const size_t batch_size = finish_idx - start_idx;
    if (batch_size == 0) {
        return std::nullopt;
    }
    if (batch_size != 1) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "grammar logits processor only supports single sequence decoding");
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = inputs.finished_mask.data_ptr<bool>();
        if (finished[start_idx]) {
            return std::nullopt;
        }
    }

    std::lock_guard<std::mutex> lock(state_mutex_);
    const auto                  logits_row = inputs.logits[start_idx];
    if (!logits_row.defined() || logits_row.dim() != 1 || logits_row.stride(0) != 1) {
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "grammar logits processor requires contiguous 1D logits rows");
    }
    auto grammar_vocab_size_or = validateVocabSize(*matcher_, static_cast<size_t>(logits_row.size(0)), "decode");
    if (!grammar_vocab_size_or.ok()) {
        matcher_->markFinished();
        return grammar_vocab_size_or.status();
    }
    auto error = decode_mask_builder_->apply(
        logits_row, *matcher_, committed_output_len_, eos_token_id_, grammar_vocab_size_or.value());
    if (error.hasError()) {
        return error;
    }
    return std::nullopt;
}

std::optional<ErrorInfo> GrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (!matcher_) {
        return std::nullopt;
    }
    RTP_LLM_CHECK(new_tokens.dim() == 2);
    RTP_LLM_CHECK(new_tokens.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(new_tokens.size(1) >= num_new_tokens);
    RTP_LLM_CHECK(new_tokens.is_contiguous());

    const int batch_size = static_cast<int>(new_tokens.size(0));
    // Keep parity with process(): this processor owns one matcher state machine,
    // so multi-sequence updates would corrupt parser state.
    if (batch_size != 1) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "grammar logits processor only supports single sequence decoding");
    }
    const auto* data = new_tokens.data_ptr<int32_t>();

    std::lock_guard<std::mutex> lock(state_mutex_);
    auto                        error = acceptCommittedLocked(data, static_cast<size_t>(num_new_tokens));
    if (error.hasError()) {
        return error;
    }
    return std::nullopt;
}

std::optional<int64_t> GrammarLogitsProcessor::committedOutputLen() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return committed_output_len_;
}

ErrorResult<int> GrammarLogitsProcessor::prepareSpeculative(const SpecLogitsProcessorRequest& request) {
    int cap_out = 0;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        auto                        cap_or = prepareSpecMask(*matcher_, eos_token_id_, request);
        if (!cap_or.ok()) {
            return cap_or.status();
        }
        cap_out = cap_or.value();
    }
    return ErrorResult<int>(std::move(cap_out));
}

ErrorInfo GrammarLogitsProcessor::acceptCommittedLocked(const int32_t* tokens, size_t n) {
    if (!matcher_ || matcher_->finished() || n == 0) {
        return ErrorInfo::OkStatus();
    }

    RTP_LLM_PROFILE_SCOPE("grammar.acceptToken");

    const int64_t old_matcher_len = matcher_->numAcceptedTokens();
    const int64_t old_output_len  = committed_output_len_;

    // A failed multi-token commit must not leave any prefix of the batch visible.
    auto rollback_commit = [this, old_matcher_len, old_output_len](const ErrorInfo& cause) -> ErrorInfo {
        const int64_t accepted_delta = matcher_->numAcceptedTokens() - old_matcher_len;
        if (accepted_delta < 0 || accepted_delta > std::numeric_limits<int>::max()) {
            matcher_->markFinished();
            return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION,
                             "grammar commit rollback range is invalid after error: " + cause.ToString());
        }

        auto rollback_error   = matcher_->rollback(static_cast<int>(accepted_delta));
        committed_output_len_ = old_output_len;
        if (rollback_error.hasError()) {
            matcher_->markFinished();
            return ErrorInfo(rollback_error.code(),
                             "grammar commit rollback failed after error: " + cause.ToString()
                                 + "; rollback_error=" + rollback_error.ToString());
        }

        return cause;
    };

    for (size_t i = 0; i < n; ++i) {
        const int32_t tok        = tokens[i];
        auto          terminated = matcher_->isTerminated();
        if (!terminated.ok()) {
            return rollback_commit(terminated.status());
        }
        if (terminated.value()) {
            // Keep the matcher TERMINATED rather than FINISHED. FINISHED makes
            // DecodeMaskBuilder stop applying a mask; a still-live stream must
            // continue to allow only EOS instead of resuming unconstrained
            // generation when min_new_tokens or ignore_eos delays completion.
            if (tok != static_cast<int32_t>(eos_token_id_)) {
                return rollback_commit(ErrorInfo(ErrorCode::GRAMMAR_NON_EOS_AFTER_TERMINAL,
                                                 "grammar received non-EOS token after terminal state"));
            }
            if (i + 1 != n) {
                return rollback_commit(ErrorInfo(ErrorCode::GRAMMAR_NON_EOS_AFTER_TERMINAL,
                                                 "grammar received additional committed tokens after terminal EOS"));
            }
            break;
        }
        auto accepted = matcher_->acceptToken(tok);
        if (!accepted.ok()) {
            return rollback_commit(accepted.status());
        }
        if (!accepted.value()) {
            return rollback_commit(ErrorInfo(ErrorCode::GRAMMAR_PARSER_REJECTED_TOKEN,
                                             "grammar commit error: parser rejected token " + std::to_string(tok)));
        }
    }

    // Matcher token count excludes EOS emitted after TERMINATED. Advance the
    // stream-facing count from its own committed state after the whole batch validates.
    committed_output_len_ = old_output_len + static_cast<int64_t>(n);
    // DecodeMaskBuilder rebuilds lazily when its cached token_len no longer matches.
    return ErrorInfo::OkStatus();
}

}  // namespace rtp_llm
