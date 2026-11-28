#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>

#include <c10/util/Exception.h>
#include <dlpack/dlpack.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#if USING_CUDA
#include <c10/cuda/CUDAGuard.h>
#endif
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
    ~DecodeMaskBuilder() {
#if USING_CUDA
        auto drain = [](const std::shared_ptr<PinnedBitmaskSlot>& slot) {
            if (slot && slot->recorded && slot->copy_done.has_value() && !slot->copy_done->query()) {
                slot->copy_done->synchronize();
            }
        };
        drain(device_mask_state_.pinned_slot);
        for (const auto& slot : slot_pool_) {
            drain(slot);
        }
#endif
    }

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
                setDeviceMaskState(finished(accepted_token_len));
                return state_or.status();
            }

            setDeviceMaskState(std::move(state_or.value()));
            return applyState(logits, device_mask_state_, eos_token_id);
        } catch (const std::bad_alloc& e) {
            setDeviceMaskState(finished(accepted_token_len));
            return detail::grammarMaskBuildError("decode", e);
        } catch (const c10::Error& e) {
            setDeviceMaskState(finished(accepted_token_len));
            return detail::grammarMaskBuildError("decode", e);
        }
    }

    ErrorInfo extractBatchEntry(RtpGrammarMatcher&                      matcher,
                                int64_t                                 accepted_token_len,
                                int64_t                                 eos_token_id,
                                int32_t                                 grammar_vocab_size,
                                size_t                                  logits_vocab_size,
                                int32_t*                                bitmask_dst,
                                size_t                                  bitmask_dst_words,
                                GrammarLogitsProcessor::BatchMaskEntry& out_entry) {
        out_entry = {};
        try {
            if (device_mask_state_.token_len == accepted_token_len) {
                switch (device_mask_state_.mode) {
                    case DeviceMaskMode::FINISHED:
                        return ErrorInfo::OkStatus();
                    case DeviceMaskMode::TERMINATED:
                        if (eos_token_id < 0 || static_cast<size_t>(eos_token_id) >= logits_vocab_size) {
                            return ErrorInfo(ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB,
                                             "grammar decode: eos_token_id (" + std::to_string(eos_token_id)
                                                 + ") out of logits vocab (vocab=" + std::to_string(logits_vocab_size)
                                                 + ")");
                        }
                        out_entry.action         = GrammarLogitsProcessor::BatchMaskAction::FORCE_TOKEN;
                        out_entry.force_token_id = eos_token_id;
                        return ErrorInfo::OkStatus();
                    case DeviceMaskMode::MASK: {
                        const int32_t cached_vocab = device_mask_state_.grammar_vocab_size;
                        if (!device_mask_state_.mask_required) {
                            if (static_cast<size_t>(cached_vocab) < logits_vocab_size) {
                                out_entry.action             = GrammarLogitsProcessor::BatchMaskAction::TAIL_ONLY;
                                out_entry.grammar_vocab_size = cached_vocab;
                            }
                            return ErrorInfo::OkStatus();
                        }
                        const int32_t words = (cached_vocab + 31) / 32;
                        RTP_LLM_CHECK(bitmask_dst != nullptr && bitmask_dst_words >= static_cast<size_t>(words));
                        std::memcpy(bitmask_dst,
                                    device_mask_state_.packed_allow_mask_cpu.data_ptr<int32_t>(),
                                    static_cast<size_t>(words) * sizeof(int32_t));
                        out_entry.action             = GrammarLogitsProcessor::BatchMaskAction::BITMASK_AND_TAIL;
                        out_entry.grammar_vocab_size = cached_vocab;
                        out_entry.bitmask_words      = words;
                        return ErrorInfo::OkStatus();
                    }
                }
            }

            if (matcher.finished()) {
                setDeviceMaskState(finished(accepted_token_len));
                return ErrorInfo::OkStatus();
            }
            auto terminated = matcher.isTerminated();
            if (!terminated.ok()) {
                setDeviceMaskState(finished(accepted_token_len));
                return terminated.status();
            }
            if (terminated.value()) {
                if (eos_token_id < 0 || static_cast<size_t>(eos_token_id) >= logits_vocab_size) {
                    return ErrorInfo(ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB,
                                     "grammar decode: eos_token_id (" + std::to_string(eos_token_id)
                                         + ") out of logits vocab (vocab=" + std::to_string(logits_vocab_size) + ")");
                }
                DeviceMaskState term_state;
                term_state.token_len = accepted_token_len;
                term_state.mode      = DeviceMaskMode::TERMINATED;
                setDeviceMaskState(std::move(term_state));
                out_entry.action         = GrammarLogitsProcessor::BatchMaskAction::FORCE_TOKEN;
                out_entry.force_token_id = eos_token_id;
                return ErrorInfo::OkStatus();
            }

            const int32_t words = (grammar_vocab_size + 31) / 32;
            RTP_LLM_CHECK(bitmask_dst != nullptr && bitmask_dst_words >= static_cast<size_t>(words));
            int64_t  dl_shape[2];
            DLTensor dl     = makeSingleRowBitmaskView(bitmask_dst, words, dl_shape);
            auto     filled = matcher.fillBitmask(&dl, 0);
            if (!filled.ok()) {
                setDeviceMaskState(finished(accepted_token_len));
                return filled.status();
            }

            if (!filled.value()) {
                DeviceMaskState no_mask_state;
                no_mask_state.token_len          = accepted_token_len;
                no_mask_state.mode               = DeviceMaskMode::MASK;
                no_mask_state.mask_required      = false;
                no_mask_state.grammar_vocab_size = grammar_vocab_size;
                setDeviceMaskState(std::move(no_mask_state));
                if (static_cast<size_t>(grammar_vocab_size) < logits_vocab_size) {
                    out_entry.action             = GrammarLogitsProcessor::BatchMaskAction::TAIL_ONLY;
                    out_entry.grammar_vocab_size = grammar_vocab_size;
                }
            } else {
                // Reset any prior step's device_mask_state_ without allocating a per-stream
                // pinned buffer; the caller stages bitmask_dst directly in one shared batch buffer.
                setDeviceMaskState(DeviceMaskState{});
                out_entry.action             = GrammarLogitsProcessor::BatchMaskAction::BITMASK_AND_TAIL;
                out_entry.grammar_vocab_size = grammar_vocab_size;
                out_entry.bitmask_words      = words;
            }
            return ErrorInfo::OkStatus();
        } catch (const std::bad_alloc& e) {
            setDeviceMaskState(finished(accepted_token_len));
            return detail::grammarMaskBuildError("decode", e);
        } catch (const c10::Error& e) {
            setDeviceMaskState(finished(accepted_token_len));
            return detail::grammarMaskBuildError("decode", e);
        }
    }

private:
    enum class DeviceMaskMode {
        MASK,
        TERMINATED,
        FINISHED,
    };

#if USING_CUDA
    struct PinnedBitmaskSlot {
        torch::Tensor                cpu_buf;
        int32_t                      words = 0;
        std::optional<torch::Event>  copy_done;
        bool                         recorded = false;
    };
#else
    struct PinnedBitmaskSlot {};
#endif

    struct DeviceMaskState {
        DeviceMaskMode                     mode          = DeviceMaskMode::FINISHED;
        int64_t                            token_len     = -1;
        bool                               mask_required = false;
        torch::Tensor                      packed_allow_mask_cpu;
        std::shared_ptr<PinnedBitmaskSlot> pinned_slot;
        int32_t                            grammar_vocab_size = 0;
    };

    static constexpr size_t kMaxPinnedPoolSize = 256;

    void setDeviceMaskState(DeviceMaskState next) {
        retireSlot(std::move(device_mask_state_.pinned_slot));
        device_mask_state_ = std::move(next);
    }

    void retireSlot(std::shared_ptr<PinnedBitmaskSlot> slot) {
#if USING_CUDA
        if (!slot) {
            return;
        }
        if (slot_pool_.size() >= kMaxPinnedPoolSize) {
            auto front = std::move(slot_pool_.front());
            slot_pool_.pop_front();
            if (front && front->recorded && front->copy_done.has_value() && !front->copy_done->query()) {
                front->copy_done->synchronize();
            }
        }
        slot_pool_.push_back(std::move(slot));
#else
        (void)slot;
#endif
    }

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

        const int32_t words = (grammar_vocab_size + 31) / 32;
#if USING_CUDA
        auto slot    = acquirePinnedSlot(words);
        auto bitmask = slot->cpu_buf.narrow(1, 0, words);
#else
        std::shared_ptr<PinnedBitmaskSlot> slot;
        auto                               bitmask = prepareCpuMask(words);
#endif
        int64_t dl_shape[2];
        auto    dl =
            makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), static_cast<int32_t>(bitmask.size(1)), dl_shape);
        auto filled = matcher.fillBitmask(&dl, 0);
        if (!filled.ok()) {
            retireSlot(std::move(slot));
            return filled.status();
        }

        state.mode               = DeviceMaskMode::MASK;
        state.mask_required      = filled.value();
        state.grammar_vocab_size = grammar_vocab_size;
        if (filled.value()) {
            state.packed_allow_mask_cpu = std::move(bitmask);
            state.pinned_slot           = std::move(slot);
        } else {
            state.packed_allow_mask_cpu = torch::Tensor{};
            retireSlot(std::move(slot));
        }
        return ErrorResult<DeviceMaskState>(std::move(state));
    }

#if USING_CUDA
    std::shared_ptr<PinnedBitmaskSlot> acquirePinnedSlot(int32_t words) {
        for (auto it = slot_pool_.begin(); it != slot_pool_.end();) {
            auto& entry = *it;
            const bool ready = !entry->recorded || !entry->copy_done.has_value() || entry->copy_done->query();
            if (!ready) {
                ++it;
                continue;
            }
            auto slot = std::move(entry);
            it        = slot_pool_.erase(it);
            if (slot && slot->words >= words) {
                slot->cpu_buf.fill_(-1);
                slot->recorded = false;
                return slot;
            }
        }
        if (slot_pool_.size() >= kMaxPinnedPoolSize) {
            auto front = std::move(slot_pool_.front());
            slot_pool_.pop_front();
            if (front && front->recorded && front->copy_done.has_value() && !front->copy_done->query()) {
                front->copy_done->synchronize();
            }
            if (front && front->words >= words) {
                front->cpu_buf.fill_(-1);
                front->recorded = false;
                return front;
            }
        }
        auto slot     = std::make_shared<PinnedBitmaskSlot>();
        slot->cpu_buf = at::full({1, words}, -1, at::TensorOptions().dtype(at::kInt).pinned_memory(true));
        slot->words   = words;
        return slot;
    }
#else
    torch::Tensor prepareCpuMask(int32_t words) {
        if (!reusable_bitmask_cpu_.defined() || reusable_mask_words_ < words) {
            reusable_bitmask_cpu_ = at::full({1, words}, -1, at::dtype(at::kInt));
            reusable_mask_words_  = words;
        } else {
            reusable_bitmask_cpu_.fill_(-1);
        }
        return reusable_bitmask_cpu_.narrow(1, 0, words);
    }
#endif

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
                c10::cuda::CUDAGuard device_guard(logits.device());
                const int64_t        words = state.packed_allow_mask_cpu.size(1);
                if (!reusable_bitmask_gpu_.defined() || reusable_bitmask_gpu_.device() != logits.device()
                    || reusable_bitmask_gpu_.size(1) < words) {
                    reusable_bitmask_gpu_ =
                        torch::empty({1, words}, torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));
                }
                mask = reusable_bitmask_gpu_.narrow(1, 0, words);
                mask.copy_(state.packed_allow_mask_cpu, /*non_blocking=*/true);
                if (state.pinned_slot) {
                    if (!state.pinned_slot->copy_done.has_value()) {
                        state.pinned_slot->copy_done.emplace(cuda_graph::makeGraphEvent());
                    }
                    state.pinned_slot->copy_done->record(cuda_graph::graphGetCurrentStream());
                    state.pinned_slot->recorded = true;
                }
            }
#endif
            runtimeApplyPackedMaskLogits(logits, mask, mask_vocab_size);
        }
        if (mask_vocab_size < logits_vocab_size) {
            const float fill_val = (logits.scalar_type() == torch::kFloat16 || logits.scalar_type() == torch::kBFloat16) ?
                                       -std::numeric_limits<float>::infinity() :
                                       BaseLogitsProcessor::neg_inf;
            logits
                .narrow(
                    0, static_cast<int64_t>(mask_vocab_size), static_cast<int64_t>(logits_vocab_size - mask_vocab_size))
                .fill_(fill_val);
        }
        return ErrorInfo::OkStatus();
    }

    static ErrorInfo forceEos(const torch::Tensor& logits, int64_t token_id) {
        if (token_id < 0 || token_id >= logits.size(0)) {
            return ErrorInfo(ErrorCode::GRAMMAR_EOS_OUT_OF_VOCAB,
                             "grammar decode: eos_token_id (" + std::to_string(token_id)
                                 + ") out of logits vocab (vocab=" + std::to_string(logits.size(0)) + ")");
        }
        const float fill_val = (logits.scalar_type() == torch::kFloat16 || logits.scalar_type() == torch::kBFloat16) ?
                                   -std::numeric_limits<float>::infinity() :
                                   BaseLogitsProcessor::neg_inf;
        logits.fill_(fill_val);
        logits[token_id] = 0.0f;
        return ErrorInfo::OkStatus();
    }

    DeviceMaskState device_mask_state_{};
#if USING_CUDA
    std::deque<std::shared_ptr<PinnedBitmaskSlot>> slot_pool_;
#else
    torch::Tensor reusable_bitmask_cpu_;
    int32_t       reusable_mask_words_ = 0;
#endif
    torch::Tensor reusable_bitmask_gpu_;
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

std::optional<ErrorInfo> GrammarLogitsProcessor::prepareBatchMaskEntry(const SamplerInputs& inputs,
                                                                       size_t               row_idx,
                                                                       int32_t*             bitmask_dst,
                                                                       size_t               bitmask_dst_words,
                                                                       BatchMaskEntry&      out_entry) {
    out_entry = {};
    if (!matcher_) {
        return std::nullopt;
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = inputs.finished_mask.data_ptr<bool>();
        if (finished[row_idx]) {
            return std::nullopt;
        }
    }

    std::lock_guard<std::mutex> lock(state_mutex_);
    const auto logits_row =
        inputs.logits.defined() && inputs.logits.dim() >= 1 && static_cast<int64_t>(row_idx) < inputs.logits.size(0) ?
            inputs.logits[static_cast<int64_t>(row_idx)] :
            torch::Tensor();
    if (!logits_row.defined() || logits_row.dim() != 1 || logits_row.stride(0) != 1) {
        return ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "grammar logits processor requires contiguous 1D logits rows");
    }
    const size_t logits_vocab_size = static_cast<size_t>(logits_row.size(0));
    auto         grammar_vocab_size_or = validateVocabSize(*matcher_, logits_vocab_size, "decode");
    if (!grammar_vocab_size_or.ok()) {
        matcher_->markFinished();
        return grammar_vocab_size_or.status();
    }
    auto error = decode_mask_builder_->extractBatchEntry(*matcher_,
                                                         committed_output_len_,
                                                         eos_token_id_,
                                                         grammar_vocab_size_or.value(),
                                                         logits_vocab_size,
                                                         bitmask_dst,
                                                         bitmask_dst_words,
                                                         out_entry);
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
