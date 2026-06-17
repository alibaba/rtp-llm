#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "rtp_llm/cpp/models/logits_processor/BitmaskUtils.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "rtp_llm/cpp/models/logits_processor/grammar_kernels/xgrammar_kernels.h"
#endif

namespace rtp_llm {

namespace {

enum class VerifyRowState {
    Active,
    Finished,
    Terminated,
};

}  // namespace

GrammarLogitsProcessor::GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher>    matcher,
                                               int64_t                               eos_token_id,
                                               LogitsProcessorFactory::ErrorReporter error_reporter):
    matcher_(std::move(matcher)), eos_token_id_(eos_token_id) {
    if (error_reporter) {
        setErrorReporter(std::move(error_reporter));
    }
}

GrammarLogitsProcessor::~GrammarLogitsProcessor() = default;

GrammarLogitsProcessor::DeviceMaskState GrammarLogitsProcessor::getDeviceMaskState(const c10::Device& device,
                                                                                   bool stream_lock_held) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    last_mask_device_ = device;

    if (device_mask_state_.has_value() && device_mask_state_->token_len == accepted_token_len_) {
        return *device_mask_state_;
    }

    device_mask_state_ = buildDeviceMaskStateLocked(device, stream_lock_held);
    return *device_mask_state_;
}

GrammarLogitsProcessor::DeviceMaskState GrammarLogitsProcessor::buildDeviceMaskStateLocked(const c10::Device& device,
                                                                                           bool stream_lock_held) {
    DeviceMaskState state;
    state.token_len = accepted_token_len_;

    if (!matcher_ || matcher_->finished()) {
        state.mode = DeviceMaskMode::Skip;
        return state;
    }
    if (matcher_->isTerminated()) {
        state.mode = DeviceMaskMode::ForceEOS;
        return state;
    }

    const int32_t grammar_vocab_size = matcher_->vocabSize();
    if (grammar_vocab_size <= 0) {
        state.mode = DeviceMaskMode::Skip;
        return state;
    }

    const int32_t words = (grammar_vocab_size + 31) / 32;
    if (!reusable_bitmask_cpu_.defined() || reusable_mask_words_ < words) {
        // Must be pinned: a pageable source silently strips non_blocking from copy_.
        reusable_bitmask_cpu_ = at::full({1, words}, -1, at::dtype(at::kInt)).pin_memory();
        reusable_mask_words_  = words;
    } else {
        reusable_bitmask_cpu_.fill_(-1);
    }
    auto     bitmask = reusable_bitmask_cpu_.narrow(1, 0, words);
    int64_t  dl_shape[2];
    DLTensor dl = makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), words, dl_shape);
    if (!matcher_->fillBitmask(&dl, 0)) {
        // Indeterminate matcher state: finish + report instead of allowing schema-illegal output.
        reported_error_.store(true, std::memory_order_relaxed);
        matcher_->markFinished();
        reportErrorViaReporter(ErrorCode::EXECUTION_EXCEPTION,
                               "grammar matcher fillBitmask failed; matcher state corrupted",
                               stream_lock_held);
        state.mode = DeviceMaskMode::Skip;
        return state;
    }

    state.grammar_vocab_size = grammar_vocab_size;
#if USING_CUDA
    if (device.is_cuda()) {
        if (!reusable_bitmask_gpu_.defined() || reusable_bitmask_gpu_.size(1) < words
            || reusable_bitmask_gpu_.device() != device) {
            reusable_bitmask_gpu_ = torch::empty({1, words}, bitmask.options().device(device));
        }
        reusable_bitmask_gpu_.copy_(bitmask, /*non_blocking=*/true);
        state.packed_bitmask = reusable_bitmask_gpu_.narrow(1, 0, words);
        state.mode           = DeviceMaskMode::Mask;
        return state;
    }
#endif

    if (!reusable_vocab_mask_cpu_.defined() || reusable_vocab_mask_cpu_.size(0) < grammar_vocab_size) {
        auto mask_options        = torch::TensorOptions().dtype(torch::kBool);
        reusable_vocab_mask_cpu_ = torch::empty({grammar_vocab_size}, mask_options);
    }
    auto           vocab_mask  = reusable_vocab_mask_cpu_.narrow(0, 0, grammar_vocab_size);
    bool*          mask_ptr    = vocab_mask.data_ptr<bool>();
    const int32_t* bitmask_ptr = bitmask.data_ptr<int32_t>();
    const size_t   words_sz    = static_cast<size_t>(words);
    for (int32_t token_id = 0; token_id < grammar_vocab_size; ++token_id) {
        mask_ptr[token_id] = !bitmaskAllowsToken(bitmask_ptr, words_sz, token_id);
    }

    state.mode = DeviceMaskMode::Mask;
    publishMaskToDevice(state, vocab_mask, device);
    return state;
}

void GrammarLogitsProcessor::publishMaskToDevice(DeviceMaskState&   state,
                                                 torch::Tensor      vocab_mask,
                                                 const c10::Device& device) {
    if (!device.is_cuda()) {
        state.vocab_mask = vocab_mask;
        return;
    }

    state.vocab_mask = vocab_mask.to(device, /*non_blocking=*/true);
}

void GrammarLogitsProcessor::applyDeviceMaskState(const torch::Tensor& logits, const DeviceMaskState& state) {
    if (state.mode == DeviceMaskMode::Skip) {
        return;
    }
    if (state.mode == DeviceMaskMode::ForceEOS) {
        forceToken(logits, eos_token_id_);
        return;
    }
    // DeviceMaskMode::Mask

#if USING_CUDA
    if (state.packed_bitmask.defined() && logits.is_cuda()) {
        auto logits_2d = logits.unsqueeze(0);
        invokeApplyXGrammarBitmaskInplace(logits_2d,
                                          state.packed_bitmask,
                                          static_cast<int64_t>(state.grammar_vocab_size),
                                          at::cuda::getCurrentCUDAStream().stream());
        // Tail [grammar_vocab, model_vocab) must be -inf to match CPU path.
        const int64_t model_vocab   = logits.size(0);
        const int64_t grammar_vocab = static_cast<int64_t>(state.grammar_vocab_size);
        if (grammar_vocab > 0 && model_vocab > grammar_vocab) {
            logits.narrow(0, grammar_vocab, model_vocab - grammar_vocab).fill_(BaseLogitsProcessor::neg_inf);
        }
        return;
    }
#endif

    if (!state.vocab_mask.defined()) {
        return;
    }
    auto mask = state.vocab_mask;
    if (mask.device() != logits.device()) {
        mask = mask.to(logits.device(), /*non_blocking=*/true);
    }
    const int64_t mask_vocab_size = std::min<int64_t>(logits.size(0), mask.size(0));
    if (mask_vocab_size > 0) {
        logits.narrow(0, 0, mask_vocab_size)
            .masked_fill_(mask.narrow(0, 0, mask_vocab_size), BaseLogitsProcessor::neg_inf);
    }
    if (mask.size(0) < logits.size(0)) {
        logits.narrow(0, mask.size(0), logits.size(0) - mask.size(0)).fill_(BaseLogitsProcessor::neg_inf);
    }
}

void GrammarLogitsProcessor::forceToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        return;
    }
    // 其余位置已是 -inf，被强制位置只要任意 finite 值即可让 softmax 归一为 1.0；
    // 用 0.0f 而非魔法常数 1，避免后续被读成"该 token 的真实 logit"。
    logits.fill_(BaseLogitsProcessor::neg_inf);
    logits[token_id] = 0.0f;
}

void GrammarLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (!matcher_) {
        return;
    }
    const size_t batch_size = finish_idx - start_idx;
    if (batch_size == 0) {
        return;
    }
    if (batch_size != 1) {
        RTP_LLM_LOG_WARNING("grammar logits processor unexpected batch_size=%zu", batch_size);
        return;
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = inputs.finished_mask.data_ptr<bool>();
        if (finished[start_idx]) {
            return;
        }
    }

    auto logits = inputs.logits.narrow(0, start_idx, 1);
    // Sampler::forward() runs without holding the stream's mutex_; reporter must take the lock.
    auto state = getDeviceMaskState(logits.device(), /*stream_lock_held=*/false);
    applyDeviceMaskState(logits[0], state);
}

void GrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (!matcher_ || matcher_->finished()) {
        return;
    }

    RTP_LLM_CHECK(new_tokens.dim() == 2);
    RTP_LLM_CHECK(new_tokens.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(new_tokens.size(1) >= num_new_tokens);
    RTP_LLM_CHECK(new_tokens.is_contiguous());

    const int            batch_size = static_cast<int>(new_tokens.size(0));
    const int            stride     = static_cast<int>(new_tokens.size(1));
    const auto*          data       = new_tokens.data_ptr<int32_t>();
    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<size_t>(batch_size * num_new_tokens));
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_new_tokens; ++j) {
            tokens.push_back(data[i * stride + j]);
        }
    }

    RTP_LLM_PROFILE_SCOPE("grammar.acceptToken");

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        // Single-pass: accept until matcher rejects, terminates, or EOS post-terminal.
        // updateStatus runs with the stream mutex_ already held — reporter uses the
        // no-lock variant so buildDeviceMaskStateLocked below doesn't self-deadlock.
        for (int32_t tok : tokens) {
            if (matcher_->isTerminated()) {
                matcher_->markFinished();
                if (tok == static_cast<int32_t>(eos_token_id_)) {
                    accepted_token_len_ = matcher_->numAcceptedTokens() + 1;  // EOS not fed to matcher
                } else {
                    reportErrorViaReporter(ErrorCode::INVALID_PARAMS,
                                           "grammar received non-EOS token after terminal state",
                                           /*stream_lock_held=*/true);
                    return;
                }
                break;
            }
            if (!matcher_->acceptToken(tok)) {
                reported_error_.store(true, std::memory_order_relaxed);
                matcher_->markFinished();
                reportErrorViaReporter(ErrorCode::INVALID_PARAMS,
                                       "grammar commit error: parser rejected token " + std::to_string(tok),
                                       /*stream_lock_held=*/true);
                return;
            }
            accepted_token_len_ = matcher_->numAcceptedTokens();
        }
        device_mask_state_ = buildDeviceMaskStateLocked(last_mask_device_.value_or(c10::Device(c10::DeviceType::CPU)),
                                                        /*stream_lock_held=*/true);
    }
}

void GrammarLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& /* src_batch_indices */) {}

bool GrammarLogitsProcessor::isSpecVerifyEligible() const {
    return matcher_ != nullptr && !reported_error_.load(std::memory_order_relaxed);
}

int GrammarLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    if (!matcher_ || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return request.propose_step;
    }
    if (reported_error_.load(std::memory_order_relaxed)) {
        return 0;
    }
    if (request.bitmask_size_int32 < static_cast<size_t>((request.vocab_size + 31) / 32)) {
        RTP_LLM_LOG_WARNING("[grammar] tryAcceptAndFillBitmask: bitmask buffer too small "
                            "(words=%zu vocab=%zu); skipping verify",
                            request.bitmask_size_int32,
                            request.vocab_size);
        reported_error_.store(true, std::memory_order_relaxed);
        reportErrorViaReporter(ErrorCode::EXECUTION_EXCEPTION,
                               "grammar MTP verify: bitmask buffer smaller than model vocab",
                               /*stream_lock_held=*/false);
        return 0;
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    // Don't short-circuit terminated here: row[0] must be filled EOS-only first or
    // the merged target mask stays allow-all and commit-side trips on non-EOS post-terminal.

    const int  P               = request.propose_step;
    const auto W               = request.bitmask_size_int32;
    int        accepted_prefix = 0;
    int        cap             = P;

    // Degrade gracefully if grammar vocab exceeds model bitmask (was previously a hard CHECK).
    {
        const int32_t grammar_vocab_size = matcher_->vocabSize();
        if (grammar_vocab_size > 0 && SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size) > W) {
            RTP_LLM_LOG_WARNING("[grammar] tryAcceptAndFillBitmask: grammar vocab (%d) exceeds "
                                "model vocab bitmask (%zu words); reporting stream error",
                                grammar_vocab_size,
                                W);
            reported_error_.store(true, std::memory_order_relaxed);
            matcher_->markFinished();
            reportErrorViaReporter(ErrorCode::EXECUTION_EXCEPTION,
                                   "grammar vocab exceeds model vocab in MTP verify (grammar="
                                       + std::to_string(grammar_vocab_size) + ", model_words=" + std::to_string(W)
                                       + ")",
                                   /*stream_lock_held=*/false);
            return 0;
        }
    }

    // forceTokenInBitmask asserts on out-of-range token_id; surface eos misconfiguration
    // as a stream error here instead of crashing the worker on the first verify row.
    if (eos_token_id_ < 0 || static_cast<size_t>(eos_token_id_ / 32) >= W) {
        RTP_LLM_LOG_WARNING("[grammar] tryAcceptAndFillBitmask: eos_token_id (%ld) out of bitmask range "
                            "(words=%zu); reporting stream error",
                            static_cast<long>(eos_token_id_),
                            W);
        reported_error_.store(true, std::memory_order_relaxed);
        matcher_->markFinished();
        reportErrorViaReporter(ErrorCode::EXECUTION_EXCEPTION,
                               "grammar MTP verify: eos_token_id (" + std::to_string(eos_token_id_)
                                   + ") out of model vocab bitmask (words=" + std::to_string(W) + ")",
                               /*stream_lock_held=*/false);
        return 0;
    }

    auto fill_row = [&](int32_t* row) -> VerifyRowState {
        std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
        if (matcher_->finished()) {
            // EOS-only so the merged target mask is bounded for this row.
            forceTokenInBitmask(row, W, eos_token_id_);
            return VerifyRowState::Finished;
        }
        if (matcher_->isTerminated()) {
            forceTokenInBitmask(row, W, eos_token_id_);
            return VerifyRowState::Terminated;
        }

        const int32_t grammar_vocab_size = matcher_->vocabSize();
        const size_t  grammar_words      = SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size);

        int64_t  dl_shape[2];
        DLTensor dl = makeSingleRowBitmaskView(row, static_cast<int32_t>(grammar_words), dl_shape);
        if (!matcher_->fillBitmask(&dl, 0)) {
            reported_error_.store(true, std::memory_order_relaxed);
            matcher_->markFinished();
            forceTokenInBitmask(row, W, eos_token_id_);
            reportErrorViaReporter(ErrorCode::EXECUTION_EXCEPTION,
                                   "grammar matcher fillBitmask failed during MTP verify; matcher state corrupted",
                                   /*stream_lock_held=*/false);
            return VerifyRowState::Finished;
        }
        clearBitmaskTokenRange(row, W, grammar_vocab_size, static_cast<int64_t>(request.vocab_size));
        return VerifyRowState::Active;
    };

    const auto reasoner_snapshot    = matcher_->reasonerSnapshot();
    auto       rollback_provisional = [&]() noexcept {
        if (accepted_prefix > 0) {
            matcher_->rollback(accepted_prefix);
        }
        matcher_->restoreReasoner(reasoner_snapshot);
    };
    auto fail_with = [&](std::string what) {
        rollback_provisional();
        matcher_->markFinished();
        reported_error_.store(true, std::memory_order_relaxed);
        if (request.bitmask_cpu_out != nullptr && request.bitmask_size_int32 > 0) {
            forceTokenInBitmask(request.bitmask_cpu_out, W, eos_token_id_);
        }
        reportErrorViaReporter(ErrorCode::EXECUTION_EXCEPTION,
                               "grammar MTP verify exception: " + std::move(what),
                               /*stream_lock_held=*/false);
    };

    try {
        for (int offset = 0; offset <= P; ++offset) {
            int32_t*             row       = request.bitmask_cpu_out + offset * W;
            const VerifyRowState row_state = fill_row(row);
            if (offset == P) {
                break;
            }
            if (row_state == VerifyRowState::Terminated || row_state == VerifyRowState::Finished) {
                cap = offset;
                break;
            }

            const int32_t draft_token = request.draft_tokens[offset];
            if (draft_token < 0 || static_cast<size_t>(draft_token) >= request.vocab_size
                || !bitmaskAllowsToken(row, W, draft_token)) {
                cap = offset;
                break;
            }
            if (!matcher_->acceptToken(draft_token)) {
                cap = offset;
                break;
            }
            ++accepted_prefix;
        }
    } catch (const std::exception& e) {
        fail_with(e.what());
        return 0;
    } catch (...) {
        fail_with("unknown");
        return 0;
    }
    // Verify never accumulates state on the matcher; commit happens via updateStatus.
    rollback_provisional();
    return cap;
}

}  // namespace rtp_llm
