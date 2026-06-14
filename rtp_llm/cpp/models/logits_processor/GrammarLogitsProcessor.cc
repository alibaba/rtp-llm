#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA
#include "rtp_llm/cpp/models/logits_processor/grammar_kernels/xgrammar_kernels.h"
#endif

namespace rtp_llm {

namespace {

DLTensor makeSingleRowBitmaskView(int32_t* data, int32_t words, int64_t shape_out[2]) {
    DLTensor dl;
    dl.data        = data;
    dl.device      = DLDevice{kDLCPU, 0};
    dl.ndim        = 2;
    dl.dtype       = DLDataType{kDLInt, 32, 1};
    shape_out[0]   = 1;
    shape_out[1]   = words;
    dl.shape       = shape_out;
    dl.strides     = nullptr;
    dl.byte_offset = 0;
    return dl;
}

bool bitmaskAllowsToken(const int32_t* bitmask, int32_t token_id) {
    const int32_t word = bitmask[token_id / 32];
    return (static_cast<uint32_t>(word) & (1u << (token_id % 32))) != 0u;
}

enum class VerifyRowState {
    Active,
    Finished,
    Terminated,
};

void clearTokenFromBitmask(int32_t* bitmask, size_t words, int64_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    bitmask[token_id / 32] &= ~(1u << (token_id % 32));
}

void forceTokenInBitmask(int32_t* bitmask, size_t words, int64_t token_id) {
    std::fill_n(bitmask, words, 0);
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    bitmask[token_id / 32] |= (1u << (token_id % 32));
}

void clearBitmaskTokenRange(int32_t* bitmask, size_t words, int64_t begin_token, int64_t end_token) {
    if (begin_token < 0 || end_token <= begin_token) {
        return;
    }
    for (int64_t token_id = begin_token; token_id < end_token; ++token_id) {
        clearTokenFromBitmask(bitmask, words, token_id);
    }
}

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

void reportInvalidParams(const LogitsProcessorFactory::ErrorReporter& reporter, const std::string& msg) {
    if (reporter) {
        // Caller is the LogitsProcessorFactory, which runs without the stream lock.
        reporter(ErrorCode::INVALID_PARAMS, msg, /*stream_lock_held=*/false);
    }
}

void reportGenerateTimeout(const LogitsProcessorFactory::ErrorReporter& reporter, const std::string& msg) {
    if (reporter) {
        reporter(ErrorCode::GENERATE_TIMEOUT, msg, /*stream_lock_held=*/false);
    }
}

bool GrammarLogitsProcessor::advanceMatcher(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) {
        return true;
    }
    if (!matcher_) {
        RTP_LLM_LOG_WARNING("[grammar] advanceMatcher: matcher not installed");
        return false;
    }

    for (int32_t tok : tokens) {
        if (matcher_->isTerminated() || matcher_->finished()) {
            break;
        }
        if (!matcher_->acceptToken(tok)) {
            reported_error_.store(true, std::memory_order_relaxed);
            matcher_->markFinished();
            RTP_LLM_LOG_WARNING("[grammar] advanceMatcher rejected token %d num_accepted=%ld",
                                tok,
                                matcher_->numAcceptedTokens());
            if (error_reporter_) {
                reportErrorViaReporter(ErrorCode::INVALID_PARAMS,
                                       "grammar commit error: parser rejected token " + std::to_string(tok),
                                       /*stream_lock_held=*/true);
            }
            return false;
        }
    }

    return true;
}

void GrammarLogitsProcessor::syncAcceptedTokenLenLocked() {
    accepted_token_len_ = matcher_ ? matcher_->numAcceptedTokens() : 0;
}

void GrammarLogitsProcessor::rebuildDeviceMaskStateLocked(bool stream_lock_held) {
    device_mask_state_ =
        buildDeviceMaskStateLocked(last_mask_device_.value_or(c10::Device(c10::DeviceType::CPU)), stream_lock_held);
}

GrammarLogitsProcessor::DeviceMaskState GrammarLogitsProcessor::getDeviceMaskState(const c10::Device& device,
                                                                                   bool stream_lock_held) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    last_mask_device_ = device;

    if (device_mask_state_.mode != DeviceMaskMode::UNSET && device_mask_state_.token_len == accepted_token_len_
        && device_mask_state_.device == device) {
        return device_mask_state_;
    }

    device_mask_state_ = buildDeviceMaskStateLocked(device, stream_lock_held);
    return device_mask_state_;
}

GrammarLogitsProcessor::DeviceMaskState GrammarLogitsProcessor::buildDeviceMaskStateLocked(const c10::Device& device,
                                                                                           bool stream_lock_held) {
    DeviceMaskState state;
    state.token_len = accepted_token_len_;
    state.device    = device;

    if (!matcher_ || matcher_->finished()) {
        state.mode = DeviceMaskMode::FINISHED;
        return state;
    }
    if (matcher_->isTerminated()) {
        state.mode = DeviceMaskMode::TERMINATED;
        return state;
    }

    const int32_t grammar_vocab_size = matcher_->vocabSize();
    if (grammar_vocab_size <= 0) {
        state.mode = DeviceMaskMode::NOOP;
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
        state.mode = DeviceMaskMode::FINISHED;
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
        state.mode           = DeviceMaskMode::MASK;
        // Re-record onto the per-processor event; avoids per-token event allocation.
        if (!reusable_ready_event_) {
            reusable_ready_event_ = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        }
        reusable_ready_event_->record(cuda_graph::graphGetCurrentStream());
        state.ready_event = reusable_ready_event_;
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
    for (int32_t token_id = 0; token_id < grammar_vocab_size; ++token_id) {
        mask_ptr[token_id] = !bitmaskAllowsToken(bitmask_ptr, token_id);
    }

    state.mode = DeviceMaskMode::MASK;
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
#if USING_CUDA
    if (!reusable_ready_event_) {
        reusable_ready_event_ = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    }
    reusable_ready_event_->record(cuda_graph::graphGetCurrentStream());
    state.ready_event = reusable_ready_event_;
#endif
}

void GrammarLogitsProcessor::applyDeviceMaskState(const torch::Tensor& logits, const DeviceMaskState& state) {
    switch (state.mode) {
        case DeviceMaskMode::FINISHED:
        case DeviceMaskMode::NOOP:
        case DeviceMaskMode::UNSET:
            return;
        case DeviceMaskMode::TERMINATED:
            forceToken(logits, eos_token_id_);
            return;
        case DeviceMaskMode::MASK:
            break;
    }

#if USING_CUDA
    if (state.packed_bitmask.defined() && logits.is_cuda()) {
        if (state.ready_event) {
            state.ready_event->block(cuda_graph::graphGetCurrentStream());
        }
        auto logits_2d = logits.unsqueeze(0);
        invokeApplyXGrammarBitmaskInplace(logits_2d,
                                          state.packed_bitmask,
                                          static_cast<int64_t>(state.grammar_vocab_size),
                                          cuda_graph::graphGetCurrentStream().stream());
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
#if USING_CUDA
    if (state.ready_event && logits.is_cuda()) {
        state.ready_event->block(cuda_graph::graphGetCurrentStream());
    }
#endif
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
        // Log once at WARN, then DEBUG, to avoid flooding for misconfigured streams.
        if (!warned_multi_seq_unsupported_) {
            warned_multi_seq_unsupported_ = true;
            RTP_LLM_LOG_WARNING("grammar logits processor only supports single sequence decoding "
                                "(this warning will not repeat for this stream)");
        } else {
            RTP_LLM_LOG_DEBUG("grammar logits processor skipping batch_size=%zu", batch_size);
        }
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
        // Terminated + EOS → finish; any other token after terminated → error.
        bool   stop_after_eos        = false;
        bool   report_after_terminal = false;
        size_t active_count          = 0;
        for (int32_t tok : tokens) {
            if (matcher_ && matcher_->isTerminated()) {
                if (tok == static_cast<int32_t>(eos_token_id_)) {
                    matcher_->markFinished();
                    stop_after_eos = true;
                } else {
                    matcher_->markFinished();
                    report_after_terminal = true;
                }
                break;
            }
            ++active_count;
        }
        if (report_after_terminal) {
            reportErrorViaReporter(ErrorCode::INVALID_PARAMS,
                                   "grammar received non-EOS token after terminal state",
                                   /*stream_lock_held=*/true);
            return;
        }
        tokens.resize(active_count);
        if (!stop_after_eos && !advanceMatcher(tokens)) {
            return;
        }
        syncAcceptedTokenLenLocked();
        if (stop_after_eos) {
            // EOS was not forwarded to the matcher; track it here to match committed count.
            ++accepted_token_len_;
        }
        // updateStatus() is called from GenerateStream::updateLogitProcessorStatus while
        // the stream's mutex_ is already held (see update()/updateWithoutLock/specUpdate),
        // so any reporter call from inside rebuild must use the no-lock variant.
        rebuildDeviceMaskStateLocked(/*stream_lock_held=*/true);
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

    bool rolled_back          = false;
    auto rollback_provisional = [&]() {
        if (rolled_back) {
            return;  // idempotent: success-path rollback may be retried by the catch arm.
        }
        rolled_back = true;
        if (accepted_prefix > 0) {
            matcher_->rollback(accepted_prefix);
        }
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
                || !bitmaskAllowsToken(row, draft_token)) {
                cap = offset;
                break;
            }
            if (!matcher_->acceptToken(draft_token)) {
                cap = offset;
                break;
            }
            ++accepted_prefix;
            // Don't early-break on terminated/finished here; next iteration's fill_row
            // must emit the correct EOS-only row instead of leaving allow-all stale.
        }

        rollback_provisional();
    } catch (const std::exception& e) {
        // Confine the failure to this stream; spec executor has no try/catch.
        const bool partial_rolled_back = rolled_back;
        try {
            rollback_provisional();
        } catch (...) {
            // swallow; markFinished below.
        }
        matcher_->markFinished();
        (void)partial_rolled_back;
        reported_error_.store(true, std::memory_order_relaxed);
        if (request.bitmask_cpu_out != nullptr && request.bitmask_size_int32 > 0) {
            forceTokenInBitmask(request.bitmask_cpu_out, W, eos_token_id_);
        }
        reportErrorViaReporter(ErrorCode::EXECUTION_EXCEPTION,
                               std::string("grammar MTP verify exception: ") + e.what(),
                               /*stream_lock_held=*/false);
        return 0;
    } catch (...) {
        try {
            rollback_provisional();
        } catch (...) {}
        matcher_->markFinished();
        reported_error_.store(true, std::memory_order_relaxed);
        if (request.bitmask_cpu_out != nullptr && request.bitmask_size_int32 > 0) {
            forceTokenInBitmask(request.bitmask_cpu_out, W, eos_token_id_);
        }
        reportErrorViaReporter(ErrorCode::EXECUTION_EXCEPTION,
                               "grammar MTP verify unknown exception",
                               /*stream_lock_held=*/false);
        return 0;
    }
    return cap;
}

}  // namespace rtp_llm
