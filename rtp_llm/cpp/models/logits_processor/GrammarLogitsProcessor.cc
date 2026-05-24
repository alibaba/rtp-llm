#include "rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.h"

#include <algorithm>
#include <limits>

#include <dlpack/dlpack.h>

#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#if USING_CUDA || USING_ROCM
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#endif
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

DLTensor makeSingleRowBitmaskView(int32_t* data, int32_t words) {
    DLTensor dl;
    dl.data   = data;
    dl.device = DLDevice{kDLCPU, 0};
    dl.ndim   = 2;
    dl.dtype  = DLDataType{kDLInt, 32, 1};
    static thread_local int64_t shape[2];
    shape[0]       = 1;
    shape[1]       = words;
    dl.shape       = shape;
    dl.strides     = nullptr;
    dl.byte_offset = 0;
    return dl;
}

bool bitmaskAllowsToken(const int32_t* bitmask, int32_t token_id) {
    const int32_t word = bitmask[token_id / 32];
    return (static_cast<uint32_t>(word) & (1u << (token_id % 32))) != 0u;
}

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

}  // namespace

GrammarLogitsProcessor::GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                               int64_t                            eos_token_id,
                                               ErrorReporter                      error_reporter):
    matcher_(std::move(matcher)), eos_token_id_(eos_token_id), error_reporter_(std::move(error_reporter)) {}

void GrammarLogitsProcessor::prepareNormalAsyncUpdate(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (num_new_tokens <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    pending_async_token_len_ = std::max(pending_async_token_len_, accepted_token_len_ + num_new_tokens);
    if (new_tokens.defined() && new_tokens.is_cuda()) {
        last_mask_device_ = new_tokens.device();
    }
}

int64_t GrammarLogitsProcessor::acceptedTokenLen() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return accepted_token_len_;
}

GrammarLogitsProcessor::DeviceMaskState GrammarLogitsProcessor::getDeviceMaskState(const c10::Device& device) {
    std::unique_lock<std::mutex> lock(state_mutex_);
    last_mask_device_ = device;
    if (pending_async_token_len_ > accepted_token_len_) {
        state_cv_.wait(lock, [this]() {
            return pending_async_token_len_ <= accepted_token_len_ || reported_error_.load() || !matcher_
                   || matcher_->finished();
        });
    }

    if (device_mask_state_.mode != DeviceMaskMode::UNSET && device_mask_state_.token_len == accepted_token_len_
        && device_mask_state_.device == device) {
        return device_mask_state_;
    }

    device_mask_state_ = buildDeviceMaskStateLocked(device);
    return device_mask_state_;
}

GrammarLogitsProcessor::DeviceMaskState GrammarLogitsProcessor::buildDeviceMaskStateLocked(const c10::Device& device) {
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
    if (matcher_->isPassthroughForMask()) {
        state.mode = DeviceMaskMode::PASSTHROUGH;
        return state;
    }

    const int32_t grammar_vocab_size = matcher_->vocabSize();
    if (grammar_vocab_size <= 0) {
        state.mode = DeviceMaskMode::NOOP;
        return state;
    }

    const int32_t words   = (grammar_vocab_size + 31) / 32;
    auto          bitmask = at::full({1, words}, -1, at::dtype(at::kInt));
    DLTensor      dl      = makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), words);
    if (!matcher_->fillBitmask(&dl, 0)) {
        state.mode = DeviceMaskMode::NOOP;
        return state;
    }

    auto mask_options = torch::TensorOptions().dtype(torch::kBool);
    if (device.is_cuda()) {
        mask_options = mask_options.pinned_memory(true);
    }
    auto           vocab_mask  = torch::empty({grammar_vocab_size}, mask_options);
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
#if USING_CUDA || USING_ROCM
    state.ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    state.ready_event->record(cuda_graph::graphGetCurrentStream());
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
        case DeviceMaskMode::PASSTHROUGH:
            maskToken(logits, eos_token_id_);
            return;
        case DeviceMaskMode::MASK:
            break;
    }

    if (!state.vocab_mask.defined()) {
        return;
    }
#if USING_CUDA || USING_ROCM
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

void GrammarLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (!matcher_) {
        return;
    }
    const size_t batch_size = finish_idx - start_idx;
    if (batch_size == 0) {
        return;
    }
    if (batch_size != 1) {
        reportErrorOnce(
            ErrorCode::INVALID_PARAMS, "grammar logits processor only supports single sequence decoding", false);
        return;
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = reinterpret_cast<const bool*>(inputs.finished_mask.data_ptr());
        if (finished[start_idx]) {
            return;
        }
    }

    auto logits = inputs.logits.narrow(0, start_idx, 1);
    auto state  = getDeviceMaskState(logits.device());
    applyDeviceMaskState(logits[0], state);
}

void GrammarLogitsProcessor::processSpeculative(const SamplerInputs&        inputs,
                                                size_t                      start_idx,
                                                size_t                      finish_idx,
                                                const std::vector<int32_t>& draft_prefix) {
    if (draft_prefix.empty()) {
        process(inputs, start_idx, finish_idx);
        return;
    }
    if (!matcher_) {
        return;
    }
    if (finish_idx - start_idx != 1) {
        reportErrorOnce(
            ErrorCode::INVALID_PARAMS, "grammar speculative logits processor only supports single row masking", false);
        return;
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = reinterpret_cast<const bool*>(inputs.finished_mask.data_ptr());
        if (finished[start_idx]) {
            return;
        }
    }

    auto            logits = inputs.logits.narrow(0, start_idx, 1);
    DeviceMaskState state;
    int             accepted_prefix = 0;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (matcher_->finished()) {
            return;
        }
        for (const int32_t token_id : draft_prefix) {
            if (!matcher_->acceptToken(token_id)) {
                matcher_->rollback(accepted_prefix);
                state.mode = DeviceMaskMode::TERMINATED;
                applyDeviceMaskState(logits[0], state);
                return;
            }
            ++accepted_prefix;
            if (matcher_->isTerminated()) {
                break;
            }
        }
        state = buildDeviceMaskStateLocked(logits.device());
        matcher_->rollback(accepted_prefix);
    }
    applyDeviceMaskState(logits[0], state);
}

bool GrammarLogitsProcessor::isSpecVerifyEligible() const {
    return matcher_ != nullptr && !reported_error_.load(std::memory_order_relaxed);
}

int GrammarLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    if (!matcher_ || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return request.propose_step;
    }

    const int  P               = request.propose_step;
    const auto W               = request.bitmask_size_int32;
    int        accepted_prefix = 0;
    int        cap             = P;

    auto fill_row = [&](int32_t* row) {
        std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
        if (matcher_->finished()) {
            return;
        }
        if (matcher_->isTerminated()) {
            forceTokenInBitmask(row, W, eos_token_id_);
            return;
        }
        if (matcher_->isPassthroughForMask()) {
            clearTokenFromBitmask(row, W, eos_token_id_);
            return;
        }

        DLTensor dl = makeSingleRowBitmaskView(row, static_cast<int32_t>(W));
        if (!matcher_->fillBitmask(&dl, 0) && matcher_->isPassthroughForMask()) {
            std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
            clearTokenFromBitmask(row, W, eos_token_id_);
        }
    };

    for (int offset = 0; offset <= P; ++offset) {
        int32_t* row = request.bitmask_cpu_out + offset * W;
        fill_row(row);
        if (offset == P) {
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
    }

    matcher_->rollback(accepted_prefix);
    return cap;
}

void GrammarLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    (void)src_batch_indices;
}

void GrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (num_new_tokens <= 0) {
        return;
    }
    if (new_tokens.dim() != 2 || new_tokens.size(0) != 1 || new_tokens.size(1) < num_new_tokens) {
        reportErrorOnce(ErrorCode::INVALID_PARAMS, "grammar accept expects one row with num_new_tokens columns", true);
        state_cv_.notify_all();
        return;
    }

    auto tokens_cpu       = new_tokens.is_cuda() ? new_tokens.cpu() : new_tokens;
    tokens_cpu            = tokens_cpu.to(torch::kInt32).contiguous();
    const auto* token_ptr = tokens_cpu.data_ptr<int32_t>();

    std::unique_lock<std::mutex> lock(state_mutex_);
    if (!matcher_ || matcher_->finished()) {
        state_cv_.notify_all();
        return;
    }

    for (int32_t i = 0; i < num_new_tokens; ++i) {
        const int32_t token_id = token_ptr[i];
        if (matcher_->isTerminated()) {
            if (token_id != eos_token_id_) {
                reportErrorOnce(ErrorCode::INVALID_PARAMS,
                                "grammar received non-EOS token after terminal state " + std::to_string(token_id),
                                true);
                return;
            }
            ++accepted_token_len_;
            matcher_->markFinished();
            break;
        }
        if (!matcher_->acceptToken(token_id)) {
            matcher_->markFinished();
            device_mask_state_ =
                buildDeviceMaskStateLocked(last_mask_device_.value_or(c10::Device(c10::DeviceType::CPU)));
            state_cv_.notify_all();
            reportErrorOnce(ErrorCode::INVALID_PARAMS,
                            "grammar accept_token error: parser rejected token " + std::to_string(token_id),
                            true);
            return;
        }
        ++accepted_token_len_;
        if (matcher_->isTerminated()) {
            break;
        }
    }

    device_mask_state_ = buildDeviceMaskStateLocked(last_mask_device_.value_or(c10::Device(c10::DeviceType::CPU)));
    state_cv_.notify_all();
}

void GrammarLogitsProcessor::reportErrorOnce(ErrorCode          error_code,
                                             const std::string& error_msg,
                                             bool               stream_lock_held) {
    if (reported_error_.exchange(true)) {
        return;
    }
    state_cv_.notify_all();
    if (error_reporter_) {
        error_reporter_(error_code, error_msg, stream_lock_held);
        return;
    }
    RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
}

void GrammarLogitsProcessor::forceToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        reportErrorOnce(ErrorCode::INVALID_PARAMS, "grammar terminal token is out of logits vocab range", false);
        return;
    }
    logits.fill_(BaseLogitsProcessor::neg_inf);
    logits[token_id] = 1;
}

void GrammarLogitsProcessor::maskToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        return;
    }
    logits[token_id] = BaseLogitsProcessor::neg_inf;
}

}  // namespace rtp_llm
