#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <dlpack/dlpack.h>
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/engine_base/grammar/XGrammarBackend.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class RtpGrammarMatcher;
class GenerateInput;
struct SamplerInputs;

class GrammarLogitsProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    explicit GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher>    matcher,
                                    int64_t                               eos_token_id   = 0,
                                    LogitsProcessorFactory::ErrorReporter error_reporter = {});

    ~GrammarLogitsProcessor() override;

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;

    bool isStateful() const override {
        return true;
    }

    bool isSpecVerifyEligible() const override;
    int  tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;

    int64_t acceptedTokenLen() const override {
        return accepted_token_len_;
    }

private:
    // Skip:     no-op (matcher absent / finished / vocab=0 — caller leaves logits untouched).
    // ForceEOS: matcher terminated; force EOS in logits.
    // Mask:     real bitmask present; apply to logits.
    enum class DeviceMaskMode {
        Skip,
        ForceEOS,
        Mask,
    };

    struct DeviceMaskState {
        DeviceMaskMode                mode      = DeviceMaskMode::Skip;
        int64_t                       token_len = -1;
        torch::Tensor                 vocab_mask;
        torch::Tensor                 packed_bitmask;
        int32_t                       grammar_vocab_size = 0;
        std::shared_ptr<torch::Event> ready_event;
    };

    // stream_lock_held: true when the caller already holds the GenerateStream's mutex_
    // (e.g. updateStatus path); false otherwise (process / spec verify paths). Plumbed
    // through to reportErrorViaReporter so the reporter picks the matching reportError
    // variant — passing the wrong value self-deadlocks since std::mutex is non-recursive.
    DeviceMaskState getDeviceMaskState(const c10::Device& device, bool stream_lock_held);
    DeviceMaskState buildDeviceMaskStateLocked(const c10::Device& device, bool stream_lock_held);
    void            publishMaskToDevice(DeviceMaskState& state, torch::Tensor vocab_mask, const c10::Device& device);
    void            applyDeviceMaskState(const torch::Tensor& logits, const DeviceMaskState& state);
    void            forceToken(const torch::Tensor& logits, int64_t token_id);

    std::shared_ptr<RtpGrammarMatcher> matcher_;

    mutable std::mutex            state_mutex_;
    std::atomic_bool              reported_error_{false};
    int64_t                       eos_token_id_       = 0;
    int64_t                       accepted_token_len_ = 0;
    std::optional<c10::Device>     last_mask_device_;
    std::optional<DeviceMaskState> device_mask_state_;
    torch::Tensor                 reusable_bitmask_cpu_;
    torch::Tensor                 reusable_bitmask_gpu_;
    torch::Tensor                 reusable_vocab_mask_cpu_;
    int32_t                       reusable_mask_words_ = 0;
    std::shared_ptr<torch::Event> reusable_ready_event_;
};

}  // namespace rtp_llm
