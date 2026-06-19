#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <dlpack/dlpack.h>
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.h"

namespace rtp_llm {

class GrammarLogitsProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    explicit GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                    int64_t                            eos_token_id = 0);

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
    // Drives applyDeviceMaskState. Mirrors RtpGrammarMatcher's lifecycle so the
    // mode alone determines the action — no second lookup against the matcher.
    //   UNSET        : never built (initial / cache miss); next call must build.
    //   NOOP         : matcher absent or grammar vocab=0 — leave logits alone.
    //   MASK         : real bitmask present; apply to logits.
    //   PASSTHROUGH  : reasoning-passthrough segment — only suppress EOS.
    //   TERMINATED   : matcher reached terminal state — force EOS.
    //   FINISHED     : matcher finalized (post-terminal commit / hard error) —
    //                  leave logits alone; commit path will surface end-of-stream.
    enum class DeviceMaskMode {
        UNSET,
        NOOP,
        MASK,
        PASSTHROUGH,
        TERMINATED,
        FINISHED,
    };

    struct DeviceMaskState {
        DeviceMaskMode mode      = DeviceMaskMode::UNSET;
        int64_t        token_len = -1;
        c10::Device    device    = c10::Device(c10::DeviceType::CPU);
        torch::Tensor  vocab_mask;  // [grammar_vocab_size] bool, true=disallow
        int32_t        grammar_vocab_size = 0;
    };

    DeviceMaskState getDeviceMaskState(const c10::Device& device);
    DeviceMaskState buildDeviceMaskStateLocked(const c10::Device& device);
    void            publishMaskToDevice(DeviceMaskState& state, torch::Tensor vocab_mask, const c10::Device& device);
    void            applyDeviceMaskState(const torch::Tensor& logits, const DeviceMaskState& state);
    void            forceToken(const torch::Tensor& logits, int64_t token_id);
    void            maskToken(const torch::Tensor& logits, int64_t token_id);

    std::shared_ptr<RtpGrammarMatcher> matcher_;

    mutable std::mutex         state_mutex_;
    int64_t                    eos_token_id_       = 0;
    int64_t                    accepted_token_len_ = 0;
    std::optional<c10::Device> last_mask_device_;
    DeviceMaskState            device_mask_state_{};
    torch::Tensor              reusable_bitmask_cpu_;
    torch::Tensor              reusable_vocab_mask_cpu_;
    int32_t                    reusable_mask_words_ = 0;
};

}  // namespace rtp_llm
