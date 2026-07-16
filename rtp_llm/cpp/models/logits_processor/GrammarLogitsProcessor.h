#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class RtpGrammarMatcher;

class GrammarLogitsProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    using ErrorReporter = std::function<void(ErrorCode, const std::string&, bool)>;

    GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                           int64_t                            eos_token_id,
                           ErrorReporter                      error_reporter = nullptr);
    ~GrammarLogitsProcessor() override = default;

    void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void processSpeculative(const SamplerInputs&        inputs,
                            size_t                      start_idx,
                            size_t                      finish_idx,
                            const std::vector<int32_t>& draft_prefix) override;
    void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;
    bool isSpecVerifyEligible() const override;
    int  tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;
    bool isStateful() const override {
        return true;
    }
    bool supportsNormalAsyncDeviceState() const override {
        return true;
    }
    void    prepareNormalAsyncUpdate(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;
    int64_t acceptedTokenLen() const override;

private:
    enum class DeviceMaskMode {
        UNSET,
        NOOP,
        MASK,
        PASSTHROUGH,
        TERMINATED,
        FINISHED,
    };

    struct DeviceMaskState {
        DeviceMaskMode                mode      = DeviceMaskMode::UNSET;
        int64_t                       token_len = -1;
        c10::Device                   device    = c10::Device(c10::DeviceType::CPU);
        torch::Tensor                 vocab_mask;
        std::shared_ptr<torch::Event> ready_event;
    };

private:
    DeviceMaskState getDeviceMaskState(const c10::Device& device);
    DeviceMaskState buildDeviceMaskStateLocked(const c10::Device& device);
    void            publishMaskToDevice(DeviceMaskState& state, torch::Tensor vocab_mask, const c10::Device& device);
    void            applyDeviceMaskState(const torch::Tensor& logits, const DeviceMaskState& state);
    void            reportErrorOnce(ErrorCode error_code, const std::string& error_msg, bool stream_lock_held);
    void            forceToken(const torch::Tensor& logits, int64_t token_id);
    void            maskToken(const torch::Tensor& logits, int64_t token_id);

private:
    mutable std::mutex                 state_mutex_;
    std::condition_variable            state_cv_;
    std::shared_ptr<RtpGrammarMatcher> matcher_;
    int64_t                            eos_token_id_;
    ErrorReporter                      error_reporter_;
    std::atomic_bool                   reported_error_          = false;
    int64_t                            accepted_token_len_      = 0;
    int64_t                            pending_async_token_len_ = 0;
    std::optional<c10::Device>         last_mask_device_;
    DeviceMaskState                    device_mask_state_;
};

using GrammarLogitsProcessorPtr = std::shared_ptr<GrammarLogitsProcessor>;

}  // namespace rtp_llm
