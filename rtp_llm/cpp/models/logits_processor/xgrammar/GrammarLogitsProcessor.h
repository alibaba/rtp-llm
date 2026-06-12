#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <dlpack/dlpack.h>
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarCompiler.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/ReasoningGate.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class RtpGrammarMatcher;
class GenerateStream;
class GenerateInput;
struct SamplerInputs;
using GenerateStreamPtr = std::shared_ptr<GenerateStream>;

class GrammarLogitsProcessor: public BaseLogitsProcessor, public SpecLogitsProcessor {
public:
    explicit GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                    GenerateStreamPtr                  stream = nullptr);

    ~GrammarLogitsProcessor() override;

    static BaseLogitsProcessorPtr tryCreatePending(const std::shared_ptr<GenerateInput>& input);

    void         process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) override;
    void         updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) override;
    void         updateMultiSeqStatus(const std::vector<int>& src_batch_indices) override;
    bool         needsPreparation() const override;
    PrepareState prepare(GenerateStream& stream) override;
    void         setStream(const std::shared_ptr<GenerateStream>& stream) override { stream_ = stream; }

    bool isStateful() const override { return true; }

    bool isSpecVerifyEligible() const override;
    int  tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) override;

    RtpGrammarMatcher* grammarMatcher() const noexcept { return matcher_.get(); }
    bool               isGrammarMatcherComplete() const noexcept;

private:
    GrammarLogitsProcessor() = default;

    enum class DeviceMaskMode {
        UNSET,
        NOOP,
        MASK,
        TERMINATED,
        FINISHED,
        PASSTHROUGH,  // think-body: allow all model-vocab tokens; parser frozen.
    };

    struct DeviceMaskState {
        DeviceMaskMode                mode      = DeviceMaskMode::UNSET;
        int64_t                       token_len = -1;
        c10::Device                   device    = c10::Device(c10::DeviceType::CPU);
        torch::Tensor                 vocab_mask;
        torch::Tensor                 packed_bitmask;
        int32_t                       grammar_vocab_size = 0;
        std::shared_ptr<torch::Event> ready_event;
    };

    enum class MatcherAdvance { Commit, ReplayOnly };
    enum class Kind { FailFast, Compile };

    DeviceMaskState getDeviceMaskState(const c10::Device& device);
    DeviceMaskState buildDeviceMaskStateLocked(const c10::Device& device);
    void            publishMaskToDevice(DeviceMaskState& state, torch::Tensor vocab_mask, const c10::Device& device);
    void            applyDeviceMaskState(const torch::Tensor& logits, const DeviceMaskState& state);
    void            forceToken(const torch::Tensor& logits, int64_t token_id);
    void            maskToken(const torch::Tensor& logits, int64_t token_id);
    void            syncAcceptedTokenLenLocked();
    void            rebuildDeviceMaskStateLocked();

    bool advanceMatcher(const std::vector<int32_t>& tokens, MatcherAdvance mode, bool caller_holds_stream_lock = false);
    void logGrammarLifecycle(const char* phase, const char* source, int64_t stream_id = -1) const;
    void installMatcher(GenerateStream& stream, const GrammarReadyPayload& payload);
    bool inReasoningPassthrough() const noexcept;

    Kind        kind_      = Kind::Compile;
    ErrorCode   fail_code_ = ErrorCode::INVALID_PARAMS;
    std::string fail_msg_;

    GrammarKeyCpp                           key_;
    std::shared_future<GrammarReadyPayload> future_;
    std::chrono::steady_clock::time_point   deadline_;
    bool                                    resolved_ = false;

    // Reasoning gate inputs captured at admission. require_reasoning_ is true
    // when the request started inside a think body; the processor publishes a
    // PASSTHROUGH (allow-all) mask and skips matcher advancement until the
    // ReasoningGate observes think_end_token_ids_ in the token stream.
    bool             require_reasoning_   = false;
    std::vector<int> think_end_token_ids_;

    std::shared_ptr<RtpGrammarMatcher> matcher_;
    std::unique_ptr<ReasoningGate>     reasoning_gate_;
    std::weak_ptr<GenerateStream>      stream_;

    mutable std::mutex          state_mutex_;
    std::atomic_bool            reported_error_{false};
    int64_t                     eos_token_id_              = 0;
    // Total tokens advanced by the processor (matcher accepts + gate-passthrough
    // observes). accepted_token_len_ reflects this; matcher_->numAcceptedTokens()
    // alone would miss tokens consumed while the reasoning gate was passthrough.
    int64_t                     total_advanced_              = 0;
    int64_t                     accepted_token_len_          = 0;
    std::optional<c10::Device>  last_mask_device_;
    DeviceMaskState             device_mask_state_;
    torch::Tensor               reusable_bitmask_cpu_;
    torch::Tensor               reusable_bitmask_gpu_;
    torch::Tensor               reusable_vocab_mask_cpu_;
    int32_t                     reusable_mask_words_ = 0;
    // Reused per-stream cuda event: re-recorded on every getDeviceMaskState()
    // instead of std::make_shared<torch::Event> per token. Lazily allocated on
    // first CUDA path through the processor; never freed during the stream's
    // lifetime since it's only a few bytes of CUDA event handle.
    std::shared_ptr<torch::Event> reusable_ready_event_;
    // True once reusable_bitmask_cpu_ has been moved to pinned memory. Pinning
    // converts the H2D copy_ into a real async DMA (otherwise PyTorch silently
    // does a sync pin+copy on a pageable source, defeating non_blocking=true).
    bool                        reusable_bitmask_cpu_pinned_ = false;
    // Per-stream once-flag for the unsupported batch_size!=1 path. Without
    // this the WARN at process() fires on every decode step for the lifetime
    // of an offending stream and drowns the engine log.
    bool                        warned_multi_seq_unsupported_ = false;
};

}  // namespace rtp_llm
