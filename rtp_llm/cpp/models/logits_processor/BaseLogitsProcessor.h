#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

struct SpecLogitsProcessorRequest;

enum class MtpProcessorMode : uint8_t {
    UNSUPPORTED,
    SPEC_VERIFY,
};

struct MtpProcessorCapability {
    MtpProcessorMode mode   = MtpProcessorMode::UNSUPPORTED;
    std::string_view reason = "processor supports normal decoding only";
};

class BaseLogitsProcessor {
public:
    BaseLogitsProcessor() = default;
    virtual ~BaseLogitsProcessor() {}
    static const float neg_inf;

public:
    virtual std::optional<ErrorInfo> process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) = 0;
    virtual void                     updateMultiSeqStatus(const std::vector<int>& src_batch_indices)           = 0;

    // Called exactly once for every token batch successfully appended to GenerateStream's
    // authoritative history, including a batch that finishes the stream. Normal and
    // speculative decoding share this callback and invoke it before publishing output.
    virtual std::optional<ErrorInfo> updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) = 0;

    // MTP support is instance-specific; processors can opt into speculative verification.
    virtual MtpProcessorCapability mtpCapability() const {
        return {};
    }

    // Implementations must restore committed state before returning. The returned cap
    // is in [0, propose_step]. This hook is called only in SPEC_VERIFY mode.
    virtual ErrorResult<int> prepareSpeculative(const SpecLogitsProcessorRequest& request);

    // Stateful processors expose their committed length for stream/processor parity
    // checks. Stateless processors return std::nullopt.
    virtual std::optional<int64_t> committedOutputLen() const {
        return std::nullopt;
    }

    void          memFill(const torch::Tensor& new_tokens_logits, size_t vocab_size, size_t index);
    void          maskLogits(torch::Tensor& new_token_logits, const torch::Tensor& vocab_mask);
    torch::Tensor generateVocabMask(size_t                                  batch_size,
                                    size_t                                  vocab_size,
                                    const std::vector<std::vector<size_t>>& batch_candidate_token_ids);
};

typedef std::shared_ptr<BaseLogitsProcessor> BaseLogitsProcessorPtr;

}  // namespace rtp_llm
