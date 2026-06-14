#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorErrorReporter.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

class GenerateStream;

class BaseLogitsProcessor {
public:
    BaseLogitsProcessor() = default;
    virtual ~BaseLogitsProcessor() {}
    static const float neg_inf;

public:
    virtual void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) = 0;
    virtual void updateMultiSeqStatus(const std::vector<int>& src_batch_indices) = 0;

    virtual bool isStateful() const {
        return false;
    }
    virtual bool supportsNormalAsyncDeviceState() const {
        return false;
    }
    virtual void prepareNormalAsyncUpdate(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
        (void)new_tokens;
        (void)num_new_tokens;
    }
    virtual int64_t acceptedTokenLen() const {
        return 0;
    }

    virtual void setStream(const std::shared_ptr<GenerateStream>& /*stream*/) {}

    void setErrorReporter(LogitsProcessorErrorReporter cb) {
        error_reporter_ = std::move(cb);
    }
    void reportErrorViaReporter(ErrorCode code, const std::string& msg, bool stream_lock_held) {
        if (error_reporter_) {
            error_reporter_(code, msg, stream_lock_held);
        }
    }

    // Engine invariant: called exactly once per token committed by GenerateStream
    // (update / specUpdate / disagg replay). Spec/disagg paths must not bypass it.
    virtual void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) = 0;

    void          memFill(const torch::Tensor& new_tokens_logits, size_t vocab_size, size_t index);
    void          maskLogits(torch::Tensor& new_token_logits, const torch::Tensor& vocab_mask);
    torch::Tensor generateVocabMask(size_t                                  batch_size,
                                    size_t                                  vocab_size,
                                    const std::vector<std::vector<size_t>>& batch_candidate_token_ids);

protected:
    LogitsProcessorErrorReporter error_reporter_;
};

typedef std::shared_ptr<BaseLogitsProcessor> BaseLogitsProcessorPtr;

}  // namespace rtp_llm
