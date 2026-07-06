#pragma once

#include <vector>

#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

class BaseLogitsProcessor {
public:
    BaseLogitsProcessor() = default;
    virtual ~BaseLogitsProcessor() {}
    static const float neg_inf;

public:
    virtual void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) = 0;
    virtual void processSpeculative(const SamplerInputs&        inputs,
                                    size_t                      start_idx,
                                    size_t                      finish_idx,
                                    const std::vector<int32_t>& draft_prefix) {
        (void)draft_prefix;
        process(inputs, start_idx, finish_idx);
    }
    virtual void updateMultiSeqStatus(const std::vector<int>& src_batch_indices)       = 0;
    virtual void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) = 0;
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
    virtual int64_t thinkContentTokenLen() const {
        return -1;
    }
    void          memFill(const torch::Tensor& new_tokens_logits, size_t vocab_size, size_t index);
    void          maskLogits(torch::Tensor& new_token_logits, const torch::Tensor& vocab_mask);
    torch::Tensor generateVocabMask(size_t                                  batch_size,
                                    size_t                                  vocab_size,
                                    const std::vector<std::vector<size_t>>& batch_candidate_token_ids);
};

typedef std::shared_ptr<BaseLogitsProcessor> BaseLogitsProcessorPtr;

}  // namespace rtp_llm
