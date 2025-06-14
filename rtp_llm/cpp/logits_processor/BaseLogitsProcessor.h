#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/dataclass/SampleInfos.h"
#include "rtp_llm/cpp/dataclass/Query.h"



namespace rtp_llm {

class BaseLogitsProcessor {
public:
    BaseLogitsProcessor(rtp_llm::DeviceBase* device);
    virtual ~BaseLogitsProcessor() {}
    static const float neg_inf;

public:
    virtual void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) = 0;
    virtual void beamSearchLogitProcessorUpdate(const std::vector<int>& beam_idx_vec) = 0;
    virtual void updateLogitProcessorStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) = 0;
    void memFill(const rtp_llm::BufferPtr& new_tokens_logits, size_t vocab_size, size_t index);
    void maskLogits(const rtp_llm::BufferPtr& new_tokens_logits, const rtp_llm::BufferPtr& vocab_mask);
    rtp_llm::BufferPtr generateVocabMask(const std::vector<size_t>& dshape, const std::vector<size_t>& candidate_token_ids);

protected:
    rtp_llm::DeviceBase* device_;
};

typedef std::shared_ptr<BaseLogitsProcessor> BaseLogitsProcessorPtr;

} // namespace rtp_llm