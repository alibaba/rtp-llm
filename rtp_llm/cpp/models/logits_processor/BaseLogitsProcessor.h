#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/logits_processor/PrefixToCandidateTokens.h"

namespace rtp_llm {

class BaseLogitsProcessor {
public:
    BaseLogitsProcessor(rtp_llm::DeviceBase* device);
    virtual ~BaseLogitsProcessor() {}
    static const float neg_inf;

public:
    virtual void       process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx)  = 0;
    virtual void       updateMultiSeqStatus(const std::vector<int>& src_batch_indices)            = 0;
    virtual void       updateStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) = 0;
    void               memFill(const rtp_llm::BufferPtr& new_tokens_logits, size_t vocab_size, size_t index);
    void               maskLogits(const rtp_llm::BufferPtr& new_token_logits, const rtp_llm::BufferPtr& vocab_mask);
    rtp_llm::BufferPtr generateVocabMask(size_t                                  batch_size,
                                         size_t                                  vocab_size,
                                         const std::vector<std::vector<size_t>>& batch_candidate_token_ids);
    void               sparseMaskLogits(const rtp_llm::BufferPtr& new_tokens_logits,
                                        const rtp_llm::BufferPtr& batch_idx,
                                        const rtp_llm::BufferPtr& vocab_mask);
    std::vector<rtp_llm::BufferPtr> generateSparseVocabMask(
        size_t batch_size, size_t vocab_size, const std::vector<std::vector<size_t>>& batch_candidate_token_ids);

    void                            weightLogits(const rtp_llm::BufferPtr& new_token_logits,
                                                 const rtp_llm::BufferPtr& batch_idx,
                                                 const rtp_llm::BufferPtr& vocab_idx,
                                                 const rtp_llm::BufferPtr& vocab_weight);
    std::vector<rtp_llm::BufferPtr> generateVocabWeight(
        size_t batch_size, size_t vocab_size, const std::vector<const TokenWeights*>& batch_candidate_token_weights);

protected:
    rtp_llm::DeviceBase* device_;
};

typedef std::shared_ptr<BaseLogitsProcessor> BaseLogitsProcessorPtr;

}  // namespace rtp_llm