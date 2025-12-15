#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

using namespace std;

namespace rtp_llm {

const float BaseLogitsProcessor::neg_inf = -std::numeric_limits<float>::max();

BaseLogitsProcessor::BaseLogitsProcessor(rtp_llm::DeviceBase* device): device_(device) {};

void BaseLogitsProcessor::memFill(const rtp_llm::BufferPtr& new_tokens_logits, size_t vocab_size, size_t index) {
    auto shapes = new_tokens_logits->shape();
    RTP_LLM_CHECK(shapes.size() == 1);
    auto tensor = Buffer2torchTensor(*new_tokens_logits, false);
    tensor.fill_(neg_inf);
    tensor[index] = 1;
}

rtp_llm::BufferPtr BaseLogitsProcessor::generateVocabMask(
    size_t batch_size, size_t vocab_size, const std::vector<std::vector<size_t>>& batch_candidate_token_ids) {
    RTP_LLM_CHECK(batch_candidate_token_ids.size() == batch_size);
    std::vector<uint8_t> vocab_mask_cpu(batch_size * vocab_size, 1);

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const auto& candidate_token_ids = batch_candidate_token_ids[batch_idx];
        for (const auto& token_id : candidate_token_ids) {
            if (token_id < vocab_size) {
                vocab_mask_cpu[batch_idx * vocab_size + token_id] = 0;
            }
        }
    }

    BufferPtr vocab_mask_buffer_cpu = vector2Buffer(vocab_mask_cpu);
    auto      buffer_reshape        = vocab_mask_buffer_cpu->reshape({batch_size, vocab_size});
    return device_->clone({buffer_reshape, rtp_llm::AllocationType::DEVICE});
}
void BaseLogitsProcessor::maskLogits(const rtp_llm::BufferPtr& new_tokens_logits,
                                     const rtp_llm::BufferPtr& vocab_mask) {
    RTP_LLM_CHECK(new_tokens_logits->shape().size() == 2);
    RTP_LLM_CHECK(vocab_mask->shape().size() == 2);
    RTP_LLM_CHECK(new_tokens_logits->shape()[0] == vocab_mask->shape()[0]);
    RTP_LLM_CHECK(new_tokens_logits->shape()[1] == vocab_mask->shape()[1]);
    device_->maskLogits(*new_tokens_logits, *vocab_mask);
}

rtp_llm::BufferPtr BaseLogitsProcessor::generateVocabWeight(
    size_t                                                batch_size,
    size_t                                                vocab_size,
    const std::vector<std::unordered_map<size_t, float>>& batch_candidate_token_weights) {
    RTP_LLM_CHECK(batch_candidate_token_weights.size() == batch_size);
    std::vector<float> vocab_weight_cpu(batch_size * vocab_size, -INFINITY);

    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const std::unordered_map<size_t, float>& candidate_token_weight = batch_candidate_token_weights[batch_idx];
        for (auto it = candidate_token_weight.begin(); it != candidate_token_weight.end(); ++it) {
            vocab_weight_cpu[batch_idx * vocab_size + it->first] = it->second;
        }
    }

    BufferPtr vocab_weight_buffer_cpu = vector2Buffer(vocab_weight_cpu);
    auto      buffer_reshape          = vocab_weight_buffer_cpu->reshape({batch_size, vocab_size});
    return device_->clone({buffer_reshape, rtp_llm::AllocationType::DEVICE});
}

void BaseLogitsProcessor::weightLogits(const rtp_llm::BufferPtr& new_tokens_logits,
                                       const rtp_llm::BufferPtr& vocab_weight) {
    RTP_LLM_CHECK(new_tokens_logits->shape().size() == 2);
    RTP_LLM_CHECK(vocab_weight->shape().size() == 2);
    RTP_LLM_CHECK(new_tokens_logits->shape()[0] == vocab_weight->shape()[0]);
    RTP_LLM_CHECK(new_tokens_logits->shape()[1] == vocab_weight->shape()[1]);
    device_->weightLogits(*new_tokens_logits, *vocab_weight);
}

}  // namespace rtp_llm
