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

std::vector<rtp_llm::BufferPtr> BaseLogitsProcessor::generateVocabWeight(
    size_t batch_size, size_t vocab_size, const std::vector<const TokenWeights*>& batch_candidate_token_weights) {
    RTP_LLM_CHECK(batch_candidate_token_weights.size() == batch_size);
    std::vector<rtp_llm::BufferPtr> result;

    int total_num = 0;
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        if (batch_candidate_token_weights[batch_idx] != nullptr) {
            total_num += batch_candidate_token_weights[batch_idx]->token_ids.size();
        }
    }
    std::vector<int>   h_batch_indices(total_num);  // batch id
    std::vector<int>   h_vocab_indices(total_num);  // vocab index
    std::vector<float> h_vocab_weight(total_num);   // weight value

    int offset = 0;
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const TokenWeights* token_weight_ptr = batch_candidate_token_weights[batch_idx];
        if (token_weight_ptr != nullptr) {
            std::copy(token_weight_ptr->token_ids.begin(),
                      token_weight_ptr->token_ids.end(),
                      h_vocab_indices.begin() + offset);
            std::copy(
                token_weight_ptr->weights.begin(), token_weight_ptr->weights.end(), h_vocab_weight.begin() + offset);
            int end_idx = offset + token_weight_ptr->token_ids.size();
            std::fill(h_batch_indices.begin() + offset, h_batch_indices.begin() + end_idx, batch_idx);
            offset = end_idx;
        }
    }

    BufferPtr d_batch_indices = vector2Buffer(h_batch_indices);
    BufferPtr d_vocab_indices = vector2Buffer(h_vocab_indices);
    BufferPtr d_vocab_weight  = vector2Buffer(h_vocab_weight);

    result.push_back(device_->clone({*d_batch_indices, rtp_llm::AllocationType::DEVICE}));
    result.push_back(device_->clone({*d_vocab_indices, rtp_llm::AllocationType::DEVICE}));
    result.push_back(device_->clone({*d_vocab_weight, rtp_llm::AllocationType::DEVICE}));
    return result;
}

void BaseLogitsProcessor::weightLogits(const rtp_llm::BufferPtr& new_tokens_logits,
                                       const rtp_llm::BufferPtr& batch_idx,
                                       const rtp_llm::BufferPtr& vocab_idx,
                                       const rtp_llm::BufferPtr& vocab_weight) {
    RTP_LLM_CHECK(new_tokens_logits->shape().size() == 2);
    RTP_LLM_CHECK(batch_idx->shape().size() == 1);
    RTP_LLM_CHECK(vocab_idx->shape().size() == batch_idx->shape().size());
    RTP_LLM_CHECK(vocab_weight->shape().size() == batch_idx->shape().size());
    device_->weightLogits(*new_tokens_logits, *batch_idx, *vocab_idx, *vocab_weight);
}

}  // namespace rtp_llm
