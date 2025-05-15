#include "maga_transformer/cpp/logits_processor/BaseLogitsProcessor.h"
#include "maga_transformer/cpp/core/torch_utils/BufferTorchUtils.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"

using namespace std;

namespace rtp_llm {

const float BaseLogitsProcessor::neg_inf = -std::numeric_limits<float>::max();

BaseLogitsProcessor::BaseLogitsProcessor(rtp_llm::DeviceBase* device) : device_(device) {};

void BaseLogitsProcessor::memFill(const rtp_llm::BufferPtr& new_tokens_logits, size_t vocab_size, size_t index) {
    auto shapes = new_tokens_logits->shape();
    RTP_LLM_CHECK(shapes.size() == 1);
    auto tensor = Buffer2torchTensor(*new_tokens_logits, false);
    tensor.fill_(neg_inf);
    tensor[index] = 1;
}

rtp_llm::BufferPtr BaseLogitsProcessor::generateVocabMask(const std::vector<size_t>& dshape, const std::vector<size_t>& candidate_token_ids) {
    RTP_LLM_CHECK(dshape.size() == 1);
    std::vector<uint8_t> vocab_mask_cpu(dshape[0], 0);
    for (size_t i = 0; i < dshape[0]; i++) {
        vocab_mask_cpu[i] = 1;
    }
    for (const auto& token_id: candidate_token_ids) {
        vocab_mask_cpu[token_id] = 0;
    }
    BufferPtr vocab_mask_buffer_cpu = vector2Buffer(vocab_mask_cpu);
    return device_->clone({*vocab_mask_buffer_cpu, rtp_llm::AllocationType::DEVICE});
}

void BaseLogitsProcessor::maskLogits(const rtp_llm::BufferPtr& new_tokens_logits, const rtp_llm::BufferPtr& vocab_mask) {
    RTP_LLM_CHECK(new_tokens_logits->shape().size() == 1);
    RTP_LLM_CHECK(vocab_mask->shape().size() == 1);
    RTP_LLM_CHECK(new_tokens_logits->shape()[0] == vocab_mask->shape()[0]);
    device_->maskLogits(*new_tokens_logits, *vocab_mask);
}

}

