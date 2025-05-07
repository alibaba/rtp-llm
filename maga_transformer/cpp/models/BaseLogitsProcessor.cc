#include "maga_transformer/cpp/models/BaseLogitsProcessor.h"
#include "maga_transformer/cpp/core/torch_utils/BufferTorchUtils.h"
#include "maga_transformer/cpp/utils/AssertUtils.h"

using namespace std;


namespace rtp_llm {

const float BaseLogitsProcessor::neg_inf = -std::numeric_limits<float>::max();

BaseLogitsProcessor::BaseLogitsProcessor(rtp_llm::DeviceBase* device) : device_(device) {};

void BaseLogitsProcessor::memFill(rtp_llm::BufferPtr new_tokens_logits, size_t vocab_size, size_t index) {
    auto shapes = new_tokens_logits->shape();
    RTP_LLM_CHECK(shapes.size() == 1);
    auto tensor = Buffer2torchTensor(*new_tokens_logits, false);
    tensor.fill_(neg_inf);
    tensor[index] = 1;
}

}

