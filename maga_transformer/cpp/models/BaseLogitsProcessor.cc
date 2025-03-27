#include "maga_transformer/cpp/models/BaseLogitsProcessor.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

BaseLogitsProcessor::BaseLogitsProcessor(ft::DeviceBase* device) : device_(device) {};

void BaseLogitsProcessor::memFill(ft::BufferPtr new_tokens_logits, size_t vocab_size, size_t index) {
    device_->bufMemset(*new_tokens_logits, 0);
    auto tensor = Buffer2torchTensor(*new_tokens_logits, false);
    tensor[index] = 1;
}

}

