#include "rtp_llm/cpp/models/logits_processor/BeamSearchLogitsProcessor.h"

namespace rtp_llm {

BeamSearchLogitsProcessor::BeamSearchLogitsProcessor(rtp_llm::DeviceBase* device): BaseLogitsProcessor(device) {}

std::shared_ptr<BeamSearchLogitsProcessor> BeamSearchLogitsProcessor::fromGenerateInput(
    rtp_llm::DeviceBase* device, std::shared_ptr<GenerateInput> generate_input, int64_t eos_token_id) {

    if (!generate_input->generate_config->hasNumBeams()) {
        return nullptr;
    }

    auto processor_ptr           = std::make_shared<BeamSearchLogitsProcessor>(device);
    processor_ptr->eos_token_id_ = eos_token_id;

    return processor_ptr;
}

void BeamSearchLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    size_t batch_size = finish_idx - start_idx;
    size_t vocab_size = inputs.logits->shape()[1];

    auto logits            = inputs.logits->slice(start_idx, batch_size);
    auto finished_mask     = inputs.finished_mask->slice(start_idx, finish_idx - start_idx);
    auto finished_mask_ptr = finished_mask->data<bool>();

    // return early when no beam needs processing
    if (!std::any_of(finished_mask_ptr, finished_mask_ptr + batch_size, [](bool v) { return v; })) {
        return;
    }

    // mask all logits of the finished beam except the eos token
    auto logit_mask_host =
        device_->allocateBuffer({DataType::TYPE_UINT8, {batch_size, vocab_size}, AllocationType::HOST});

    auto logit_mask_host_ptr = logit_mask_host->data<uint8_t>();
    memset(logit_mask_host_ptr, 0, batch_size * vocab_size * sizeof(uint8_t));

    for (size_t idx = 0; idx < batch_size; ++idx) {
        if (finished_mask_ptr[idx]) {
            auto cur_logit_mask_host_ptr = logit_mask_host_ptr + idx * vocab_size;
            memset(cur_logit_mask_host_ptr, 1, vocab_size * sizeof(uint8_t));
            cur_logit_mask_host_ptr[eos_token_id_] = 0;
        }
    }

    auto logit_mask = device_->clone({*logit_mask_host, AllocationType::DEVICE});

    maskLogits(logits, logit_mask);
}

void BeamSearchLogitsProcessor::beamSearchLogitProcessorUpdate(const std::vector<int>& beam_idx_vec) {
    // do nothing
}

void BeamSearchLogitsProcessor::updateLogitProcessorStatus(const rtp_llm::BufferPtr& new_tokens,
                                                           int32_t                   num_new_tokens) {
    // do nothing
}

}  // namespace rtp_llm