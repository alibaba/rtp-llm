#include "rtp_llm/cpp/models/logits_processor/MultiSeqLogitsProcessor.h"

namespace rtp_llm {

MultiSeqLogitsProcessor::MultiSeqLogitsProcessor(rtp_llm::DeviceBase* device): BaseLogitsProcessor(device) {}

std::shared_ptr<MultiSeqLogitsProcessor> MultiSeqLogitsProcessor::fromGenerateInput(
    rtp_llm::DeviceBase* device, std::shared_ptr<GenerateInput> generate_input, int64_t eos_token_id) {

    if (generate_input->generate_config->num_return_sequences <= 1 && !generate_input->generate_config->hasNumBeams()) {
        return nullptr;
    }

    auto processor_ptr           = std::make_shared<MultiSeqLogitsProcessor>(device);
    processor_ptr->eos_token_id_ = eos_token_id;

    return processor_ptr;
}

void MultiSeqLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    size_t batch_size = finish_idx - start_idx;

    auto logits            = inputs.logits->slice(start_idx, batch_size);
    auto finished_mask     = inputs.finished_mask->slice(start_idx, finish_idx - start_idx);
    auto finished_mask_ptr = finished_mask->data<bool>();

    // return early when no sequence needs processing
    if (!std::any_of(finished_mask_ptr, finished_mask_ptr + batch_size, [](bool v) { return v; })) {
        return;
    }

    // mask all logits of the finished sequences except the eos token
    auto finished_mask_host = device_->allocateBuffer({DataType::TYPE_UINT8, {batch_size}, AllocationType::HOST});

    auto finished_mask_host_ptr = finished_mask_host->data<uint8_t>();
    memset(finished_mask_host_ptr, 0, batch_size * sizeof(uint8_t));

    for (size_t idx = 0; idx < batch_size; ++idx) {
        if (finished_mask_ptr[idx]) {
            finished_mask_host_ptr[idx] = 1;
        }
    }

    auto               finished_mask_device = device_->clone({*finished_mask_host, AllocationType::DEVICE});
    FinishedMaskParams params;
    params.finished_mask = finished_mask_device;
    params.end_token_id  = eos_token_id_;
    params.logits        = logits;

    finishedMaskLogits(params);
}

void MultiSeqLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    // do nothing
}

void MultiSeqLogitsProcessor::updateStatus(const rtp_llm::BufferPtr& new_tokens, int32_t num_new_tokens) {
    // do nothing
}

}  // namespace rtp_llm