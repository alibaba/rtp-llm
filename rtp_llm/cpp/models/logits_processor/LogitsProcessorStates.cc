#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"

#include <algorithm>
#include <utility>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#if USING_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

using namespace std;

namespace rtp_llm {

LogitsProcessorStates::LogitsProcessorStates() {};

namespace {

bool isProcessorApplied(const SamplerInputs& inputs, const SpecLogitsProcessorId& processor_id) {
    if (!processor_id.valid()) {
        return false;
    }
    return std::find(inputs.spec_applied_processors.begin(), inputs.spec_applied_processors.end(), processor_id)
           != inputs.spec_applied_processors.end();
}

void recordSpecTensorUseOnCurrentStream(const torch::Tensor& tensor) {
#if USING_CUDA
    if (tensor.defined() && tensor.is_cuda()) {
        c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(),
                                                      at::cuda::getCurrentCUDAStream(tensor.device().index()));
    }
#else
    (void)tensor;
#endif
}

}  // namespace

void LogitsProcessorStates::batchProcess(const SamplerInputs& inputs) {
    const bool has_spec_mask = inputs.phase == LogitsProcessorPhase::MTP_VERIFY && inputs.spec_vocab_mask_gpu.defined();
    if (has_spec_mask) {
        if (inputs.spec_mask_ready_event) {
            inputs.spec_mask_ready_event->block(cuda_graph::graphGetCurrentStream());
        }
        recordSpecTensorUseOnCurrentStream(inputs.spec_vocab_mask_gpu);
        RTP_LLM_CHECK_WITH_INFO(inputs.logits.dim() == 2 && inputs.spec_vocab_mask_gpu.dim() == 2,
                                "MTP spec mask and logits must both be 2-D");
        RTP_LLM_CHECK_WITH_INFO(inputs.logits.size(0) == inputs.spec_vocab_mask_gpu.size(0),
                                "MTP spec mask rows must match logits rows");
        RTP_LLM_CHECK_WITH_INFO(inputs.spec_vocab_mask_gpu.size(1) <= inputs.logits.size(1),
                                "MTP spec mask width must not exceed logits width");
        // The grammar/constraint mask is defined over the tokenizer's real
        // vocabulary. LM-head logits can include TP-alignment columns, which
        // are never valid tokens and must be masked in the same constrained
        // path. Keep the real-vocab mask compact and fill the suffix directly.
        const int64_t real_vocab_size = inputs.spec_vocab_mask_gpu.size(1);
        inputs.logits.narrow(/*dim=*/1, /*start=*/0, real_vocab_size)
            .masked_fill_(inputs.spec_vocab_mask_gpu, BaseLogitsProcessor::neg_inf);
        if (real_vocab_size < inputs.logits.size(1)) {
            inputs.logits
                .narrow(/*dim=*/1, /*start=*/real_vocab_size, /*length=*/inputs.logits.size(1) - real_vocab_size)
                .fill_(BaseLogitsProcessor::neg_inf);
        }
    }

    for (size_t i = 0; i < logits_processors_.size(); i++) {
        if (has_spec_mask && std::dynamic_pointer_cast<SpecLogitsProcessor>(logits_processors_[i]) != nullptr
            && i < processor_ids_.size() && isProcessorApplied(inputs, processor_ids_[i])) {
            continue;
        }
        if (draft_prefixes_[i].empty()) {
            logits_processors_[i]->process(inputs, intervals_[i].first, intervals_[i].second);
        } else {
            logits_processors_[i]->processSpeculative(
                inputs, intervals_[i].first, intervals_[i].second, draft_prefixes_[i]);
        }
    }
}

void LogitsProcessorStates::insert(
    const BaseLogitsProcessorPtr& ptr, size_t start, size_t finish, uint64_t stream_id, size_t processor_idx) {
    logits_processors_.push_back(ptr);
    intervals_.push_back(std::make_pair(start, finish));
    draft_prefixes_.emplace_back();
    processor_ids_.push_back({stream_id, processor_idx});
}

void LogitsProcessorStates::insertSpeculative(const BaseLogitsProcessorPtr& ptr,
                                              size_t                        start,
                                              size_t                        finish,
                                              std::vector<int32_t>          draft_prefix,
                                              uint64_t                      stream_id,
                                              size_t                        processor_idx) {
    logits_processors_.push_back(ptr);
    intervals_.push_back(std::make_pair(start, finish));
    draft_prefixes_.push_back(std::move(draft_prefix));
    processor_ids_.push_back({stream_id, processor_idx});
}

}  // namespace rtp_llm
