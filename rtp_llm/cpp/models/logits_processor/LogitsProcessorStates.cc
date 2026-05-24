#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"

#include <utility>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

using namespace std;

namespace rtp_llm {

LogitsProcessorStates::LogitsProcessorStates() {};

void LogitsProcessorStates::batchProcess(const SamplerInputs& inputs) {
    const bool has_spec_mask = inputs.phase == LogitsProcessorPhase::MTP_VERIFY && inputs.spec_vocab_mask_gpu.defined();
    if (has_spec_mask) {
        if (inputs.spec_mask_ready_event) {
            inputs.spec_mask_ready_event->block(cuda_graph::graphGetCurrentStream());
        }
        inputs.logits.masked_fill_(inputs.spec_vocab_mask_gpu, BaseLogitsProcessor::neg_inf);
    }

    for (size_t i = 0; i < logits_processors_.size(); i++) {
        if (has_spec_mask && std::dynamic_pointer_cast<SpecLogitsProcessor>(logits_processors_[i]) != nullptr) {
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

void LogitsProcessorStates::insert(const BaseLogitsProcessorPtr& ptr, size_t start, size_t finish) {
    logits_processors_.push_back(ptr);
    intervals_.push_back(std::make_pair(start, finish));
    draft_prefixes_.emplace_back();
}

void LogitsProcessorStates::insertSpeculative(const BaseLogitsProcessorPtr& ptr,
                                              size_t                        start,
                                              size_t                        finish,
                                              std::vector<int32_t>          draft_prefix) {
    logits_processors_.push_back(ptr);
    intervals_.push_back(std::make_pair(start, finish));
    draft_prefixes_.push_back(std::move(draft_prefix));
}

}  // namespace rtp_llm
