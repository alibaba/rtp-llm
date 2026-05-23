#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"

#include <utility>

using namespace std;

namespace rtp_llm {

LogitsProcessorStates::LogitsProcessorStates() {};

void LogitsProcessorStates::batchProcess(const SamplerInputs& inputs) {
    for (size_t i = 0; i < logits_processors_.size(); i++) {
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
