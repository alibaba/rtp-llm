#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"

#include <utility>

using namespace std;

namespace rtp_llm {

LogitsProcessorStates::LogitsProcessorStates() {};

void LogitsProcessorStates::batchProcess(const SamplerInputs& inputs) {
    for (size_t i = 0; i < logits_processors_.size(); i++) {
        if (inputs.spec_applied_processors.count(logits_processors_[i].get())) {
            continue;
        }
        // process() never throws; errors are stored on the processor (hasError()/error())
        // and pulled by the owning GenerateStream on its next update tick.
        logits_processors_[i]->process(inputs, intervals_[i].first, intervals_[i].second);
    }
}

void LogitsProcessorStates::insert(const BaseLogitsProcessorPtr& ptr, size_t start, size_t finish) {
    logits_processors_.push_back(ptr);
    intervals_.push_back(std::make_pair(start, finish));
}

}  // namespace rtp_llm
