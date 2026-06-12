#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.h"

using namespace std;

namespace rtp_llm {

LogitsProcessorStates::LogitsProcessorStates(){};

void LogitsProcessorStates::setIntervalError(std::vector<std::optional<ErrorInfo>>& errors,
                                             const std::pair<size_t, size_t>&       interval,
                                             const ErrorInfo&                       error) {
    const size_t begin = std::min(interval.first, errors.size());
    const size_t end   = std::min(interval.second, errors.size());
    for (size_t row = begin; row < end; ++row) {
        if (!errors[row].has_value()) {
            errors[row] = error;
        }
    }
}

std::vector<std::optional<ErrorInfo>> LogitsProcessorStates::batchProcess(const SamplerInputs& inputs) {
    std::vector<std::optional<ErrorInfo>> processor_errors(inputs.batch_size);
    for (const auto& invocation : invocations_) {
        const auto& interval = invocation.interval;
        auto        error    = invocation.processor->process(inputs, interval.first, interval.second);
        if (error.has_value()) {
            setIntervalError(processor_errors, interval, error.value());
        }
    }
    return processor_errors;
}

void LogitsProcessorStates::insert(const BaseLogitsProcessorPtr& ptr, size_t start, size_t finish) {
    invocations_.push_back({ptr, std::make_pair(start, finish)});
}

}  // namespace rtp_llm
