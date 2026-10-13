#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

class LogitsProcessorStates {
public:
    LogitsProcessorStates();
    virtual ~LogitsProcessorStates() {}

public:
    std::vector<std::optional<ErrorInfo>> batchProcess(const SamplerInputs& inputs);
    void                                  insert(const BaseLogitsProcessorPtr& ptr, size_t start, size_t finish);

private:
    static void setIntervalError(std::vector<std::optional<ErrorInfo>>& errors,
                                 const std::pair<size_t, size_t>&       interval,
                                 const ErrorInfo&                       error);

    struct Invocation {
        BaseLogitsProcessorPtr    processor;
        std::pair<size_t, size_t> interval;
    };

    std::vector<Invocation> invocations_;
};

typedef std::shared_ptr<LogitsProcessorStates> LogitsProcessorStatesPtr;

}  // namespace rtp_llm
