#pragma once

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/dataclass/SampleInfos.h"
#include "rtp_llm/cpp/dataclass/Query.h"
#include "rtp_llm/cpp/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

class LogitsProcessorStates {
public:
    LogitsProcessorStates();
    virtual ~LogitsProcessorStates() {}

public:
    void batchProcess(const SamplerInputs& inputs);
    void insert(const BaseLogitsProcessorPtr& ptr, size_t start, size_t finish);

private:
    std::vector<BaseLogitsProcessorPtr>    logits_processors_;
    std::vector<std::pair<size_t, size_t>> intervals_;
};

typedef std::shared_ptr<LogitsProcessorStates> LogitsProcessorStatesPtr;

}  // namespace rtp_llm