#pragma once

#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

class LogitsProcessorStates {
public:
    LogitsProcessorStates();
    virtual ~LogitsProcessorStates() {}

public:
    void batchProcess(const SamplerInputs& inputs);
    void insert(const BaseLogitsProcessorPtr& ptr, size_t start, size_t finish);
    void insertSpeculative(const BaseLogitsProcessorPtr& ptr,
                           size_t                        start,
                           size_t                        finish,
                           std::vector<int32_t>          draft_prefix);

private:
    std::vector<BaseLogitsProcessorPtr>    logits_processors_;
    std::vector<std::pair<size_t, size_t>> intervals_;
    std::vector<std::vector<int32_t>>      draft_prefixes_;
};

typedef std::shared_ptr<LogitsProcessorStates> LogitsProcessorStatesPtr;

}  // namespace rtp_llm
