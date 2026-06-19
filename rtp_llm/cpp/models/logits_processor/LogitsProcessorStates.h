#pragma once

#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class LogitsProcessorStates {
public:
    // Caller-provided sink for per-processor errors. Empty by default; populated
    // via insert() so the runner stays decoupled from GenerateStream and avoids
    // the (logits_processor → stream → logits_processor) circular Bazel dep.
    using ErrorSink = std::function<void(ErrorCode, const std::string&)>;

    LogitsProcessorStates();
    virtual ~LogitsProcessorStates() {}

public:
    void batchProcess(const SamplerInputs& inputs);
    void insert(const BaseLogitsProcessorPtr& ptr,
                size_t                        start,
                size_t                        finish,
                uint64_t                      stream_id     = SpecLogitsProcessorId::kInvalidStreamId,
                size_t                        processor_idx = std::numeric_limits<size_t>::max(),
                ErrorSink                     error_sink    = {});

private:
    std::vector<BaseLogitsProcessorPtr>    logits_processors_;
    std::vector<std::pair<size_t, size_t>> intervals_;
    std::vector<SpecLogitsProcessorId>     processor_ids_;
    std::vector<ErrorSink>                 error_sinks_;
};

typedef std::shared_ptr<LogitsProcessorStates> LogitsProcessorStatesPtr;

}  // namespace rtp_llm
