#pragma once

#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

class BatchGrammarScratchPool {
public:
    BatchGrammarScratchPool() = default;
    ~BatchGrammarScratchPool();

    BatchGrammarScratchPool(const BatchGrammarScratchPool&)            = delete;
    BatchGrammarScratchPool& operator=(const BatchGrammarScratchPool&) = delete;

    struct Slot {
        torch::Tensor               pinned_cpu;
        torch::Tensor               device_gpu;
        int64_t                     capacity_ints = 0;
        int                         device_index  = -1;
#if USING_CUDA
        std::optional<torch::Event> done_event;
        bool                        recorded = false;
#endif
    };

    std::shared_ptr<Slot> acquire(int device_index, int64_t required_ints);
    void                  retire(std::shared_ptr<Slot> slot);

private:
    static constexpr size_t           kMaxSlots = 64;
    std::mutex                        mutex_;
    std::deque<std::shared_ptr<Slot>> pool_;
};

class LogitsProcessorStates {
public:
    LogitsProcessorStates();
    virtual ~LogitsProcessorStates() {}

public:
    std::vector<std::optional<ErrorInfo>> batchProcess(const SamplerInputs& inputs);
    std::vector<std::optional<ErrorInfo>> batchProcess(const SamplerInputs&     inputs,
                                                       BatchGrammarScratchPool* scratch_pool);
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
