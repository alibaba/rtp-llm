#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"
#include "rtp_llm/cpp/models/context_parallel/RoundRobinProcessor.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"

namespace rtp_llm {

std::unique_ptr<IContextParallelProcessor> ContextParallelProcessorFactory::create(CPProcessorType type) {
    switch (type) {
        case CPProcessorType::ZIG_ZAG:
            return std::make_unique<ZigZagProcessor>();
        case CPProcessorType::ROUND_ROBIN:
            return std::make_unique<RoundRobinProcessor>();
        default:
            return std::make_unique<ZigZagProcessor>();
    }
}
}  // namespace rtp_llm
