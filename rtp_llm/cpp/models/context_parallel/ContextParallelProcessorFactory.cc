#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"

namespace rtp_llm {

std::unique_ptr<IContextParallelProcessor> ContextParallelProcessorFactory::create(ProcessorType type) {
    switch (type) {
        case ProcessorType::ZIG_ZAG:
            return std::make_unique<ZigZagProcessor>();
        default:
            return std::make_unique<ZigZagProcessor>();
    }
}
}  // namespace rtp_llm
