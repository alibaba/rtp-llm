#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"

namespace rtp_llm {

std::unique_ptr<IContextParallelProcessor> ContextParallelProcessorFactory::create(
    ProcessorType type, const ParallelismConfig& parallelism_config, int cp_chunk_alignment) {
    switch (type) {
        case ProcessorType::ZIG_ZAG:
            return std::make_unique<ZigZagProcessor>(parallelism_config, cp_chunk_alignment);
        default:
            return std::make_unique<ZigZagProcessor>(parallelism_config, cp_chunk_alignment);
    }
}
}  // namespace rtp_llm
