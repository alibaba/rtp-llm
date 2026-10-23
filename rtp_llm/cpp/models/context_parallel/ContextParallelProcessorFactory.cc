#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"

namespace rtp_llm {

std::unique_ptr<IContextParallelProcessor> ContextParallelProcessorFactory::create(
    ProcessorType type, const ParallelismConfig& parallelism_config, bool prefer_local_hidden_states,
    bool rebuild_default_position_ids_on_prefix) {
    switch (type) {
        case ProcessorType::ZIG_ZAG:
            return std::make_unique<ZigZagProcessor>(parallelism_config, prefer_local_hidden_states, rebuild_default_position_ids_on_prefix);
        default:
            return std::make_unique<ZigZagProcessor>(parallelism_config, prefer_local_hidden_states, rebuild_default_position_ids_on_prefix);
    }
}
}  // namespace rtp_llm
