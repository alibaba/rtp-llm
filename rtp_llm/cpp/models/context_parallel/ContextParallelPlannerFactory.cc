#include "rtp_llm/cpp/models/context_parallel/ContextParallelPlannerBase.h"
#include "rtp_llm/cpp/models/context_parallel/ZigzagPlanner.h"

namespace rtp_llm {

std::unique_ptr<IContextParallelPlanner> ContextParallelPlannerFactory::create(PlannerType type) {
    switch (type) {
        case PlannerType::ZIG_ZAG:
            return std::make_unique<ZigZagPlanner>();
        default:
            return std::make_unique<ZigZagPlanner>();
    }
}
}  // namespace rtp_llm
